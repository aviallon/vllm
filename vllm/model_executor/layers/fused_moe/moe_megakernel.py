# SPDX-License-Identifier: Apache-2.0
"""Qwen3MoEMegakernel — forward pass du bloc MoE routé en kernel(s) fusionnés.

Campagne MEGAKERNEL MoE INT8 (2026-10-27), design + analyse du triangle
parallélisme/poids/réduction :
`research/moe-megakernel-campaign-2026-10-27/REPORT.md` §5.

Deux stratégies (gate `VLLM_MOE_MEGAKERNEL_STRATEGY`) :

* **``pair`` (v1, défaut)** — ``qwen3_moe_pair_fusion_kernel`` : gemm1 →
  silu+mul → quant per-token → gemm2 fusionnés en UN lancement par couche,
  intermédiaires en registres (acc1 [BM,2I] résident, contrat quant per-token
  IDENTIQUE au livré), sortie ``intermediate_cache3`` bf16 + ``moe_sum``
  inchangé. Grid = (blocs × tuiles de sortie) ; le poids w1[e] est relu par
  les tuiles de sortie (filtre L2 — la réutilisation est le pari mesuré au
  microbench). Supprime : quant_a1 (si x bf16) + gemm1 + silu_q + gemm2 →
  1 lancement ; les allers-retours HBM cache1/cache2 disparaissent.

* **``atomic`` (v2, la vision complète)** — ``qwen3_moe_megakernel`` : le
  pipeline entier + somme pondérée experts par atomique fp32 locale +
  finalize (cast bf16) — cache1/cache2/cache3 ET `moe_sum` supprimés.
  Grid = (blocs × chunks intermédiaires) ; la réduction sur les chunks est
  l'atomique. ⚠ gfx90a abaisse les `atomicrmw fadd` fp32 en boucles CAS
  (`global_atomic_cmpswap` — mesuré au codegen) : le coût est à MESURER au
  microbench avant armer (piège atomics fine-grained mémoire PAIRE = NOP
  silencieux sur MI200 PCIe : l'accumulateur est local, le piège ne
  s'applique pas).  **RÉGRESSÉ au wall (2026-10-29) : +11/+23/+60/+54 % en
  c1/c8/c32/c128 — la masse CAS O(M·topk·n_chunks·K) est le facteur
  limitant (M32-M128-EXTENSION.md §6).**

* **``tree`` (v3, ANTIATOM-DESIGN.md — le bras de la campagne
  anti-atomics)** — **zéro `tl.atomic_add`, zéro memset, zéro finalize** :
  réduction à deux niveaux sans accumulateur partagé.  pass1 (grille
  blocs×chunks) : gemm1 chunk → silu fp32 → ``mid_ws [M·topk, I]`` fp32 ;
  pass2 (grille blocs×tuiles K) : prologue de quant PER-TOKEN (amax sur I
  entier — contrat livré verbatim) puis UN dot pleine-I → rangées de
  PAIRES ``ws [M·topk, K]`` bf16 (un seul écrivain par rangée) ; pass3
  (grille M×tuiles K) : ``out[t] = Σ_s ws[t·topk+s]`` (fp32 → bf16).  Le
  workspace ``ws`` est le ``cache3`` livré : la réduction des experts reste
  un aller-retour HBM (le triangle du REPORT §5 — sans atomics c'est le
  seul plancher, la voie « propriétaire unique par rangée » meurt du
  parallélisme aux petits M : 1 CTA à c1).  Gain vs ``atomic`` = toute la
  masse CAS + memset + finalize + le gemm2 chunké (mesuré : ~40 % du pic
  de bande) remplacé par le dot pleine-I livré.  **Déterminisme bit-exact
  run-to-run** (ordre de sommation fixe) — contrat testé.  Écart nommé vs
  le livré : silu en fp32 (le livré arrondit en bf16) ; sinon même contrat
  de quant (per-token, amax entier).

Le triangle (documenté §5 du REPORT) : la parallélisme de streaming c1 vient
de découper les colonnes des poids ; une CTA fusionnée couvre soit la largeur
intermédiaire entière (gros travail/CTA, réutilisation L2 de w1), soit un
chunk (réduction inter-CTA = atomiques). Les deux stratégies mesurent les deux
branches du triangle ; le wall est l'oracle.

Contrats numériques — écarts NOMMÉS vs le chemin livré (les gates de
cohérence par tâche jugent, doctrine 2026-10-24) :

1. silu sur accumulateur **fp32** (le chemin livré arrondit en bf16 d'abord) ;
2. stratégie ``atomic`` seulement : quant de l'intermédiaire par
   **(rangée, chunk)** au lieu de per-token sur I entier (le CTA ne voit que
   son chunk) — composé en fp32 dans l'épilogue ; la stratégie ``pair`` garde
   le contrat per-token EXACT ;
3. stratégie ``atomic`` : somme des contributions par **atomique fp32**
   (ordre non déterministe au dernier ULP entre runs — gate de déterminisme).

Écarts d'ASSOCIATION flottante (classe 1 ULP, nommés — revue mimo W1) :
l'ordre des multiplications diffère du livré sans changer les valeurs
significatives : gemm1 compose `(acc·s1g)·sx` là où le livré fait
`((acc·xs)·w1_scale)` ; gemm2 compose `(acc2·(s_row·wt))·s2` contre
`((acc2·s2)·w2s)·wt` ; les sommes partielles int8 par chunk s'accumulent
en fp32 dans un autre ordre.  Classe 1 ULP, non jugable à 3 %.

Identiques : quant de x (expressions de `_per_token_quant_int8`), échelles
per-channel, poids routé fp32 avant cast, sortie bf16 [M, K].

Gates : `VLLM_MOE_MEGAKERNEL` (défaut OFF = le chemin livré verbatim) ;
`VLLM_MOE_MEGAKERNEL_STRATEGY` = `pair` | `atomic` | `tree` (défaut `pair`).

TOUR 2 — incréments 2+3 (spec verbatim :
`research/moe-megakernel-campaign-2026-10-27/TOUR2-DESIGN.md` §6) :

* **INC 2 (gate `VLLM_MOE_MEGAKERNEL_NORM`, défaut OFF)** — norm+quant v2
  dans le PROLOGUE du kernel (mode ROUTING, 3 passes sur rangées L2-hot,
  BM=1) : `x_f → s = fp32(x_f)+fp32(res) → res_out=bf16(s) → var →
  y_bf16 → amax → scale=amax/127 (SANS clamp) → q=clamp(nearbyint(y_bf16·
  inv_s))`.  Le contrat numérique est le natif v2 VERBATIM (store
  `ar-rms-quant-v2-spec`, §3.2) — ⚠ **PAS** le clamp FLT_EPSILON (la
  « mauvaise référence » de v1) ; `nearbyint` half-even (libdevice), PAS le
  `round` half-away de `_per_token_quant_int8` — écart NOMMÉ.  Le router voit
  `y_bf16` (la sortie de norm bf16, comme le router livré) ; l'échelle de
  déquant de gemm1 = `scale`.
* **INC 3 (gate `VLLM_MOE_MEGAKERNEL_AR`, défaut OFF, exige NORM)** — phase 0
  AR in-kernel (primitives PROUVÉES, store `triton-3way-cap` 2026-10-22 +
  `ar_norm_quant.py` phase 0) : publish = **store LOCAL** de ma rangée (⚠ les
  stores pairs nus sont une ANOMALIE mesurée — jamais), séquence dérivée par
  **slot** in-kernel (jamais un scalar arg : cuit dans le graphe = replays
  périmés silencieux), xchg release/sys de MON flag, spin `!= seq` (fail
  loud : hang > corruption silencieuse) acquire/sys, puis **peer-loads**
  seuls (i64 cast load-bearing).  ⚠ DESIGN FINDING de la grille chunked : la
  table de flags est par **(rangée, chunk)** `[M, n_chunks]` int32, éléments
  paddés **256 B** — pas par rangée comme le 3-way : avec la grille
  (token × chunk) un flag par rangée ferait xchg la séquence 32×
  (corruption).  Chaque CTA (t,c) xchg SON flag (t,c), spin sur le flag PAIR
  (t,c) — symétrique et sans race.

Contrats INC 2/3 : `res_out` exige un tampon **DISTINCT** de `res` (la grille
chunked relit `res` pendant que d'autres CTAs écrivent — l'in-place du 3-way
est la seule chose que la structure interdit) ; sortie int8 (contrat v2')
inchangée.  Repli = chemin livré (les gates s'empilent ROUTING→NORM→AR ;
sans les entrées du prologue le site d'appel moteur replie sur la chaîne
livrée — le raccordement au graphe réel est l'étape nommée restante).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, NamedTuple

import torch
import triton
import triton.language as tl

from vllm import envs

logger = logging.getLogger(__name__)

__all__ = [
    "qwen3_moe_pair_fusion_kernel",
    "qwen3_moe_megakernel",
    "qwen3_moe_mk_tree_pass1_kernel",
    "qwen3_moe_mk_tree_pass2_kernel",
    "moe_megakernel_reduce_kernel",
    "moe_megakernel_finalize",
    "fused_moe_megakernel",
    "moe_megakernel_enabled",
    "moe_megakernel_strategy",
    "moe_megakernel_flags",
    "moe_megakernel_eligible",
    "moe_megakernel_scales_ok",
    "moe_megakernel_record_call",
    "moe_megakernel_call_stats",
    "normalize_w_scale",
    "MoeMegakernelComm",
    "FLAG_ELEMS",
    "AR_MAX_CTAS",
]

# Plafond de co-résidence de la grille AR (inc 3) : le spin inter-rangs
# (`!= seq`, fail loud) exige que TOUTES les CTA de la grille finissent par
# tourner sur CHAQUE rang — une CTA en attente de scheduling derrière des
# CTAs qui spin = deadlock de scheduling entre rangs (les rangs n'ont pas
# les mêmes CTA résidentes au même instant).  Grille = M × n_chunks CTAs ;
# au-delà du plafond = repli sur le chemin livré (compté + logué).  96 =
# conservateur vs l'occupation mesurée au smoke (à ré-étalonner avec le
# compteur d'occupation du build inc 2/3 — R2 : instrument nommé).
def _ar_max_ctas() -> int:
    # Borné par la résidence MESURÉE (smoke : shared 65 536 → 1 WG/CU →
    # 104 CU sur MI210) : monter la variable d'env au-delà convertirait le
    # fail-loud en hang (revue GLM m1) — clampé, jamais silencieux.
    req = _env_int("VLLM_MOE_MEGAKERNEL_AR_MAX_CTAS", 96)
    if req > AR_RESIDENT_CTAS_HARD:
        _log_once(
            "mk-ar-max-ctas-clamped",
            "[moe-megakernel] AR_MAX_CTAS=%d > résidence mesurée %d — "
            "bridé à %d (au-delà : deadlock de scheduling entre rangs)",
            req, AR_RESIDENT_CTAS_HARD, AR_RESIDENT_CTAS_HARD,
        )
        return AR_RESIDENT_CTAS_HARD
    return req


AR_MAX_CTAS = 96
AR_RESIDENT_CTAS_HARD = 104


# ---------------------------------------------------------------------------
# lecture des gates — CANAUX DOUBLES (piège « déclaré mais jamais consommé »)
# ---------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    """Gate booléenne : ``os.environ`` d'abord, puis l'attribut ``envs``.

    ⚠ Les variables ``VLLM_MOE_MEGAKERNEL*`` ne sont PAS enregistrées dans
    ``vllm.envs`` : le ``getattr(envs, ...)`` seul les lit donc comme
    TOUJOURS False — la gate existerait, l'env serait posé, et rien ne
    s'allumerait JAMAIS.  C'est exactement ce qui a laissé le bras muet au
    wall A/B du 2026-10-28 (warning moteur « Unknown vLLM environment
    variable » = preuve que l'attribut n'existe pas).  Le canal
    ``os.environ`` (opérateur) est prépondérant ; l'attribut ``envs`` reste
    le canal de test.  Même contrat que ``dense_mfma_enabled()``.
    """
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() in ("1", "true")
    return bool(getattr(envs, name, False))


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw.strip()
    return str(getattr(envs, name, default))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return int(raw.strip())
    return int(getattr(envs, name, default))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@triton.jit
def _round_i8(x):
    """Rounding du contrat `_per_token_quant_int8` (libdevice.round,
    half-away-from-zero).  Chemin livré — inchangé."""
    return tl.extra.hip.libdevice.round(x).to(tl.int8)


@triton.jit
def _rint_clamp_i8(v):
    """Contrat NATIF v2 `dynamic_scaled_int8_quant` : `nearbyint` half-even
    (libdevice — PAS `round` half-away) + saturation [-128, 127].
    ⚠ L'écart nommé tour 2 : le prologue NORM passe à ce contrat ; le chemin
    livré `_per_token_quant_int8` garde `_round_i8` (TOUR2-DESIGN §6.1)."""
    q = tl.extra.hip.libdevice.nearbyint(v)
    return tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)


# Éléments int32 entre deux flags de la table AR : 256 B de padding par
# (rangée, chunk) — la spec tour 2 (§6.2) exige l'absence de ligne de cache
# partagée entre slots, sinon les CTAs d'une même rangée se corrompent la
# séquence.  (Le 3-way utilisait 128 B ; ici c'est 256 B, verbatim spec.)
FLAG_ELEMS = 256 // 4


# INC-A (tour 3, mandat MFMA — TOUR3-FULL-FORWARD-DESIGN.md §3.4) :
# politique de tuiles qui abaisse TOUS les dots en v_mfma (jamais v_dot4).
# Mesure tour 1 : BM=1 tombe en v_dot4 (6-19x plus lent) ; le padding BM=16
# (lignes pad masquées) a déjà divisé par 2 le temps du kernel pair.  Règle :
# tout tl.dot a BM>=16 ET une dim de contraction >=16.  CI est la
# contraction de gemm2 (et la tuile N de gemm1) : CI=8 tombait en micro-dots
# v_dot4 ; CI=32 le remet en MFMA — et AMÉLIORE le contrat chunké (échelles
# par (rangée, chunk) plus proches du per-token livré).  MK_BN2 borne l'acc2
# de gemm2 côté sortie : l'acc2 [BM,K] plein de la v1 dépassait le fichier
# de registres à BM=16 (512/lane = cap = spill).
MK_BM = 16
MK_CI = 32
MK_BN2 = 64
# configs `tree` (sweep moemk_tree_sweep_bench, MI210, shapes réels) :
# BKO=256 + num_warps=2/num_stages=1 = les configs mesurées ; CI=64 (n_chunks
# = 4) était la meilleure pour pass1 ; BN2=64 = la tuile de sortie du kernel
# livré (le dot pleine-I de pass2 en a besoin pour son budget de registres).
MK_TREE_BKO = 256
MK_TREE_CI = 64
# Stockage bf16 de l'intermédiaire post-silu (passe tuiles, 2026-10-29) :
# moitié moins de trafic sur mid_ws, même contrat de quant per-token.
# MESURÉ NEUTRE en vitesse (tilepass sweep F) et légèrement lossy —
# resté OFF ; le knob est testé (deux canaux) pour toute reprise.
MK_TREE_MID_BF16 = False
# Seuil de régime pass2 groupé (passe de tuiles 2026-10-29) :
# PLAFONDS-TUILES.md §2/§4 — au-delà, BN2=128 + warps 4 + stages 3
# (mesuré : M=128 424->383 µs, M=48 331->312 ; sous le seuil la config
# historique BN2=64/warps 2/stages 1 est optimale : M=32 246 vs 255).
MK_TREE_GROUPED_BIGM = 40
# Forme de RÉFÉRENCE du repli : la config r10 (tree_default_config) n'est
# mesurée/compilable que sur la forme decode locale TP=2 (I=256).  Toute
# autre forme exige une entrée de table (sinon : repli chaîne livrée —
# règle de couverture de moe_megakernel_eligible, 2026-10-30).
MK_TREE_FALLBACK_I = 256


def tree_pass2_config(M: int, naive: bool) -> tuple[int, int | None, int, int]:
    """Régime de lancement de pass2 (passe de tuiles, mesures
    `moemk_tilepass_bench`/`moemk_tilepass_probe2`, log de preuve
    evidence/tilepass-*.log).  Retourne (BN2, LOCAL_TILES, num_warps,
    num_stages) ; LOCAL_TILES=None => n_tiles entier (grille 1-D blocs),
    1 => grille 2-D (blocs, tuiles) du régime naive.  FONCTION PARTAGÉE
    launcher/tests — l'égalité bit-exacte harnais↔launcher est prouvée par
    `test_tree_pipeline_matches_launcher` (garde anti-harnais-faux).

    Régimes (un seul bloc de config par régime ; le régime M≥40 est un
    TRIPLET (BN2, nw, ns) — le point isolé (64,4,3) mesure ~5 µs de part
    BN2, le reste est ns 2→3 ; probe3 P1) :
      * naive (M·topk·4 ≤ E)      : 64, 1, 2, 1  — INCHANGÉ (c1/c8) ;
      * grouped M < BIGM (c32)    : 64, None, 2, 1 — INCHANGÉ (mesuré
        optimal : BN2=128 perd ~10 µs à M=32) ;
      * grouped M ≥ BIGM (c128)   : 128, None, 4, 3 — NOUVEAU (−41 µs
        @M=128, −18 @M=48 ; « ns=3 compile pour pass2 et échoue en LDS
        pour pass1 » = observation mesurée, pas un mécanisme vérifié).
    ⚠ Le champ LOCAL_TILES n'a que deux valeurs : 1 (grille 2-D naive) et
    None (= n_tiles entier, grille 1-D) — et il est CONSOMMÉ par les DEUX
    consommateurs (launcher + harnais de tests le résolvent en n_tiles) :
    un futur renvoi explicite (ex. 8) serait donc réellement appliqué
    (revue deepseek m6 — le champ n'est pas mort).
    """
    if naive:
        return 64, 1, 2, 1
    if M < MK_TREE_GROUPED_BIGM:
        return 64, None, 2, 1
    return 128, None, 4, 3


# ---------------------------------------------------------------------------
# TABLE INDEXÉE PAR M (2026-10-30) — configs de lancement `tree` par régime
# ---------------------------------------------------------------------------
# Miroir du mécanisme des tables MoE LIVRÉES (`get_moe_configs` de
# `fused_moe.py`) : un JSON par forme LOCALE des poids, nommé à la convention
# livrée `E={E},N={N},device_name={dev},dtype=int8_w8a8,…`, cherché d'abord
# dans `VLLM_TUNED_CONFIG_FOLDER` puis dans le dossier `configs/` du paquet.
# Ici la clé interne est K (comme livré) et la valeur est une LISTE
# d'entrées de régime indexées par M (bornes `m_max`, première correspondance
# gagnante — la même sémantique que le croisement mesuré M≈36-40).
#
# Champ d'une entrée (TOUS consommés — canari
# `test_tree_table_entry_fields_consumed`) :
#   m_max     : borne haute du régime (None = au-delà) ; naive flag filtre
#   BM        : tuile de rangées (align grouped + grilles pass1/pass2)
#   BKO/CI    : tuiles pass1 (K/intermédiaire)
#   nw1/ns1   : pass1 num_warps/num_stages
#   BN2/LOCAL_TILES : pass2 tuile de sortie / tuiles par CTA
#               (None = n_tiles entier, grille 1-D ; naive force 1)
#   nw2/ns2   : pass2 num_warps/num_stages
#
# Toute entrée absente/invalide → REPLI = config r10 (`tree_default_config`),
# logué (fail loud sans casser le service) — la table ne casse jamais le
# chemin mesuré.
class TreeTileConfig(NamedTuple):
    BM: int
    BKO: int
    CI: int
    nw1: int
    ns1: int
    BN2: int
    LOCAL_TILES: int | None
    nw2: int
    ns2: int
    source: str


_ENTRY_KEYS = ("m_max", "naive", "BM", "BKO", "CI", "nw1", "ns1",
               "BN2", "LOCAL_TILES", "nw2", "ns2")


def tree_default_config(M: int, naive: bool, source: str) -> TreeTileConfig:
    """Config COURANTE (r10) — le repli garanti de la table.  Pass1 =
    constants mesurées (BM=16, BKO=256, CI=64, nw/ns=2/1) ; pass2 = le
    régime `tree_pass2_config` (fonction partagée launcher/tests r10)."""
    bn2, lt, nw2, ns2 = tree_pass2_config(M, naive)
    return TreeTileConfig(MK_BM, MK_TREE_BKO, MK_TREE_CI, 2, 1,
                          bn2, lt, nw2, ns2, source)


def tree_config_file_name(E: int, N: int) -> str:
    """Nom de table à la convention LIVRÉE (`get_config_file_name`), suffixe
    `megakernel-tree` pour ne jamais entrer en collision avec la table du
    kernel fused_moe (même dossier `configs/`)."""
    from vllm.utils.platform_utils import get_device_name_as_file_name
    return (f"E={E},N={N},device_name={get_device_name_as_file_name()},"
            "dtype=int8_w8a8,megakernel-tree.json")


_TREE_TABLE_CACHE: dict[tuple[int, int], tuple[dict | None, str]] = {}


def tree_config_cache_clear() -> None:
    """Vide le cache de tables (tests / changement de table à chaud)."""
    _TREE_TABLE_CACHE.clear()


def get_tree_config_table(E: int, N: int) -> tuple[dict | None, str]:
    """(table K->entrées, chemin source) — lookup identique aux tables
    livrées : `VLLM_TUNED_CONFIG_FOLDER` d'abord (priorité user), puis le
    dossier `configs/` du paquet.  (None, chemin) si absente."""
    key = (E, N)
    if key in _TREE_TABLE_CACHE:
        return _TREE_TABLE_CACHE[key]
    json_file_name = tree_config_file_name(E, N)
    paths = []
    user_folder = getattr(envs, "VLLM_TUNED_CONFIG_FOLDER", None)
    if user_folder:
        paths.append(os.path.join(user_folder, json_file_name))
    paths.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "configs", json_file_name))
    table: dict | None = None
    source = paths[-1]
    for path in paths:
        if os.path.exists(path):
            source = path
            with open(path) as f:
                raw = json.load(f)
            raw.pop("triton_version", None)
            raw.pop("comment", None)
            table = {int(k): v for k, v in raw.items()}
            n_ent = sum(len(v) for v in table.values())
            _log_once(
                f"mk-tree-table-{E}-{N}",
                "[moe-megakernel] tree table chargée (E=%d,I=%d) : %d "
                "entrée(s) depuis %s (K: %s)",
                E, N, n_ent, path, sorted(table),
            )
            break
    _TREE_TABLE_CACHE[key] = (table, source)
    return table, source


def _tree_entry_ok(ent: Any) -> bool:
    """Validation fail loud d'une entrée de table (champs + types)."""
    if not isinstance(ent, dict):
        return False
    if any(k not in ent for k in _ENTRY_KEYS):
        return False
    for k in ("BM", "BKO", "CI", "nw1", "ns1", "BN2", "nw2", "ns2"):
        v = ent[k]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            return False
    if ent["LOCAL_TILES"] is not None:
        v = ent["LOCAL_TILES"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            return False
    m_max = ent["m_max"]
    if m_max is not None and (not isinstance(m_max, int)
                              or isinstance(m_max, bool) or m_max < 1):
        return False
    return isinstance(ent["naive"], bool)


def tree_select_config(M: int, naive: bool, E: int, I: int, K: int
                       ) -> TreeTileConfig:
    """SÉLECTION de config de lancement `tree` — table indexée par M, clé =
    forme locale des poids (E, N=I), repli = config r10.  Fonction PURE
    (M, naive, E, I, K) : le site moteur (align grouped) et le launcher
    appellent la MÊME sélection — toute désynchronisation est attrapée par
    l'assert align↔grille du launcher (fail loud).

    Observabilité (leçon gate muette) : la config choisie est portée par la
    ligne `tree regime` du launcher (src=table|repli:…) et le chargement de
    la table est logué une fois par forme."""
    if not _env_flag("VLLM_MOE_MEGAKERNEL_TREE_TABLE"):
        return tree_default_config(M, naive, "repli:knob-off")
    table, source = get_tree_config_table(E, I)
    if table is None:
        _log_once(
            f"mk-tree-table-miss-{E}-{I}",
            "[moe-megakernel] tree table absente (%s) — config r10 (repli)",
            source,
        )
        return tree_default_config(M, naive, "repli:table-absente")
    entries = table.get(K)
    if not entries:
        logger.warning_once(
            "moe-megakernel: tree table sans entrée K=%d (%s) — config r10 "
            "(repli)", K, source)
        return tree_default_config(M, naive, "repli:k-absent")
    for ent in entries:
        if not _tree_entry_ok(ent):
            logger.warning_once(
                "moe-megakernel: entrée de tree table invalide (%s) — "
                "config r10 (repli)", source)
            return tree_default_config(M, naive, "repli:entree-invalide")
        if bool(ent["naive"]) != naive:
            continue
        if ent["m_max"] is not None and M > ent["m_max"]:
            continue
        return TreeTileConfig(
            ent["BM"], ent["BKO"], ent["CI"], ent["nw1"], ent["ns1"],
            ent["BN2"], ent["LOCAL_TILES"], ent["nw2"], ent["ns2"],
            "table")
    return tree_default_config(M, naive, "repli:entree-absente")


def moe_megakernel_align_block_m(M: int, naive: bool, E: int, I: int,
                                 K: int) -> int:
    """BLOCK_SIZE_M du tri grouped (`_prepare_expert_assignment`) — le site
    moteur DOIT aligner avec LE MÊME BM que la grille de lancement, sinon
    les blocs ne mappent pas les paires (garde : assert align↔grille dans le
    launcher).  Non-tree : MK_BM (comportement r10 inchangé)."""
    if moe_megakernel_strategy() != "tree":
        return MK_BM
    return tree_select_config(M, naive, E, I, K).BM


@triton.jit
def _mk_norm_chunk(x_rows, peer_rows, res_rows, norm_w_ptr,
                   offs, valid, rstd,
                   K: tl.constexpr, AR: tl.constexpr):
    """Chunk (s, y_bf16) de la chaîne norm+quant v2 (contrat §3.2 verbatim) :
    xf = bf16(fp32(x_self)+fp32(x_peer)) (AR) ou x tel quel ;
    s  = fp32(xf) + fp32(res)  (residual add EN FP32) ;
    y_bf16 = bf16(s · rstd · w_fp32)  (unique arrondi de sortie de norm).
    Retourne (s, y_bf16) — `s` NON arrondi (var = mean(s·s) en fp32)."""
    m = valid[:, None] & (offs < K)[None, :]
    xk = tl.load(x_rows[:, None] + offs[None, :], mask=m, other=0.0)
    if AR:
        pb = tl.load(peer_rows[:, None] + offs[None, :], mask=m, other=0.0)
        xf = (xk.to(tl.float32) + pb.to(tl.float32)).to(tl.bfloat16)
    else:
        xf = xk
    rs = tl.load(res_rows[:, None] + offs[None, :], mask=m, other=0.0)
    s = xf.to(tl.float32) + rs.to(tl.float32)
    wc = tl.load(norm_w_ptr + offs, mask=offs < K, other=0.0)
    yb = ((s * rstd) * wc.to(tl.float32)).to(tl.bfloat16)
    return s, yb


@triton.jit
def _mk_ar_rendezvous(x_row_base, slot_base, row, chunk, flag_p, pflag_p,
                      n_chunks: tl.constexpr, flag_elems: tl.constexpr,
                      K: tl.constexpr, BLOCK_H: tl.constexpr):
    """Phase 0 AR (inc 3) — protocole en deux moitiés :

    * moitié PUBLICATION (store `triton-3way-cap` 2026-10-22 +
      `ar_norm_quant.py` phase 0) : séquence dérivée PAR SLOT in-kernel
      (jamais un scalar arg : cuit dans le graphe = replays périmés
      silencieux), publish LOCAL dans le tampon de parité (⚠ les stores
      pairs nus sont une ANOMALIE mesurée — jamais), xchg release/sys de
      MON flag, spin `!= seq` (fail loud) acquire/sys sur le flag pair
      avec un RMW (non hoistable).
    * moitié CONSOMMATION (revue GLM 2026-10-27 B1) : **double-tampon
      ping-pong** — le publish de l'appel N écrit le tampon `seq & 1` et
      les peer-loads de l'appel N lisent le tampon `seq & 1` du pair :
      l'appel N+1 (deux côtés) travaille dans l'AUTRE tampon.  Pas de
      course de réutilisation entre appels, sans handshake supplémentaire.
      Complétude : l'appel N+2 du pair (seul à pouvoir réécrire mon tampon
      de parité N) est bloqué par SON spin `!= seq` de l'appel N+1 tant que
      mon flag N+1 n'est pas xchgé — or mon appel N+1 n'existe qu'après
      mon appel N (ordre de flux) : l'avance du pair est plafonnée à <1
      appel.  (Un ack de fin par slot — variante (a) — s'est révélé
      INSUFFISANT : la ressource est par RANGÉE, lue par toutes les CTA
      (t,·) du pair, alors que l'ack est par (rangée, chunk) — attrapé par
      le test à x varié : res_out bit-exact mais q déchiré en fin de
      kernel.  C'est le test falsifiable de la revue GLM M1 qui l'a vu.)

    Retourne `seq` (l'appelant en déduit le tampon de parité des
    peer-loads).
    """
    f_off = (row * n_chunks + chunk) * flag_elems
    cur = tl.atomic_add(flag_p + f_off, 0, sem="acquire", scope="sys")
    seq = cur + 1
    # publish LOCAL dans MON tampon de parité
    slot_row = slot_base + (row * 2 + (seq & 1)) * K
    for base in range(0, K, BLOCK_H):
        offs = base + tl.arange(0, BLOCK_H)
        m = offs < K
        xv = tl.load(x_row_base + offs, mask=m, other=0.0)
        tl.store(slot_row + offs, xv, mask=m)
    # séquence par slot + release
    tl.atomic_xchg(flag_p + f_off, seq, sem="release", scope="sys")
    # moitié PUBLICATION : le pair a-t-il PUBLIÉ ?
    v = tl.atomic_add(pflag_p + f_off, 0, sem="acquire", scope="sys")
    while v != seq:
        v = tl.atomic_add(pflag_p + f_off, 0, sem="acquire", scope="sys")
    return seq


# ---------------------------------------------------------------------------
# stratégie `pair` (v1) — gemm1+silu+quant+gemm2, sortie cache3
# ---------------------------------------------------------------------------
@triton.jit
def qwen3_moe_pair_fusion_kernel(
    # tenseurs
    x_ptr,           # [M, K] int8 (quantifié, échelle x_scale)
    x_scale_ptr,     # [M, 1] fp32 per-token
    w1_ptr,          # [E, 2I, K] int8, K contiguous
    w1_scale_ptr,    # [E, 2I] fp32 per-channel
    w2_ptr,          # [E, K, I] int8, I contiguous
    w2_scale_ptr,    # [E, K] fp32 per-channel
    topk_w_ptr,      # [M, top_k] fp32
    topk_ids_ptr,    # [M, top_k] int32
    sorted_token_ids_ptr,  # [EM] int32 (grouped) ; ignoré en NAIVE
    expert_ids_ptr,        # [n_blocs] int32 (grouped) ; ignoré en NAIVE
    c3_ptr,          # [num_pairs, K] bf16 (intermediate_cache3)
    # sentinelles de padding au-delà : scalaire (jamais une copie H2D —
    # non capturable en graphe CUDA)
    num_valid,
    # strides
    stride_xm,
    stride_w1e,
    stride_w1n,
    stride_w2e,
    stride_w2k,
    stride_c3m,
    # scalaires
    top_k: tl.constexpr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BKO: tl.constexpr,
    BN2: tl.constexpr,
    NAIVE: tl.constexpr,
):
    """1 CTA = (bloc de paires, tuile de sortie K).

    gemm1 : x [BM,K] × w1[e, 2I, K] -> acc_g/acc_u [BM,I] fp32 RÉSIDENTS
    silu  : silu(gate)·up [BM,I] fp32
    quant : amax par rangée sur I ENTIER (contrat per-token livré)
    gemm2 : q [BM,I] × w2[e, K-tuile, I] -> acc2 [BM,BN2] fp32
    épilogue : poids routé fp32 × échelle w2 -> c3 (bf16) — moe_sum aval
               inchangé (contrat de sortie identique au livré).
    """
    pid = tl.program_id(0)
    tile = tl.program_id(1)

    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        # 1 paire valide (ligne 0) + lignes PAD masquées : les dots restent
        # [BM, ...] = MFMA, comme le kernel livré (BLOCK_M=16 avec 1 rangée
        # valide) — BM=1 tombait en v_dot4 (mesuré : 6-19x plus lent).
        pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)

    tokens = pairs // top_k
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
    # un seul expert par CTA-bloc, indexé par le pid (les lignes pad sont
    # masquées) — en naive le launcher passe topk_ids.view(-1) comme
    # expert_ids, exactement le contrat du kernel livré.
    e = tl.load(expert_ids_ptr + pid)
    e64 = e.to(tl.int64)

    offs_k = tl.arange(0, BKO)
    offs_2i = tl.arange(0, 2 * I)

    # ---------------------------------------------------------------
    # gemm1 : boucle K (hidden) — acc gate/up [BM, I] résidents
    # ---------------------------------------------------------------
    acc_g = tl.zeros((BM, I), dtype=tl.float32)
    acc_u = tl.zeros((BM, I), dtype=tl.float32)
    x_rows = x_ptr + tokens.to(tl.int64) * stride_xm
    for k in range(0, tl.cdiv(K, BKO)):
        x = tl.load(x_rows[:, None] + (k * BKO + offs_k)[None, :],
                    mask=valid[:, None], other=0)
        # gate et up CHARGE-EN-SEQUENCE : leurs tuiles B ([BKO,I] chacune)
        # ne doivent pas être vivantes ensemble — sinon la fenêtre LDS
        # somme (2×32 Ko + staging = OutOfResources 65 664 > 65 536,
        # mesuré sur GPU).  En séquence : shared = max = 32 Ko.
        wg = tl.load(w1_ptr + e64 * stride_w1e
                     + tl.arange(0, I)[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        acc_g += tl.dot(x, wg)
        wu = tl.load(w1_ptr + e64 * stride_w1e
                     + (I + tl.arange(0, I))[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        acc_u += tl.dot(x, wu)

    sx = tl.load(x_scale_ptr + tokens.to(tl.int64), mask=valid, other=0.0)
    s1g = tl.load(w1_scale_ptr + e64 * (2 * I) + tl.arange(0, I))
    s1u = tl.load(w1_scale_ptr + e64 * (2 * I) + I + tl.arange(0, I))
    acc_g = acc_g * s1g * sx[:, None]
    acc_u = acc_u * s1u * sx[:, None]

    # ---------------------------------------------------------------
    # silu+mul puis quant per-token (contrat livré : amax sur I entier)
    # ---------------------------------------------------------------
    mid = (acc_g * tl.sigmoid(acc_g)) * acc_u
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    sc = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])

    # ---------------------------------------------------------------
    # gemm2 : tuile de sortie `tile`, poids routé dans l'épilogue
    # ---------------------------------------------------------------
    offs_o = tile * BN2 + tl.arange(0, BN2)
    w2t = tl.load(w2_ptr + e64 * stride_w2e
                  + tl.arange(0, I)[:, None]
                  + offs_o[None, :] * stride_w2k)
    acc2 = tl.zeros((BM, BN2), dtype=tl.float32) + tl.dot(q, w2t)
    s2 = tl.load(w2_scale_ptr + e64 * K + offs_o)
    contrib = acc2 * (sc * wt)[:, None] * s2[None, :]

    # sortie cache3 bf16, rangées = indices de paire (contrat livré)
    c3_ptrs = (c3_ptr + pairs_i64[:, None] * stride_c3m
               + offs_o[None, :])
    tl.store(c3_ptrs, contrib, mask=valid[:, None])


# ---------------------------------------------------------------------------
# stratégie `atomic` (v2) — pipeline entier + somme pondérée atomique
# ---------------------------------------------------------------------------
@triton.jit
def _expert_chunk_work(
    x_rows, valid, inv_amax, sx_rows,
    w1_ptr, w1_scale_ptr, w2_ptr, w2_scale_ptr,
    out_acc_ptr, tokens,
    e64, w_s,
    stride_xm, stride_w1e, stride_w1n, stride_w2e, stride_w2k,
    offs_i, offs_k,
    top_k: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BKO: tl.constexpr, CI: tl.constexpr,
    BN2: tl.constexpr,
    X_IS_INT8: tl.constexpr = False,
    # --- INC 2/3 (tour 2) : source de q = chaîne norm+quant v2 ---
    NORM: tl.constexpr = False,
    AR: tl.constexpr = False,
    res_rows=0, peer_rows=0, norm_w_ptr=0,
    rstd=0.0, dbg_q_ptr=0, do_dbg_q=0,
):
    """Travail d'expert : gemm1(chunk) -> silu -> quant (rangée,chunk) ->
    gemm2 -> contribution pondérée atomique.  Intermédiaires en registres.

    Triple contrat d'entrée :
    * X_IS_INT8=True  : x int8 déjà quantifié (chemin tour 1), déquant par
      `sx_rows` (per-token) ;
    * X_IS_INT8=False : x bf16 (ROUTING) re-quantifié à la volée — contrat
      `_per_token_quant_int8` via `inv_amax` = 127/absmax du prologue ;
    * NORM=True       : x est la sortie de la chaîne norm+quant v2 — par
      chunk `y_bf16` recomputé (pas de slicing de tenseurs de registres,
      piège constexpr) puis `q = clamp(nearbyint(y_bf16·inv_s))` (contrat
      v2, half-even) ; `inv_amax` porte `inv_s`, `sx_rows` porte `scale`
      (échelle de déquant = scale, verbatim spec §6.1).  AR ajoute le
      peer-load dans xf.  `do_dbg_q` publie q (surface de test, contrat).

    INC-A (MFMA) : gemm2 est tuilé en sortie BN2 (contraction CI>=16 =
    v_mfma, acc2 borné [BM,BN2] — l'acc2 [BM,K] plein spillait à BM=16) ;
    `e64`/`w_s` peuvent être SCALAIRES (broadcast de la rangée 0 en mode
    ROUTING à BM=16 : les lignes pad partagent l'expert — sinon le gather
    des poids deviendrait ×BM et détruirait le trafic poids).
    """
    acc_g = tl.zeros((BM, CI), dtype=tl.float32)
    acc_u = tl.zeros((BM, CI), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BKO)):
        if NORM:
            # inc 2 : q recomputé par chunk depuis la chaîne norm+quant v2
            _s, yb = _mk_norm_chunk(
                x_rows, peer_rows, res_rows, norm_w_ptr,
                k * BKO + offs_k, valid, rstd, K=K, AR=AR)
            xq = _rint_clamp_i8(yb.to(tl.float32) * inv_amax[:, None])
            if do_dbg_q:
                tl.store(
                    dbg_q_ptr + tokens.to(tl.int64)[:, None] * K
                    + (k * BKO + offs_k)[None, :], xq,
                    mask=(valid[:, None]
                          & ((k * BKO + offs_k) < K)[None, :]))
        else:
            xk = tl.load(x_rows[:, None] + (k * BKO + offs_k)[None, :],
                         mask=valid[:, None], other=0)
            if X_IS_INT8:
                xq = xk
            else:
                xq = _round_i8(xk.to(tl.float32) * inv_amax[:, None])
        wg = tl.load(w1_ptr + e64 * stride_w1e
                     + offs_i[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        wu = tl.load(w1_ptr + e64 * stride_w1e
                     + (I + offs_i)[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        acc_g += tl.dot(xq, wg)
        acc_u += tl.dot(xq, wu)

    s1g = tl.load(w1_scale_ptr + e64 * (2 * I) + offs_i)
    s1u = tl.load(w1_scale_ptr + e64 * (2 * I) + I + offs_i)
    acc_g = acc_g * s1g * sx_rows[:, None]
    acc_u = acc_u * s1u * sx_rows[:, None]

    mid = (acc_g * tl.sigmoid(acc_g)) * acc_u
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    sc = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])

    # ---------------------------------------------------------------
    # gemm2 : tuiles de sortie BN2 (INC-A MFMA — contraction CI>=16,
    # acc2 borné [BM,BN2]), poids routé dans l'épilogue
    # ---------------------------------------------------------------
    offs_t = tl.arange(0, BN2)
    for base in range(0, K, BN2):
        o = base + offs_t
        m_o = o < K
        w2t = tl.load(w2_ptr + e64 * stride_w2e
                      + offs_i[:, None]
                      + o[None, :] * stride_w2k)
        acc2 = tl.zeros((BM, BN2), dtype=tl.float32) + tl.dot(q, w2t)
        s2 = tl.load(w2_scale_ptr + e64 * K + o, mask=m_o, other=0.0)
        contrib = acc2 * (sc * w_s)[:, None] * s2[None, :]
        tl.atomic_add(out_acc_ptr + tokens.to(tl.int64)[:, None] * K
                      + o[None, :], contrib,
                      mask=valid[:, None] & m_o[None, :],
                      sem="relaxed", scope="gpu")


@triton.jit
def qwen3_moe_megakernel(
    x_ptr,
    x_scale_ptr,
    w1_ptr,
    w1_scale_ptr,
    w2_ptr,
    w2_scale_ptr,
    topk_w_ptr,
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    out_acc_ptr,     # [M, K] fp32 pré-zérodé (accumulation atomique)
    num_valid,
    router_w_ptr,    # [E, K] bf16 (poids du router, K contiguous) — ROUTING
    dbg_ids_ptr,     # [M, top_k] int32 — sélection topk (contrat routing)
    dbg_w_ptr,       # [M, top_k] fp32
    stride_xm,
    stride_w1e,
    stride_w1n,
    stride_w2e,
    stride_w2k,
    # --- INC 2 (norm+quant v2, gate NORM) ---
    res_ptr,         # [M, K] bf16 (tampon DISTINCT de res_out)
    res_out_ptr,     # [M, K] bf16 — SORTIE : nouveau residual bf16(s)
    norm_w_ptr,      # [K] fp32
    eps,             # fp32
    dbg_scale_ptr,   # [M] fp32 — surface de test : scale = amax/127
    dbg_q_ptr,       # [M, K] int8 — surface de test : q du prologue
    # --- INC 3 (AR in-kernel, gate AR) : adresses brutes int64 ---
    slot_ptr,        # mon slot symétrique [M, K] bf16 (publish LOCAL)
    peer_x_ptr,      # slot du pair (handles IPC) — peer-LOADS seuls
    flag_ptr,        # table [M, n_chunks] int32, 256 B/slot
    peer_flag_ptr,   # table du pair
    top_k: tl.constexpr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BKO: tl.constexpr,
    CI: tl.constexpr,
    BN2: tl.constexpr,
    NAIVE: tl.constexpr,
    ROUTING: tl.constexpr = False,
    RENORM: tl.constexpr = True,
    E: tl.constexpr = 256,
    NORM: tl.constexpr = False,
    AR: tl.constexpr = False,
    FLAG_ELEMS_C: tl.constexpr = 64,
):
    """1 CTA = (bloc de paires, chunk intermédiaire) — ou (token, chunk)
    avec ROUTING (incrément tour 2 : router+topk absorbés).

    Même pipeline que la stratégie `pair` mais le CTA ne couvre que `CI`
    colonnes intermédiaires (quant par (rangée, chunk) — écart nommé 2) et
    somme sa contribution par atomique fp32 locale dans out_acc (le `moe_sum`
    est absorbé). Chaque octet de poids est lu une fois (axe chunks sans
    redondance).
    """
    pid = tl.program_id(0)
    chunk = tl.program_id(1)

    # Inc 3 : pointeurs pairs en adresses brutes int64 -> pointeurs.  La
    # promotion i64 est LOAD-BEARING (le JIT infère i32 sinon ;
    # tt.int_to_ptr rejette i32 — mesuré 2026-10-22).  Casts toujours
    # présents (comme le 3-way) : les dummies 0 ne sont jamais déréférencés.
    slot_p = tl.cast(slot_ptr.to(tl.int64), tl.pointer_type(tl.bfloat16))
    peer_p = tl.cast(peer_x_ptr.to(tl.int64), tl.pointer_type(tl.bfloat16))
    flag_p = tl.cast(flag_ptr.to(tl.int64), tl.pointer_type(tl.int32))
    pflag_p = tl.cast(peer_flag_ptr.to(tl.int64), tl.pointer_type(tl.int32))

    offs_i = chunk * CI + tl.arange(0, CI)
    offs_k = tl.arange(0, BKO)

    if ROUTING:
        # ---- INCRÉMENT 1 : router + topk absorbés (CTA = (token, chunk)) --
        # prologue : router dot (bf16 x, contrat du router livré) -> softmax
        # fp32 -> topk (argmax itératif, expressions topk_softmax) -> quant
        # per-token de x (contrat `_per_token_quant_int8`).  Puis boucle des
        # experts de CE token dans la CTA (travail partagé ci-dessous) ->
        # somme pondérée atomique.  BM=1 (1 token/CTA — les lignes pad ne
        # partagent pas d'expert : voir TOUR2-DESIGN §0).
        tokens = tl.where(tl.arange(0, BM) == 0, pid, num_valid)
        valid = tokens < num_valid
        x_rows = x_ptr + tokens.to(tl.int64) * stride_xm
        res_rows = res_ptr + tokens.to(tl.int64) * K
        offs_e = tl.arange(0, E)

        # peer_rows : tampon de PARITÉ du pair (ping-pong, moitié
        # consommation) — n'est connu qu'après dérivation de `ar_seq`.
        peer_rows = peer_p + tokens.to(tl.int64) * (2 * K)

        if AR:
            # ---- INC 3 : phase 0 AR — rendezvous par slot (rangée, chunk),
            # table de flags paddée 256 B (design finding : un flag par
            # rangée ferait xchg la séquence 32× — corruption).  Deux
            # moitiés : publication (spin d'arrivée) ET consommation
            # (double-tampon ping-pong — voir `_mk_ar_rendezvous`).
            ar_seq = _mk_ar_rendezvous(
                x_ptr + pid.to(tl.int64) * stride_xm,
                slot_p,
                pid, chunk, flag_p, pflag_p,
                n_chunks=tl.cdiv(I, CI), flag_elems=FLAG_ELEMS_C,
                K=K, BLOCK_H=256)
            peer_rows = peer_rows + (ar_seq & 1) * K

        if NORM:
            # ---- INC 2 : prologue norm+quant v2 en 3 passes (contrat §3.2
            # VERBATIM — pas le clamp FLT_EPSILON ; nearbyint half-even).
            # BM=1 : la rangée est L2-hot, les passes la re-parcourent.
            # Passe 1 : s = fp32(x_f)+fp32(res) -> var (fp32) ; sortie
            # res_out = bf16(s) (un seul écrivain : chunk 0 — le reste du
            # graphe relit res tant que les autres tournent).
            var_acc = 0.0
            for base in range(0, K, 256):
                offs_h = base + tl.arange(0, 256)
                s, _yb = _mk_norm_chunk(
                    x_rows, peer_rows, res_rows, norm_w_ptr,
                    offs_h, valid, 1.0, K=K, AR=AR)
                var_acc += tl.sum(s * s)
                if chunk == 0:
                    tl.store(res_out_ptr + tokens.to(tl.int64)[:, None] * K
                             + offs_h[None, :], s.to(tl.bfloat16),
                             mask=valid[:, None] & (offs_h < K)[None, :])
            var = var_acc / K
            rstd = tl.math.rsqrt(var + eps)
            # Passe 2 : y_bf16 = bf16(s·rstd·w_fp32) -> amax (ordre libre)
            amax = tl.zeros((BM,), dtype=tl.float32)
            for base in range(0, K, 256):
                offs_h = base + tl.arange(0, 256)
                _s, yb = _mk_norm_chunk(
                    x_rows, peer_rows, res_rows, norm_w_ptr,
                    offs_h, valid, rstd, K=K, AR=AR)
                amax = tl.maximum(
                    amax, tl.max(tl.abs(yb.to(tl.float32)), axis=1))
            # scale = amax/127 SANS clamp ; inv_s avec le cas zéro
            # (0 si amax==0, 1 division) — dynamic_scaled_int8_quant verbatim
            scale_v = amax / 127.0
            inv_amax = tl.where(amax == 0.0, tl.zeros_like(amax),
                                127.0 / amax)
            sx_rows = scale_v
            if chunk == 0:
                tl.store(dbg_scale_ptr + tokens.to(tl.int64), scale_v,
                         mask=valid)
        else:
            # inc 1 (inchangé) : la quant de x est faite à la volée dans la
            # boucle d'experts ; rstd sans objet (dummy transmis au helper)
            rstd = 0.0

        # passe router : tuile K dédiée (BKO_R) — la tuile B du dot
        # [BKO_R, E] bf16 = 32 Ko (fenêtre OK) ; le dot bf16×bf16->fp32 est
        # le contrat exact du router livré (GateLinear).  En NORM le router
        # voit y_bf16 (la sortie de norm bf16 — comme le router livré).
        logits = tl.zeros((BM, E), dtype=tl.float32)
        amax_x = tl.full((BM,), 1e-10, dtype=tl.float32)
        offs_kr = tl.arange(0, 64)
        for k in range(0, tl.cdiv(K, 64)):
            kk = k * 64 + offs_kr
            if NORM:
                _s, xin = _mk_norm_chunk(
                    x_rows, peer_rows, res_rows, norm_w_ptr,
                    kk, valid, rstd, K=K, AR=AR)
            else:
                xk = tl.load(x_rows[:, None] + kk[None, :],
                             mask=valid[:, None], other=0)
                xin = xk.to(tl.bfloat16)
                amax_x = tl.maximum(
                    amax_x, tl.max(tl.abs(xk.to(tl.float32)), axis=1))
            wr = tl.load(router_w_ptr + kk[:, None]
                         + offs_e[None, :] * K)
            logits += tl.dot(xin, wr)

        # softmax fp32 (mêmes opérations que topk_softmax, ordre près)
        p = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        if not NORM:
            # échelle de déquant per-token de x (contrat livré)
            inv_amax = 127.0 / amax_x
            sx_rows = amax_x / 127.0

        # topk : pass A = somme des poids sélectionnés (renormalize) ;
        # pass B = boucle d'experts (même ordre de sélection, déterministe).
        sum_sel = tl.zeros((BM,), dtype=tl.float32)
        p_work = p
        for _s in tl.static_range(top_k):
            e_t = tl.argmax(p_work, axis=1).to(tl.int32)
            picked = tl.sum(
                tl.where(offs_e[None, :] == e_t[:, None], p, 0.0), axis=1)
            sum_sel += picked
            p_work = tl.where(offs_e[None, :] == e_t[:, None], -1.0, p_work)

        p_work = p
        for _s in tl.static_range(top_k):
            e_t = tl.argmax(p_work, axis=1).to(tl.int32)
            w_s = tl.sum(
                tl.where(offs_e[None, :] == e_t[:, None], p, 0.0), axis=1)
            if RENORM:
                w_s = w_s / sum_sel
            p_work = tl.where(offs_e[None, :] == e_t[:, None], -1.0, p_work)
            # contrat de routing : publier la sélection (tests de contrat)
            tl.store(dbg_ids_ptr + tokens.to(tl.int64) * top_k + _s, e_t,
                     mask=valid)
            tl.store(dbg_w_ptr + tokens.to(tl.int64) * top_k + _s, w_s,
                     mask=valid)
            # INC-A (MFMA) : à BM=16 les lignes pad DOIVENT partager
            # l'expert de la ligne 0 — 1 seul token valide par CTA (ligne 0,
            # invariant ROUTING) ; si chaque ligne tirait son expert, le
            # gather des poids deviendrait ×BM (16 experts distincts par
            # tuile) et détruirait le trafic poids.  Broadcast SCALAIRE
            # (somme d'un seul élément = exact) de la sélection ligne 0.
            e0 = tl.sum(tl.where(tl.arange(0, BM) == 0, e_t, 0))
            w0 = tl.sum(tl.where(tl.arange(0, BM) == 0, w_s, 0.0))
            # q n'a pas de dépendance à l'expert : publié une seule fois
            # (première itération, chunk 0) — surface de test (contrat).
            do_dbg = (chunk == 0) if _s == 0 else 0
            _expert_chunk_work(
                x_rows, valid, inv_amax, sx_rows,
                w1_ptr, w1_scale_ptr, w2_ptr, w2_scale_ptr,
                out_acc_ptr, tokens,
                e0.to(tl.int64), w0,
                stride_xm, stride_w1e, stride_w1n, stride_w2e, stride_w2k,
                offs_i, offs_k,
                top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, CI=CI, BN2=BN2,
                X_IS_INT8=False,
                NORM=NORM, AR=AR,
                res_rows=res_rows, peer_rows=peer_rows,
                norm_w_ptr=norm_w_ptr, rstd=rstd,
                dbg_q_ptr=dbg_q_ptr, do_dbg_q=do_dbg,
            )
    else:
        # ---- chemin tour 1 : 1 expert par CTA-bloc (topk consommé) ----
        offs_pair = pid * BM + tl.arange(0, BM)
        if NAIVE:
            # 1 paire valide (ligne 0) + lignes PAD masquées : les dots
            # restent [BM, ...] = MFMA, comme le kernel livré.
            pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
        else:
            pairs = tl.load(sorted_token_ids_ptr + offs_pair)
        valid = pairs < num_valid
        pairs_i64 = pairs.to(tl.int64)

        tokens = pairs // top_k
        wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
        # un seul expert par CTA-bloc, indexé par le pid (les lignes pad sont
        # masquées) — en naive le launcher passe topk_ids.view(-1) comme
        # expert_ids, exactement le contrat du kernel livré.
        e = tl.load(expert_ids_ptr + pid)
        e64 = e.to(tl.int64)

        x_rows = x_ptr + tokens.to(tl.int64) * stride_xm
        sx = tl.load(x_scale_ptr + tokens.to(tl.int64), mask=valid, other=0.0)
        _expert_chunk_work(
            x_rows, valid, tl.zeros((BM,), dtype=tl.float32), sx,
            w1_ptr, w1_scale_ptr, w2_ptr, w2_scale_ptr,
            out_acc_ptr, tokens,
            e64, wt,
            stride_xm, stride_w1e, stride_w1n, stride_w2e, stride_w2k,
            offs_i, offs_k,
            top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, CI=CI, BN2=BN2,
            X_IS_INT8=True,
        )


@triton.jit
def moe_megakernel_finalize_kernel(
    out_acc_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr,
):
    """Cast final fp32 -> bf16 (contrat de sortie de `moe_sum`)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    v = tl.load(out_acc_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, v, mask=mask)


@triton.jit
def moe_megakernel_finalize_int8_kernel(
    out_acc_ptr, q_ptr, s_ptr, K: tl.constexpr, BLOCK: tl.constexpr,
):
    """Requant sortant per-token (contrat `_per_token_quant_int8`) —
    INTERFACE consommée par le motif v2' megakernel→AR int8 (2b).

    1 programme = 1 rangée (token) : absmax sur K entier (clamp 1e-10),
    s = absmax/127, q = round(v·127/absmax) (libdevice.round, même
    rounding que le chemin livré). Échelle par (rangée) = cas particulier
    de (rangée, tuile) avec une seule tuile (scale_tile = K, dérivé à
    l'exécution par l'op AR int8).
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(out_acc_ptr + pid * K + offs, mask=offs < K, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(v)), 1e-10)
    s = amax / 127.0
    q = _round_i8(v * (127.0 / amax))
    tl.store(q_ptr + pid * K + offs, q, mask=offs < K)
    tl.store(s_ptr + pid, s)


# stratégie `tree` (v3, ANTIATOM-DESIGN.md) — ZÉRO atomic, zéro memset,
# zéro finalize : réduction à deux niveaux sans accumulateur partagé.
#
# DEUX contrats de quant coexistent — et `tree` choisit le MEILLEUR :
# `atomic` ne voit qu'un chunk intermédiaire par CTA ⇒ échelle par
# (rangée, chunk) et gemm2 chunké (4-8 dots séquentiels de contraction
# CI — mesuré : le gemm2 tombe à ~40 % du pic de bande passante).
# `tree`, lui, stocke l'intermédiaire post-silu en fp32 (mid_ws) et
# quantifie au token ENTIER dans le prologue de pass2 — le contrat livré
# VERBATIM (amax sur I entier, `_round_i8`) — ce qui libère le gemm2 :
# UN dot pleine-I [16, I] x [I, BN2], exactement le dataflow du kernel
# livré (mesuré : c'est là que se cachait le gap structurel, PAS dans les
# atomics — moemk_decomp_bench).
#
#   pass1  qwen3_moe_mk_tree_pass1   grid (blocs x chunks)
#          gemm1 chunk (acc [16, 2CI] fp32) -> silu fp32 (mid) ->
#          store mid_ws [M·topk, I] fp32   (aucun atomic, aucune échelle)
#   pass2  qwen3_moe_mk_tree_pass2   grid (blocs x tuiles K)
#          prologue : amax sur I entier -> q int8 + s_row (contrat livré)
#          acc2 [16, BN2] = dot(q, w2_tuile) · s_row · s_w2 · w_topk
#          -> ws [M·topk, K] bf16 (rangées de paires, un seul écrivain)
#   pass3  moe_megakernel_reduce      grid (M x tuiles K)
#          out[t] = Σ_s ws[t·topk+s] (fp32 -> bf16) = moe_sum + cast fusionnés
#
# Déterminisme bit-exact run-to-run (ordre fixe) ; écarts nommés vs le
# livré : silu en fp32 (le livré arrondit en bf16 dans cache1) — sinon le
# contrat de quant est le SIEN.  I doit être une puissance de 2 (tuile
# pleine-I) — asserté au launcher (fail loud).
# ---------------------------------------------------------------------------
@triton.jit
def qwen3_moe_mk_tree_pass1_kernel(
    x_ptr,            # [M, K] int8 (quantifié, échelle x_scale)
    x_scale_ptr,      # [M] fp32 per-token
    w1_ptr,           # [E, 2I, K] int8, K contiguous
    w1_scale_ptr,     # [E, 2I] fp32 per-channel (contigu)
    mid_ws_ptr,       # SORTIE [num_pairs, I] post-silu (fp32, ou bf16 si
                      # MID_BF16 — échelles composées, quant dans pass2)
    sorted_token_ids_ptr,  # [EM] int32 (grouped) ; ignoré en NAIVE
    expert_ids_ptr,        # [n_blocs] int32 ; ignoré en NAIVE
    num_valid,
    stride_xm,
    stride_w1e,
    stride_w1n,
    top_k: tl.constexpr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BKO: tl.constexpr,
    CI: tl.constexpr,
    NAIVE: tl.constexpr,
    MID_BF16: tl.constexpr,
):
    """1 CTA = (bloc de paires, chunk intermédiaire) — gemm1 chunké puis
    silu fp32 ; sortie = la tranche [chunk·CI,(chunk+1)·CI) de la rangée de
    paire post-silu, écrite UNE fois par son CTA propriétaire (store nu,
    aucun atomic ; n_chunks CTAs par rangée, colonnes disjointes — pas de
    course, mais pas d'exclusivité de rangée non plus).  Aucune échelle de
    quant ici : la quant est per-token dans pass2 (contrat livré)."""
    pid = tl.program_id(0)
    chunk = tl.program_id(1)

    # C1 (revue deepseek 2026-10-29) : les blocs de PADDING du régime
    # grouped (moe_align remplit expert_ids de -1 au-delà des blocs utiles)
    # sortent AVANT tout chargement — comme le kernel livré (`npp` early
    # exit).  Sans cette garde : calcul complet + chargements poids SANS
    # masque hors bornes (e64=-1) + trafic garbage (~37 % de la grille à
    # M=32 — mesuré dans le surcoût de site) ; le livré, lui, les saute.
    e = tl.load(expert_ids_ptr + pid)
    if e < 0:
        return
    e64 = e.to(tl.int64)

    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)
    tokens = pairs // top_k

    x_rows = x_ptr + tokens.to(tl.int64) * stride_xm
    sx = tl.load(x_scale_ptr + tokens.to(tl.int64), mask=valid, other=0.0)
    offs_i = chunk * CI + tl.arange(0, CI)
    offs_k = tl.arange(0, BKO)

    acc_g = tl.zeros((BM, CI), dtype=tl.float32)
    acc_u = tl.zeros((BM, CI), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BKO)):
        # mimo C1 : masque sur les colonnes K — pour K non multiple de BKO
        # la dernière itération lirait x du token suivant et w1 de l'expert
        # suivant (corruption silencieuse, gate K%128 trop permissive).
        kk = k * BKO + offs_k
        m_k = kk < K
        xk = tl.load(x_rows[:, None] + kk[None, :],
                     mask=valid[:, None] & m_k[None, :], other=0)
        wg = tl.load(w1_ptr + e64 * stride_w1e
                     + offs_i[None, :] * stride_w1n
                     + kk[:, None], mask=m_k[:, None], other=0)
        acc_g += tl.dot(xk, wg)
        wu = tl.load(w1_ptr + e64 * stride_w1e
                     + (I + offs_i)[None, :] * stride_w1n
                     + kk[:, None], mask=m_k[:, None], other=0)
        acc_u += tl.dot(xk, wu)

    s1g = tl.load(w1_scale_ptr + e64 * (2 * I) + offs_i)
    s1u = tl.load(w1_scale_ptr + e64 * (2 * I) + I + offs_i)
    acc_g = acc_g * s1g * sx[:, None]
    acc_u = acc_u * s1u * sx[:, None]

    mid = (acc_g * tl.sigmoid(acc_g)) * acc_u
    # MID_BF16 (passe tuiles 2026-10-29) : stockage bf16 de l'intermédiaire
    # post-silu — moitié moins de trafic HBM sur mid_ws, et le point
    # d'arrondi bf16 rejoint la classe du livré (cache1 bf16).  Le quant
    # reste PER-TOKEN sur I entier dans pass2 (contrat livré verbatim) :
    # pass2 recharge et convertit en fp32 avant amax/_round_i8.
    if MID_BF16:
        tl.store(mid_ws_ptr + pairs_i64[:, None] * I + offs_i[None, :],
                 mid.to(tl.bfloat16), mask=valid[:, None])
    else:
        tl.store(mid_ws_ptr + pairs_i64[:, None] * I + offs_i[None, :], mid,
                 mask=valid[:, None])


@triton.jit
def qwen3_moe_mk_tree_pass2_kernel(
    mid_ws_ptr,       # [num_pairs, I] post-silu (fp32, ou bf16 si MID_BF16)
    w2_ptr,           # [E, K, I] int8, I contiguous
    w2_scale_ptr,     # [E, K] fp32 per-channel (contigu)
    topk_w_ptr,       # [M, top_k] fp32 (poids routés — épilogue)
    ws_ptr,           # SORTIE [num_pairs, K] bf16 (rangées de paires)
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_valid,
    stride_w2e,
    stride_w2k,
    top_k: tl.constexpr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN2: tl.constexpr,
    LOCAL_TILES: tl.constexpr,
    NAIVE: tl.constexpr,
    MID_BF16: tl.constexpr,
):
    """1 CTA = 1 bloc de paires ; `LOCAL_TILES` tuiles de sortie traitées par
    CTA (régime mesuré : LOCAL_TILES=1 + grille (blocs, tuiles) gagne aux
    PETITS M — la quant redondante est absorbée, parité mesurée à M=8 ;
    LOCAL_TILES=K/BN2 + grille (blocs,) gagne aux GRANDS M — le quant est
    calculé UNE fois par bloc).  Le quant est PER-TOKEN (amax sur I entier —
    contrat livré verbatim) et chaque tuile fait UN dot pleine-I :
    acc2 [16, BN2] = dot(q [16,I], w2_tuile [I,BN2]) · s_row · s_w2 ·
    w_topk, avec un seul store nu de la rangée de paire.  Le tuilage du dot
    est celui du kernel livré (BN2=64, contraction I entière) — mesuré :
    les dots chunkés (contraction CI) tombent à ~40 % du pic."""
    pid = tl.program_id(0)
    # groupe de tuiles : le décalage est programme_id(1) * LOCAL_TILES
    # (PIÈGE vécu 2026-10-29 : sans le facteur, les groupes se recouvrent
    # et les tuiles hautes ne sont jamais écrites — garbage en sortie,
    # attrapé par test_tree_poisoned_workspaces_no_gaps).
    tile0 = tl.program_id(1) * LOCAL_TILES

    # C1 (revues deepseek/mimo) : blocs de padding grouped (expert_ids = -1) →
    # sortie AVANT tout chargement (voir pass1).
    e = tl.load(expert_ids_ptr + pid)
    if e < 0:
        return
    e64 = e.to(tl.int64)

    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)

    # --- quant per-token (contrat `_per_token_quant_int8`, amax sur I) ---
    offs_i_full = tl.arange(0, I)
    mid = tl.load(mid_ws_ptr + pairs_i64[:, None] * I + offs_i_full[None, :],
                  mask=valid[:, None], other=0.0)
    if MID_BF16:
        # le stockage est bf16 (arrondi post-silu, classe du cache1 livré) ;
        # l'arithmétique du quant reste fp32 (comme silu_q livré).
        mid = mid.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    s_row = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])

    offs_t = tl.arange(0, BN2)
    for tt in range(0, LOCAL_TILES):
        o = (tile0 + tt) * BN2 + offs_t
        m_o = o < K
        s2 = tl.load(w2_scale_ptr + e64 * K + o, mask=m_o, other=0.0)
        # mimo S3 : masque sur la tuile de sortie (K non multiple de BN2)
        w2t = tl.load(w2_ptr + e64 * stride_w2e
                      + offs_i_full[:, None]
                      + o[None, :] * stride_w2k,
                      mask=m_o[None, :], other=0)
        acc2 = tl.zeros((BM, BN2), dtype=tl.float32) + tl.dot(q, w2t)
        contrib = acc2 * (s_row * wt)[:, None] * s2[None, :]
        tl.store(ws_ptr + pairs_i64[:, None] * K + o[None, :], contrib,
                 mask=valid[:, None] & m_o[None, :])


@triton.jit
def moe_megakernel_reduce_kernel(
    ws_ptr,           # [num_pairs, K] bf16 (rangées de paires)
    out_ptr,          # [M, K] bf16 (ou fp32 si OUT_FP32)
    top_k: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    """Réduction topk + cast final FUSIONNÉS (le `moe_sum` + finalize du
    chemin livré, en un lancement).  1 programme = (token, tuile K) :
    out[t] = Σ_{s<top_k} ws[t·topk+s] — accumulateur fp32, ordre de
    sommation FIXE ⇒ déterminisme bit-exact run-to-run.  OUT_FP32 : sortie
    fp32 (interface INT8_OUT / `moe_megakernel_output_int8`).
    """
    t = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    m = offs < K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, top_k):
        # i64 : M·topk·K > 2^30 sortirait de l'int32 (doctrine du module)
        p = (t.to(tl.int64) * top_k + s) * K
        v = tl.load(ws_ptr + p + offs, mask=m, other=0.0)
        acc += v
    if OUT_FP32:
        tl.store(out_ptr + t * K + offs, acc, mask=m)
    else:
        tl.store(out_ptr + t * K + offs, acc.to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------------------
# API python
# ---------------------------------------------------------------------------
_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


# ---------------------------------------------------------------------------
# compteur d'appels bras/repli (travail nommé F1/W1, M32-M128-EXTENSION §8.5)
# ---------------------------------------------------------------------------
# `warning_once` est STRUCTURELLEMENT muet après le 1er message : il ne peut
# pas prouver quel chemin tourne pendant un run.  Ce compteur, lui, journalise
# la DISTRIBUTION des appels par (issue, M, régime) — les deux falsifiants
# nommés (compteur d'appels bras/repli + distribution de M).  ⚠ Compte les
# invocations du SITE python (capture de graphe comprise) — pas les replays
# du graphe (pas de Python derrière) ; la lecture « appels du site » est la
# seule honnête.  Une ligne de résumé est émise au 1er appel puis toutes les
# 200 invocations.
_CALL_STATS: dict[tuple[str, int, str], int] = {}
_CALL_TOTAL = 0


def moe_megakernel_record_call(outcome: str, M: int,
                               naive: bool | None) -> None:
    global _CALL_TOTAL
    regime = "naive" if naive else ("grouped" if naive is not None else "-")
    key = (outcome, M, regime)
    _CALL_STATS[key] = _CALL_STATS.get(key, 0) + 1
    _CALL_TOTAL += 1
    if _CALL_TOTAL == 1 or _CALL_TOTAL % 200 == 0:
        # deux lignes : les issues `bras:` ne doivent JAMAIS pouvoir être
        # évincées par une pluie de clés `repli:M=…` (revue deepseek W6) ;
        # + histogramme M borné (le falsifiant « distribution de M »).
        bras = sorted((kv for kv in _CALL_STATS.items()
                       if kv[0][0].startswith("bras:")),
                      key=lambda kv: -kv[1])
        repli = sorted((kv for kv in _CALL_STATS.items()
                        if not kv[0][0].startswith("bras:")),
                       key=lambda kv: -kv[1])
        # histogramme M borné (quantiles pondérés, sans expansion)
        pairs_ms = sorted((k[1], v) for k, v in _CALL_STATS.items())
        total = sum(v for _, v in pairs_ms)
        if total:
            acc, p50, lo = 0, pairs_ms[-1][0], pairs_ms[0][0]
            for m, v in pairs_ms:
                acc += v
                if acc >= total // 2:
                    p50 = m
                    break
            hist = f"M min={lo} p50={p50} max={pairs_ms[-1][0]}"
        else:
            hist = "M -"
        logger.info(
            "[moe-megakernel] appels-site #%d (%s) bras: %s",
            _CALL_TOTAL, hist,
            ", ".join(f"{k[0]}|M={k[1]}|{k[2]}={v}" for k, v in bras)
            or "AUCUN",
        )
        logger.info(
            "[moe-megakernel] appels-site #%d replis: %s",
            _CALL_TOTAL,
            ", ".join(f"{k[0]}|M={k[1]}|{k[2]}={v}" for k, v in repli[:14])
            or "aucun",
        )


def moe_megakernel_call_stats() -> dict[tuple[str, int, str], int]:
    """Copie du compteur (surface de test)."""
    return dict(_CALL_STATS)


def moe_megakernel_reset_call_stats() -> None:
    """Remise à zéro (tests uniquement — le processus moteur ne réinitialise
    jamais : le compteur est la mémoire du run)."""
    global _CALL_TOTAL
    _CALL_STATS.clear()
    _CALL_TOTAL = 0


def moe_megakernel_enabled() -> bool:
    return _env_flag("VLLM_MOE_MEGAKERNEL")


def moe_megakernel_strategy() -> str:
    return _env_str("VLLM_MOE_MEGAKERNEL_STRATEGY", "pair")


def moe_megakernel_flags() -> tuple[bool, bool, bool]:
    """(ROUTING, NORM, AR) — les 3 gates empilées (spec §6.2, défaut OFF).

    * ROUTING : `VLLM_MOE_MEGAKERNEL_ROUTING=1` OU le mécanisme incrément 1
      `VLLM_MOE_MEGAKERNEL_STRATEGY=routing` (rétro-compat) ;
    * NORM    : `VLLM_MOE_MEGAKERNEL_NORM=1` (inc 2) ;
    * AR      : `VLLM_MOE_MEGAKERNEL_AR=1` (inc 3, exige NORM).
    """
    routing = (
        _env_flag("VLLM_MOE_MEGAKERNEL_ROUTING")
        or moe_megakernel_strategy() == "routing"
    )
    norm = _env_flag("VLLM_MOE_MEGAKERNEL_NORM")
    ar = _env_flag("VLLM_MOE_MEGAKERNEL_AR")
    return routing, norm, ar


@triton.jit
def _mk_zero_flags_kernel(ptr, n, BLOCK: tl.constexpr):
    p = tl.cast(ptr.to(tl.int64), tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    for base in range(0, n, BLOCK):
        o = base + offs
        tl.store(p + o, tl.zeros([BLOCK], dtype=tl.int32), mask=o < n)


def _mk_zero_flags(ptr: int, n: int) -> None:
    block = 1024
    _mk_zero_flags_kernel[((n + block - 1) // block,)](ptr, n, BLOCK=block)


class MoeMegakernelComm:
    """Buffers symétriques pour l'AR in-kernel (inc 3).

    Même mécanisme IPC que `ArNormQuantComm` (handles hipIpc via
    `CustomAllreduce.create_shared_buffer`, le chemin d'all-reduce custom de
    prod) ; layout de flags celui de la spec tour 2 : table `[rows, n_chunks]`
    int32 avec **256 B par slot** (`FLAG_ELEMS`) — pas de ligne de cache
    partagée entre slots (design finding : un flag par rangée ferait xchg la
    séquence n_chunks× — corruption).  Séquence initiale 0 -> première
    séquence dérivée in-kernel = 1.
    """

    def __init__(self, group: Any, rows: int, hidden: int, n_chunks: int) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.rows = rows
        self.hidden = hidden
        self.n_chunks = n_chunks
        # double-tampon PING-PONG par rangée (moitié consommation du
        # handshake — voir `_mk_ar_rendezvous`) : layout [rows, 2, hidden]
        # bf16 ; l'appel de séquence `seq` utilise le tampon `seq & 1`.
        data_bytes = rows * 2 * hidden * 2  # bf16 × 2 tampons de parité
        flag_bytes = rows * n_chunks * FLAG_ELEMS * 4
        # registered BEFORE any graph capture; pointers are process-lifetime
        self.data_ptrs = CustomAllreduce.create_shared_buffer(
            data_bytes, group=group
        )
        self.flag_ptrs = CustomAllreduce.create_shared_buffer(
            flag_bytes, group=group
        )
        rank = torch.distributed.get_rank(group=group)
        peer = 1 - rank
        assert len(self.data_ptrs) == 2, "TP=2 only (v2 scope)"
        self.rank = rank
        self.data_ptr = self.data_ptrs[rank]
        self.peer_data_ptr = self.data_ptrs[peer]
        self.flag_ptr = self.flag_ptrs[rank]
        self.peer_flag_ptr = self.flag_ptrs[peer]
        _mk_zero_flags(self.flag_ptr, rows * n_chunks * FLAG_ELEMS)
        # ⚠ les flags du pair sont ZÉROS par SON propre constructeur
        # (construction symétrique) — jamais d'écriture vers la mémoire
        # PAIRE : les stores pairs nus sont l'anomalie mesurée du probe
        # 2026-10-22 (65536/65536 mismatch).
        # BARRIÈRE post-init (revue mimo) : sans elle, un rang peut lancer
        # l'appel 1 pendant que l'autre zero-encore sa table — des octets
        # périmés == 1 feraient passer le spin de publication avant la
        # vraie publication (xf = mélange x + garbage).  Le vote collectif
        # du `get_decision` de ar_norm_quant joue ce rôle côté 3-way.
        import torch.distributed as dist
        dist.barrier(group=group)


def moe_megakernel_eligible(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_int8_w8a8: bool,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
    quantization_emulation: bool,
    activation_is_silu: bool,
    clamp_limit_is_none: bool,
    has_lora: bool,
    has_bias: bool,
    expert_map: Any,
    apply_router_weight_on_input: bool,
    a1q_scale: torch.Tensor | None,
) -> str | None:
    """None si éligible, sinon la raison du repli (comptée/loguée).

    Revue deepseek W5 : TOUT ce que le launcher exige (stratégie connue,
    I puissance de 2 pour `tree`, sortie bf16 contiguë) doit être une
    raison de repli ICI — un assert dans la forward tuerait le moteur au
    lieu de replier.  Les asserts du launcher restent en défense en
    profondeur.
    """
    if not moe_megakernel_enabled():
        return "gate off"
    if moe_megakernel_strategy() not in ("pair", "atomic", "tree",
                                          "routing"):
        return f"strategie inconnue {moe_megakernel_strategy()!r}"
    if not (use_int8_w8a8 and per_act_token_quant):
        return "not int8 per-token w8a8"
    if block_shape is not None or quantization_emulation:
        return "blockwise/emulated quant"
    if not (activation_is_silu and clamp_limit_is_none):
        return "activation not plain SILU"
    if has_lora or has_bias:
        return "lora/bias"
    if expert_map is not None:
        return "expert_map (EP)"
    if apply_router_weight_on_input:
        return "router weight on input"
    if hidden_states.dtype not in (torch.int8, torch.bfloat16):
        return f"hidden dtype {hidden_states.dtype}"
    if hidden_states.dtype is torch.int8 and a1q_scale is None:
        return "int8 hidden without a1q_scale"
    K = w1.size(2)
    I = w2.size(2)
    # gates de shape (revue mimo C1/W8) : congruentes avec les TUILES
    # réelles, pas avec celles d'une époque antérieure.  Les kernels tree
    # masquent leurs chargements par K (tout K est sûr), mais pair/atomic
    # non : on garde les congruences qui les protègent, et tree exige I
    # puissance de 2 (tl.arange(0, I) pleine-I).
    if K % 128 != 0:
        return f"K {K} not multiple of BKO"
    if I % MK_CI != 0:
        return f"I {I} not multiple of CI={MK_CI}"
    if moe_megakernel_strategy() == "tree" and I & (I - 1):
        return f"tree: I {I} not a power of 2 (tuile pleine-I)"
    if hidden_states.size(0) > _env_int(
        "VLLM_MOE_MEGAKERNEL_MAX_TOKENS", 8
    ):
        # code de raison STABLE (mimo W5) — M est déjà dans la clé du
        # compteur ; une raison interpolée fragmente le dictionnaire.
        return "M-above-threshold"
    if moe_megakernel_strategy() == "tree" and I != MK_TREE_FALLBACK_I:
        # TABLE indexée par M (2026-10-30) : la config de REPLI (r10) n'est
        # validée que sur la forme de référence I=256 — à I=512 (prefill-EP)
        # elle ne compile même PAS (LDS 128 Ko > 64 Ko, mesuré 2026-10-30).
        # Règle de repli EXPLICITE (choix opérateur 10:23 : « un repli vers
        # le chemin standard suffit pour le prefill ») : une forme hors table
        # ET hors forme de référence tombe sur la chaîne livrée, comptée et
        # loguée — jamais sur une config muette qui casserait au compile.
        # (APRÈS M-above-threshold : cette raison historique garde son sens.)
        naive_cov = (expert_map is None
                     and hidden_states.size(0) * topk_ids.size(1) * 4
                     <= w1.size(0))
        sel = tree_select_config(hidden_states.size(0), naive_cov,
                                 w1.size(0), I, K)
        if sel.source.startswith("repli:"):
            return "tree-forme-non-couverte"
    return None


def moe_megakernel_finalize(
    out_acc: torch.Tensor, output: torch.Tensor
) -> None:
    n = output.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    moe_megakernel_finalize_kernel[grid](out_acc, output, n=n, BLOCK=BLOCK)


def moe_megakernel_output_int8(
    out_acc: torch.Tensor, M: int, K: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sortie INT8 du megakernel (interface v2', gate
    VLLM_MOE_MEGAKERNEL_INT8_OUT).

    Contrat (documenté REPORT §5.4) :
      q     : int8 [M, K] contiguous, q[b,k] = round(out[b,k]/s[b])
      s     : fp32 [M, 1], s[b] = max|out[b,:]|/127 (clamp 1e-10)
    Rounding et expressions = `_per_token_quant_int8` (chemin livré).
    """
    q = torch.empty((M, K), device=out_acc.device, dtype=torch.int8)
    s = torch.empty((M, 1), device=out_acc.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    moe_megakernel_finalize_int8_kernel[(M,)](out_acc, q, s, K=K, BLOCK=BLOCK)
    return q, s


def moe_megakernel_int8_out_enabled() -> bool:
    # lecture dual-channel (revue mimo S1) : un getattr(envs,...) nu ici
    # serait EXACTEMENT le site qui redeviendrait muet si la variable
    # n'est pas enregistrée — unifier avec _env_flag.
    return _env_flag("VLLM_MOE_MEGAKERNEL_INT8_OUT")


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrat `_per_token_quant_int8` (mêmes expressions)."""
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )
    return per_token_quant_int8(x)


def normalize_w_scale(t: torch.Tensor | None) -> torch.Tensor | None:
    """Contrat d'entrée des échelles poids : [E, N] OU [E, N, 1].

    Le loader quark int8 MoE passe les échelles per-channel en 3-D
    [E, N, 1] (`quark_moe.py` : « Per-channel scales: 2D [E, N] -> 3D
    [E, N, 1] for the int8 MoE kernel ») — le kernel megakernel indexe un
    [E, N] contigu.  Squeeze de la dernière dimension unitaire (vue
    contiguë conservée).  Toute autre forme est laissée telle quelle —
    `moe_megakernel_scales_ok` la refuse ensuite (fail loud).

    ⚠ Leçon 2026-10-29 (gate muette n°2) : le site moteur repliait
    SILENCIEUSEMENT sur ce décalage de forme (« repli (echelles poids non
    per-channel) ») pendant que les tests passaient (échelles 2-D
    synthétiques) — la forme réelle du checkpoint n'avait jamais été
    exercée.  Le contrat accepte désormais les DEUX layouts, et un test de
    contrat les couvre tous les deux.
    """
    if t is not None and t.dim() == 3 and t.size(2) == 1:
        return t.squeeze(-1)
    return t


def moe_megakernel_scales_ok(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
) -> bool:
    """Échelles poids per-channel exploitables par le kernel (après
    `normalize_w_scale`) : w1_scale [E, 2I], w2_scale [E, K]."""
    w1_scale = normalize_w_scale(w1_scale)
    w2_scale = normalize_w_scale(w2_scale)
    return not (
        w1_scale is None
        or w1_scale.dim() != 2
        or w1_scale.size(1) != 2 * w2.size(2)
        or w2_scale is None
        or w2_scale.dim() != 2
        or w2_scale.size(1) != w1.size(2)
    )


def fused_moe_megakernel(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    naive: bool,
    sorted_token_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    cache3: torch.Tensor | None = None,
    router_w: torch.Tensor | None = None,
    top_k_override: int | None = None,
    dbg_ids: torch.Tensor | None = None,
    dbg_w: torch.Tensor | None = None,
    # --- INC 2 (gate NORM) : prologue norm+quant v2 ---
    res: torch.Tensor | None = None,
    norm_w: torch.Tensor | None = None,
    eps: float = 1e-6,
    res_out: torch.Tensor | None = None,
    dbg_scale: torch.Tensor | None = None,
    dbg_q: torch.Tensor | None = None,
    # --- INC 3 (gate AR) : handles IPC (MoeMegakernelComm) ---
    ar_comm: Any | None = None,
) -> None:
    """Exécute la forward pass MoE routée fusionnée.

    stratégie `pair`   : 1 lancement (+ moe_sum aval par l'appelant) ;
    stratégie `atomic` : 2 lancements (kernel + finalize), sortie directe.
    Sortie : `output` [M, K] bf16 (contrat `moe_sum` livré).
    """
    M = hidden_states.size(0)
    # normalisation du layout des échelles (quark passe [E, N, 1]) — le
    # kernel indexe [E, N] contigu ; voir normalize_w_scale.
    w1_scale = normalize_w_scale(w1_scale)
    w2_scale = normalize_w_scale(w2_scale)
    if topk_ids is not None:
        top_k = topk_ids.size(1)
    elif topk_weights is not None:
        top_k = topk_weights.size(1)
    else:
        assert top_k_override is not None, "topk absent: top_k_override requis"
        top_k = top_k_override
    K = w1.size(2)
    I = w2.size(2)
    strategy = moe_megakernel_strategy()
    routing, norm_flag, ar_flag = moe_megakernel_flags()

    # Empilement des gates (spec §6.2 : ROUTING → NORM → AR) — fail loud.
    if ar_flag:
        assert norm_flag, "AR exige NORM (empilement ROUTING→NORM→AR)"
    if norm_flag or ar_flag:
        assert routing, "NORM/AR exigent le mode ROUTING"

    if routing:
        x, x_scale = hidden_states, None  # le prologue requantifie
    elif hidden_states.dtype is torch.bfloat16:
        x, x_scale = _quantize_rows(hidden_states)
    else:
        x = hidden_states
        x_scale = a1q_scale

    num_valid = M * top_k
    if naive:
        n_blocks = M * top_k
    else:
        assert sorted_token_ids is not None and expert_ids is not None
        n_blocks = triton.cdiv(sorted_token_ids.size(0), MK_BM)
    BM = 16  # lignes pad (MFMA) — naive ET grouped, comme le kernel livré
    BKO = 128

    if routing:
        # INCRÉMENTS 1-3 (tour 2) : router+topk absorbés — CTA =
        # (token, chunk), boucle d'experts in-kernel, sortie int8 (contrat
        # v2') + copie bf16 pour le contrat aval immédiat.  Régime naive
        # uniquement (le tri grouped est un travail de données-dépendant —
        # TOUR2-DESIGN §0).  NORM = prologue norm+quant v2 ; AR = phase 0
        # in-kernel (spec §6).
        assert router_w is not None, "routing: router_w requis"
        assert naive, "routing in-kernel: regime naive only"
        assert hidden_states.dtype is torch.bfloat16, (
            "routing in-kernel: bf16 x requis (le router livré voit le bf16)")
        # INC-A MFMA : CI>=16 (contraction gemm2) et BM=16 (lignes pad) =
        # v_mfma au lieu de v_dot4 (6-19x plus lent, mesuré tour 1).
        CI = MK_CI
        n_chunks = triton.cdiv(I, CI)
        if norm_flag:
            assert res is not None and norm_w is not None and res_out is not None, (
                "NORM: res/norm_w/res_out requis (contrat v2 §3.2)")
            assert res_out.data_ptr() != res.data_ptr(), (
                "NORM: res_out doit être DISTINCT de res — la grille chunked "
                "relit res pendant que d'autres CTAs écrivent res_out ; "
                "l'in-place du 3-way est la seule chose que la structure "
                "interdit (double-buffering du flux residual)")
            assert res.shape == (M, K) and res_out.shape == (M, K), (
                "NORM: res/res_out doivent être [M, K]")
            assert norm_w.numel() == K, "NORM: norm_w doit être [K]"
            # le kernel indexe avec stride K implicite (revue GLM m2) :
            # un tenseur transposé/slicé lirait silencieusement de faux
            # offsets — exigé contiguous, comme le livré.
            assert (res.is_contiguous() and res_out.is_contiguous()
                    and norm_w.is_contiguous()), (
                "NORM: res/res_out/norm_w doivent être contigus")
        if ar_flag:
            assert ar_comm is not None, "AR: ar_comm (handles IPC) requis"
            assert ar_comm.hidden == K and ar_comm.n_chunks == n_chunks, (
                f"AR: comm {ar_comm.hidden}x{ar_comm.n_chunks} != "
                f"K={K} n_chunks={n_chunks}")
            assert ar_comm.rows >= M, "AR: comm trop petite pour M"
            # plafond de co-résidence : le spin inter-rangs exige que toute
            # la grille finisse par tourner sur les DEUX rangs (sinon
            # deadlock de scheduling entre rangs).  Au-delà : fail loud
            # côté API (AssertionError) — le FUTUR câblage convertit ce
            # raise en repli livré par appel (le site moteur replie déjà en
            # amont, donc injoignable aujourd'hui).
            n_ctas = M * n_chunks
            if n_ctas > _ar_max_ctas():
                _log_once(
                    "mk-ar-over-grid",
                    "[moe-megakernel] AR hors plafond de co-résidence "
                    "(%d CTAs > %d) — l'appelant doit replier sur la "
                    "chaîne livrée",
                    n_ctas, _ar_max_ctas(),
                )
                raise AssertionError(
                    f"AR: {n_ctas} CTAs > plafond {_ar_max_ctas()} "
                    "(co-résidence requise par le spin) — repli livré")
        out_acc = torch.zeros((M, K), device=output.device,
                              dtype=torch.float32)
        grid = (M, n_chunks)
        # x bf16 TEL QUEL (le prologue requantifie — contrat
        # `_per_token_quant_int8` ou v2 selon NORM) ; échelle factice.
        dbg_i = dbg_ids if dbg_ids is not None else torch.zeros(
            M, top_k, device=output.device, dtype=torch.int32)
        dbg_ww = dbg_w if dbg_w is not None else torch.zeros(
            M, top_k, device=output.device)
        dev = output.device
        res_t = res if res is not None else torch.zeros(
            1, 1, device=dev, dtype=torch.bfloat16)
        res_out_t = res_out if res_out is not None else torch.zeros(
            1, 1, device=dev, dtype=torch.bfloat16)
        norm_w_t = norm_w if norm_w is not None else torch.zeros(
            1, device=dev)
        dbg_scale_t = dbg_scale if dbg_scale is not None else torch.zeros(
            M, device=dev)
        # ⚠ les surfaces dbg sont ÉCRITES par le kernel en mode NORM (revue
        # mimo B-OOB) : les dummies doivent avoir la VRAIE taille — un
        # tenseur de 1 élément = écritures hors allocation (corruption GPU).
        dbg_q_t = dbg_q if dbg_q is not None else torch.zeros(
            M, K, device=dev, dtype=torch.int8)
        if ar_comm is not None:
            slot_ptr = ar_comm.data_ptr
            peer_x_ptr = ar_comm.peer_data_ptr
            flag_ptr = ar_comm.flag_ptr
            peer_flag_ptr = ar_comm.peer_flag_ptr
        else:
            slot_ptr = peer_x_ptr = flag_ptr = peer_flag_ptr = 0
        qwen3_moe_megakernel[grid](
            hidden_states, torch.zeros(1, 1, device=dev),
            w1, w1_scale, w2, w2_scale,
            topk_weights if topk_weights is not None else
            torch.zeros(1, device=dev),
            topk_ids if topk_ids is not None else
            torch.zeros(1, device=dev, dtype=torch.int32),
            torch.zeros(1, device=dev, dtype=torch.int32),
            torch.zeros(1, device=dev, dtype=torch.int32),
            out_acc,
            M,
            router_w,
            dbg_i, dbg_ww,
            x.stride(0), w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            res_t, res_out_t, norm_w_t, eps, dbg_scale_t, dbg_q_t,
            slot_ptr, peer_x_ptr, flag_ptr, peer_flag_ptr,
            top_k=top_k, K=K, I=I, BM=MK_BM, BKO=128, CI=CI, BN2=MK_BN2,
            NAIVE=True,
            ROUTING=True, RENORM=True, E=w1.size(0),
            NORM=norm_flag, AR=ar_flag, FLAG_ELEMS_C=FLAG_ELEMS,
            num_warps=4, num_stages=1,
        )
        q_out, s_out = moe_megakernel_output_int8(out_acc, M, K)
        output.copy_((q_out.float() * s_out).to(output.dtype))
        _log_once(
            "mk-routing" + ("-norm" if norm_flag else "")
            + ("-ar" if ar_flag else ""),
            "[moe-megakernel] routing in-kernel ACTIVE (NORM=%s AR=%s "
            "M=%d topk=%d E=%d K=%d)",
            norm_flag, ar_flag, M, top_k, w1.size(0), K,
        )
        return

    if strategy == "atomic":
        # INC-A MFMA : CI>=16 (contraction gemm2 en v_mfma au lieu des
        # micro-dots v_dot4 de CI=8) + sortie tuilée BN2 (registres bornés).
        CI = MK_CI
        out_acc = torch.zeros((M, K), device=output.device,
                              dtype=torch.float32)
        grid = (n_blocks, triton.cdiv(I, CI))
        qwen3_moe_megakernel[grid](
            x, x_scale, w1, w1_scale, w2, w2_scale,
            topk_weights, topk_ids,
            sorted_token_ids if sorted_token_ids is not None else topk_ids,
            expert_ids if expert_ids is not None else topk_ids,
            out_acc,
            num_valid,
            w1.new_zeros((1, 1)),  # router_w factice (non lu hors ROUTING)
            w1.new_zeros((1,), dtype=torch.int32),  # dbg_ids factice
            w1.new_zeros((1,)),                      # dbg_w factice
            x.stride(0), w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            # inc 2/3 : dummies (NORM/AR exigent ROUTING — asserté ci-dessus)
            w1.new_zeros((1, 1), dtype=torch.bfloat16),   # res
            w1.new_zeros((1, 1), dtype=torch.bfloat16),   # res_out
            w1.new_zeros((1,)),                          # norm_w
            0.0,                                         # eps
            w1.new_zeros((1,)),                          # dbg_scale
            w1.new_zeros((1,), dtype=torch.int8),         # dbg_q
            0, 0, 0, 0,                                  # slot/peer/flags
            top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, CI=CI, BN2=MK_BN2,
            NAIVE=naive,
            num_warps=4,
        )
        if getattr(envs, "VLLM_MOE_MEGAKERNEL_INT8_OUT", False):
            # variante interface v2' : requant per-token depuis l'acc fp32
            # (qualité max) au lieu du cast bf16.
            q_out, s_out = moe_megakernel_output_int8(out_acc, M, K)
            _log_once(
                "mk-int8out",
                "[moe-megakernel] sortie INT8 (interface v2') ACTIVE "
                "(M=%d K=%d)", M, K,
            )
            # la copie vers `output` (bf16) maintient le contrat aval
            # immédiat ; le motif v2' consommera (q_out, s_out) directement
            # quand il sera câblé.
            output.copy_((q_out.float() * s_out).to(output.dtype))
            return
        moe_megakernel_finalize(out_acc, output)
    elif strategy == "tree":
        # ANTIATOM-DESIGN.md §2 (révisé par la décomposition) : 3
        # lancements, ZÉRO atomic / memset / finalize.  Le quant est
        # PER-TOKEN (contrat livré verbatim) dans le prologue de pass2 —
        # plus de contrat chunké : le gemm2 est un dot pleine-I (le
        # dataflow du kernel livré, mesuré 2,4× plus rapide que les dots
        # chunkés).  BN2=64 = tuile de sortie livrée ; BKO=256/CI=64 =
        # configs mesurées (sweep moemk_tree_sweep_bench).  Depuis la passe
        # TABLE (2026-10-30) : la config de lancement vient de
        # `tree_select_config` (table indexée par M, clé = forme locale des
        # poids — comme les tables livrées) avec repli garanti sur la config
        # r10 (tree_default_config).
        tcfg = tree_select_config(M, naive, w1.size(0), I, K)
        CI = min(tcfg.CI, I)
        n_chunks = triton.cdiv(I, CI)
        num_pairs = M * top_k
        assert output.dtype is torch.bfloat16 and output.is_contiguous(), (
            "tree: sortie bf16 contiguë requise (contrat moe_sum livré)")
        assert I & (I - 1) == 0, (
            f"tree: I={I} doit être une puissance de 2 (tuile pleine-I)")
        # mid_ws = intermédiaire post-silu [M·topk, I] (L2-hot, lu par
        # les tuiles K) — fp32, ou bf16 si MK_TREE_MID_BF16 (même contrat
        # de quant per-token, arrondi bf16 post-silu) ; ws = rangées de
        # paires bf16 (= le cache3 livré, mais PLEINEMENT écrit par pass2 —
        # aucun memset, invariant : le tri grouped couvre chaque paire
        # exactement une fois).
        mid_ws = torch.empty((num_pairs, I), device=output.device,
                             dtype=(torch.bfloat16 if MK_TREE_MID_BF16
                                    else torch.float32))
        ws = torch.empty((num_pairs, K), device=output.device,
                         dtype=torch.bfloat16)
        s1 = sorted_token_ids if sorted_token_ids is not None else topk_ids
        e1 = expert_ids if expert_ids is not None else topk_ids
        # align↔grille : le BM du tri grouped DOIT être celui de la sélection
        # (le site appelle moe_megakernel_align_block_m — même fonction
        # PURE).  L'égalité expert_ids.numel() == n_blocks est SHARP : le tri
        # alloue cdiv(sorted.size(0), align_bm) blocs ; toute désync entre
        # align et grille casse ICI (fail loud), pas en garbage silencieux.
        grid_blocks = M * top_k if naive else triton.cdiv(
            sorted_token_ids.size(0), tcfg.BM)
        assert e1.numel() == grid_blocks, (
            f"tree: align/grille desynchronises (expert_ids={e1.numel()} "
            f"blocs vs grille={grid_blocks}, BM={tcfg.BM}) — le tri grouped "
            "doit utiliser moe_megakernel_align_block_m")
        qwen3_moe_mk_tree_pass1_kernel[(grid_blocks, n_chunks)](
            x, x_scale, w1, w1_scale, mid_ws, s1, e1, num_valid,
            x.stride(0), w1.stride(0), w1.stride(1),
            top_k=top_k, K=K, I=I, BM=tcfg.BM, BKO=tcfg.BKO, CI=CI,
            NAIVE=naive, MID_BF16=MK_TREE_MID_BF16,
            num_warps=tcfg.nw1, num_stages=tcfg.ns1,
        )
        n_tiles = triton.cdiv(K, tcfg.BN2)
        # régime de lancement = table indexée par M (repli r10 =
        # tree_pass2_config) : le champ LOCAL_TILES est CONSOMMÉ dans les
        # DEUX régimes (None = n_tiles entier, grille 1-D ; 1 = grille 2-D
        # (blocs, tuiles) du régime naive r10 ; tout autre = groupes de
        # tuiles).  Mesures : LT=n_tiles gagne aux grands M groupés
        # (PLAFONDS-TUILES §1) ; le cas naive 1-D est balayé par la table
        # (axe du mandat LOCAL_TILES=n_tiles).
        local_tiles = (n_tiles if tcfg.LOCAL_TILES is None
                       else tcfg.LOCAL_TILES)
        grid2 = (grid_blocks,) if local_tiles == n_tiles else (
            grid_blocks, triton.cdiv(n_tiles, local_tiles))
        # PREUVE DE BRANCHEMENT du régime (revue deepseek M1) : une ligne
        # par régime exécuté, avec la config RÉELLEMENT lancée (BM/CI
        # compris) et sa SOURCE (table | repli:…) — le wall la lit ; un
        # canari de test vérifie qu'elle porte la config consommée
        # (test_tree_launch_regime_log_canary).
        _log_once(
            f"mk-tree-regime-{tcfg.BM}-{tcfg.CI}-{tcfg.BKO}-{tcfg.BN2}-"
            f"{local_tiles}-{tcfg.nw1}-{tcfg.ns1}-{tcfg.nw2}-{tcfg.ns2}-"
            f"{tcfg.source}",
            "[moe-megakernel] tree regime M=%d naive=%s BM=%d CI=%d BN2=%d "
            "LOCAL_TILES=%s nw1/ns1=%d/%d nw2/ns2=%d/%d src=%s",
            M, naive, tcfg.BM, tcfg.CI, tcfg.BN2, local_tiles,
            tcfg.nw1, tcfg.ns1, tcfg.nw2, tcfg.ns2, tcfg.source,
        )
        qwen3_moe_mk_tree_pass2_kernel[grid2](
            mid_ws, w2, w2_scale, topk_weights, ws, s1, e1, num_valid,
            w2.stride(0), w2.stride(1),
            top_k=top_k, K=K, I=I, BM=tcfg.BM, BN2=tcfg.BN2,
            LOCAL_TILES=local_tiles, NAIVE=naive, MID_BF16=MK_TREE_MID_BF16,
            num_warps=tcfg.nw2, num_stages=tcfg.ns2,
        )
        if moe_megakernel_int8_out_enabled():
            # variante interface v2' : le reduce écrit fp32, puis requant
            # per-token (contrat existant, inchangé).
            out_acc = torch.empty((M, K), device=output.device,
                                  dtype=torch.float32)
            moe_megakernel_reduce_kernel[(M, triton.cdiv(K, 64))](
                ws, out_acc, top_k=top_k, K=K, BLOCK=64,
                OUT_FP32=True, num_warps=4,
            )
            q_out, s_out = moe_megakernel_output_int8(out_acc, M, K)
            _log_once(
                "mk-int8out",
                "[moe-megakernel] sortie INT8 (interface v2') ACTIVE "
                "(M=%d K=%d)", M, K,
            )
            output.copy_((q_out.float() * s_out).to(output.dtype))
            return
        moe_megakernel_reduce_kernel[(M, triton.cdiv(K, 64))](
            ws, output, top_k=top_k, K=K, BLOCK=64,
            OUT_FP32=False, num_warps=4,
        )
    else:
        assert strategy == "pair", f"stratégie inconnue: {strategy!r}"
        # BN2=64 + num_warps=8 = config mesurée au smoke (shared 32 Ko,
        # VGPR 256) ; BN2=256 déborde la fenêtre LDS (65 536 + 128 B de
        # staging = OutOfResources mesuré sur GPU).
        BN2 = 64
        assert cache3 is not None, "stratégie pair: cache3 requis"
        grid = (n_blocks, triton.cdiv(K, BN2))
        qwen3_moe_pair_fusion_kernel[grid](
            x, x_scale, w1, w1_scale, w2, w2_scale,
            topk_weights, topk_ids,
            sorted_token_ids if sorted_token_ids is not None else topk_ids,
            expert_ids if expert_ids is not None else topk_ids,
            cache3,
            num_valid,
            x.stride(0), w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1), cache3.stride(1),
            top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, BN2=BN2, NAIVE=naive,
            num_warps=8,
            # liveness séquentielle gate/up exigée (fenêtre LDS = max, pas
            # somme) — le pipeliner ne doit pas pré-charger wg(k+1) pendant
            # wu(k) : num_stages=1.
            num_stages=1,
        )
        # moe_sum aval = le chemin livré (contrat identique).
        from vllm import _custom_ops as ops
        ops.moe_sum(cache3, output)

    _log_once(
        "mk-on",
        "[moe-megakernel] Qwen3MoEMegakernel ACTIVE (strategy=%s M=%d "
        "topk=%d E=%d I=%d K=%d naive=%s)",
        strategy, M, top_k, w1.size(0), I, K, naive,
    )
