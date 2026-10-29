# SPDX-License-Identifier: Apache-2.0
"""INC-C — lm_head fp32 + gather logits + SAMPLING shard-aware (campagne
MEGAKERNEL full-forward, MI210 gfx90a, Triton 3.8 HIP).

Mandat : `research/moe-megakernel-campaign-2026-10-27/TOUR3-FULL-FORWARD-DESIGN.md`
§3.5 (contrat shard-aware) et §5.3 (plafond) ; contrainte shard-aware et crash
OOB historique : `research/ar-vol-count-2026-10-25/ETAPE-A` §3.4 +
`ETAPE-B2` §3/§5 (``MinTokensLogitsProcessor.index_put_`` avec des ids
GLOBAUX sur logits de largeur shard : ``eos=248044`` → ``248044 % 124160 =
123884`` = la colonne de la rangée SUIVANTE, une écriture hors allocation
pour la dernière rangée — UB CUDA observée en crash SIGABRT des deux rangs).

Aujourd'hui (chemin livré) : lm_head ColumnParallel (shard 124 160/rang) →
all-gather NCCL [M,248320] → cast fp32 → sampler Python pleine-vocab
(processeurs, top-k/p, tirage).  Les deux rangs calculent le même token en
redondance ; le gather = 42,8 % des octets du pas (ETAPE-A §2.3).

Cible INC-C (spec ETAPE-A §3.4, inchangée) :

1. ``mk_lm_head_logits`` : logits shard **fp32** [M, vocab_local] — le cast
   fp32 est absorbé dans l'épilogue du GEMM MFMA (``moe_megakernel_dense``) ;
2. ``apply_shard_aware_processors`` : toute la logique indexée par vocab mappe
   ids GLOBAUX → colonnes locales (``col = gid - vocab_start_index``) avec
   masque de store **borné** (``0 <= col < vocab_local``) — le crash OOB
   historique est structurellement impossible (le store ne peut pas sortir
   du shard ; prouvé par test négatif + mutation) ;
3. ``shard_topk`` : top-K' locaux (K'=32-64, ``DEFAULT_K_PRIME``) → mini
   gather [M, 2K'] via ``gather_fn`` injectable (défaut = all-gather
   torch.distributed si initialisé, sinon **repli compté** mono-rang) ;
4. top-k/p + softmax + tirage **Philox par (request_id_hash, step)** —
   avancé IDENTIQUEMENT sur les deux rangs (contre-based, sans état : même
   graine de part et d'autre ⇒ mêmes uniforms ⇒ même token) ;
5. replis comptés + loggés (one-shot) vers gather complet : ``num_logprobs
   == -1``, ``logprob_token_ids`` hors shard, processeurs non couverts, noyau
   top-p non couvert, absence de troncature.

Écarts NOMMÉS vs le livré (aucun n'est bit-exact par construction ; la
doctrine 2026-10-24 : la bit-exactitude n'est pas un critère, la cohérence
par tâche l'est) :

* **mapeage RNG différent** : le tirage Philox par (request_id_hash, step)
  remplace les ``torch.Generator`` du livré — **même distribution** (loi
  softmax renormalisée sur les survivants top-k/p), pas les mêmes tokens pour
  temperature > 0.  Jugé par les gates de tâche (needles/greedy/AIME25),
  JAMAIS par comparaison de logprobs inter-impls.
* **greedy et logprobs = identiques** (argmax avec déblocage d'ex aequo par
  plus petit gid = le ``argmax`` livré (première colonne = plus petit id) ;
  logprobs par décomposition logsumexp exacte ``Z = logaddexp(Z_shard0,
  Z_shard1)`` — identiques à l'ordre de réduction fp32 près).
* **sémantique des logprobs** : ``log_softmax`` des logits traités
  (post-température, post-processeurs, **PRÉ-troncature** top-k/p).  Le mode
  livré ``processed_logits`` masque avant de rapporter les logprobs : écart
  nommé, hors critère (interdiction de juger la distribution aux logprobs).
* **écart NOMMÉ** : les ex aequo fp32 exacts à la frontière top-k/top-p
  entre shards — le livré garde tous les ex aequo (masque par valeur), le
  chemin sharded ne voit que ceux présents dans le merged set ; mesure nulle
  sur des logits continus, nommé pour les distributions à masses discrètes.
* **lm_head fp32 direct** vs livré (bf16 puis cast) : meilleur d'un arrondi
  bf16, jamais pire.
* **sans troncature** (ni top-k ni top-p) : le support de la loi livrée est
  le vocab entier — irréductible à 2K' candidats ⇒ repli (nommé).

Gate ``VLLM_MOE_MEGAKERNEL_SAMPLING`` (défaut OFF) — **AUCUN câblage moteur**
(comme INC-A) : ce module ne touche ni sampler.py, ni builtin.py, ni envs.py.
Gate OFF = repli = chemin livré **verbatim** (``delivered_fn`` est rappelé
tel quel — identité de repli prouvée par test d'identité d'objet).

⚠ La variable n'est PAS enregistrée dans ``vllm.envs`` : le
``getattr(envs, ...)`` seul du pattern ``moe_megakernel_enabled()`` la
lirait comme TOUJOURS False — la gate existerait mais ne s'allumerait jamais
(piège « déclaré mais jamais consommé »).  Deux canaux, dans cet ordre :

1. ``os.environ`` — canal opérateur actuel, prépondérant ;
2. attribut ``envs.VLLM_MOE_MEGAKERNEL_SAMPLING`` — canal de test / futur
   enregistrement.
"""

from __future__ import annotations

import logging
import operator
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.moe_megakernel_dense import (
    dense_mfma_gemm,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_K_PRIME",
    "FALLBACK_COUNTERS",
    "ProcessorOutcome",
    "SampleResult",
    "ShardContractError",
    "ShardLogprobs",
    "ShardProcessorState",
    "ShardRequestParams",
    "apply_shard_aware_processors",
    "default_gather_fn",
    "global_to_local",
    "mk_lm_head_logits",
    "mk_sampling_enabled",
    "philox_uniforms",
    "shard_sample",
    "shard_topk",
]

# K' du mandat (§3.5-3 : K'=32-64 configurable) : le merged set couvre le
# top-K' GLOBAL de façon exacte (preuve : un shard contribue au plus K'
# éléments au top-K' global ⇒ top-K' global ⊆ union des top-K' locaux).
DEFAULT_K_PRIME = 64

# Seuil greedy/random du livré (sampler.py `_SAMPLING_EPS`) : temp < eps ⇒
# rangée greedy, température remplacée par 1.0 dans le chemin livré.
_SAMPLING_EPS = 1e-5

_INF = float("-inf")

# Compteur de replis, instrumentation runtime (« compteur d'appels » du
# mandat) : incrémenté par CHAQUE repli, tous motifs confondus.  Les tests
# l'observent ; les clés = motifs de repli nommés ci-dessous.
FALLBACK_COUNTERS: dict[str, int] = {}

_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


def _note_fallback(state: ShardProcessorState | None, reason: str, detail: str) -> None:
    """Comptabilise + logge (one-shot) un repli vers le chemin livré."""
    FALLBACK_COUNTERS[reason] = FALLBACK_COUNTERS.get(reason, 0) + 1
    if state is not None:
        state.counters[reason] = state.counters.get(reason, 0) + 1
    _log_once(
        f"mk-sampling-fallback-{reason}",
        "[mk-sampling] repli gather complet (chemin livré) — motif %s : %s",
        reason,
        detail,
    )


class ShardContractError(RuntimeError):
    """Contrat shard-aware violé en mode strict (fail loud)."""


def mk_sampling_enabled() -> bool:
    """Gate INC-C ``VLLM_MOE_MEGAKERNEL_SAMPLING`` (défaut OFF = repli livré).

    ⚠ La variable n'est PAS enregistrée dans ``vllm.envs`` (INC-C : aucun
    câblage, envs.py intouché) : le ``getattr(envs, ...)`` seul du pattern
    ``moe_megakernel_enabled()`` la lirait comme TOUJOURS False — la gate
    existerait mais ne s'allumerait jamais (piège « déclaré mais jamais
    consommé »).  Deux canaux, dans cet ordre :

    1. ``os.environ`` — le canal opérateur actuel, prépondérant ;
    2. attribut ``envs.VLLM_MOE_MEGAKERNEL_SAMPLING`` — canal de test / du
       futur enregistrement (pattern ``moe_megakernel_enabled()``).
    """
    raw = os.environ.get("VLLM_MOE_MEGAKERNEL_SAMPLING")
    if raw is not None:
        return raw.strip().lower() in ("1", "true")
    return bool(getattr(envs, "VLLM_MOE_MEGAKERNEL_SAMPLING", False))


# ===========================================================================
# 1. lm_head fp32 (le cast est absorbé dans l'épilogue du GEMM MFMA)
# ===========================================================================
def _lm_head_delivered(
    x: torch.Tensor,
    lm_w_int8: torch.Tensor,
    lm_w_scale: torch.Tensor,
    out: torch.Tensor,
    a_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Repli = chaîne livrée VERBATIM : lm_head bf16 (`triton_scaled_mm*`)
    puis cast fp32 (le kernel de cast livré = une conversion bf16→fp32,
    bit-exacte : la copie vers `out` fp32 en est l'identité numérique)."""
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
        triton_scaled_mm_fused_quant,
    )

    M, N = x.shape[0], lm_w_int8.shape[1]
    if x.dtype == torch.int8:
        assert a_scale is not None, (
            "repli livré : x int8 sans a_scale (contrat per-token requis)"
        )
        res = triton_scaled_mm(x, lm_w_int8, a_scale, lm_w_scale, torch.bfloat16)
    else:
        res, _ = triton_scaled_mm_fused_quant(x, lm_w_int8, lm_w_scale, torch.bfloat16)
    out[:M, :N].copy_(res)  # cast bf16→fp32 = le kernel de cast livré
    return out


def mk_lm_head_logits(
    x: torch.Tensor,
    lm_w_int8: torch.Tensor,
    lm_w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    a_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """lm_head fp32 sharded : out [M, vocab_local] fp32 = x[M,K] × W_int8.

    * ``x`` bf16 [M,K] (quant per-token fusionnée à la volée) ou int8
      pré-quantifié (alors ``a_scale`` [M] per-token REQUIS) ;
    * ``lm_w_int8`` int8 [K, vocab_local] (shard de lm_head, ColumnParallel),
      ``lm_w_scale`` fp [vocab_local] per-channel ;
    * ``out`` fp32 [≥M, ≥vocab_local] pré-alloué (alloué ici si absent).

    Gate ``VLLM_MOE_MEGAKERNEL_SAMPLING`` OFF → repli = chaîne livrée
    verbatim (lm_head bf16 + cast fp32), compté + loggé (one-shot).
    Gate ON → ``dense_mfma_gemm`` avec sortie fp32 (le cast fp32 est absorbé
    dans l'épilogue : meilleur d'un arrondi bf16 vs le livré — écart NOMMÉ).

    Launcher public du mandat : ``mk_lm_head_logits(x, lm_w_int8, lm_w_scale,
    out)``.
    """
    assert x.dim() == 2 and lm_w_int8.dim() == 2, (
        f"x et lm_w_int8 doivent être 2D : {tuple(x.shape)}, {tuple(lm_w_int8.shape)}"
    )
    M, K = x.shape
    Kw, vocab_local = lm_w_int8.shape
    assert Kw == K, (
        f"K désaligné : x {tuple(x.shape)} vs lm_w_int8 {tuple(lm_w_int8.shape)}"
    )
    assert x.dtype in (torch.bfloat16, torch.int8), f"x dtype hors contrat : {x.dtype}"
    assert lm_w_int8.dtype == torch.int8, (
        f"lm_w_int8 doit être int8 : {lm_w_int8.dtype}"
    )
    assert lm_w_scale.is_floating_point() and lm_w_scale.numel() == vocab_local, (
        f"lm_w_scale doit être fp [vocab_local={vocab_local}] : "
        f"{tuple(lm_w_scale.shape)} {lm_w_scale.dtype}"
    )
    lm_w_scale = lm_w_scale.reshape(-1)
    if x.dtype == torch.int8:
        assert a_scale is not None and a_scale.numel() == M, (
            "x int8 pré-quantifié exige a_scale [M] per-token (contrat scaled_mm)"
        )
        a_scale = a_scale.reshape(-1)
    if out is None:
        out = torch.empty((M, vocab_local), dtype=torch.float32, device=x.device)
    assert out.dtype == torch.float32 and out.dim() == 2, (
        f"out doit être fp32 2D : {out.dtype} {tuple(out.shape)}"
    )
    assert out.shape[0] >= M and out.shape[1] >= vocab_local, (
        f"out {tuple(out.shape)} trop petit pour [{M},{vocab_local}]"
    )
    for name, t in (("x", x), ("lm_w_int8", lm_w_int8), ("lm_w_scale", lm_w_scale)):
        assert t.device == x.device, f"{name} sur {t.device} != x sur {x.device}"
    assert out.device == x.device, f"out sur {out.device} != x sur {x.device}"

    if not mk_sampling_enabled():
        _note_fallback(
            None,
            "lm_head_gate_off",
            "gate VLLM_MOE_MEGAKERNEL_SAMPLING OFF — lm_head livré + cast fp32",
        )
        return _lm_head_delivered(x, lm_w_int8, lm_w_scale, out, a_scale)

    _log_once(
        f"mk-lm-head-{x.dtype}-{M}x{vocab_local}x{K}",
        "[mk-sampling] lm_head fp32 MFMA ACTIF : x %s [%d,%d] x W int8 "
        "[%d,%d], sortie fp32 (cast absorbé dans l'épilogue)",
        x.dtype,
        M,
        K,
        K,
        vocab_local,
    )
    return dense_mfma_gemm(x, lm_w_int8, lm_w_scale, a_scale=a_scale, out=out)


# ===========================================================================
# 2. Processeurs shard-aware (ids GLOBAUX → colonnes locales, store borné)
# ===========================================================================
def global_to_local(gid: int, vocab_start_index: int) -> int:
    """Mapping id global → colonne locale : ``col = gid - vocab_start_index``.

    C'est LE correctif du crash OOB historique (ETAPE-B2 §3) : le livré écrit
    les ids globaux directement dans les colonnes d'un tenseur de largeur
    SHARD — ``index = row*124160 + 248044`` tombe à la colonne 123884 de la
    rangée suivante (et hors allocation pour la dernière rangée).
    """
    return gid - vocab_start_index


@dataclass
class ShardRequestParams:
    """Contrat d'une requête pour les processeurs shard-aware.

    Tous les ids sont **GLOBAUX** (vocab_size_global) — c'est le contrat du
    chemin livré ; le mapping vers les colonnes locales est le travail de
    ``apply_shard_aware_processors``.

    * ``stop_token_ids`` : ids masqués à -inf (MinTokensLogitsProcessor /
      min_tokens — le ``index_put_`` du crash historique) ;
    * ``banned_token_ids`` : ids masqués (bad_words / forbidden) ;
    * ``bias_token_ids`` : (id, delta) additifs (logit_bias, forced ids,
      pénalties fréquence/présence déjà réduites en deltas par l'appelant) ;
    * ``allowed_token_ids`` : liste blanche (None = pas de liste ; ensemble
      vide = contradiction de contrat → fail loud) ;
    * ``required_logprob_token_ids`` : ids dont la logprob de sortie est
      EXIGÉE (``logprob_token_ids`` livré) — hors shard ⇒ repli gather
      complet (spec §3.5-5) ou fail loud en mode strict ;
    * ``exotic`` : noms de processeurs non couverts (MinP, spec-decode
      ``apply_with_spec_decode``, restauration structured-output…) ⇒ repli.
    """

    stop_token_ids: tuple[int, ...] = ()
    banned_token_ids: tuple[int, ...] = ()
    bias_token_ids: tuple[tuple[int, float], ...] = ()
    allowed_token_ids: frozenset[int] | None = None
    required_logprob_token_ids: tuple[int, ...] = ()
    exotic: tuple[str, ...] = ()
    num_logprobs: int = 0  # 0 = aucune, n > 0 = top-n (couvert si n <= K'),
    # -1 = pleine-vocab → repli gather complet (motif num_logprobs_all)


@dataclass
class ShardProcessorState:
    """État de travail + compteur de replis par motif (instrumentation)."""

    counters: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n


@dataclass
class ProcessorOutcome:
    """Résultat de ``apply_shard_aware_processors``.

    * ``needs_full_gather`` : repli demandé (motif dans ``reason``) ;
    * ``masked`` : nombre d'écritures -inf effectuées dans ce shard ;
    * ``skipped_out_of_shard`` : ids masquants hors shard (normaux : c'est
      l'AUTRE rang qui les masque — l'union des deux shards = le masque
      global ; comptés pour observabilité) ;
    * ``required_local`` : (row, gid, col) des logprobs exigées couvertes par
      CE shard — consommées par ``shard_sample`` (aucun input déclaré n'est
      ignoré) ;
    * ``required_out_of_shard`` : (row, gid) exigés hors shard.
    """

    needs_full_gather: bool = False
    reason: str | None = None
    masked: int = 0
    skipped_out_of_shard: int = 0
    # biais (logit_bias/forced) hors shard : ignorés par CE rang (l'autre
    # rang les applique) — comptés pour l'observabilité (revue mimo S4)
    skipped_bias_out_of_shard: int = 0
    required_local: list[tuple[int, int, int]] = field(default_factory=list)
    required_out_of_shard: list[tuple[int, int]] = field(default_factory=list)
    # (row, gid) exigés couverts par le rang de sortie mais PAS par ce shard :
    # valeur calculée au rang de sortie (assemblage des sorties là-bas)
    required_elsewhere: list[tuple[int, int]] = field(default_factory=list)


def apply_shard_aware_processors(
    logits_shard: torch.Tensor,
    vocab_start_index: int,
    vocab_size_global: int,
    params: Sequence[ShardRequestParams] | ShardRequestParams,
    state: ShardProcessorState,
    *,
    strict: bool = False,
    output_vocab_start: int | None = None,
) -> ProcessorOutcome:
    """Applique la logique indexée par vocab EN shard-aware (spec §3.5-2).

    Chaque id global ``gid`` est mappé en colonne locale ``col =
    vocab_start_index`` via ``global_to_local`` ; le store est **borné** :
    seuls les ``0 <= col < vocab_local`` sont écrits, jamais ailleurs (le
    crash OOB de ``MinTokensLogitsProcessor.index_put_`` est structurellement
    impossible ici — test négatif + test de mutation à l'appui).

    * ids MASQUANTS (stop/banned) hors shard : **pas une erreur** — l'autre
      rang les masque ; l'union des deux rangs = le masque global exact.
      Comptés dans ``skipped_out_of_shard`` ;
    * ids EXIGÉS (``required_logprob_token_ids``, logprob de sortie) hors
      shard : le contrat exige une valeur que ce shard ne possède pas ⇒
      ``strict=True`` → ``ShardContractError`` (fail loud) ; sinon repli
      compté+loggé vers gather complet (``needs_full_gather``, motif
      ``logprob_token_ids_oob``) — spec §3.5-5.  ⚠ La décision est portée
      par le shard du **rang de sortie** (``output_vocab_start``, défaut =
      mon shard = mono-rang) : tous les rangs évaluent la MÊME condition sur
      des données répliquées ⇒ **décision symétrique sans collectif** — un
      repli d'un seul rang désynchroniserait tokens ET collectifs (revue GLM
      2026-10-28, critique 1).  Un id couvert par le rang de sortie mais pas
      par le mien est laissé au rang de sortie (``required_elsewhere`` : la
      valeur est assemblée là où sortent les sorties) ;
    * processeurs exotiques (``params.exotic`` non vide) ⇒ repli compté+loggé
      (motif ``processor_exotic``) — spec §3.5-5 ;
    * ``allowed_token_ids`` vide = contradiction (le livré masquerait tout) ⇒
      fail loud dans les deux modes.

    ⚠ Retours anticipés (exotic, required hors shard de sortie) : le shard
    peut être PARTIELLEMENT muté (les écritures des lignes déjà parcourues
    ne sont PAS annulées) — sans conséquence pour le repli (le chemin livré
    reprend les logits pleine-vocab), mais l'appelant ne doit pas réutiliser
    ce shard comme s'il était vierge (revue GLM 2026-10-28).  L'identité de
    repli porte sur le ``delivered_fn`` verbatim, pas sur ce tampon.
    """
    M, vocab_local = logits_shard.shape
    assert logits_shard.dim() == 2, (
        f"logits_shard doit être 2D : {tuple(logits_shard.shape)}"
    )
    assert vocab_local > 0, f"shard vide : vocab_local={vocab_local}"
    assert isinstance(vocab_start_index, int) and vocab_start_index >= 0, (
        f"vocab_start_index doit être un entier ≥ 0 : {vocab_start_index!r}"
    )
    assert vocab_start_index + vocab_local <= vocab_size_global, (
        f"shard [{vocab_start_index},{vocab_start_index + vocab_local}) hors du "
        f"vocab global [0,{vocab_size_global})"
    )
    if isinstance(params, ShardRequestParams):
        rows_params: list[ShardRequestParams] = [params] * M
    else:
        rows_params = list(params)
    assert len(rows_params) == M, (
        f"params : {len(rows_params)} requêtes pour M={M} lignes de logits"
    )
    # shard du rang de sortie (décision symétrique de repli) : défaut = mon
    # shard (mono-rang / rang de sortie = moi)
    out_start = vocab_start_index if output_vocab_start is None else output_vocab_start
    assert isinstance(out_start, int) and out_start >= 0, (
        f"output_vocab_start doit être un entier ≥ 0 : {out_start!r}"
    )
    assert out_start + vocab_local <= vocab_size_global, (
        f"shard de sortie [{out_start},{out_start + vocab_local}) hors du "
        f"vocab global [0,{vocab_size_global})"
    )

    outcome = ProcessorOutcome()
    dev = logits_shard.device

    # Collecte en TENSEURS (une liste blanche pleine-vocab = jusqu'à
    # vocab_local entrées par ligne : des listes Python seraient hors contrat
    # de coût ; le store reste borné par le filtre de colonnes ci-dessous).
    mask_rows_t: list[torch.Tensor] = []
    mask_cols_t: list[torch.Tensor] = []
    bias_rows_t: list[torch.Tensor] = []
    bias_cols_t: list[torch.Tensor] = []
    bias_deltas_t: list[torch.Tensor] = []
    cols_all = torch.arange(vocab_local, device=dev, dtype=torch.long)

    for row, p in enumerate(rows_params):
        if p.exotic:
            outcome.needs_full_gather = True
            outcome.reason = "processor_exotic"
            _note_fallback(
                state,
                "processor_exotic",
                f"processeurs non couverts (ligne {row}) : {list(p.exotic)}",
            )
            return outcome
        if p.allowed_token_ids is not None:
            assert len(p.allowed_token_ids) > 0, (
                f"allowed_token_ids vide (ligne {row}) : contradiction de contrat "
                f"(le livré masquerait tout le vocab)"
            )
        # ids masquants (stop/min_tokens, bad_words) : mapping borné
        local_cols = []
        for gid in (*p.stop_token_ids, *p.banned_token_ids):
            gid = _check_gid(gid, vocab_size_global, row)
            col = global_to_local(gid, vocab_start_index)
            if 0 <= col < vocab_local:
                local_cols.append(col)
            else:
                outcome.skipped_out_of_shard += 1
        if local_cols:
            mask_rows_t.append(
                torch.full((len(local_cols),), row, dtype=torch.long, device=dev)
            )
            mask_cols_t.append(torch.tensor(local_cols, dtype=torch.long, device=dev))
        # ids additifs (logit_bias / forced / pénalties) : même mapping
        add_cols = []
        add_deltas = []
        for gid, delta in p.bias_token_ids:
            gid = _check_gid(gid, vocab_size_global, row)
            col = global_to_local(gid, vocab_start_index)
            if 0 <= col < vocab_local:
                add_cols.append(col)
                add_deltas.append(float(delta))
            else:
                outcome.skipped_bias_out_of_shard += 1
        if add_cols:
            bias_rows_t.append(
                torch.full((len(add_cols),), row, dtype=torch.long, device=dev)
            )
            bias_cols_t.append(torch.tensor(add_cols, dtype=torch.long, device=dev))
            bias_deltas_t.append(
                torch.tensor(add_deltas, dtype=torch.float32, device=dev)
            )
        # liste blanche : complément local (l'union des shards = le masque
        # global ; hors shard = non autorisé localement, l'autre rang gère)
        if p.allowed_token_ids is not None:
            _lo, _hi = min(p.allowed_token_ids), max(p.allowed_token_ids)
            assert _lo >= 0 and _hi < vocab_size_global, (
                f"allowed_token_ids hors [0,{vocab_size_global}) (ligne {row}) : "
                f"[{_lo}, {_hi}]"
            )
            allowed_t = torch.tensor(
                sorted(p.allowed_token_ids), dtype=torch.long, device=dev
            )
            not_allowed = ~torch.isin(cols_all + vocab_start_index, allowed_t)
            n_na = int(not_allowed.sum())
            if n_na:
                mask_rows_t.append(
                    torch.full((n_na,), row, dtype=torch.long, device=dev)
                )
                mask_cols_t.append(cols_all[not_allowed].contiguous())
        # ids exigés (logprobs de sortie) — décision portée par le shard du
        # rang de sortie (SYMÉTIQUE entre rangs, voir docstring)
        for gid in p.required_logprob_token_ids:
            gid = _check_gid(gid, vocab_size_global, row)
            out_col = global_to_local(gid, out_start)
            if not (0 <= out_col < vocab_local):
                outcome.required_out_of_shard.append((row, gid))
                if strict:
                    raise ShardContractError(
                        f"contrat exige le token global {gid} (ligne {row}) hors "
                        f"du shard de sortie [{out_start},"
                        f"{out_start + vocab_local}) — impossible en shard "
                        f"sampling : repli gather complet requis"
                    )
                outcome.needs_full_gather = True
                outcome.reason = "logprob_token_ids_oob"
                _note_fallback(
                    state,
                    "logprob_token_ids_oob",
                    f"token global {gid} (ligne {row}) exigé hors shard de "
                    f"sortie (out_start={out_start}, local={vocab_local})",
                )
                return outcome
            col = global_to_local(gid, vocab_start_index)
            if 0 <= col < vocab_local:
                outcome.required_local.append((row, gid, col))
            else:
                outcome.required_elsewhere.append((row, gid))

    if mask_rows_t:
        rows_t = torch.cat(mask_rows_t)
        cols_t = torch.cat(mask_cols_t)
        # MASQUE DE STORE BORNÉ — garde structurelle contre le OOB : le
        # mapping peut produire un col hors shard (gid de l'autre rang), il
        # est rejeté AVANT tout store (ETAPE-B2 §5-1).
        _assert_store_bounded(cols_t, vocab_local, "masque")
        logits_shard.index_put_(
            (rows_t, cols_t),
            torch.full((), _INF, dtype=logits_shard.dtype, device=dev),
        )
        outcome.masked = int(rows_t.numel())
    if bias_rows_t:
        rows_t = torch.cat(bias_rows_t)
        cols_t = torch.cat(bias_cols_t)
        _assert_store_bounded(cols_t, vocab_local, "biais")
        deltas_t = torch.cat(bias_deltas_t).to(logits_shard.dtype)
        logits_shard.index_put_((rows_t, cols_t), deltas_t, accumulate=True)
    state.bump("processor_rows", M)
    return outcome


def _check_gid(gid: int, vocab_size_global: int, row: int) -> int:
    """Valide un id global et le normalise en int Python.

    ``operator.index`` accepte les entiers « réels » (int Python, entiers
    numpy/torch) mais refuse floats/strings — pas de coercion muette (revue
    mimo S5).  Retourne l'id normalisé (les ids numpy doivent rester
    utilisables dans le mapping).
    """
    try:
        g = operator.index(gid)
    except TypeError:
        raise AssertionError(
            f"token id global doit être un entier : {gid!r} (ligne {row})"
        ) from None
    assert 0 <= g < vocab_size_global, (
        f"token id global hors [0,{vocab_size_global}) : {g!r} (ligne {row})"
    )
    return g


def _assert_store_bounded(cols_t: torch.Tensor, vocab_local: int, what: str) -> None:
    """Masque de store BORNÉ : aucune colonne hors [0, vocab_local) ne peut
    être écrite — la garde structurelle contre le OOB historique
    (ETAPE-B2 §5-1).  Extraite pour être testable directement (une garde
    jamais exécutée ne prouve rien).
    """
    bad = cols_t[(cols_t < 0) | (cols_t >= vocab_local)]
    assert bad.numel() == 0, (
        f"colonne hors shard détectée avant store ({what}) : "
        f"{bad[:8].tolist()} (vocab_local={vocab_local})"
    )


# ===========================================================================
# 3. Top-K' local + mini gather
# ===========================================================================
def shard_topk(
    logits_shard: torch.Tensor,
    k_local: int,
    vocab_start_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K' local → (valeurs [M,k], ids GLOBAUX [M,k]).

    Les ids retournés sont GLOBAUX (``col + vocab_start_index``) : le merged
    set du mini gather parle ids globaux des deux côtés.  Ex aequo à la
    frontière K' : sélection arbitraire mais déterministe (``torch.topk``) —
    écart NOMMÉ module (voir docstring).
    """
    assert logits_shard.dim() == 2, (
        f"logits_shard doit être 2D : {tuple(logits_shard.shape)}"
    )
    M, vocab_local = logits_shard.shape
    k = min(k_local, vocab_local)
    assert k >= 1, f"k_local={k_local} hors contrat (shard {vocab_local} colonnes)"
    vals, cols = torch.topk(logits_shard, k, dim=1)
    return vals, cols.to(torch.int64) + int(vocab_start_index)


def default_gather_fn(
    values: torch.Tensor,
    ids: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mini gather [M, R·k] — défaut du mandat §3.5-3.

    * torch.distributed initialisé → all-gather des candidats (valeurs, ids
      globaux) et des lse de shard [M,1] par rang → concaténation dim=1 ;
    * sinon → **repli compté** (one-shot) mono-rang : merged = local.  En
      multi-rang réel sans collectif ce serait incomplet : le motif
      ``gather_unavailable`` est compté pour que l'appelant voie le repli.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        _note_fallback(
            None,
            "gather_unavailable",
            "torch.distributed non initialisé — mini gather replié en mono-rang",
        )
        return values, ids, lse

    world = dist.get_world_size()
    if world == 1:
        return values, ids, lse

    backend = dist.get_backend()

    def _gather(t: torch.Tensor) -> list[torch.Tensor]:
        # gloo ne transporte pas les tenseurs CUDA : passage CPU transparent
        # (le mini gather est ~10 Ko — le coût est hors graphe de toute façon).
        if t.is_cuda and backend == "gloo":
            t = t.cpu()
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous())
        if out[0].device != values.device:
            out = [o.to(values.device) for o in out]
        return out

    vals_list = _gather(values)
    ids_list = _gather(ids)
    lse_list = _gather(lse)
    return (
        torch.cat(vals_list, dim=1),
        torch.cat(ids_list, dim=1),
        torch.cat(lse_list, dim=1),
    )


# ===========================================================================
# 4. Tirage Philox par (request_id_hash, step) — avancé identiquement sur
#    les deux rangs (contre-based, sans état)
# ===========================================================================
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_ROUNDS = 10


def _mul_lo_hi(a: Any, b: Any) -> tuple[Any, Any]:
    """(lo32, hi32) du produit a·b pour a,b uint32 (int ou tenseurs int64).

    Décomposition 16 bits : le produit plein 32×32 peut dépasser int64
    (2^64) — les sommes intermédiaires restent < 2^50.
    """
    x = (a & 0xFFFF) * (b & 0xFFFF)
    y = (a >> 16) * (b & 0xFFFF) + (a & 0xFFFF) * (b >> 16)
    z = (a >> 16) * (b >> 16)
    lo_full = x + (y << 16)
    lo = lo_full & 0xFFFFFFFF
    hi = (z + (lo_full >> 32)) & 0xFFFFFFFF
    return lo, hi


def philox_uniforms(
    request_id_hashes: torch.Tensor,
    step: int,
    n: int,
    device: torch.device | str,
) -> torch.Tensor:
    """[M, n] uniforms fp32 dans [0,1) — Philox 4x32-10 (Random123).

    Clé = ``(request_id_hash lo32, hi32)`` ; compteur = ``(step, bloc, 0, 0)``
    (4 sorties u32 par invocation, ``bloc = col // 4``).  Fonction PURE de
    ``(request_id_hash, step)`` : avancée identique sur les deux rangs, sans
    état — même stream des deux côtés ⇒ même tirage (le mapeage RNG diffère
    du livré : écart NOMMÉ module).

    ``request_id_hashes`` : int64 [M] dans [0, 2^63) (fail loud sinon).
    """
    M = request_id_hashes.shape[0]
    assert request_id_hashes.dtype == torch.int64, (
        f"request_id_hashes doit être int64 : {request_id_hashes.dtype}"
    )
    assert bool((request_id_hashes >= 0).all()), (
        "request_id_hashes doit être ≥ 0 (int64 non signé de fait)"
    )
    assert 0 <= step < 2**32, f"step hors [0, 2^32) : {step}"
    assert n >= 1, f"n={n} doit être ≥ 1"

    n_blocks = (n + 3) // 4
    h = request_id_hashes.to(torch.int64)
    k0 = (h & 0xFFFFFFFF)[:, None].expand(M, n_blocks)
    k1 = ((h >> 32) & 0xFFFFFFFF)[:, None].expand(M, n_blocks)
    c0 = torch.full((M, n_blocks), step & 0xFFFFFFFF, dtype=torch.int64, device=device)
    c1 = (
        torch.arange(n_blocks, dtype=torch.int64, device=device)[None, :]
        .expand(M, n_blocks)
        .contiguous()
    )
    c2 = torch.zeros((M, n_blocks), dtype=torch.int64, device=device)
    c3 = torch.zeros((M, n_blocks), dtype=torch.int64, device=device)

    for _ in range(_PHILOX_ROUNDS):
        lo0, hi0 = _mul_lo_hi(_PHILOX_M0, c0)
        lo1, hi1 = _mul_lo_hi(_PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + _PHILOX_W0) & 0xFFFFFFFF
        k1 = (k1 + _PHILOX_W1) & 0xFFFFFFFF

    out = torch.stack((c0, c1, c2, c3), dim=-1).reshape(M, n_blocks * 4)[:, :n]
    return ((out.to(torch.float64) + 0.5) * (2.0**-32)).to(torch.float32)


# ===========================================================================
# 5. Shard sampling (top-k/p + softmax + tirage) avec replis comptés
# ===========================================================================
@dataclass
class ShardLogprobs:
    """Logprobs du chemin sharded (sémantique : ``log_softmax`` des logits
    traités — post-température, post-processeurs, PRÉ-troncature ; écart
    NOMMÉ vs le mode ``processed_logits`` livré, voir docstring module).

    ``required`` : logprobs des ids exigés couverts par ce shard (contract
    ``logprob_token_ids``), alignées sur ``required_ids`` = (row, gid).
    """

    chosen: torch.Tensor  # [M] fp32 — logprob du token tiré
    top_ids: torch.Tensor  # [M, n] int64 (globaux)
    top_logprobs: torch.Tensor  # [M, n] fp32
    required_ids: tuple[tuple[int, int], ...] = ()
    required: torch.Tensor | None = None  # [R] fp32


@dataclass
class SampleResult:
    """Résultat de ``shard_sample`` (ou du repli verbatim ``delivered_fn``).

    ``used_shard_path=False`` ⇒ ``tokens``/``logprobs`` viennent du repli
    (chemin livré) et ``fallback_reason`` nomme le motif.
    """

    tokens: torch.Tensor | None  # [M] int64
    logprobs: ShardLogprobs | None
    used_shard_path: bool
    fallback_reason: str | None = None


def _fallback_result(
    state: ShardProcessorState | None,
    reason: str,
    delivered_fn: Callable[[], Any] | None,
) -> Any:
    """Repli = chemin livré VERBATIM : ``delivered_fn()`` rappelé tel quel
    (identité d'objet — le diff du fallback est vide, méthode
    ``audit_fallback_identity``).  Sans ``delivered_fn`` : signal nu
    (``used_shard_path=False``) pour que l'appelant fasse son gather complet."""
    if delivered_fn is not None:
        return delivered_fn()
    return SampleResult(
        tokens=None, logprobs=None, used_shard_path=False, fallback_reason=reason
    )


def _argmax_lowest_id(vals: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Argmax parmi les candidats fusionnés, déblocage d'ex aequo par le plus
    petit id global — identique au ``argmax`` livré (première colonne = plus
    petit id du vocab).  Rangée entièrement masquée (-inf partout) : renvoie
    le plus petit id du merged set (le livré renvoie 0) — deux ordures
    divergentes dans un cas déjà hors contrat (écart NOMMÉ)."""
    max_v = vals.max(dim=1, keepdim=True).values
    is_max = vals == max_v
    big = torch.iinfo(torch.int64).max
    return torch.where(is_max, ids, torch.full_like(ids, big)).min(dim=1).values


def shard_sample(
    logits_shard: torch.Tensor,
    vocab_start_index: int,
    vocab_size_global: int,
    params: Sequence[ShardRequestParams] | ShardRequestParams,
    state: ShardProcessorState,
    *,
    k_prime: int = DEFAULT_K_PRIME,
    gather_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    | None = None,
    request_id_hashes: torch.Tensor | None = None,
    step: int = 0,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    delivered_fn: Callable[[], Any] | None = None,
    output_vocab_start: int | None = None,
) -> Any:
    """Sampling shard-aware complet (spec §3.5) — gate OFF ou repli =
    ``delivered_fn()`` verbatim.

    Pipeline gate-ON : processeurs shard-aware → top-K' local → mini gather
    [M, 2K'] → top-k/p (sémantique livrée ``apply_top_k_top_p_pytorch`` :
    masque par valeur, top-p = « gardé iff masse strictement meilleure < p »,
    au moins le max conservé) + softmax renormalisé sur les survivants →
    tirage Philox (request_id_hash, step) par inverse-CDF → mêmes tokens sur
    tous les rangs.

    Greedy (temp < eps) : argmax fusionné = identique au livré.  Logprobs :
    décomposition logsumexp exacte (``logZ = logaddexp(lse_shard…)``) =
    identiques à l'ordre de réduction fp32 près.

    Replis comptés + loggés (one-shot) vers gather complet :
    ``gate_off``, ``num_logprobs_all`` (num_logprobs == -1),
    ``logprobs_mixed`` (num_logprobs hétérogènes dans le batch — cas réel
    ordinaire : repli, pas de crash), ``logprobs_topn_uncovered``
    (num_logprobs > K'), ``logprob_token_ids_oob`` (décision SYMÉTIQUE
    entre rangs, portée par ``output_vocab_start``), ``processor_exotic``,
    ``top_k_uncovered`` (top_k > K'), ``top_p_uncovered`` (noyau non couvert
    par le merged set), ``no_truncation`` (ni top-k ni top-p : support =
    vocab entier).

    Contrats d'appel NOMMÉS (revue GLM 2026-10-28) :

    * ``temperature=None`` ⇒ **tout le batch est greedy** (argmax).  Ce n'est
      PAS « température 1.0 » : pour du tirage sans température, passer un
      tenseur de 1.0 (le livré interprète l'absence de température
      différemment — écart NOMMÉ, l'appelant doit être explicite) ;
    * ``request_id_hashes`` doit être **unique par rangée du batch** (assert
      fail loud) : le tirage est une fonction pure de (hash, step) — deux
      rangées même hash = tirages identiques pour toujours (n>1 samples,
      branches de beam : le hash doit incorporer l'index d'échantillon).
    """
    M, vocab_local = logits_shard.shape
    assert logits_shard.dim() == 2 and logits_shard.is_floating_point(), (
        f"logits_shard doit être flottant 2D : {tuple(logits_shard.shape)} "
        f"{logits_shard.dtype}"
    )
    assert vocab_local > 0 and vocab_size_global > 0, (
        f"vocab vide : local={vocab_local} global={vocab_size_global}"
    )
    assert vocab_start_index >= 0, (
        f"vocab_start_index doit être ≥ 0 : {vocab_start_index}"
    )
    assert vocab_start_index + vocab_local <= vocab_size_global, (
        f"shard [{vocab_start_index},{vocab_start_index + vocab_local}) hors "
        f"[0,{vocab_size_global})"
    )
    assert k_prime >= 1, f"k_prime={k_prime} hors contrat (K'=32-64 mandat)"
    if isinstance(params, ShardRequestParams):
        rows_params: list[ShardRequestParams] = [params] * M
    else:
        rows_params = list(params)
    assert len(rows_params) == M, (
        f"params : {len(rows_params)} requêtes pour M={M} lignes"
    )

    if not mk_sampling_enabled():
        _note_fallback(
            state,
            "gate_off",
            "gate VLLM_MOE_MEGAKERNEL_SAMPLING OFF — chemin livré verbatim",
        )
        return _fallback_result(state, "gate_off", delivered_fn)

    # --- processeurs shard-aware (mapping global→local, store borné) --------
    outcome = apply_shard_aware_processors(
        logits_shard,
        vocab_start_index,
        vocab_size_global,
        rows_params,
        state,
        output_vocab_start=output_vocab_start,
    )
    if outcome.needs_full_gather:
        return _fallback_result(state, outcome.reason, delivered_fn)

    # --- logprobs : replis nommés (spec §3.5-5) ---------------------------
    nls = {p.num_logprobs for p in rows_params}
    if len(nls) > 1:
        # cas réel ordinaire (batch mixte logprobs/non) : repli compté+loggé,
        # PAS un crash (revue GLM 2026-10-28, critique 2)
        _note_fallback(
            state,
            "logprobs_mixed",
            f"num_logprobs hétérogènes dans le batch : {sorted(nls)}",
        )
        return _fallback_result(state, "logprobs_mixed", delivered_fn)
    num_logprobs = nls.pop()
    if num_logprobs == -1:
        _note_fallback(
            state,
            "num_logprobs_all",
            "num_logprobs == -1 (logprobs pleine-vocab) — gather complet requis",
        )
        return _fallback_result(state, "num_logprobs_all", delivered_fn)
    k_loc = min(k_prime, vocab_local)
    if num_logprobs > k_loc:
        _note_fallback(
            state,
            "logprobs_topn_uncovered",
            f"num_logprobs={num_logprobs} > K'={k_loc} — le merged set ne "
            f"couvre pas le top-n global",
        )
        return _fallback_result(state, "logprobs_topn_uncovered", delivered_fn)

    # --- température (livré : < eps ⇒ greedy, température forcée à 1.0) ----
    if temperature is None:
        temp = torch.ones(M, device=logits_shard.device)
    else:
        assert temperature.shape == (M,), (
            f"temperature doit être [{M}] : {tuple(temperature.shape)}"
        )
        temp = torch.where(
            temperature < _SAMPLING_EPS,
            torch.ones_like(temperature),
            temperature,
        )
    is_random = temperature is not None and bool((temperature >= _SAMPLING_EPS).any())
    # masque de rangées aléatoires depuis la température D'ORIGINE (le
    # remplacement < eps → 1.0 ne doit pas déclasser une rangée greedy en
    # rangée tirée)
    if temperature is None:
        is_random_row = torch.zeros(M, dtype=torch.bool, device=logits_shard.device)
    else:
        is_random_row = temperature >= _SAMPLING_EPS

    # --- top-K' local → mini gather → lse décomposé ------------------------
    top_vals, top_ids = shard_topk(logits_shard, k_loc, vocab_start_index)
    lse_shard = torch.logsumexp(logits_shard / temp[:, None], dim=1)
    gf = gather_fn or default_gather_fn
    m_vals, m_ids, m_lse = gf(top_vals, top_ids, lse_shard.unsqueeze(1))
    assert m_vals.shape == m_ids.shape and m_vals.shape[0] == M, (
        f"gather_fn a renvoyé une forme incohérente : valeurs "
        f"{tuple(m_vals.shape)} vs ids {tuple(m_ids.shape)} (M={M})"
    )
    assert m_lse.shape[0] == M and m_lse.dim() == 2, (
        f"gather_fn : lse incohérent {tuple(m_lse.shape)} (attendu [M,·])"
    )
    # température sur les candidats fusionnés (mono : scaling par rangée,
    # invariance du top-K' local — même set que sur logits tempérés)
    m_vals_t = m_vals / temp[:, None]
    logZ = torch.logsumexp(m_lse, dim=1)  # Z du vocab entier, EXACT par
    # décomposition (logaddexp associatif à l'ordre fp32 près)

    # --- greedy : argmax fusionné, identique au livré ----------------------
    greedy_tokens = _argmax_lowest_id(m_vals_t, m_ids)

    tokens = greedy_tokens
    if is_random:
        drawn, fallback = _sample_random(
            m_vals_t,
            m_ids,
            logZ,
            state,
            request_id_hashes,
            step,
            is_random_row,
            top_k,
            top_p,
            k_loc,
            delivered_fn,
            greedy_tokens,
        )
        if fallback is not None:
            return fallback
        tokens = drawn

    logprobs = None
    if num_logprobs > 0 or outcome.required_local or outcome.required_elsewhere:
        # required_elsewhere ⇒ le résultat REFLETE le contrat même quand la
        # valeur est laissée au rang de sortie (required_ids=() de ce côté) —
        # aucun input déclaré n'est silencieusement perdu (canary)
        logprobs = _compute_logprobs(
            m_vals_t,
            m_ids,
            logZ,
            tokens,
            num_logprobs,
            outcome,
            logits_shard / temp[:, None],
        )

    state.bump("shard_path_calls")
    _log_once(
        "mk-sampling-active",
        "[mk-sampling] SAMPLING shard-aware ACTIF (VLLM_MOE_MEGAKERNEL_SAMPLING"
        "=1) : %d lignes, K'=%d, mini gather %d candidats/rang, tirage Philox "
        "(request_id_hash, step)",
        M,
        k_loc,
        k_loc,
    )
    return SampleResult(tokens=tokens, logprobs=logprobs, used_shard_path=True)


def _sample_random(
    m_vals_t: torch.Tensor,
    m_ids: torch.Tensor,
    logZ: torch.Tensor,
    state: ShardProcessorState,
    request_id_hashes: torch.Tensor | None,
    step: int,
    is_random_row: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    k_loc: int,
    delivered_fn: Callable[[], Any] | None,
    greedy_tokens: torch.Tensor,
) -> tuple[torch.Tensor | None, Any | None]:
    """Top-k/p + softmax + tirage Philox sur le merged set.

    Retourne ``(tokens, None)`` ou ``(None, résultat_de_repli)``.
    """
    M, C = m_vals_t.shape
    dev = m_vals_t.device

    # tri composite (valeur décroissante, id croissant) — déterministe, même
    # ordre des deux rangs (merged identique) et sémantique de déblocage
    # d'ex aequo NOMMÉE (écart vs le livré : tri positionnel).
    id_asc = torch.argsort(m_ids, dim=1, stable=True)
    vals_by_id = torch.gather(m_vals_t, 1, id_asc)
    val_desc = torch.argsort(vals_by_id, dim=1, descending=True, stable=True)
    order = torch.gather(id_asc, 1, val_desc)
    v_s = torch.gather(m_vals_t, 1, order)
    i_s = torch.gather(m_ids, 1, order)

    if top_k is not None:
        assert top_k.shape == (M,), f"top_k doit être [{M}] : {tuple(top_k.shape)}"
        if bool((top_k > k_loc).any()):
            _note_fallback(
                state,
                "top_k_uncovered",
                f"top_k max {int(top_k.max())} > K'={k_loc} — le merged set "
                f"ne couvre pas le top-k global",
            )
            return None, _fallback_result(state, "top_k_uncovered", delivered_fn)
    has_p = top_p is not None and bool((top_p < 1.0).any())
    if top_p is not None:
        assert top_p.shape == (M,), f"top_p doit être [{M}] : {tuple(top_p.shape)}"
        assert bool(((top_p >= 0.0) & (top_p <= 1.0)).all()), (
            f"top_p hors [0,1] : min={float(top_p.min())} max={float(top_p.max())}"
        )
    if top_k is None and not has_p:
        _note_fallback(
            state,
            "no_truncation",
            "ni top-k ni top-p : le support de la loi livrée est le vocab "
            "entier — irréductible à 2K' candidats",
        )
        return None, _fallback_result(state, "no_truncation", delivered_fn)

    # --- masque top-k (livré : valeur >= k-ième plus grande = conservé) ----
    if top_k is not None:
        k_eff = top_k.clamp(min=1).to(torch.int64)
        v_k = v_s.gather(1, (k_eff - 1).unsqueeze(1))  # [M,1]
        keep = v_s >= v_k  # les ex aequo à la frontière sont conservés
    else:
        keep = torch.ones_like(v_s, dtype=torch.bool)

    # --- masque top-p (livré : gardé iff masse strictement meilleure < p) --
    if has_p:
        neg = torch.where(keep, v_s, torch.full_like(v_s, _INF))
        if top_k is not None:
            # livré : softmax renormalisé sur les survivants top-k
            lse_kept = torch.logsumexp(neg, dim=1)
            q = torch.exp(v_s - lse_kept[:, None]) * keep
        else:
            # livré : softmax du vocab entier — décomposition exacte via logZ.
            # GARDE DE COUVERTURE : la masse du top-K' GLOBAL (les K'
            # premiers du merged, = top-K' global) doit être ≥ p, sinon le
            # noyau s'étend au-delà du merged set ⇒ repli (spec §3.5-5).
            q = torch.exp(v_s - logZ[:, None]) * keep
            mass_top = q[:, :k_loc].sum(dim=1)
            if bool((mass_top < top_p).any()):
                _note_fallback(
                    state,
                    "top_p_uncovered",
                    f"masse du top-K' global {float(mass_top.min()):.6f} < "
                    f"p={float(top_p.min()):.6f} — le noyau top-p dépasse le "
                    f"merged set (2K' candidats)",
                )
                return None, _fallback_result(state, "top_p_uncovered", delivered_fn)
        prefix = torch.cumsum(q, dim=1) - q  # masse strictement meilleure
        keep_p = prefix < top_p[:, None]
        keep_p[:, 0] = True  # livré : au moins le max conservé
        keep = keep & keep_p

    # --- softmax renormalisé sur les survivants + tirage Philox ------------
    law_logits = torch.where(keep, v_s, torch.full_like(v_s, _INF))
    law = torch.exp(law_logits - torch.logsumexp(law_logits, dim=1)[:, None])
    cum = torch.cumsum(law, dim=1)
    assert request_id_hashes is not None, (
        "tirage aléatoire requis : request_id_hashes [M] manquant "
        "(Philox par (request_id_hash, step))"
    )
    assert request_id_hashes.shape == (M,), (
        f"request_id_hashes doit être [{M}] : {tuple(request_id_hashes.shape)}"
    )
    # Contrat d'unicité (fail loud) : même hash = même stream Philox = mêmes
    # tirages pour toujours (revue GLM 2026-10-28) — le hash doit incorporer
    # l'index d'échantillon (n>1, branches de beam).  Coût : un sync prototype
    # (à déplacer hors chemin critique au câblage in-kernel).
    assert int(torch.unique(request_id_hashes).numel()) == M, (
        "request_id_hashes doit être unique par rangée du batch : "
        "(hash, step) est la graine du tirage — les doublons donneraient "
        "des tirages identiques pour toujours"
    )
    u = philox_uniforms(request_id_hashes, step, 1, dev)  # [M,1]
    n_before = (cum <= u).sum(dim=1).clamp(max=C - 1)
    drawn = i_s.gather(1, n_before.unsqueeze(1)).squeeze(1)
    return torch.where(is_random_row, drawn, greedy_tokens), None


def _compute_logprobs(
    m_vals_t: torch.Tensor,
    m_ids: torch.Tensor,
    logZ: torch.Tensor,
    tokens: torch.Tensor,
    num_logprobs: int,
    outcome: ProcessorOutcome,
    logits_shard_t: torch.Tensor,
) -> ShardLogprobs:
    """log_softmax des logits traités (PRÉ-troncature) par décomposition :
    logprob(gid) = logit_tempéré(gid) − logZ, logZ = logaddexp des lse de
    shard (exact).  Top-n = top-n du merged set (exact si n ≤ K' — garde en
    amont).  Les ids EXIGÉS sont lus dans les logits du SHARD (colonne
    locale) — pas dans le merged set (où une colonne est un rang de candidat,
    pas un id de vocab).
    """
    M, C = m_vals_t.shape
    # logit du token tiré : GATHER par indice — pas de masque arithmétique
    # (l'ancien `(vals * onehot).sum()` donnait 0 × -inf = NaN dès qu'un
    # candidat du merged set est masqué à -inf — revue mimo C3 2026-10-29,
    # atteint via allowed_token_ids + num_logprobs > 0)
    pos = m_ids == tokens[:, None]
    assert bool((pos.sum(dim=1) == 1).all()), (
        "token tiré absent ou dupliqué dans le merged set — contrat violé"
    )
    idx = pos.to(torch.int64).argmax(dim=1)
    chosen_logit = m_vals_t.gather(1, idx.unsqueeze(1)).squeeze(1)
    chosen = chosen_logit - logZ

    top_ids = torch.empty((M, 0), dtype=torch.int64, device=m_ids.device)
    top_logprobs = torch.empty((M, 0), dtype=torch.float32, device=m_ids.device)
    if num_logprobs > 0:
        id_asc = torch.argsort(m_ids, dim=1, stable=True)
        vals_by_id = torch.gather(m_vals_t, 1, id_asc)
        val_desc = torch.argsort(vals_by_id, dim=1, descending=True, stable=True)
        order = torch.gather(id_asc, 1, val_desc)
        top_ids = torch.gather(m_ids, 1, order[:, :num_logprobs])
        top_logprobs = (
            torch.gather(m_vals_t, 1, order[:, :num_logprobs]) - logZ[:, None]
        )

    required = None
    required_ids: tuple[tuple[int, int], ...] = ()
    if outcome.required_local:
        rows = torch.tensor(
            [r for r, _, _ in outcome.required_local],
            dtype=torch.long,
            device=m_ids.device,
        )
        cols = torch.tensor(
            [c for _, _, c in outcome.required_local],
            dtype=torch.long,
            device=m_ids.device,
        )
        required = logits_shard_t[rows, cols] - logZ[rows]
        required_ids = tuple((r, g) for r, g, _ in outcome.required_local)
    return ShardLogprobs(
        chosen=chosen,
        top_ids=top_ids,
        top_logprobs=top_logprobs,
        required_ids=required_ids,
        required=required,
    )
