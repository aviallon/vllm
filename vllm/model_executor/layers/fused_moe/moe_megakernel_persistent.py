# SPDX-License-Identifier: Apache-2.0
"""INC-D v0 — boucle PERSISTANTE multi-couches : un lancement, N couches.

Campagne MEGAKERNEL full-forward 2026-10-28 (MI210 gfx90a, Triton 3.8 HIP) ;
mandat opérateur 09:15 : « INC-D EN PRIORITÉ — le jalon = UN launch qui
couvre les 40 couches (même avec des phases internes non optimales) + un
microbench wall vs chaîne livrée ».
`research/moe-megakernel-campaign-2026-10-27/TOUR3-FULL-FORWARD-DESIGN.md`
§3.1-3.3 (architecture persistante).

Ce module livre le **mécanisme** INC-D :

1. **Grille persistante de P CTAs** (P ≤ co-résidence mesurée 104 — garde
   fail loud héritée de `AR_MAX_CTAS`) qui parcourt une boucle de
   N_LAYERS couches DANS le kernel — un seul lancement ;
2. **File de travail** par phase (réclamation `atomic_add` — modèle Cohere
   §2.2 du design) : chaque CTA prend des items tant qu'il en reste ;
3. **Barrière de phase par compteurs globaux** (arrivée + epoch, RMW
   acquire/release — le mécanisme `_mk_ar_rendezvous` prouvé au tour 2) ;
   le dernier arrivé reset les deux files et avance l'epoch ;
4. **Replay CPU supprimé** : 1 lancement pour N_LAYERS × 2 phases au lieu
   de 2×N_LAYERS lancements — la variable (L) du design (1 322
   nœuds/token/rang, replay CPU 7-9,8 ms/pas).

**Couverture v0 (NOMMÉE, pas cachée)** : deux phases câblées en cascade
réelle (la sortie de la couche L est l'entrée de la couche L+1) :

* **A — ADD_NORM** : fused add-RMSNorm v2, contrat `_mk_norm_chunk`
  verbatim : ``s = fp32(x)+fp32(res)`` (add EN FP32), ``var = mean(s·s)``
  fp32, ``rstd = rsqrt(var+eps)``, ``y = bf16(s·rstd·w)``, ``res' = s``
  (NON arrondi) — 1 item = 1 rangée ;
* **B — GEMM dense W8A8 MFMA** : quant per-token fusionnée (passe 1
  absmax / passe 2 ``nearbyint`` half-even + clamp, épilogue × scale_a ×
  scale_b — contrat ``scaled_mm_fused_quant_kernel`` verbatim = le corps
  INC-A) — 1 item = 1 tuile [BM,BN], dot MFMA (mandat §3.4).

Les phases attention core (INC-B), GDN (INC-B livré, corps prêt à
inline), MoE (tour 1-2) et lm_head+sampling (INC-C) s'insèrent dans la
MÊME boucle comme opcodes supplémentaires.

⚠ **Écarts NOMMÉS** :
* **chaîne carrée v0** : ``N == K`` (la forme out-proj réelle
  [2048→2048]) — le tampon d'activation est unique et doit servir
  d'entrée à la couche suivante ; les shapes non carrées demandent deux
  tampons de largeurs distinctes (suite nommée) ;
* **poids partagés entre couches** dans le microbench (tables de
  pointeurs répétées) : le wall comparatif est nettoyé du trafic poids
  (même cache des deux bras) mais le plancher d'octets réel du modèle
  (1,7-2,5 Go/pas) n'est PAS mesuré ici ;
* **barrières grossières** (une par phase et par couche) : le design
  §3.3 vise des compteurs par dépendance chunkée ; leur coût est dans le
  wall mesuré (classe (L)/(K) à optimiser) ;
* **spin de barrière** : fenêtre de deadlock élargie (O3) — garde de
  co-résidence P ≤ 104 + plafond de spins fail loud + test de garde
  dédié + mutation-tests des compteurs.

Gate ``VLLM_MOE_MEGAKERNEL_PERSISTENT`` (défaut OFF, canaux doubles —
piège « déclaré mais jamais consommé » vécu le 2026-10-28) — aucun
câblage moteur ; gate OFF = repli = chaîne livrée verbatim (compté).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import triton
import triton.language as tl

from vllm import envs

logger = logging.getLogger(__name__)

__all__ = [
    "qwen3_mk_persistent_v0_kernel",
    "mk_phase_barrier",
    "mk_persistent_v0",
    "mk_persistent_fallback",
    "mk_persistent_enabled",
    "MK_P_RESIDENT_HARD",
]

# Co-résidence mesurée (smoke tour 2 : shared 65 Ko → 1 WG/CU → 104 CU) —
# au-delà, une CTA peut rester bloquée derrière des CTAs qui spinent :
# deadlock de scheduling.  Même discipline que AR_MAX_CTAS.
# Plafond de spins : au-delà, fail loud (drapeau lu côté hôte) — jamais une
# boucle infinie silencieuse (doctrine : hang > corruption, mais un hang de
# TEST serait indéfini — le plafond le rend observable et borné).  Le
# chemin normal ne spinne jamais (µs) ; le plafond ne sert qu'aux pannes.
# ⚠ Borné en ITERATIONS, pas en secondes (revue deepseek F5) : calibré bas
# (~0,1-0,3 s de spin contentieux) pour qu'un test de mutation reste borné
# en temps ; injectable (`spin_limit`) pour les tests.
MK_P_SPIN_LIMIT = 500_000


def _resident_hard() -> int:
    """Co-résidence effective : min(104 mesuré tour 2, CU du device).

    ⚠ Revue deepseek F6 : le 104 vient du smoke d'un AUTRE noyau (shared
    65 Ko) ; sur un device à moins de CU, la garde codée en dur passerait
    alors que des CTAs ne sont pas co-résidentes → spin jusqu'au plafond
    (fail-stop tardif) au lieu d'un refus avant lancement.  Dérivé du
    device, la garde refuse AVANT (fail loud)."""
    try:
        idx = torch.cuda.current_device()
        sm = torch.cuda.get_device_properties(idx).multi_processor_count
    except Exception:
        sm = 104
    return min(104, int(sm))


MK_P_RESIDENT_HARD = 104  # plafond absolu (mesure tour 2)

P_BM = 16  # BM MFMA (mandat §3.4 : BM≥16, lignes pad masquées)
P_BN = 64
P_BK = 128
P_NUM_WARPS = 4
P_NUM_STAGES = 2

PHASE_ADD_NORM = tl.constexpr(0)
PHASE_GEMM = tl.constexpr(1)
N_PHASES = 2  # usage python (shapes)
PHASE_SLOTS = tl.constexpr(2)  # usage kernel (indexation) — un global
# non-constexpr dans un @triton.jit = NameError (piège connu)


# ---------------------------------------------------------------------------
# gates — canaux DOUBLES (os.environ prépondérant, puis attribut envs).
# Le getattr(envs,...) seul lit les variables NON ENREGISTERIES comme
# toujours False : c'est le bras muet du wall A/B 2026-10-28.
# ---------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() in ("1", "true")
    return bool(getattr(envs, name, False))


def mk_persistent_enabled() -> bool:
    return _env_flag("VLLM_MOE_MEGAKERNEL_PERSISTENT")


# ---------------------------------------------------------------------------
# mécanisme : barrière de phase (compteurs globaux, modèle tour 2)
# ---------------------------------------------------------------------------
@triton.jit
def mk_phase_barrier(
    arrive_ptr,  # [N_PHASES] i32 — compteurs d'arrivée (reset par phase)
    epoch_ptr,  # [N_PHASES] i32 — epochs (le dernier avance)
    fail_ptr,  # [1] i32 — drapeau fail loud (spin limit)
    phase: tl.constexpr,  # index de phase (constante par spécialisation)
    n_workers: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
    MUT_SKIP_RESET: tl.constexpr = False,
):
    """Arrivée + attente de fin de phase (barrière à compteurs).

    * chaque CTA qui termine incrémente ``arrive`` (RMW release) ;
    * la dernière (== n_workers) reset ``arrive`` puis avance ``epoch``
      (release) — **mutation ``MUT_SKIP_RESET``** : le reset de ``arrive``
      est sauté → ``n == n_workers`` n'est plus jamais vrai → la barrière
      ne converge plus → le plafond de spins fail loud se déclenche (le
      test attend cet échec borné) ;
    * les autres attendent le changement d'epoch (RMW acquire) avec
      plafond de spins : au-delà, ``fail = max(fail, 1)`` (release) —
      Triton interdit ``return`` dans un ``while``, la boucle porte donc
      la condition de sortie (échec = sortie avec epoch inchangé).

    ⚠ La FILE de travail n'est PAS reset ici (contrairement au premier
    design) : les claims sont des RMW `relaxed` et un reset par le
    dernier arrivé ouvrait une fenêtre de course avec les claims précoces
    de la phase suivante (items perdus en double — mesuré : des rangées
    entières sautées, dépendant du timing/nb de CTAs).  Le compteur court
    librement, item = claim − base_couche ; il n'y a donc rien à reset.
    La seule chose qui se reset est ``arrive`` (les incréments de la
    phase suivante ne peuvent pas précéder son propre epoch-bump).
    """
    cur_epoch = tl.atomic_add(epoch_ptr + phase, 0, sem="acquire", scope="sys")
    # acq_rel (et non release) : l'arête acquire sur l'incrémentation
    # établit formellement le happens-before « stores du producteur ->
    # loads du consommateur » pour les payloads qui transitent par cette
    # barrière (revue mimo W4 : en release-seul la visibilité des données
    # reposait sur la cohérence L2 seule, sûre en pratique sur gfx90a
    # mais non formelle).
    n = tl.atomic_add(arrive_ptr + phase, 1, sem="acq_rel", scope="sys") + 1
    if n == n_workers:
        if not MUT_SKIP_RESET:
            tl.atomic_xchg(arrive_ptr + phase, 0, sem="release", scope="sys")
        tl.atomic_xchg(epoch_ptr + phase, cur_epoch + 1, sem="release", scope="sys")
    else:
        spins = tl.zeros([1], dtype=tl.int32)
        e = tl.atomic_add(epoch_ptr + phase, 0, sem="acquire", scope="sys")
        while (e == cur_epoch) & (tl.sum(spins) <= SPIN_LIMIT):
            e = tl.atomic_add(epoch_ptr + phase, 0, sem="acquire", scope="sys")
            spins = spins + 1
        # sortie avec epoch inchangé = plafond atteint -> fail loud
        tl.atomic_max(
            fail_ptr, (e == cur_epoch).to(tl.int32), sem="release", scope="sys"
        )


@triton.jit
def qwen3_mk_persistent_v0_kernel(
    # tampons d'activations (contigus, largeur K)
    x_ptr,  # [2, M, K] bf16 — ping-pong d'entrée/sortie par couche
    y_ptr,  # [M, K] bf16 — scratch de la sortie de norm (phase A -> B)
    res_ptr,  # [2, M, K] bf16 — résiduel ping-pong
    # tables de pointeurs par couche (int64 -> pointer, cast load-bearing)
    normw_tab,  # [n_layers] ptr -> [K] bf16
    w_tab,  # [n_layers] ptr -> [K, N] int8 (contigu, stride (N,1))
    ws_tab,  # [n_layers] ptr -> [N] fp32
    # mécanisme
    arrive_ptr,  # [N_PHASES] i32
    epoch_ptr,  # [N_PHASES] i32
    work_ptr,  # [N_PHASES] i32
    fail_ptr,  # [1] i32
    n_items_norm,  # items phase A = M
    n_items_gemm,  # items phase B = cdiv(M,BM) * cdiv(N,BN)
    # scalaires
    M,
    N,
    n_layers,
    eps,
    # formes
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    N_WORKERS: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
    MUT_SKIP_RESET: tl.constexpr = False,
    MUT_WRONG_LAYER: tl.constexpr = False,  # hook : poids de la couche L+1
):
    """Un CTA = un travailleur persistant ; boucle sur les couches.

    Par couche L : phase A (ADD_NORM, items = rangées) → barrière →
    phase B (GEMM, items = tuiles) → barrière.  Lecture couche L dans
    ``x_ptr[L&1]``, écriture couche L+1 dans ``x_ptr[(L+1)&1]`` (tampon
    mort de la couche L — la phase A de L a fini avant la phase B).
    """
    pid = tl.program_id(axis=0)
    if pid >= N_WORKERS:
        return

    for L in tl.range(0, n_layers):
        lwin = L
        if MUT_WRONG_LAYER:
            lwin = tl.where(L + 1 >= n_layers, n_layers - 1, L + 1)
        lw = tl.load(normw_tab + lwin).to(tl.pointer_type(tl.bfloat16))
        lb = tl.load(w_tab + lwin).to(tl.pointer_type(tl.int8))
        lbs = tl.load(ws_tab + lwin).to(tl.pointer_type(tl.float32))

        xin = x_ptr + (L & 1) * M * K
        y_out = y_ptr
        res_in = res_ptr + (L & 1) * M * K
        res_out = res_ptr + ((L + 1) & 1) * M * K

        # ---------------- phase A : ADD_NORM (items = rangées) ----------
        # file PAR (couche, phase) : un slot de curseur neuf par couple —
        # chaque item est réclamé exactement une fois, les claims en trop
        # (CTA surnuméraires) sortent simplement de la boucle ; AUCUN reset
        # nulle part (le reset par le dernier arrivé était une fenêtre de
        # course avec les claims précoces — mesuré, items perdus).
        wslot = work_ptr + L * PHASE_SLOTS + PHASE_ADD_NORM
        item = tl.atomic_add(wslot, 1, sem="relaxed", scope="sys")
        while item < n_items_norm:
            r = item
            offs = tl.arange(0, K).to(tl.int64)
            x = tl.load(xin + r * K + offs, mask=offs < K, other=0.0).to(
                tl.float32
            )
            re = tl.load(res_in + r * K + offs, mask=offs < K, other=0.0).to(
                tl.float32
            )
            s_ = x + re
            var = tl.sum(s_ * s_) / K
            # rsqrt (et NON 1/sqrt) : primitive IDENTIQUE à torch.rsqrt du
            # repli — 1/sqrt en deux ops diverge d'1 ULP aux frontières
            # d'arrondi, et une chaîne de 40 couches en résiduel amplifie
            # ce grain en O(1) (mesuré : 0 exact jusqu'à ~20 couches puis
            # cascade — pas un bug de synchro, un écart de primitive).
            rstd = tl.math.rsqrt(var + eps)
            w = tl.load(lw + offs, mask=offs < K, other=0.0).to(tl.float32)
            y = (s_ * rstd * w).to(tl.bfloat16)
            tl.store(y_out + r * K + offs, y, mask=offs < K)
            tl.store(res_out + r * K + offs, s_.to(tl.bfloat16), mask=offs < K)
            item = tl.atomic_add(wslot, 1, sem="relaxed", scope="sys")
        mk_phase_barrier(
            arrive_ptr, epoch_ptr, fail_ptr, PHASE_ADD_NORM,
            n_workers=N_WORKERS, SPIN_LIMIT=SPIN_LIMIT,
            MUT_SKIP_RESET=MUT_SKIP_RESET,
        )

        # ---------------- phase B : GEMM dense W8A8 MFMA ----------------
        xout = x_ptr + ((L + 1) & 1) * M * K
        num_pid_n = tl.cdiv(N, BN)
        wslot_b = work_ptr + L * PHASE_SLOTS + PHASE_GEMM
        item = tl.atomic_add(wslot_b, 1, sem="relaxed", scope="sys")
        while item < n_items_gemm:
            pid_m = item // num_pid_n
            pid_n = item % num_pid_n
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            mmask = offs_m < M
            nmask = offs_n < N
            # passe 1 — absmax per-row (quant per-token fusionnée,
            # contrat scaled_mm_fused_quant verbatim)
            amax = tl.zeros((BM,), dtype=tl.float32)
            k_off = tl.arange(0, BK)
            for _k1 in tl.range(0, tl.cdiv(K, BK), num_stages=2):
                mk1 = k_off < K
                a_f = tl.load(
                    y_out + offs_m[:, None] * K + k_off[None, :],
                    mask=mmask[:, None] & mk1[None, :], other=0.0,
                )
                amax = tl.maximum(amax, tl.max(tl.abs(a_f), axis=1))
                k_off = k_off + BK
            inv_s = tl.where(amax == 0.0, 0.0, 127.0 / amax)
            scale_a = amax / 127.0
            # passe 2 — quant (nearbyint half-even + clamp) + dot MFMA
            acc = tl.zeros((BM, BN), dtype=tl.int32)
            offs_k = tl.arange(0, BK)
            for _k in tl.range(0, tl.cdiv(K, BK), num_stages=2):
                mk2 = offs_k < K
                a_f = tl.load(
                    y_out + offs_m[:, None] * K + offs_k[None, :],
                    mask=mmask[:, None] & mk2[None, :], other=0.0,
                )
                a_r = tl.extra.hip.libdevice.nearbyint(a_f * inv_s[:, None])
                a_q = tl.maximum(tl.minimum(a_r, 127.0), -128.0).to(tl.int8)
                b = tl.load(
                    lb + offs_k[:, None] * N + offs_n[None, :],
                    mask=mk2[:, None] & nmask[None, :], other=0,
                )
                acc = tl.dot(a_q, b, acc, out_dtype=tl.int32)
                offs_k = offs_k + BK
            sb = tl.load(lbs + offs_n, mask=nmask, other=0.0)
            acc_f = scale_a[:, None] * acc.to(tl.float32) * sb[None, :]
            tl.store(
                xout + offs_m[:, None] * N + offs_n[None, :],
                acc_f.to(tl.bfloat16),
                mask=mmask[:, None] & nmask[None, :],
            )
            item = tl.atomic_add(wslot_b, 1, sem="relaxed", scope="sys")
        mk_phase_barrier(
            arrive_ptr, epoch_ptr, fail_ptr, PHASE_GEMM,
            n_workers=N_WORKERS, SPIN_LIMIT=SPIN_LIMIT,
            MUT_SKIP_RESET=MUT_SKIP_RESET,
        )


# ---------------------------------------------------------------------------
# API python
# ---------------------------------------------------------------------------
_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


def mk_persistent_fallback(
    x: torch.Tensor,
    res: torch.Tensor,
    norm_ws: list[torch.Tensor],
    ws: list[torch.Tensor],
    w_scales: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Repli = chaîne livrée VERBATIM (gate OFF) : add-RMSNorm v2 puis
    ``triton_scaled_mm_fused_quant`` par couche, séquentiel (le livré
    alloue toujours sa sortie : le repli retourne un tenseur neuf)."""
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    cur_x = x.clone()
    cur_res = res.clone()
    for nw, w, wscl in zip(norm_ws, ws, w_scales):
        s = cur_x.float() + cur_res.float()
        var = (s * s).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        y = (s * rstd * nw.float()).to(torch.bfloat16)
        cur_x, _ = triton_scaled_mm_fused_quant(y, w, wscl, torch.bfloat16)
        cur_res = s.to(torch.bfloat16)
    return cur_x


def mk_persistent_v0(
    x: torch.Tensor,
    res: torch.Tensor,
    norm_ws: list[torch.Tensor],
    ws: list[torch.Tensor],
    w_scales: list[torch.Tensor],
    eps: float = 1e-6,
    n_workers: int = 64,
    spin_limit: int = MK_P_SPIN_LIMIT,
    out: torch.Tensor | None = None,
    # hooks de mutation (tests uniquement)
    _mut_skip_reset: bool = False,
    _mut_wrong_layer: bool = False,
) -> torch.Tensor:
    """Boucle persistante v0 — UN lancement pour N couches (INC-D).

    * ``x``/``res`` [M, K] bf16 (entrée couche 0) ; résultat [M, N] bf16 ;
    * ``norm_ws[L]`` [K] bf16, ``ws[L]`` int8 [K, N], ``w_scales[L]`` [N]
      fp32 — les poids peuvent être PARTAGÉS entre couches (mêmes
      tenseurs dans les tables de pointeurs) ;
    * chaîne carrée v0 : **``N == K``** (écart NOMMÉ — voir docstring
      module) ;
    * ``n_workers`` = P CTAs persistantes — **plafonné à la co-résidence
      mesurée** (assert fail loud : au-delà, deadlock de spin).

    Gate ``VLLM_MOE_MEGAKERNEL_PERSISTENT`` OFF → repli verbatim.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16, (
        f"x doit être bf16 2D : {x.dtype} {tuple(x.shape)}"
    )
    M, K = x.shape
    n_layers = len(norm_ws)
    assert n_layers == len(ws) == len(w_scales), (
        f"tables de poids incohérentes : {n_layers}/{len(ws)}/{len(w_scales)}"
    )
    assert n_layers > 0, "n_layers doit être >= 1"
    N = ws[0].shape[1]
    assert N == K, (
        f"chaîne carrée v0 : N={N} doit être == K={K} (écart NOMMÉ, "
        f"docstring module)"
    )
    assert K % 16 == 0 and (K & (K - 1)) == 0, (
        f"K={K} doit être une puissance de 2 multiple de 16 : le corps "
        f"norm utilise tl.arange(0, K) (exige pow2) et le dot MFMA exige "
        f"K%16==0"
    )
    for L in range(n_layers):
        assert norm_ws[L].shape == (K,), (
            f"norm_ws[{L}] doit être [{K}] : {tuple(norm_ws[L].shape)}"
        )
        assert ws[L].shape == (K, N) and ws[L].dtype == torch.int8, (
            f"ws[{L}] doit être int8 [{K},{N}] : "
            f"{ws[L].dtype} {tuple(ws[L].shape)}"
        )
        assert ws[L].stride(1) == 1, f"ws[{L}] doit être contigu : {ws[L].stride()}"
        assert w_scales[L].numel() == N and w_scales[L].is_floating_point(), (
            f"w_scales[{L}] doit être fp [{N}] : {tuple(w_scales[L].shape)}"
        )
    resident = _resident_hard()
    assert 1 <= n_workers <= resident, (
        f"n_workers={n_workers} hors [1, {resident}] — au-delà de la "
        f"co-résidence (min(104 mesuré, CU du device)), deadlock de "
        f"scheduling entre CTAs qui spinent (garde AR_MAX_CTAS, design "
        f"O3).  Repli : réduire n_workers."
    )
    for name, t in (
        ("x", x), ("res", res),
        *[(f"norm_ws[{L}]", norm_ws[L]) for L in range(n_layers)],
        *[(f"ws[{L}]", ws[L]) for L in range(n_layers)],
        *[(f"w_scales[{L}]", w_scales[L]) for L in range(n_layers)],
    ):
        assert t.device == x.device, f"{name} sur {t.device} != x sur {x.device}"

    if not mk_persistent_enabled():
        _log_once(
            "mk-persistent-fallback",
            "[mk-persistent] gate VLLM_MOE_MEGAKERNEL_PERSISTENT OFF — "
            "repli sur la chaîne livrée (add-RMSNorm + scaled_mm par couche)",
        )
        return mk_persistent_fallback(x, res, norm_ws, ws, w_scales, eps)

    device = x.device
    if out is None:
        out_t = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    else:
        assert out.shape == (M, N) and out.dtype == torch.bfloat16, (
            f"out doit être bf16 [{M},{N}] : {out.dtype} {tuple(out.shape)}"
        )
        out_t = out

    # tampons : x ping-pong [2,M,K], y scratch [M,K], res ping-pong [2,M,K]
    xbuf = torch.zeros((2, M, K), dtype=torch.bfloat16, device=device)
    xbuf[0] = x
    ybuf = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
    resbuf = torch.zeros((2, M, K), dtype=torch.bfloat16, device=device)
    resbuf[0] = res

    def _ptr_table(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
        )

    nw_tab = _ptr_table(norm_ws)
    w_tab = _ptr_table(ws)
    ws_tab = _ptr_table(w_scales)

    arrive = torch.zeros(N_PHASES, dtype=torch.int32, device=device)
    epoch = torch.zeros(N_PHASES, dtype=torch.int32, device=device)
    work = torch.zeros((n_layers, N_PHASES), dtype=torch.int32, device=device)
    fail = torch.zeros(1, dtype=torch.int32, device=device)

    _log_once(
        f"mk-persistent-{n_layers}x{M}x{K}",
        "[mk-persistent] boucle persistante ACTIVE "
        "(VLLM_MOE_MEGAKERNEL_PERSISTENT=1) : %d couches, M=%d K=%d, "
        "P=%d CTAs, 1 lancement (%d phases/couche)",
        n_layers, M, K, n_workers, N_PHASES,
    )

    n_items_norm = M
    n_items_gemm = triton.cdiv(M, P_BM) * triton.cdiv(N, P_BN)
    qwen3_mk_persistent_v0_kernel[(n_workers,)](
        xbuf, ybuf, resbuf,
        nw_tab, w_tab, ws_tab,
        arrive, epoch, work, fail,
        n_items_norm, n_items_gemm,
        M, N, n_layers, eps,
        K=K, BM=P_BM, BN=P_BN, BK=P_BK,
        N_WORKERS=n_workers, SPIN_LIMIT=int(spin_limit),
        num_warps=P_NUM_WARPS, num_stages=P_NUM_STAGES,
        MUT_SKIP_RESET=bool(_mut_skip_reset),
        MUT_WRONG_LAYER=bool(_mut_wrong_layer),
    )
    torch.cuda.synchronize()
    assert int(fail[0]) == 0, (
        "mk-persistent : plafond de spins atteint (fail loud) — barrière "
        "non convergée (compteurs corrompus ou mutation MUT_SKIP_RESET)"
    )
    out_t.copy_(xbuf[n_layers & 1])
    return out_t
