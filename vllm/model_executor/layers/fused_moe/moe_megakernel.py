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
  s'applique pas).

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

Identiques : quant de x (expressions de `_per_token_quant_int8`), échelles
per-channel, poids routé fp32 avant cast, sortie bf16 [M, K].

Gates : `VLLM_MOE_MEGAKERNEL` (défaut OFF = le chemin livré verbatim) ;
`VLLM_MOE_MEGAKERNEL_STRATEGY` = `pair` | `atomic` (défaut `pair`).
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import triton
import triton.language as tl

from vllm import envs

logger = logging.getLogger(__name__)

__all__ = [
    "qwen3_moe_pair_fusion_kernel",
    "qwen3_moe_megakernel",
    "moe_megakernel_finalize",
    "fused_moe_megakernel",
    "moe_megakernel_enabled",
    "moe_megakernel_strategy",
    "moe_megakernel_eligible",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@triton.jit
def _round_i8(x):
    """Rounding du contrat `_per_token_quant_int8` (libdevice.round)."""
    return tl.extra.hip.libdevice.round(x).to(tl.int8)


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
    num_valid_ptr,   # [1] int32 (sentinelles de padding au-delà)
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
        pairs = offs_pair
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    num_valid = tl.load(num_valid_ptr)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)

    tokens = pairs // top_k
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
    if NAIVE:
        e = tl.load(topk_ids_ptr + pairs_i64, mask=valid, other=0)
    else:
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
        wg = tl.load(w1_ptr + e64 * stride_w1e
                     + tl.arange(0, I)[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        wu = tl.load(w1_ptr + e64 * stride_w1e
                     + (I + tl.arange(0, I))[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        acc_g += tl.dot(x, wg)
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
    num_valid_ptr,
    stride_xm,
    stride_w1e,
    stride_w1n,
    stride_w2e,
    stride_w2k,
    top_k: tl.constexpr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BKO: tl.constexpr,
    CI: tl.constexpr,
    NAIVE: tl.constexpr,
):
    """1 CTA = (bloc de paires, chunk intermédiaire).

    Même pipeline que la stratégie `pair` mais le CTA ne couvre que `CI`
    colonnes intermédiaires (quant par (rangée, chunk) — écart nommé 2) et
    somme sa contribution par atomique fp32 locale dans out_acc (le `moe_sum`
    est absorbé). Chaque octet de poids est lu une fois (axe chunks sans
    redondance).
    """
    pid = tl.program_id(0)
    chunk = tl.program_id(1)

    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        pairs = offs_pair
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    num_valid = tl.load(num_valid_ptr)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)

    tokens = pairs // top_k
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
    if NAIVE:
        e = tl.load(topk_ids_ptr + pairs_i64, mask=valid, other=0)
    else:
        e = tl.load(expert_ids_ptr + pid)
    e64 = e.to(tl.int64)

    offs_i = chunk * CI + tl.arange(0, CI)
    offs_k = tl.arange(0, BKO)
    offs_o = tl.arange(0, K)

    acc_g = tl.zeros((BM, CI), dtype=tl.float32)
    acc_u = tl.zeros((BM, CI), dtype=tl.float32)
    x_rows = x_ptr + tokens.to(tl.int64) * stride_xm
    for k in range(0, tl.cdiv(K, BKO)):
        x = tl.load(x_rows[:, None] + (k * BKO + offs_k)[None, :],
                    mask=valid[:, None], other=0)
        wg = tl.load(w1_ptr + e64 * stride_w1e
                     + (chunk * CI + offs_i)[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        wu = tl.load(w1_ptr + e64 * stride_w1e
                     + (I + chunk * CI + offs_i)[None, :] * stride_w1n
                     + (k * BKO + offs_k)[:, None])
        acc_g += tl.dot(x, wg)
        acc_u += tl.dot(x, wu)

    sx = tl.load(x_scale_ptr + tokens.to(tl.int64), mask=valid, other=0.0)
    s1g = tl.load(w1_scale_ptr + e64 * (2 * I) + chunk * CI + offs_i)
    s1u = tl.load(w1_scale_ptr + e64 * (2 * I) + I + chunk * CI + offs_i)
    acc_g = acc_g * s1g * sx[:, None]
    acc_u = acc_u * s1u * sx[:, None]

    mid = (acc_g * tl.sigmoid(acc_g)) * acc_u
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    sc = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])

    # gemm2 : I entier du chunk, toute la ligne de sortie
    w2t = tl.load(w2_ptr + e64 * stride_w2e
                  + offs_i[:, None]
                  + offs_o[None, :] * stride_w2k)
    acc2 = tl.zeros((BM, K), dtype=tl.float32) + tl.dot(q, w2t)
    s2 = tl.load(w2_scale_ptr + e64 * K + offs_o)
    contrib = acc2 * (sc * wt)[:, None] * s2[None, :]

    # scope="gpu" : accumulateur local au GPU (pas d'accès pair — le piège
    # atomics fine-grained PCIe = NOP silencieux sur MI200 ne s'applique pas).
    tl.atomic_add(out_acc_ptr + tokens.to(tl.int64)[:, None] * K
                  + offs_o[None, :], contrib, mask=valid[:, None],
                  sem="relaxed", scope="gpu")


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


# ---------------------------------------------------------------------------
# API python
# ---------------------------------------------------------------------------
_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


def moe_megakernel_enabled() -> bool:
    return bool(getattr(envs, "VLLM_MOE_MEGAKERNEL", False))


def moe_megakernel_strategy() -> str:
    return str(getattr(envs, "VLLM_MOE_MEGAKERNEL_STRATEGY", "pair"))


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
    """None si éligible, sinon la raison du repli (comptée/loguée)."""
    if not moe_megakernel_enabled():
        return "gate off"
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
    if K % 128 != 0:
        return f"K {K} not multiple of BKO"
    if I % 8 != 0:
        return f"I {I} not multiple of CI"
    if hidden_states.size(0) > getattr(
        envs, "VLLM_MOE_MEGAKERNEL_MAX_TOKENS", 8
    ):
        return f"M={hidden_states.size(0)} above threshold"
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
    assert K % 1024 == 0, f"K={K} not multiple of finalize BLOCK"
    moe_megakernel_finalize_int8_kernel[(M,)](out_acc, q, s, K=K, BLOCK=1024)
    return q, s


def moe_megakernel_int8_out_enabled() -> bool:
    return bool(getattr(envs, "VLLM_MOE_MEGAKERNEL_INT8_OUT", False))


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrat `_per_token_quant_int8` (mêmes expressions)."""
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )
    return per_token_quant_int8(x)


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
) -> None:
    """Exécute la forward pass MoE routée fusionnée.

    stratégie `pair`   : 1 lancement (+ moe_sum aval par l'appelant) ;
    stratégie `atomic` : 2 lancements (kernel + finalize), sortie directe.
    Sortie : `output` [M, K] bf16 (contrat `moe_sum` livré).
    """
    M = hidden_states.size(0)
    top_k = topk_ids.size(1)
    K = w1.size(2)
    I = w2.size(2)
    strategy = moe_megakernel_strategy()

    if hidden_states.dtype is torch.bfloat16:
        x, x_scale = _quantize_rows(hidden_states)
    else:
        x = hidden_states
        x_scale = a1q_scale

    num_valid = M * top_k
    if naive:
        n_blocks = M * top_k
    else:
        assert sorted_token_ids is not None and expert_ids is not None
        n_blocks = triton.cdiv(sorted_token_ids.size(0), 16)
    BM = 1 if naive else 16
    BKO = 128

    if strategy == "atomic":
        CI = 8
        out_acc = torch.zeros((M, K), device=output.device,
                              dtype=torch.float32)
        grid = (n_blocks, triton.cdiv(I, CI))
        qwen3_moe_megakernel[grid](
            x, x_scale, w1, w1_scale, w2, w2_scale,
            topk_weights, topk_ids,
            sorted_token_ids if sorted_token_ids is not None else topk_ids,
            expert_ids if expert_ids is not None else topk_ids,
            out_acc,
            torch.tensor([num_valid], device=output.device,
                         dtype=torch.int32),
            x.stride(0), w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, CI=CI, NAIVE=naive,
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
    else:
        BN2 = 256
        assert cache3 is not None, "stratégie pair: cache3 requis"
        grid = (n_blocks, triton.cdiv(K, BN2))
        qwen3_moe_pair_fusion_kernel[grid](
            x, x_scale, w1, w1_scale, w2, w2_scale,
            topk_weights, topk_ids,
            sorted_token_ids if sorted_token_ids is not None else topk_ids,
            expert_ids if expert_ids is not None else topk_ids,
            cache3,
            torch.tensor([num_valid], device=output.device,
                         dtype=torch.int32),
            x.stride(0), w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1), cache3.stride(1),
            top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, BN2=BN2, NAIVE=naive,
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
