# SPDX-License-Identifier: Apache-2.0
"""GEMM dense W8A8 int8 en MFMA — moteur INC-A (campagne MEGAKERNEL full-forward).

Campagne MEGAKERNEL 2026-10-27 (MI210 gfx90a, Triton 3.8.0 HIP) ; mandat
INC-A « GEMM denses MFMA » :
`research/moe-megakernel-campaign-2026-10-27/TOUR3-FULL-FORWARD-DESIGN.md`
§5.1/§6.  Politique de tuiles = **§3.4 VERBATIM** : tout `tl.dot` est en forme
MFMA — BM≥16 (lignes pad MASQUÉES quand M<16 : BM=1 tombait en v_dot4,
mesuré 6-19× plus lent au tour 1) et dim de contraction ≥16, multiple de 16.
La vérification se fait AU CODEGEN (test ASM : `v_mfma` présent, `v_dot4`
absent) — un smoke qui ne regarde pas l'asm ne prouve rien sur le mandat.

``qwen3_dense_mfma_gemm_kernel`` : C = f(A[M,K], B_int8[K,N], b_scale[N]) →
sortie bf16 [M,N].  **Contrat numérique EXACTEMENT celui de
``scaled_mm_fused_quant_kernel``** (triton_scaled_mm.py, paire (a)
2026-10-23) — expressions reprises verbatim :

* A bf16 : passe 1 absmax par rangée sur K entier (fp32 max de |x|),
  ``scale_a = amax/127``, ``inv_s = 127/amax`` (0 si amax=0) ; passe 2 quant
  par chunk en registres ``q = clamp(nearbyint(x·inv_s), -128, 127)``
  (libdevice half-EVEN — PAS le ``round`` half-away de
  ``_per_token_quant_int8``) ;
* A int8 pré-quantifié : ``a_scale[M]`` per-token fourni (passe 1 sautée) ;
* accumulation **int32 EXACTE** : l'ordre de réduction K est libre (MFMA
  K≥16 vs v_dot4 K=4 ne change pas la somme entière — seule l'épilogue
  flottante compte, et elle est identique) ;
* épilogue : ``acc fp32 × scale_a[rangée] × scale_b[colonne]`` puis cast
  bf16 — mêmes expressions, même ordre d'arrondi que le livré.

Tuiles (TOUR3 §3.4, shapes locales TP=2, K=2048) : **(BM,BN,BK) =
(16,64,128)**, ``num_warps=4``, ``num_stages=2`` (double-buffer LDS :
fenêtre BK×BN int8 = 8 Ko × 2 = 16 Ko, dans les 64 Ko/CU mesurés).  M<16 :
lignes pad masquées — le motif de ``qwen3_moe_pair_fusion_kernel`` : le
``tl.dot`` reste [16,BK], forme MFMA.  N quelconque accepté (masque de
colonnes), K multiple de 16 (queue K masquée).

Gate ``VLLM_MOE_MEGAKERNEL_DENSE`` (défaut OFF) — **AUCUN câblage moteur** :
ce module ne touche ni triton_moe.py, ni envs.py, ni aucun chemin prod.
Gate OFF = repli = le chemin livré VERBATIM (`triton_scaled_mm` /
`triton_scaled_mm_fused_quant`).  Seul le launcher public
``dense_mfma_gemm`` est exposé.
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
    "qwen3_dense_mfma_gemm_kernel",
    "dense_mfma_gemm",
    "dense_mfma_enabled",
]

# Tuiles MFMA du mandat (TOUR3-FULL-FORWARD-DESIGN.md §3.4) : BM=16 (lignes
# pad masquées — le tl.dot reste [16,BK] = forme MFMA), BN=64, BK=128 ;
# num_stages=2 double-bufférise la tuile B (BK×BN int8 = 8 Ko × 2 = 16 Ko).
DENSE_BM = 16
DENSE_BN = 64
DENSE_BK = 128
DENSE_NUM_WARPS = 4
DENSE_NUM_STAGES = 2


@triton.jit
def qwen3_dense_mfma_gemm_kernel(
    # tenseurs
    a_ptr,  # [M, K] bf16 (quantifié à la volée) ou int8 (pré-quantifié)
    b_ptr,  # [K, N] int8
    b_scale_ptr,  # [N] fp32 per-channel
    a_scale_ptr,  # [M] fp32 per-token (mode A int8 ; None sinon)
    c_ptr,  # [M, N] bf16
    # scalaires (arguments scalaires — jamais une copie H2D par lancement)
    M,
    N,
    K,
    # strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # tuiles / mode
    A_IS_INT8: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """1 CTA = 1 tuile de sortie [BM, BN] — boucle K en tl.dot MFMA.

    Forme MFMA mandatée (§3.4) : dot [BM≥16, BK≥16] × [BK≥16, BN≥16] — les
    lignes pad (M<BM) et colonnes pad (N<BN) sont MASQUÉES en load/store,
    pas supprimées : supprimer BM=16 faisait tomber le dot en v_dot4.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Lignes/colonnes pad masquées (motif qwen3_moe_pair_fusion_kernel) :
    # offs < M/N partout, loads other=0, stores masqués — la sentinelle de
    # sortie au-delà de [M,N] reste intacte (test dédié).
    offs_m = pid_m * BM + tl.arange(0, BM).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN).to(tl.int64)
    masks_m = offs_m < M
    masks_n = offs_n < N

    # ------------------------------------------------------------------
    # Passe 1 — absmax per-row sur K entier (mode A bf16) : la quant
    # per-token dynamique exige la rangée complète avant de former la
    # moindre valeur quantifiée.  Charges masquées other=0.0 : la queue K
    # partiel |0| = 0, comme le kernel livré qui ne parcourt que les
    # éléments valides depuis un max initial 0.0.  Expressions VERBATIM
    # `scaled_mm_fused_quant_kernel` (bit-exactness contract).
    # ------------------------------------------------------------------
    if A_IS_INT8:
        scale_a = tl.load(a_scale_ptr + offs_m, mask=masks_m, other=0.0)
        inv_s = tl.zeros((BM,), dtype=tl.float32)  # inutilisé (pas de re-quant)
    else:
        amax = tl.zeros((BM,), dtype=tl.float32)
        k_off = tl.arange(0, BK).to(tl.int64)
        a_scan = a_ptr + stride_am * offs_m[:, None] + stride_ak * k_off[None, :]
        # num_stages=2 pipeline les charges entre itérations (même budget
        # LDS que le livré : fenêtre = max des dots, ici une seule boucle).
        for _k1 in tl.range(0, tl.cdiv(K, BK), num_stages=2):
            masks_k = k_off < K
            a_f = tl.load(a_scan, mask=masks_m[:, None] & masks_k[None, :], other=0.0)
            amax = tl.maximum(amax, tl.max(tl.abs(a_f.to(tl.float32)), axis=1))
            k_off = k_off + BK
            a_scan = a_scan + BK * stride_ak

        inv_s = tl.where(amax == 0.0, 0.0, 127.0 / amax)
        scale_a = amax / 127.0

    # ------------------------------------------------------------------
    # Passe 2 — quant de A par chunk en registres (VALU) + dot MFMA int32.
    # num_stages=2 : le travail VALU du chunk k+1 loge dans les bulles
    # MFMA/load du chunk k (contrat livré, K de dot BK=128 ≥ 16).
    # ------------------------------------------------------------------
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int64)
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + stride_ak * offs_k[None, :]
    b_ptrs = b_ptr + stride_bk * offs_k[:, None] + stride_bn * offs_n[None, :]
    for _k in tl.range(0, tl.cdiv(K, BK), num_stages=2):
        masks_k = offs_k < K
        masks_a = masks_m[:, None] & masks_k[None, :]
        if A_IS_INT8:
            a_q = tl.load(a_ptrs, mask=masks_a, other=0)
        else:
            a_f = tl.load(a_ptrs, mask=masks_a, other=0.0)
            # rounding VERBATIM du livré : nearbyint half-even + saturation
            a_r = tl.extra.hip.libdevice.nearbyint(a_f.to(tl.float32) * inv_s[:, None])
            a_q = tl.maximum(tl.minimum(a_r, 127.0), -128.0).to(tl.int8)

        masks_b = masks_k[:, None] & masks_n[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)

        acc = tl.dot(a_q, b, acc, out_dtype=tl.int32)

        offs_k = offs_k + BK
        a_ptrs = a_ptrs + BK * stride_ak
        b_ptrs = b_ptrs + BK * stride_bk

    # ------------------------------------------------------------------
    # Épilogue — expressions, ordre et arrondis VERBATIM
    # `scaled_mm_fused_quant_kernel` : acc fp32 × scale_a[rangée] puis
    # × scale_b[colonne], cast bf16.  (Le livré écrit `scale_b.T * acc`
    # avec scale_b broadcast (BN,1) — multiplication fp32 commutative au
    # bit près : per-colonne ici est bit-identique.)
    # ------------------------------------------------------------------
    scale_a_t = scale_a[:, None].to(tl.float32)
    acc_f = scale_a_t * acc.to(tl.float32)
    scale_b = tl.load(b_scale_ptr + offs_n, mask=masks_n, other=0.0)
    acc_f = acc_f * scale_b[None, :].to(tl.float32)

    c = acc_f.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c, mask=masks_m[:, None] & masks_n[None, :])


# ---------------------------------------------------------------------------
# API python
# ---------------------------------------------------------------------------
_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


def dense_mfma_enabled() -> bool:
    """Gate INC-A ``VLLM_MOE_MEGAKERNEL_DENSE`` (défaut OFF = repli livré).

    ⚠ La variable n'est PAS enregistrée dans ``vllm.envs`` (INC-A : aucun
    câblage, envs.py intouché) : le ``getattr(envs, ...)`` seul du pattern
    ``moe_megakernel_enabled()`` la lirait comme TOUJOURS False — la gate
    existerait mais ne s'allumerait jamais (piège « déclaré mais jamais
    consommé »).  Deux canaux, dans cet ordre :

    1. ``os.environ`` — le canal opérateur actuel, prépondérant ;
    2. attribut ``envs.VLLM_MOE_MEGAKERNEL_DENSE`` — canal de test / du
       futur enregistrement (pattern ``moe_megakernel_enabled()``).
    """
    raw = os.environ.get("VLLM_MOE_MEGAKERNEL_DENSE")
    if raw is not None:
        return raw.strip().lower() in ("1", "true")
    return bool(getattr(envs, "VLLM_MOE_MEGAKERNEL_DENSE", False))


def _dense_mfma_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    a_scale: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Repli = chemin livré VERBATIM (gate OFF) — aucune réécriture.

    A bf16 → ``triton_scaled_mm_fused_quant`` (le kernel de référence du
    contrat) ; A int8 pré-quantifié → ``triton_scaled_mm``.  Le seul
    ajout est la copie vers un tampon ``out`` pré-alloué (le livré alloue
    toujours sa sortie).
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
        triton_scaled_mm_fused_quant,
    )

    M, N = a.shape[0], b.shape[1]
    if a.dtype == torch.int8:
        assert a_scale is not None, (
            "repli livré : A int8 sans a_scale (contrat per-token requis)"
        )
        res = triton_scaled_mm(a, b, a_scale, b_scale, torch.bfloat16)
    else:
        res, _ = triton_scaled_mm_fused_quant(a, b, b_scale, torch.bfloat16)
    if out is None:
        return res
    out[:M, :N].copy_(res)
    return out


def dense_mfma_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GEMM dense W8A8 int8 en MFMA : C[M,N] bf16 = f(A[M,K], B[K,N], b_scale).

    Contrat numérique EXACTEMENT celui de ``scaled_mm_fused_quant_kernel``
    (voir le docstring du module) :

    * ``a`` bf16 [M,K] → quant per-token absmax/127 à la volée (passe 1),
      rounding ``nearbyint`` half-even + saturation (passe 2) ;
    * ``a`` int8 [M,K] pré-quantifié → ``a_scale`` [M] ou [M,1] fp32
      per-token REQUIS ;
    * ``b`` int8 [K,N], ``b_scale`` [N] ou [N,1] fp32 per-channel ;
    * ``K % 16 == 0`` (dim de contraction MFMA ; queue K masquée), N
      quelconque (masque de colonnes).

    ``out`` (optionnel) : tampon bf16 **ou fp32** [≥M, ≥N] pré-alloué — les
    lignes et colonnes hors [M,N] ne sont PAS écrites (masques du kernel).
    Le fp32 (INC-C, `moe_megakernel_sampling.mk_lm_head_logits`) est absorbé
    dans l'épilogue du kernel (``acc_f.to(c_ptr.element_ty)`` = no-op fp32) :
    la sortie fp32 est directe, sans cast bf16 intermédiaire.  Écart NOMMÉ
    vs le repli : le repli livré arrondit en bf16 puis cast fp32 (la chaîne
    livrée), le chemin gate-ON garde le fp32 de l'acc — meilleur d'un arrondi
    bf16, jamais pire.  Retourne le tampon de sortie (alloué ici si absent,
    bf16).

    Gate ``VLLM_MOE_MEGAKERNEL_DENSE`` : OFF → repli verbatim sur le chemin
    livré (``triton_scaled_mm`` / ``triton_scaled_mm_fused_quant``).
    """
    assert a.dim() == 2 and b.dim() == 2, (
        f"A et B doivent être 2D : A {tuple(a.shape)}, B {tuple(b.shape)}"
    )
    M, K = a.shape
    Kb, N = b.shape
    assert Kb == K, f"K désaligné : A {tuple(a.shape)} vs B {tuple(b.shape)}"
    assert M > 0 and N > 0 and K > 0, f"shape vide : M={M} N={N} K={K}"
    assert K % 16 == 0, (
        f"K={K} doit être un multiple de 16 (dim de contraction MFMA, "
        f"mandat TOUR3 §3.4)"
    )
    assert b.dtype == torch.int8, f"B doit être int8 : {b.dtype}"
    assert a.dtype in (torch.int8, torch.bfloat16), (
        f"A doit être bf16 (quant à la volée) ou int8 (pré-quantifié) : {a.dtype}"
    )
    assert b_scale.is_floating_point() and b_scale.numel() == N, (
        f"b_scale doit être fp [N={N}] per-channel : "
        f"{tuple(b_scale.shape)} {b_scale.dtype}"
    )
    b_scale = b_scale.reshape(-1)
    if a.dtype == torch.int8:
        assert a_scale is not None, (
            "A int8 pré-quantifié exige a_scale [M] per-token (contrat scaled_mm)"
        )
    else:
        assert a_scale is None, (
            "A bf16 : la quant est FUSIONNÉE (passe 1) — a_scale ne doit pas être passé"
        )
    if a_scale is not None:
        assert a_scale.is_floating_point() and a_scale.numel() == M, (
            f"a_scale doit être fp [M={M}] per-token : "
            f"{tuple(a_scale.shape)} {a_scale.dtype}"
        )
        a_scale = a_scale.reshape(-1)
    for name, t in (("A", a), ("B", b), ("b_scale", b_scale)):
        assert t.device == a.device, f"{name} sur {t.device} != A sur {a.device}"
    if a_scale is not None:
        assert a_scale.device == a.device, (
            f"a_scale sur {a_scale.device} != A sur {a.device}"
        )
    if out is not None:
        assert out.dtype in (torch.bfloat16, torch.float32) and out.dim() == 2, (
            f"out doit être bf16 ou fp32 2D : {out.dtype} {tuple(out.shape)}"
        )
        assert out.shape[0] >= M and out.shape[1] >= N, (
            f"out {tuple(out.shape)} trop petit pour [{M},{N}]"
        )
        assert out.device == a.device, f"out sur {out.device} != A sur {a.device}"

    if not dense_mfma_enabled():
        _log_once(
            "dense-mfma-fallback",
            "[dense-mfma] gate VLLM_MOE_MEGAKERNEL_DENSE OFF — repli sur le "
            "chemin livré (scaled_mm)",
        )
        return _dense_mfma_fallback(a, b, b_scale, a_scale, out)

    _log_once(
        f"dense-mfma-{a.dtype}-{M}x{N}x{K}",
        "[dense-mfma] GEMM dense MFMA ACTIF (VLLM_MOE_MEGAKERNEL_DENSE=1) : "
        "A %s [%d,%d] x B int8 [%d,%d], tuile (%d,%d,%d) nw=%d ns=%d",
        a.dtype,
        M,
        K,
        K,
        N,
        DENSE_BM,
        DENSE_BN,
        DENSE_BK,
        DENSE_NUM_WARPS,
        DENSE_NUM_STAGES,
    )

    if out is None:
        out_t = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    else:
        out_t = out

    grid = (triton.cdiv(M, DENSE_BM) * triton.cdiv(N, DENSE_BN),)
    qwen3_dense_mfma_gemm_kernel[grid](
        a,
        b,
        b_scale,
        a_scale,
        out_t,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        A_IS_INT8=(a.dtype == torch.int8),
        BM=DENSE_BM,
        BN=DENSE_BN,
        BK=DENSE_BK,
        num_warps=DENSE_NUM_WARPS,
        num_stages=DENSE_NUM_STAGES,
    )
    return out_t
