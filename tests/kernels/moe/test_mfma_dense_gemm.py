# SPDX-License-Identifier: Apache-2.0
"""Tests GPU du GEMM dense MFMA W8A8 (mandat INC-A, campagne MEGAKERNEL).

Exécution réelle MI210 (gfx90a) exigée — les tests numériques sont marqués
skip sans GPU.  Auto-portant : `python3 test_mfma_dense_gemm.py` (runner
__main__) ou pytest.

Contrat testé (TOUR3-FULL-FORWARD-DESIGN.md §3.4 pour les tuiles, docstring
de `moe_megakernel_dense.py` pour le contrat numérique) :

* (a) A int8 pré-quantifié : égalité `torch.equal` avec le chemin livré
  `triton_scaled_mm` (acc int32 exacte ⇒ bit-exact attendu), et A bf16 :
  `torch.equal` avec `triton_scaled_mm_fused_quant` (contrat EXACTEMENT
  celui de `scaled_mm_fused_quant_kernel` — expressions verbatim) ;
* (b) A bf16 quant à la volée : accord ≤ 1 LSB de quant avec la référence
  torch fp64 (pleine précision, ordre de réduction hors sujet) ;
* (c) lignes pad masquées : M=1 avec BM=16 n'écrit QUE la ligne 0 (les
  lignes 1..15 et les colonnes pad gardent la sentinelle) ;
* (d) mandat MFMA au CODEGEN : `v_mfma` présent, `v_dot4` absent ;
* (e) déterminisme run-to-run ;
* gate `VLLM_MOE_MEGAKERNEL_DENSE` : canal env/canal envs, défaut OFF,
  repli = chemin livré verbatim (identité de repli) ; fail loud sur les
  entrées hors contrat.
"""

from __future__ import annotations

import os
import re
import sys

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_megakernel_dense import (
    dense_mfma_enabled,
    dense_mfma_gemm,
    qwen3_dense_mfma_gemm_kernel,
)

torch.manual_seed(20261028)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU requis (MI210 gfx90a)"
)


def _gate_on() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_DENSE"] = "1"


def _gate_off() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_DENSE"] = "0"


def make_problem(M: int, N: int, K: int, device: str = "cuda"):
    """Problème W8A8 : A bf16 [M,K], B int8 [K,N], b_scale [N] per-channel."""
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    b = torch.randint(-127, 128, (K, N), device=device, dtype=torch.int8)
    b_scale = torch.rand(N, device=device) * 0.02 + 0.001
    return a, b, b_scale


def quant_contract_torch(a: torch.Tensor):
    """Quant per-token du contrat `scaled_mm_fused_quant_kernel` (expressions
    verbatim, torch) : absmax fp32 sur K entier, scale = amax/127,
    inv_s = 127/amax (0 si amax=0), q = clamp(round(x·inv_s), -128, 127).
    `torch.round` = half-even = `nearbyint` libdevice (cf.
    test_rint_half_even_primitive du module MoE, qui prouve l'égalité des
    deux primitives aux frontières)."""
    a_f = a.float()
    amax = a_f.abs().amax(dim=1, keepdim=True)
    scale = amax / 127.0
    inv_s = torch.where(amax == 0.0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(a_f * inv_s).clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


# ===========================================================================
# (a) contrat vs le chemin livré
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 32, 2048),  # GDN in_proj_ba local (N skinny) c1
        (1, 124160, 2048),  # lm_head local TP=2, c1
        (2, 2048, 2048),  # out proj / GDN out
        (8, 4096, 2048),  # GDN in_proj_qkv
        (32, 4608, 2048),  # attn QKV proj, M=32
        (4, 100, 1040),  # N quelconque + queue K partielle (K % 128 != 0)
    ],
)
def test_contract_int8_a_vs_shipped_scaled_mm(M, N, K):
    """A int8 pré-quantifié : égalité bit-à-bit avec `triton_scaled_mm`
    (le chemin livré partagé par les deux kernels — acc int32 exacte, même
    épilogue) : `torch.equal`, pas une tolérance."""
    _gate_on()
    a, b, b_scale = make_problem(M, N, K)
    a_q, a_scale = quant_contract_torch(a)

    out = dense_mfma_gemm(a_q, b, b_scale, a_scale=a_scale.view(-1))
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    ref = triton_scaled_mm(a_q, b, a_scale, b_scale, torch.bfloat16)
    torch.cuda.synchronize()

    assert out.shape == (M, N) and out.dtype == torch.bfloat16, (
        f"sortie {tuple(out.shape)} {out.dtype} != [{M},{N}] bf16"
    )
    assert torch.equal(out, ref), (
        f"A int8 [{M},{K}]x[{K},{N}] : diff max "
        f"{float((out.float() - ref.float()).abs().max())} vs triton_scaled_mm"
        f" — le contrat n'est pas bit-exact"
    )
    # garde anti-test-vide : la référence n'est pas nulle
    assert ref.abs().max() > 0


@needs_gpu
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 32, 2048),
        (8, 2048, 2048),
        (32, 4608, 2048),
    ],
)
def test_contract_bf16_a_vs_shipped_fused_quant(M, N, K):
    """A bf16 : `torch.equal` avec `triton_scaled_mm_fused_quant` — la
    preuve que le contrat est EXACTEMENT celui de
    `scaled_mm_fused_quant_kernel` (quant fusionnée, même rounding, même
    épilogue).  Régime fusionné livré respecté : M≤32, K%256==0."""
    _gate_on()
    a, b, b_scale = make_problem(M, N, K)

    out = dense_mfma_gemm(a, b, b_scale)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    ref, _ = triton_scaled_mm_fused_quant(a, b, b_scale, torch.bfloat16)
    torch.cuda.synchronize()

    assert torch.equal(out, ref), (
        f"A bf16 [{M},{K}]x[{K},{N}] : diff max "
        f"{float((out.float() - ref.float()).abs().max())} vs "
        f"scaled_mm_fused_quant_kernel — expressions du contrat divergentes"
    )


# ===========================================================================
# (b) A bf16 quant à la volée vs référence torch fp32 — ≤ 1 LSB de quant
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 32, 2048),
        (2, 2048, 2048),
        (8, 4096, 2048),
        (32, 4608, 2048),
    ],
)
def test_contract_bf16_quant_vs_fp32_reference(M, N, K):
    """Référence pleine précision : A d'origine × B déquantifié, en fp64
    (l'ordre de réduction fp32 de la référence ne doit pas mordre).  L'écart
    autorisé = la BORNÉE DE QUANTISATION : chaque valeur de A est arrondie
    à ±0.5 LSB (LSB = a_scale, la valeur d'échelle per-token), soit
    0.5·a_scale·Σ_k |B_dq[k,n]| par sortie, + 1 ulp bf16 d'arrondi de
    sortie.  La tolérance est donc ≤ 1 LSB de quant — et l'égalité bit-à-bit
    du contrat est déjà prouvée par le test (a)/(b) vs le livré."""
    _gate_on()
    a, b, b_scale = make_problem(M, N, K)
    out = dense_mfma_gemm(a, b, b_scale)
    torch.cuda.synchronize()

    _, a_scale = quant_contract_torch(a)
    b_dq = b.double() * b_scale.double().view(1, -1)  # [K, N] fp64
    ref_full = a.double() @ b_dq  # [M, N] fp64
    bound = 0.5 * a_scale.double() * b_dq.abs().sum(0).view(1, -1)
    # marge de rounding : (scale_a·inv_s ≈ 1 ± 2^-22) + 1 ulp bf16 de sortie
    tol = bound + ref_full.abs() * 2.0**-20 + out.double().abs() * 2.0**-8 + 1e-9
    diff = (out.double() - ref_full).abs()
    bad = int((diff > tol).sum())
    assert bad == 0, (
        f"{bad} éléments au-delà de 1 LSB de quant "
        f"(M={M} N={N} K={K}) : max diff={float(diff.max()):.6g}, "
        f"max tol={float(tol.max()):.6g}"
    )
    # garde anti-test-vide : la borne de quant MORD (elle n'est pas nulle
    # ni écrasante : l'erreur de quant réelle existe et est couverte)
    assert float(bound.max()) > 0
    assert float(diff.max()) > 0, "diff nulle : la référence fp64 n'a pas été comparée"


# ===========================================================================
# (c) lignes/colonnes pad masquées (BM=16 à M=1)
# ===========================================================================
@needs_gpu
def test_pad_rows_and_cols_masked():
    """M=1 avec BM=16 : SEULE la ligne 0 est écrite ; lignes 1..15 et
    colonnes pad (au-delà de N) gardent la sentinelle — c'est le masque de
    store qui garde le tl.dot en forme MFMA à M<16 (mandat §3.4)."""
    _gate_on()
    M, N, K = 1, 32, 256
    a, b, b_scale = make_problem(M, N, K)
    out = torch.full((16, 64), -384.0, device="cuda", dtype=torch.bfloat16)
    before = out.clone()

    dense_mfma_gemm(a, b, b_scale, out=out)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    ref, _ = triton_scaled_mm_fused_quant(a, b, b_scale, torch.bfloat16)
    torch.cuda.synchronize()

    # ligne 0, colonnes 0..N-1 : calculées == référence livrée
    assert torch.equal(out[:M, :N], ref), "ligne 0 != référence livrée"
    # garde anti-test-vide : la ligne 0 a réellement été écrite
    assert bool((out[:M, :N] != before[:M, :N]).any()), (
        "ligne 0 inchangée : le test ne prouve rien"
    )
    # lignes pad 1..15 : sentinelle intacte
    assert torch.equal(out[M:, :], before[M:, :]), (
        "lignes pad (1..15) ÉCRITES — masque de rangées cassé"
    )
    # colonnes pad N..63 de la ligne 0 : sentinelle intacte
    assert torch.equal(out[:M, N:], before[:M, N:]), (
        f"colonnes pad ({N}..63) ÉCRITES — masque de colonnes cassé"
    )


# ===========================================================================
# (d) mandat MFMA au codegen : v_mfma présent, v_dot4 absent
# ===========================================================================
def _codegen_asm(bm: int = 16) -> dict[bool, str]:
    """Capture le codegen — méthode `moe_megakernel_smoke.py` (ASTSource +
    triton_compile, wheel Triton 3.8.0 ; pitfall M5 : signature COMPLÈTE et
    TOUS les constexprs explicites).  `bm` exposé pour la branche de
    MUTATION (BM=1 doit être vu en v_dot4)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as triton_compile

    try:
        from triton.runtime.driver import driver as _triton_driver

        target = _triton_driver.active.get_current_target()
    except Exception:  # noqa: BLE001 — repli : la cible de la campagne
        target = GPUTarget("hip", "gfx90a", 32)

    asm_by_mode: dict[bool, str] = {}
    for a_is_int8 in (True, False):
        sig = {
            "a_ptr": "*i8" if a_is_int8 else "*bf16",
            "b_ptr": "*i8",
            "b_scale_ptr": "*fp32",
            "a_scale_ptr": "*fp32",
            "c_ptr": "*bf16",
            "M": "i32",
            "N": "i32",
            "K": "i32",
            "stride_am": "i32",
            "stride_ak": "i32",
            "stride_bk": "i32",
            "stride_bn": "i32",
            "stride_cm": "i32",
            "stride_cn": "i32",
        }
        # constexprs EXPLICITES (ASTSource n'applique pas les défauts)
        co = {"A_IS_INT8": a_is_int8, "BM": bm, "BN": 64, "BK": 128}
        src = ASTSource(fn=qwen3_dense_mfma_gemm_kernel, signature=sig, constexprs=co)
        c = triton_compile(
            src, target=target, options={"num_warps": 4, "num_stages": 2}
        )
        asm_by_mode[a_is_int8] = c.asm.get("amdgcn", "") or ""
    return asm_by_mode


@needs_gpu
def test_asm_mfma_not_dot4():
    """MANDAT MFMA (TOUR3 §3.4) : le dot de la boucle K doit s'abaisser en
    v_mfma (K de dot = BK = 128 ≥ 16), JAMAIS en v_dot4 (la mesure tour 1 :
    v_dot4 = 6-19× plus lent).  Un test qui ne regarde pas l'asm ne prouve
    rien sur le mandat.  Le seul `tl.dot` du kernel vit dans la boucle K :
    tout `v_mfma` de l'asm EST le dot de la boucle."""
    asm_by_mode = _codegen_asm()
    assert len(asm_by_mode) == 2, "les deux modes A doivent compiler"
    for a_is_int8, asm in asm_by_mode.items():
        assert asm, f"mode A_IS_INT8={a_is_int8} : codegen amdgcn vide"
        n_mfma = len(re.findall("v_mfma", asm))
        n_dot4 = len(re.findall("v_dot4", asm))
        assert n_mfma >= 1, (
            f"mode A_IS_INT8={a_is_int8} : AUCUN v_mfma dans le codegen — "
            f"le mandat MFMA n'est pas rempli (dot tombé en micro-dots ?)"
        )
        assert n_dot4 == 0, (
            f"mode A_IS_INT8={a_is_int8} : {n_dot4} v_dot4 dans le codegen "
            f"— mandat MFMA violé (forme de dot non-MFMA : vérifier BM≥16 "
            f"et BK≥16)"
        )
    # MUTATION (revue mimo) : le test doit pouvoir DISCORDER — BM=1 (forme
    # non-MFMA mesurée tour 1 : v_dot4 = 6-19x plus lent) doit être vu en
    # v_dot4.  Sans cette branche négative, un test ASM qui passe dans les
    # deux cas ne prouverait rien.
    asm_bad = _codegen_asm(bm=1)[True]
    n_dot4_bad = len(re.findall("v_dot4", asm_bad))
    print(f"asm mutation BM=1 (attendu v_dot4) : {n_dot4_bad} occurrences")
    assert n_dot4_bad > 0, (
        "mutation: BM=1 n'émet plus de v_dot4 — le test ASM perd sa "
        "capacité à discorder (ou le codegen a changé : re-évaluer)")


# ===========================================================================
# (e) déterminisme run-to-run
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize("M,N,K", [(2, 2048, 2048), (8, 4608, 2048)])
def test_determinism_run_to_run(M, N, K):
    """Pas d'atomique dans ce kernel : deux runs doivent être bit-identiques
    (gate de déterminisme)."""
    _gate_on()
    a, b, b_scale = make_problem(M, N, K)
    a_q, a_scale = quant_contract_torch(a)

    outs_f = [dense_mfma_gemm(a, b, b_scale) for _ in range(2)]
    outs_i = [
        dense_mfma_gemm(a_q, b, b_scale, a_scale=a_scale.view(-1)) for _ in range(2)
    ]
    torch.cuda.synchronize()

    assert torch.equal(outs_f[0], outs_f[1]), (
        f"mode bf16 [{M},{N}] : runs divergents "
        f"(max diff {float((outs_f[0].float() - outs_f[1].float()).abs().max())})"
    )
    assert torch.equal(outs_i[0], outs_i[1]), (
        f"mode int8 [{M},{N}] : runs divergents "
        f"(max diff {float((outs_i[0].float() - outs_i[1].float()).abs().max())})"
    )


# ===========================================================================
# gate + repli
# ===========================================================================
def test_gate_channels_and_default_off(monkeypatch):
    """Gate `VLLM_MOE_MEGAKERNEL_DENSE`, défaut OFF.

    CANARI declared-input : `os.environ` (canal opérateur) doit allumer la
    gate même sans enregistrement dans `vllm.envs` (interdit en INC-A) —
    sinon la variable existerait sans jamais rien activer."""
    from vllm import envs

    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_DENSE", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_DENSE", False, raising=False)
    assert dense_mfma_enabled() is False, "défaut doit être OFF"

    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "1")
    assert dense_mfma_enabled() is True, (
        "VLLM_MOE_MEGAKERNEL_DENSE=1 ignoré — la gate ne s'allume jamais"
    )
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "true")
    assert dense_mfma_enabled() is True
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "0")
    assert dense_mfma_enabled() is False

    # canal envs (tests / futur enregistrement), sans variable d'env
    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_DENSE", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_DENSE", True, raising=False)
    assert dense_mfma_enabled() is True, "canal envs.VLLM_MOE_MEGAKERNEL_DENSE ignoré"


@needs_gpu
def test_fallback_gate_off_is_shipped_path(monkeypatch):
    """Gate OFF : repli = chemin livré VERBATIM.  Le kernel MFMA ne doit PAS
    tourner (compteur) et le résultat doit être celui des kernels livrés
    (identité de repli, méthode audit_fallback_identity) — y compris la
    copie vers un tampon `out` pré-alloué."""
    _gate_off()
    import vllm.model_executor.layers.fused_moe.moe_megakernel_dense as mdd

    calls = {"n": 0}

    class _CountingKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                calls["n"] += 1

            return _launch

    monkeypatch.setattr(mdd, "qwen3_dense_mfma_gemm_kernel", _CountingKernel())

    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
        triton_scaled_mm_fused_quant,
    )

    M, N, K = 2, 256, 512
    a, b, b_scale = make_problem(M, N, K)
    a_q, a_scale = quant_contract_torch(a)

    out_f = dense_mfma_gemm(a, b, b_scale)
    out_i = dense_mfma_gemm(a_q, b, b_scale, a_scale=a_scale.view(-1))
    ref_f, _ = triton_scaled_mm_fused_quant(a, b, b_scale, torch.bfloat16)
    ref_i = triton_scaled_mm(a_q, b, a_scale, b_scale, torch.bfloat16)
    torch.cuda.synchronize()

    assert calls["n"] == 0, (
        "le kernel MFMA a tourné malgré la gate OFF — le repli n'est pas le "
        "chemin livré"
    )
    assert torch.equal(out_f, ref_f), "repli bf16 != triton_scaled_mm_fused_quant"
    assert torch.equal(out_i, ref_i), "repli int8 != triton_scaled_mm"

    # branche tampon out du repli : copie bornée à [M,N]
    buf = torch.zeros(8, N, device="cuda", dtype=torch.bfloat16)
    r = dense_mfma_gemm(a, b, b_scale, out=buf)
    torch.cuda.synchronize()
    assert r is buf, "le repli doit retourner le tampon fourni"
    assert torch.equal(buf[:M, :N], ref_f), "copie repli out[:M,:N] != livré"
    assert bool((buf[M:] == 0).all()), "lignes hors [M,N] écrites par le repli"


@needs_gpu
def test_launcher_fail_loud():
    """Entrées hors contrat : assert nommés, pas de comportement silencieux."""
    _gate_on()
    a, b, b_scale = make_problem(2, 64, 256)
    a_q, a_scale = quant_contract_torch(a)

    # K non multiple de 16 (contrat de contraction MFMA)
    bad_a = torch.zeros(2, 100, device="cuda", dtype=torch.bfloat16)
    bad_b = torch.zeros(100, 64, device="cuda", dtype=torch.int8)
    with pytest.raises(AssertionError, match="multiple de 16"):
        dense_mfma_gemm(bad_a, bad_b, b_scale[:64])

    # A int8 sans a_scale
    with pytest.raises(AssertionError, match="a_scale"):
        dense_mfma_gemm(a_q, b, b_scale)

    # A bf16 avec a_scale (la quant est fusionnée : contradiction)
    with pytest.raises(AssertionError, match="a_scale"):
        dense_mfma_gemm(a, b, b_scale, a_scale=a_scale.view(-1))

    # b_scale de mauvaise taille
    with pytest.raises(AssertionError, match="b_scale"):
        dense_mfma_gemm(a, b, torch.ones(63, device="cuda"))

    # out trop petit
    with pytest.raises(AssertionError, match="out"):
        dense_mfma_gemm(
            a, b, b_scale, out=torch.zeros(2, 32, device="cuda", dtype=torch.bfloat16)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
