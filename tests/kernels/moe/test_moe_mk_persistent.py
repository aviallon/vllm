# SPDX-License-Identifier: Apache-2.0
"""Tests GPU de la boucle persistante v0 (INC-D, campagne MEGAKERNEL).

Exécution réelle MI210 (gfx90a) exigée.  Auto-portant :
`python3 test_moe_mk_persistent.py` ou pytest.

Contrats testés (docstring de `moe_megakernel_persistent.py`) :

* (a) chaîne de N couches EN CASCADE RÉELLE vs chaîne livrée
  (`triton_scaled_mm_fused_quant` par couche) — plusieurs M et N=K
  (chaîne carrée v0, écart NOMMÉ) ;
* (b) MUTATIONS « break the fix » : `MUT_SKIP_RESET` (la barrière ne
  reset/avance pas → sorties FAUSSES sans hang) et `MUT_WRONG_LAYER`
  (poids de la couche L+1 → divergence) — si une mutation reste
  invisible, c'est la revendication qui tombe ;
* (c) la GARDE de spin est testée elle-même : une barrière où moins de
  `n_workers` CTA arrivent doit TRIPPER le drapeau fail loud (borné,
  observable — pas un hang) ;
* (d) garde de co-résidence (P > 104 = assert fail loud) ;
* (e) gate 2 canaux, défaut OFF, repli = chaîne livrée (identité +
  compteur : le kernel persistant ne tourne pas) ; déterminisme.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.moe_megakernel_persistent import (
    MK_P_RESIDENT_HARD,
    mk_phase_barrier,
    mk_persistent_enabled,
    mk_persistent_fallback,
    mk_persistent_v0,
    qwen3_mk_persistent_v0_kernel,
)

torch.manual_seed(20261028)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU requis (MI210 gfx90a)"
)


def _gate_on() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_PERSISTENT"] = "1"


def _gate_off() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_PERSISTENT"] = "0"


def make_chain(
    n_layers: int = 40,
    M: int = 4,
    K: int = 256,
    device: str = "cuda",
    share_weights: bool = True,
):
    """Chaîne carrée : poids [K,K] int8 par couche (partagés par défaut)."""
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    res = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    norm_ws = [
        torch.randn(K, device=device, dtype=torch.bfloat16) * 0.2 + 1.0
        for _ in range(n_layers if not share_weights else 1)
    ]
    ws = [
        torch.randint(-127, 128, (K, K), device=device, dtype=torch.int8)
        for _ in range(n_layers if not share_weights else 1)
    ]
    w_scales = [
        torch.rand(K, device=device) * 0.02 + 0.001
        for _ in range(n_layers if not share_weights else 1)
    ]
    if share_weights:  # tables répétées : mêmes tenseurs, pointeurs répétés
        norm_ws = norm_ws * n_layers
        ws = ws * n_layers
        w_scales = w_scales * n_layers
    return x, res, norm_ws, ws, w_scales


@needs_gpu
@pytest.mark.parametrize("M,K", [(1, 256), (4, 256), (8, 512)])
def test_persistent_chain_vs_shipped(M, K):
    """40 couches en cascade réelle vs chaîne livrée — tolérance NOMMÉE.

    Le corps GEMM est bit-exact vs `triton_scaled_mm_fused_quant` (prouvé
    INC-A) MAIS la norm du fusionné réduit `mean(s·s)` en arbre alors que
    torch réduit en séquence → rstd au dernier ULP → y au dernier ULP de
    bf16 → l'échelle de quant peut basculer d'1 LSB.  Tolérance 5e-3
    (niveau bf16) ; un vrai écart de contrat (couche non chaînée, poids
    non lus) est O(1).
    """
    _gate_on()
    n_layers = 40
    x, res, norm_ws, ws, w_scales = make_chain(n_layers=n_layers, M=M, K=K)
    ref = mk_persistent_fallback(x, res, norm_ws, ws, w_scales, 1e-6)
    got = mk_persistent_v0(x, res, norm_ws, ws, w_scales, 1e-6, n_workers=64)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        got.float(), ref.float(), rtol=5e-3, atol=5e-3,
        msg="chaîne persistante 40 couches != chaîne livrée",
    )


@needs_gpu
def test_chain_actually_chains():
    """Sensibilité du harnais : 1 couche ≠ 2 couches ≠ 3 couches (une
    chaîne qui n'enchaînerait pas serait invisible sinon)."""
    _gate_on()
    x, res, nw, ws, wsc = make_chain(n_layers=3, M=2, K=256)
    outs = [
        mk_persistent_v0(x, res, nw[:L], ws[:L], wsc[:L], 1e-6, n_workers=32)
        for L in (1, 2, 3)
    ]
    assert not torch.equal(outs[0], outs[1]), "1 couche == 2 couches : la chaîne n'enchaîne pas"
    assert not torch.equal(outs[1], outs[2]), "2 couches == 3 couches : la chaîne n'enchaîne pas"


# ===========================================================================
# (b) mutations — « break the fix on purpose »
# ===========================================================================
@needs_gpu
def test_mutation_skip_reset_diverges():
    """MUT_SKIP_RESET (le reset de ``arrive`` est sauté par le dernier
    arrivé) : ``n == n_workers`` n'est plus jamais vrai dès la couche 1 →
    la barrière ne converge plus → le plafond de spins fail loud se
    déclenche et le launcher lève.  Échec BORNÉ et OBSERVABLE (pas un
    hang) — si rien ne se passait, le reset ne serait pas load-bearing."""
    _gate_on()
    x, res, nw, ws, wsc = make_chain(n_layers=4, M=4, K=256)
    with pytest.raises(AssertionError, match="plafond de spins"):
        mk_persistent_v0(
            x, res, nw, ws, wsc, 1e-6, n_workers=32, _mut_skip_reset=True
        )


@needs_gpu
def test_mutation_wrong_layer_diverges():
    """MUT_WRONG_LAYER (poids de la couche L+1) : divergence — canari
    declared-input : la table de pointeurs par couche doit être RÉELLEMENT
    consommée."""
    _gate_on()
    x, res, nw, ws, wsc = make_chain(n_layers=4, M=4, K=256, share_weights=False)
    ref = mk_persistent_fallback(x, res, nw, ws, wsc, 1e-6)
    mut = mk_persistent_v0(
        x, res, nw, ws, wsc, 1e-6, n_workers=32, _mut_wrong_layer=True
    )
    torch.cuda.synchronize()
    assert not torch.equal(mut, ref), (
        "MUT_WRONG_LAYER : sortie IDENTIQUE — la table de pointeurs par "
        "couche n'est pas consommée (poids partagés ? le test exige des "
        "poids DISTINCTS par couche)"
    )


# ===========================================================================
# (c) la garde de spin elle-même
# ===========================================================================
@triton.jit
def _guard_probe_kernel(
    arrive_ptr, epoch_ptr, work_ptr, fail_ptr,
    n_workers: tl.constexpr, SPIN_LIMIT: tl.constexpr,
):
    """Une seule CTA arrive à une barrière pour n_workers>1 : elle doit
    finir par tripper le plafond de spins (fail loud) et revenir."""
    mk_phase_barrier(
        arrive_ptr, epoch_ptr, work_ptr, fail_ptr, 0,
        n_workers=n_workers, SPIN_LIMIT=SPIN_LIMIT,
    )


@needs_gpu
def test_spin_limit_guard_trips():
    """La GARDE est testée elle-même : barrière pour n_workers=4 avec 1
    seule CTA → spin limit → drapeau fail=1, retour BORNÉ (pas de hang).
    Un garde qui ne peut pas se déclencher est une décoration."""
    dev = "cuda"
    arrive = torch.zeros(2, dtype=torch.int32, device=dev)
    epoch = torch.zeros(2, dtype=torch.int32, device=dev)
    work = torch.zeros(2, dtype=torch.int32, device=dev)
    fail = torch.zeros(1, dtype=torch.int32, device=dev)
    _guard_probe_kernel[(1,)](
        arrive, epoch, work, fail, n_workers=4, SPIN_LIMIT=1000,
    )
    torch.cuda.synchronize()
    assert int(fail[0]) == 1, (
        "plafond de spins NON atteint alors qu'1 seule CTA sur 4 arrive — "
        "la garde fail loud ne se déclenche pas (décoration)"
    )


# ===========================================================================
# (d) garde de co-résidence
# ===========================================================================
@needs_gpu
def test_grid_guard_fails_loud():
    _gate_on()
    x, res, nw, ws, wsc = make_chain(n_layers=1, M=1, K=256)
    with pytest.raises(AssertionError, match="co-résidence"):
        mk_persistent_v0(x, res, nw, ws, wsc, n_workers=MK_P_RESIDENT_HARD + 1)
    with pytest.raises(AssertionError, match="co-résidence"):
        mk_persistent_v0(x, res, nw, ws, wsc, n_workers=0)


# ===========================================================================
# (e) gate / repli / déterminisme
# ===========================================================================
def test_gate_channels_and_default_off(monkeypatch):
    """CANARI declared-input : canal os.environ prioritaire (les variables
    ne sont PAS dans vllm.envs — piège du bras muet 2026-10-28)."""
    from vllm import envs

    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_PERSISTENT", raising=False)
    monkeypatch.setattr(
        envs, "VLLM_MOE_MEGAKERNEL_PERSISTENT", False, raising=False
    )
    assert mk_persistent_enabled() is False, "défaut doit être OFF"
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_PERSISTENT", "1")
    assert mk_persistent_enabled() is True, (
        "VLLM_MOE_MEGAKERNEL_PERSISTENT=1 ignoré — gate muette"
    )
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_PERSISTENT", "0")
    assert mk_persistent_enabled() is False
    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_PERSISTENT", raising=False)
    monkeypatch.setattr(
        envs, "VLLM_MOE_MEGAKERNEL_PERSISTENT", True, raising=False
    )
    assert mk_persistent_enabled() is True, "canal envs ignoré"


@needs_gpu
def test_fallback_gate_off_is_shipped_path(monkeypatch):
    """Gate OFF : repli = chaîne livrée, le kernel persistant ne tourne
    PAS (compteur) et le résultat est celui de la chaîne livrée."""
    _gate_off()
    import vllm.model_executor.layers.fused_moe.moe_megakernel_persistent as pm

    calls = {"n": 0}

    class _CountingKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                calls["n"] += 1

            return _launch

    monkeypatch.setattr(pm, "qwen3_mk_persistent_v0_kernel", _CountingKernel())
    x, res, nw, ws, wsc = make_chain(n_layers=3, M=2, K=256)
    got = mk_persistent_v0(x, res, nw, ws, wsc, 1e-6, n_workers=16)
    torch.cuda.synchronize()
    assert calls["n"] == 0, "le kernel persistant a tourné malgré la gate OFF"
    ref = mk_persistent_fallback(x, res, nw, ws, wsc, 1e-6)
    assert torch.equal(got, ref), "repli != chaîne livrée (identité)"


@needs_gpu
def test_determinism_run_to_run():
    _gate_on()
    x, res, nw, ws, wsc = make_chain(n_layers=8, M=4, K=256)
    a = mk_persistent_v0(x, res, nw, ws, wsc, 1e-6, n_workers=64)
    b = mk_persistent_v0(x, res, nw, ws, wsc, 1e-6, n_workers=64)
    assert torch.equal(a, b), "non-déterminisme run-to-run"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
