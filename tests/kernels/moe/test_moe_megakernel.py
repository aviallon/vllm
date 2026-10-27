# SPDX-License-Identifier: Apache-2.0
"""Tests GPU du Qwen3MoEMegakernel (campagne MoE int8 2026-10-27).

Exécution réelle MI210 exigée (les tests numériques tournent sur GPU).
Auto-portant : `python3 test_moe_megakernel.py` (runner __main__) ou pytest.

Contrat testé (écarts NOMMÉS du module — pas bit-exact, la cohérence par
tâche est jugée par les gates aval) :
* stratégie `pair`   : sortie ≈ chaîne livrée (silu fp32 vs bf16 arrondi) ;
* stratégie `atomic` : idem + échelle par (rangée, chunk) + somme atomique ;
* repli : toute condition non éligible → `moe_megakernel_eligible` renvoie
  une raison et `_maybe_moe_megakernel` ne modifie rien.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_megakernel import (
    fused_moe_megakernel,
    moe_megakernel_eligible,
    moe_megakernel_enabled,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
)

torch.manual_seed(20261027)


def make_problem(M: int, top_k: int, E: int, I: int, K: int, device="cuda"):
    """Problème int8 w8a8 per-token : poids per-channel, x bf16."""
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    w1 = torch.randint(-127, 128, (E, 2 * I, K), device=device,
                       dtype=torch.int8)
    w2 = torch.randint(-127, 128, (E, K, I), device=device, dtype=torch.int8)
    w1_scale = (torch.rand(E, 2 * I, device=device) * 0.02 + 0.001)
    w2_scale = (torch.rand(E, K, device=device) * 0.02 + 0.001)
    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_weights = torch.rand(M, top_k, device=device, dtype=torch.float32)
    return x, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights


def reference_chain(x, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights,
                    naive: bool):
    """Chaîne livrée de référence (sémantique du chemin prod int8)."""
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
        silu_and_mul_per_token_quant_int8,
    )
    M, K = x.shape
    top_k = topk_ids.size(1)
    E, _, I = w2.size(0), w2.size(1), w2.size(2)

    xq, xs = per_token_quant_int8(x)
    cache3 = torch.zeros(M, top_k, K, device=x.device, dtype=torch.bfloat16)

    pairs = [(t, s) for t in range(M) for s in range(top_k)]
    if naive:
        groups = {(t, s): [(t, s)] for (t, s) in pairs}
    else:
        groups = {}
        for p in pairs:
            e = int(topk_ids[p])
            groups.setdefault(e, []).append(p)

    def run_group(rows, e):
        # gemm1 (acc fp32 -> bf16 cache1 -> silu_q -> gemm2), contrat livré
        idx = torch.tensor([r[0] * top_k + r[1] for r in rows], device=x.device)
        t_idx = torch.tensor([r[0] for r in rows], device=x.device)
        xq_r = xq[t_idx]                       # [n, K] int8
        xs_r = xs[t_idx]                       # [n, 1]
        acc = (xq_r.float() @ w1[e].float().T) * xs_r * w1_scale[e][None, :]
        cache1 = acc.to(torch.bfloat16)        # arrondi bf16 du livré
        q2, s2 = silu_and_mul_per_token_quant_int8(cache1)
        acc2 = (q2.float() @ w2[e].float().T) * s2 * w2_scale[e][None, :]
        for j, (t, s) in enumerate(rows):
            cache3[t, s] = (acc2[j].float()
                            * float(topk_weights[t, s])).to(torch.bfloat16)

    if naive:
        for (t, s) in pairs:
            run_group([(t, s)], int(topk_ids[t, s]))
    else:
        for e, rows in groups.items():
            run_group(rows, e)

    from vllm import _custom_ops as ops
    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    ops.moe_sum(cache3, out)
    return out


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = b.abs().max().clamp_min(1e-6)
    return float((a - b).abs().max() / denom)


@pytest.mark.parametrize("strategy", ["pair", "atomic"])
@pytest.mark.parametrize("M,top_k,E,I,K", [
    (1, 8, 256, 256, 2048),    # c1 réel TP=2
    (4, 8, 64, 64, 512),       # synthétique couverture
    (8, 4, 32, 32, 256),
])
def test_contract_vs_shipped(strategy, M, top_k, E, I, K):
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_STRATEGY"] = strategy
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = "64"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = strategy
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    naive = M * top_k * 4 <= E
    ref = reference_chain(x, w1, w2, w1s, w2s, ids, wts, naive)

    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    sorted_ids = expert_ids = None
    if not naive:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
        )
        sorted_ids, expert_ids, _ = _prepare_expert_assignment(
            ids, {"BLOCK_SIZE_M": 16}, M, top_k, E, None)

    cache3 = None
    if strategy == "pair":
        cache3 = torch.empty(M, top_k, K, device=x.device,
                             dtype=torch.bfloat16)
    fused_moe_megakernel(
        out, x, w1, w2, wts, ids, None, w1s, w2s,
        naive=naive, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        cache3=cache3,
    )
    torch.cuda.synchronize()

    err = _rel_err(out, ref)
    # écarts nommés : silu fp32 vs bf16 (pair) ; + échelle chunk + atomiques
    # (atomic). Tolérance = 3 % relative max (relevée = écart de contrat).
    assert err < 0.03, f"strategy={strategy} rel_err={err:.4f}"
    # non-trivialité (garde-fou anti-test-vide : la référence n'est pas nulle)
    assert ref.abs().max() > 0


@pytest.mark.parametrize("strategy", ["pair", "atomic"])
def test_determinism_run_to_run(strategy):
    """L'atomique fp32 est non-associative : deux runs ne doivent pas
    diverger au-delà du dernier ULP bf16 (gate de déterminisme)."""
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = strategy
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    M, top_k, E, I, K = 4, 8, 64, 64, 512
    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    naive = M * top_k * 4 <= E
    sorted_ids = expert_ids = None
    if not naive:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
        )
        sorted_ids, expert_ids, _ = _prepare_expert_assignment(
            ids, {"BLOCK_SIZE_M": 16}, M, top_k, E, None)
    outs = []
    for _ in range(2):
        out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
        cache3 = (torch.empty(M, top_k, K, device=x.device,
                              dtype=torch.bfloat16)
                  if strategy == "pair" else None)
        fused_moe_megakernel(
            out, x, w1, w2, wts, ids, None, w1s, w2s,
            naive=naive, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
            cache3=cache3)
        torch.cuda.synchronize()
        outs.append(out)
    # bit-identique ou écart de sommation atomique (1 ulp bf16 max)
    diff = (outs[0].float() - outs[1].float()).abs().max()
    scale = outs[0].float().abs().max().clamp_min(1e-6)
    assert float(diff / scale) < 2e-3, f"run-to-run drift {diff}"


def test_fallback_reasons():
    """Chaque condition d'inéligibilité renvoie une raison (repli explicite)."""
    M, top_k, E, I, K = 2, 8, 16, 32, 512
    x, w1, w2, w1s, w2s, ids, _ = make_problem(M, top_k, E, I, K)
    xq = x.to(torch.int8)

    def elig(**over):
        base = dict(
            hidden_states=xq, w1=w1, w2=w2, topk_ids=ids,
            use_int8_w8a8=True, per_act_token_quant=True,
            block_shape=None, quantization_emulation=False,
            activation_is_silu=True, clamp_limit_is_none=True,
            has_lora=False, has_bias=False, expert_map=None,
            apply_router_weight_on_input=False, a1q_scale=torch.ones(
                M, 1, device=x.device),
        )
        base.update(over)
        return moe_megakernel_eligible(**base)

    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 8
    assert elig() is None
    assert elig(activation_is_silu=False) is not None
    assert elig(has_lora=True) is not None
    assert elig(expert_map=object()) is not None
    assert elig(block_shape=[128, 128]) is not None
    assert elig(apply_router_weight_on_input=True) is not None
    assert elig(a1q_scale=None) is not None
    assert elig(hidden_states=x) is None          # bf16 accepté (prologue)
    assert elig(hidden_states=xq, a1q_scale=None) is not None
    big = torch.zeros(9, K, device=x.device, dtype=torch.int8)
    assert elig(hidden_states=big) is not None   # M > seuil
    envs.VLLM_MOE_MEGAKERNEL = False
    assert elig() is not None                    # gate off


def test_grouped_mode():
    """Mode grouped (M·topk·4 > E) : tri par expert préservé."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "pair"
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    M, top_k, E, I, K = 8, 4, 8, 32, 256   # M*topk*4 = 128 > 8 -> grouped
    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
    )
    sorted_ids, expert_ids, _ = _prepare_expert_assignment(
        ids, {"BLOCK_SIZE_M": 16}, M, top_k, E, None)
    ref = reference_chain(x, w1, w2, w1s, w2s, ids, wts, naive=False)

    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    cache3 = torch.empty(M, top_k, K, device=x.device, dtype=torch.bfloat16)
    fused_moe_megakernel(
        out, x, w1, w2, wts, ids, None, w1s, w2s,
        naive=False, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        cache3=cache3)
    torch.cuda.synchronize()
    assert _rel_err(out, ref) < 0.03


def test_fallback_is_shipped_path(monkeypatch):
    """Gate OFF : TritonExperts.apply ne doit PAS toucher le megakernel."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = False
    called = {"n": 0}
    import vllm.model_executor.layers.fused_moe.moe_megakernel as mkm
    monkeypatch.setattr(
        mkm, "fused_moe_megakernel",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    exp = object.__new__(TritonExperts)
    assert exp._maybe_moe_megakernel(
        None, None, None, None, None, None, None, None, False, None
    ) is False
    assert called["n"] == 0


def test_fallback_with_gate_on_does_not_raise(monkeypatch):
    """Gate ON + inéligible : le repli doit S'EXÉCUTER (y compris sa ligne
    de log) sans lever.  LEÇON 2026-10-25 : une ligne de log avec `logger`
    non défini a crash-loopé un pod prod — la ligne de repli n'avait jamais
    tourné avant expédition.  Ce test exécute exactement cette ligne."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 8

    exp = object.__new__(TritonExperts)
    exp._lora_context = None
    exp.quant_config = type("Q", (), {"use_int8_w8a8": True})()
    # per_act_token_quant est une propriété du parent (pas de setter) :
    # monkeypatch de la classe.
    monkeypatch.setattr(TritonExperts, "per_act_token_quant", True,
                        raising=False)
    monkeypatch.setattr(TritonExperts, "block_shape", None, raising=False)
    # w1_bias/w2_bias sont aussi des propriétés du parent.
    monkeypatch.setattr(TritonExperts, "w1_bias", None, raising=False)
    monkeypatch.setattr(TritonExperts, "w2_bias", None, raising=False)
    exp.quantization_emulation = False
    exp.activation_config = type("A", (), {"clamp_limit": None})()

    M, top_k, E, I, K = 9, 8, 256, 32, 256   # M=9 > seuil -> repli
    x = torch.zeros(M, K, device="cuda", dtype=torch.int8)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    ids = torch.zeros(M, top_k, device="cuda", dtype=torch.int32)
    # ne doit PAS lever (le chemin de repli s'exécute, log compris)
    assert exp._maybe_moe_megakernel(
        None, x, w1, w2, None, ids, torch.ones(M, 1, device="cuda"),
        None, False, None,
    ) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-x"]))


# ---------------------------------------------------------------------------
# INCREMENT 1 (tour 2) — routing in-kernel
# ---------------------------------------------------------------------------
def _torch_routing(x_bf16, router_w, top_k, renormalize=True):
    """Référence du routing livré : logits bf16 -> topk_softmax (le VRAI
    kernel) — la comparaison d'ids est implicite : des experts différents
    donneraient une sortie hors tolérance."""
    from vllm import _custom_ops as ops
    M = x_bf16.size(0)
    E = router_w.size(0)
    logits = (x_bf16.float() @ router_w.float().T)
    topk_weights = torch.empty(M, top_k, device=x_bf16.device,
                               dtype=torch.float32)
    topk_ids = torch.empty(M, top_k, device=x_bf16.device, dtype=torch.int32)
    tei = torch.empty(M, top_k, device=x_bf16.device, dtype=torch.int32)
    ops.topk_softmax(topk_weights, topk_ids, tei, logits, renormalize)
    return topk_weights, topk_ids


def _torch_softmax_topk(x_bf16, router_w, top_k):
    """Mêmes expressions que le prologue du kernel (softmax fp32 + argmax
    itératif + renormalize) — pour isoler un écart de sélection."""
    M = x_bf16.size(0)
    E = router_w.size(0)
    logits = (x_bf16.float() @ router_w.float().T)
    p = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    p = p / p.sum(dim=1, keepdim=True)
    ids = torch.zeros(M, top_k, device=x_bf16.device, dtype=torch.int64)
    wts = torch.zeros(M, top_k, device=x_bf16.device)
    pwork = p.clone()
    for s in range(top_k):
        e = pwork.argmax(dim=1)
        ids[:, s] = e
        wts[:, s] = p.gather(1, e[:, None])[:, 0]
        pwork.scatter_(1, e[:, None], -1.0)
    wts = wts / wts.sum(dim=1, keepdim=True)
    return wts, ids.to(torch.int32)


@pytest.mark.parametrize("M,top_k,E,I,K", [
    (1, 8, 256, 256, 2048),   # c1 réel TP=2
    (4, 8, 64, 64, 512),
])
def test_increment1_routing_vs_shipped(M, top_k, E, I, K):
    """Incrément 1 : le kernel calcule router+topk lui-même.

    Deux contrats vérifiés séparément (pour ne pas confondre un basculement
    de frontière topk avec une erreur du travail d'expert) :
    1. ROUTING : la sélection du kernel == topk_softmax du livré, sauf
       frontières d'égalité (marge 8e/9e faible — documentée) ;
    2. TRAVAIL D'EXPERT : la sortie ≈ la chaîne de référence construite avec
       la sélection DU KERNEL (tolérance serrée)."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    router_w = torch.randn(E, K, device=x.device,
                           dtype=torch.bfloat16) * 0.02

    r_wts, r_ids = _torch_routing(x, router_w, top_k)

    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    dbg_ids = torch.zeros(M, top_k, device=x.device, dtype=torch.int32)
    dbg_w = torch.zeros(M, top_k, device=x.device)
    fused_moe_megakernel(
        out, x, w1, w2, None, None, None, w1s, w2s,
        naive=True, cache3=None, router_w=router_w, top_k_override=top_k,
        dbg_ids=dbg_ids, dbg_w=dbg_w,
    )
    torch.cuda.synchronize()

    # 1. contrat ROUTING (ids), marge de frontière tolérée
    logits = x.float() @ router_w.float().T
    srt = logits.sort(dim=1, descending=True).values
    margin = srt[:, top_k - 1] - srt[:, top_k]
    ids_match = (dbg_ids.to(torch.int64) == r_ids.to(torch.int64)).all(dim=1)
    for t in range(M):
        if not bool(ids_match[t]):
            assert float(margin[t]) < 1e-2, (
                f"topk flip hors frontière (marge {float(margin[t]):.4f}): "
                f"kernel={dbg_ids[t].tolist()} ref={r_ids[t].tolist()}")
    # poids du kernel ≈ poids livrés (même sélection)
    w_err = float((dbg_w - r_wts).abs().max() / r_wts.abs().max())
    assert w_err < 2e-2, f"poids topk kernel err={w_err:.4f}"

    # 2. contrat TRAVAIL D'EXPERT : référence avec la sélection DU KERNEL
    ref = reference_chain(x, w1, w2, w1s, w2s, dbg_ids, dbg_w, naive=True)
    err = _rel_err(out, ref)
    assert err < 0.03, f"expert-work rel_err={err:.4f}"
    assert ref.abs().max() > 0


def test_increment1_topk_math_matches():
    """Le softmax+argmax du prologue reproduit les poids de topk_softmax."""
    M, top_k, E, K = 8, 8, 64, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    router_w = torch.randn(E, K, device="cuda", dtype=torch.bfloat16) * 0.02
    w_a, i_a = _torch_routing(x, router_w, top_k)
    w_b, i_b = _torch_softmax_topk(x, router_w, top_k)
    # ids identiques (pas de tirage au sort dans les données)
    assert (i_a.to(torch.int64) == i_b).all(), (
        f"topk ids divergent: {i_a} vs {i_b}")
    assert (w_a - w_b).abs().max() < 2e-3, "poids topk divergents"


def test_increment1_fallbacks():
    """Inéligibilités du mode routing : grouped, x int8, router absent."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64
    M, top_k, E, I, K = 8, 8, 8, 32, 256   # grouped (M*topk*4 > E)
    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    router_w = torch.randn(E, K, device=x.device, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fused_moe_megakernel(
            torch.zeros(M, K, device=x.device, dtype=torch.bfloat16),
            x, w1, w2, wts, ids, None, w1s, w2s,
            naive=False, router_w=router_w, top_k_override=top_k,
        )
    # router absent
    M2 = 2
    x2, w12, w22, w1s2, w2s2, ids2, wts2 = make_problem(M2, top_k, 256, I, K)
    with pytest.raises(AssertionError):
        fused_moe_megakernel(
            torch.zeros(M2, K, device=x2.device, dtype=torch.bfloat16),
            x2, w12, w22, wts2, ids2, None, w1s2, w2s2,
            naive=True, router_w=None, top_k_override=top_k,
        )
