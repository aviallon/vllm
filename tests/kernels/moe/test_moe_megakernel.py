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


# ===========================================================================
# INCRÉMENTS 2+3 (tour 2) — norm+quant v2 + AR in-kernel
# Spec verbatim : research/moe-megakernel-campaign-2026-10-27/TOUR2-DESIGN.md
# §6 ; contrat numérique = natif v2 (store ar-rms-quant-v2-spec §3.2) —
# nearbyint half-EVEN (PAS le `round` half-away du chemin livré), scale =
# amax/127 SANS clamp.
# ===========================================================================
import statistics
import time
import types

import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.moe_megakernel import (
    AR_MAX_CTAS,
    FLAG_ELEMS,
    MoeMegakernelComm,
    _rint_clamp_i8,
    _round_i8,
    moe_megakernel_flags,
)


def reference_norm_chain(x, res, w, eps, peer=None):
    """Contrat v2 §3.2 VERBATIM (torch) — retourne
    (q int8 [M,K], scale fp32 [M,1], res_out bf16 [M,K], y_bf16 [M,K]).
    `torch.round` = half-to-even = `nearbyint` (la bonne primitive)."""
    if peer is None:
        xf = x
    else:
        xf = (x.to(torch.float32) + peer.to(torch.float32)).to(
            torch.bfloat16)
    s = xf.to(torch.float32) + res.to(torch.float32)
    res_out = s.to(torch.bfloat16)
    var = (s * s).mean(dim=-1, keepdim=True)
    y = s * torch.rsqrt(var + eps)
    yb = (y * w.to(torch.float32)).to(torch.bfloat16)
    amax = yb.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    inv_s = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(yb.to(torch.float32) * inv_s).clamp(-128, 127)
    return q.to(torch.int8), scale, res_out, yb


def reference_experts_from_q(q, scale, w1, w2, w1_scale, w2_scale,
                            sel_ids, sel_w):
    """Travail d'expert depuis (q, scale) avec la sélection DU KERNEL
    (sépare la sélection du calcul — méthode incrément 1).  Accumulation
    fp32 comme le chemin `atomic` (finalize = cast bf16)."""
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        silu_and_mul_per_token_quant_int8,
    )
    M, K = q.shape
    top_k = sel_ids.size(1)
    out_acc = torch.zeros(M, K, device=q.device, dtype=torch.float32)
    for t in range(M):
        for slot in range(top_k):
            e = int(sel_ids[t, slot])
            wt = float(sel_w[t, slot])
            acc = ((q[t].to(torch.float32) @ w1[e].to(torch.float32).T)
                   * scale[t] * w1_scale[e])           # [2I]
            cache1 = acc.to(torch.bfloat16)[None]      # [1, 2I]
            q2, s2 = silu_and_mul_per_token_quant_int8(cache1)
            acc2 = ((q2.to(torch.float32) @ w2[e].to(torch.float32).T)
                    * s2 * w2_scale[e][None, :])       # [1, K]
            out_acc[t] += acc2[0].to(torch.float32) * wt
    return out_acc


def _requant_roundtrip(out_acc):
    """Miroir du chemin de sortie du mode ROUTING (v2' : requant per-token
    puis déquant pour le contrat bf16 aval)."""
    amax = out_acc.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    s = amax / 127.0
    q = (torch.round(out_acc / s) * s).to(torch.bfloat16)
    return q


@triton.jit
def _test_rint_kernel(in_ptr, near_ptr, away_ptr, n, BLOCK: tl.constexpr):
    """Contraste le primitive v2 (`_rint_clamp_i8`, half-even) avec celui du
    chemin livré (`_round_i8`, half-away) — MUTATION lisible : si le kernel
    utilisait `round` au lieu de `nearbyint`, ce test échoue."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(in_ptr + offs, mask=m, other=0.0)
    tl.store(near_ptr + offs, _rint_clamp_i8(v), mask=m)
    tl.store(away_ptr + offs, _round_i8(v), mask=m)


def test_rint_half_even_primitive():
    """Le prologue NORM arrondit half-even (`nearbyint`) — écart NOMMÉ vs le
    `round` half-away du chemin livré (TOUR2-DESIGN §6.1)."""
    vals = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -0.25, 0.25, 7.5,
            300.0, -300.0, 1000.0, -1000.0, 0.0]
    inp = torch.tensor(vals, device="cuda", dtype=torch.float32)
    n = inp.numel()
    near = torch.empty(n, device="cuda", dtype=torch.int8)
    away = torch.empty(n, device="cuda", dtype=torch.int8)
    _test_rint_kernel[((1,))](inp, near, away, n, BLOCK=16)
    torch.cuda.synchronize()
    # half-even (nearbyint) : 0.5->0, -0.5->0, 2.5->2, -2.5->-2, 3.5->4
    assert near.tolist()[:10] == [-2, -2, 0, 0, 2, 2, 4, 0, 0, 8], (
        f"nearbyint half-even violé: {near.tolist()[:10]}")
    # half-away (round, chemin livré) : 0.5->1, -0.5->-1, 2.5->3, -2.5->-3
    assert away.tolist()[:10] == [-3, -2, -1, 1, 2, 3, 4, 0, 0, 8], (
        f"contrôle round half-away inattendu: {away.tolist()[:10]}")
    # les deux primitives DIFFÈRENT aux frontières (garde anti-test-vide)
    assert (near[:10].int() != away[:10].int()).any()
    # saturation v2 [-128, 127]
    assert near.tolist()[10:] == [127, -128, 127, -128, 0]


def _run_increment23(M, top_k, E, I, K, *, norm, ar_comm=None,
                     eps=1e-6, seed=20261027, peer_x=None):
    """Exécute le megakernel ROUTING(+NORM)(+AR) et retourne tous les
    artefacts de contrat."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = bool(norm)
    envs.VLLM_MOE_MEGAKERNEL_AR = ar_comm is not None
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_NORM"] = "1" if norm else "0"
    os.environ["VLLM_MOE_MEGAKERNEL_AR"] = "1" if ar_comm is not None else "0"

    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    res = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2
    norm_w = torch.randn(K, device="cuda").float() * 0.5 + 1.0
    w1 = torch.randint(-127, 128, (E, 2 * I, K), device="cuda",
                       dtype=torch.int8)
    w2 = torch.randint(-127, 128, (E, K, I), device="cuda", dtype=torch.int8)
    w1s = (torch.rand(E, 2 * I, device="cuda") * 0.02 + 0.001)
    w2s = (torch.rand(E, K, device="cuda") * 0.02 + 0.001)
    router_w = torch.randn(E, K, device="cuda",
                           dtype=torch.bfloat16) * 0.02

    out = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    res_out = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    dbg_q = torch.empty(M, K, device="cuda", dtype=torch.int8)
    dbg_scale = torch.zeros(M, device="cuda")
    dbg_ids = torch.zeros(M, top_k, device="cuda", dtype=torch.int32)
    dbg_w = torch.zeros(M, top_k, device="cuda")
    fused_moe_megakernel(
        out, x, w1, w2, None, None, None, w1s, w2s,
        naive=True, router_w=router_w, top_k_override=top_k,
        res=res, norm_w=norm_w, eps=eps, res_out=res_out,
        dbg_scale=dbg_scale, dbg_q=dbg_q, dbg_ids=dbg_ids, dbg_w=dbg_w,
        ar_comm=ar_comm,
    )
    torch.cuda.synchronize()
    return types.SimpleNamespace(
        x=x, res=res, norm_w=norm_w, w1=w1, w2=w2, w1s=w1s, w2s=w2s,
        router_w=router_w, out=out, res_out=res_out, dbg_q=dbg_q,
        dbg_scale=dbg_scale, dbg_ids=dbg_ids, dbg_w=dbg_w)


@pytest.mark.parametrize("M,top_k,E,I,K", [
    (1, 8, 256, 256, 2048),    # c1 réel TP=2
    (4, 8, 128, 64, 512),      # synthétique couverture
])
def test_increment2_norm_quant_contract(M, top_k, E, I, K):
    """INC 2 : contrat v2 VERBATIM vs référence torch.

    Sépare (comme l'incrément 1) la sélection du calcul : le travail
    d'expert est comparé à une référence construite avec la sélection DU
    KERNEL, et le prologue (norm+quant) est comparé DIRECTEMENT via les
    surfaces dbg_q/dbg_scale/res_out."""
    r = _run_increment23(M, top_k, E, I, K, norm=True)
    q_ref, scale_ref, res_ref, _ = reference_norm_chain(
        r.x, r.res, r.norm_w, 1e-6)

    # 1. res_out = bf16(s) — BIT-EXACT (ne dépend pas de var)
    assert torch.equal(r.res_out, res_ref), "res_out != bf16(fp32(xf)+fp32(res))"

    # 2. scale = amax(y_bf16)/127 SANS clamp — classe-3 (ordre de var)
    assert torch.allclose(r.dbg_scale, scale_ref[:, 0],
                          rtol=1e-4, atol=1e-8), (
        f"scale hors contrat: {(r.dbg_scale - scale_ref[:, 0]).abs().max()}")

    # 3. q = clamp(nearbyint(y_bf16·inv_s)) — exact JUSQU'À LA REQUANT
    #    FINALE (±1 LSB aux frontières d'arrondi, classe-3 via var)
    flips = (r.dbg_q.int() - q_ref.int()).abs()
    assert (flips <= 1).all(), f"q diverge de >1 LSB: {flips.max()}"
    nflips = int((flips > 0).sum())
    print(f"q +/-1 LSB flips: {nflips}/{flips.numel()}")
    assert nflips <= flips.numel() // 100, f"trop de flips: {nflips}"

    # 4. le cas zéro est couvert par test_rint / une rangée nulle dédiée
    # 5. travail d'expert vs référence construite avec la sélection DU KERNEL
    ref_acc = reference_experts_from_q(
        q_ref, scale_ref, r.w1, r.w2, r.w1s, r.w2s, r.dbg_ids, r.dbg_w)
    ref_out = _requant_roundtrip(ref_acc)
    err = _rel_err(r.out, ref_out)
    assert err < 0.03, f"expert-work rel_err={err:.4f}"
    assert ref_out.abs().max() > 0  # garde anti-test-vide

    # 6. sélection : ids cohérents (frontières d'égalité tolérées, inc 1)
    #    — comparaison par ENSEMBLE (l'ordre de sélection du topk itératif
    #    peut différer de argsort sans que la sélection diffère)
    logits = (reference_norm_chain(r.x, r.res, r.norm_w, 1e-6)[3].float()
              @ r.router_w.float().T)
    srt = logits.sort(dim=1, descending=True).values
    margin = srt[:, top_k - 1] - srt[:, top_k]
    ids_ref = logits.argsort(dim=1, descending=True)[:, :top_k]
    sets_match = (
        r.dbg_ids.to(torch.int64).sort(dim=1).values
        == ids_ref.sort(dim=1).values).all(dim=1)
    for t in range(M):
        if not bool(sets_match[t]):
            assert float(margin[t]) < 1e-2, (
                f"topk flip hors frontière (marge {float(margin[t]):.4f})")


def test_increment2_zero_row():
    """Cas zéro : amax==0 -> inv_s=0, q=0, scale=0 (contrat v2, SANS clamp
    FLT_EPSILON — la « mauvaise référence » de v1 donnerait scale=1e-10/127)."""
    M, top_k, E, I, K = 1, 8, 256, 256, 2048
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = True
    envs.VLLM_MOE_MEGAKERNEL_AR = False
    x = torch.zeros(1, K, device="cuda", dtype=torch.bfloat16)
    res = torch.zeros(1, K, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.ones(K, device="cuda")
    w1 = torch.randint(-127, 128, (E, 2 * I, K), device="cuda",
                       dtype=torch.int8)
    w2 = torch.randint(-127, 128, (E, K, I), device="cuda", dtype=torch.int8)
    w1s = (torch.rand(E, 2 * I, device="cuda") * 0.02 + 0.001)
    w2s = (torch.rand(E, K, device="cuda") * 0.02 + 0.001)
    router_w = torch.randn(E, K, device="cuda",
                           dtype=torch.bfloat16) * 0.02
    out = torch.zeros(1, K, device="cuda", dtype=torch.bfloat16)
    res_out = torch.empty(1, K, device="cuda", dtype=torch.bfloat16)
    dbg_scale = torch.full((1,), 7.0, device="cuda")  # sentinelle
    dbg_q = torch.full((1, K), 9, device="cuda", dtype=torch.int8)
    fused_moe_megakernel(
        out, x, w1, w2, None, None, None, w1s, w2s,
        naive=True, router_w=router_w, top_k_override=top_k,
        res=res, norm_w=norm_w, eps=1e-6, res_out=res_out,
        dbg_scale=dbg_scale, dbg_q=dbg_q,
    )
    torch.cuda.synchronize()
    assert torch.all(dbg_scale == 0), f"scale zéro non nul: {dbg_scale}"
    assert torch.all(dbg_q == 0), "q zéro non nul"
    assert torch.all(res_out == 0)
    assert torch.all(out == 0)


def test_increment2_res_out_must_not_alias():
    """res_out aliasant res = corruption croisée entre CTAs (la grille
    chunked relit res pendant que d'autres CTAs écrivent) — fail loud."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = True
    envs.VLLM_MOE_MEGAKERNEL_AR = False
    M, top_k, E, I, K = 1, 8, 256, 256, 512
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    w1s = torch.ones(E, 2 * I, device="cuda")
    w2s = torch.ones(E, K, device="cuda")
    router_w = torch.zeros(E, K, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fused_moe_megakernel(
            torch.zeros(M, K, device="cuda", dtype=torch.bfloat16),
            x, w1, w2, None, None, None, w1s, w2s,
            naive=True, router_w=router_w, top_k_override=top_k,
            res=res, norm_w=torch.ones(K, device="cuda"), res_out=res,
        )


def test_gate_stacking_requires_routing_and_norm():
    """Empilement des gates (spec §6.2 : ROUTING → NORM → AR) — fail loud."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    M, top_k, E, I, K = 1, 8, 256, 256, 512
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    w1s = torch.ones(E, 2 * I, device="cuda")
    w2s = torch.ones(E, K, device="cuda")
    rw = torch.zeros(E, K, device="cuda", dtype=torch.bfloat16)
    common = dict(res=res, norm_w=torch.ones(K, device="cuda"),
                  res_out=torch.empty_like(res))

    # NORM sans ROUTING
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "pair"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = False
    envs.VLLM_MOE_MEGAKERNEL_NORM = True
    envs.VLLM_MOE_MEGAKERNEL_AR = False
    with pytest.raises(AssertionError, match="ROUTING"):
        fused_moe_megakernel(
            torch.zeros(M, K, device="cuda", dtype=torch.bfloat16),
            x, w1, w2, None, None, None, w1s, w2s,
            naive=True, router_w=rw, top_k_override=top_k, **common)

    # AR sans NORM
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = False
    envs.VLLM_MOE_MEGAKERNEL_AR = True
    with pytest.raises(AssertionError, match="NORM"):
        fused_moe_megakernel(
            torch.zeros(M, K, device="cuda", dtype=torch.bfloat16),
            x, w1, w2, None, None, None, w1s, w2s,
            naive=True, router_w=rw, top_k_override=top_k, **common)


def test_increment3_grid_bound_fails_loud():
    """Plafond de co-résidence AR : au-delà de AR_MAX_CTAS le spin peut
    deadlock entre rangs (scheduling) — fail loud, l'appelant replie."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = True
    envs.VLLM_MOE_MEGAKERNEL_AR = True
    M, top_k, E, I, K = 8, 8, 256, 256, 512   # 8 × 32 = 256 CTAs > 96
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    fake_comm = types.SimpleNamespace(
        rows=256, hidden=K, n_chunks=(I + 7) // 8,
        data_ptr=0, peer_data_ptr=0, flag_ptr=0, peer_flag_ptr=0)
    with pytest.raises(AssertionError, match="co-résidence"):
        fused_moe_megakernel(
            torch.zeros(M, K, device="cuda", dtype=torch.bfloat16),
            x, w1, w2, None, None, None,
            torch.ones(E, 2 * I, device="cuda"),
            torch.ones(E, K, device="cuda"),
            naive=True, router_w=torch.zeros(E, K, device="cuda",
                                             dtype=torch.bfloat16),
            top_k_override=top_k,
            res=res, norm_w=torch.ones(K, device="cuda"),
            res_out=torch.empty_like(res), ar_comm=fake_comm)
    assert M * ((I + 7) // 8) > AR_MAX_CTAS  # précondition du test


def test_engine_site_fallback_without_prologue_inputs(monkeypatch):
    """Le site d'appel moteur (TritonExperts.apply) n'a ni residual, ni
    poids de norm, ni poids de router, ni handles IPC : gates NORM/AR/
    ROUTING demandées => repli sur la chaîne livrée, ligne de log INCLUSE
    (chaque ligne expédiée exécutée — leçon 2026-10-25)."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64
    exp = object.__new__(TritonExperts)
    exp._lora_context = None
    exp.quant_config = type("Q", (), {"use_int8_w8a8": True})()
    monkeypatch.setattr(TritonExperts, "per_act_token_quant", True,
                        raising=False)
    monkeypatch.setattr(TritonExperts, "block_shape", None, raising=False)
    monkeypatch.setattr(TritonExperts, "w1_bias", None, raising=False)
    monkeypatch.setattr(TritonExperts, "w2_bias", None, raising=False)
    exp.quantization_emulation = False
    exp.activation_config = type("A", (), {"clamp_limit": None})()
    M, top_k, E, I, K = 2, 8, 256, 32, 256
    x = torch.zeros(M, K, device="cuda", dtype=torch.int8)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    ids = torch.zeros(M, top_k, device="cuda", dtype=torch.int32)

    def call():
        return exp._maybe_moe_megakernel(
            None, x, w1, w2, None, ids, torch.ones(M, 1, device="cuda"),
            None, False, None)

    # NORM demandé, non câblé au site -> repli + log
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_NORM", True)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_AR", False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_STRATEGY", "pair")
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_ROUTING", False)
    assert call() is False
    # ROUTING demandé (pas de router_w au site) -> repli + log
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_NORM", False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_ROUTING", True)
    assert call() is False
    assert moe_megakernel_flags() == (True, False, False)


@triton.jit
def _copy_flags_kernel(ptr, out_ptr, n, BLOCK: tl.constexpr):
    p = tl.cast(ptr.to(tl.int64), tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    for base in range(0, n, BLOCK):
        o = base + offs
        tl.store(out_ptr + o, tl.load(p + o, mask=o < n, other=0),
                 mask=o < n)


def _mk_two_rank_worker(rank, world, port, outdir, n_calls):
    """Module-level pour spawn (picklable) ; résultats en FICHIERS (pas de
    mp.Queue — mesuré 2026-10-22 : EOFError sur le resource_sharer)."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = True
    envs.VLLM_MOE_MEGAKERNEL_NORM = True
    envs.VLLM_MOE_MEGAKERNEL_AR = True
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    M, top_k, E, I, K = 1, 8, 256, 256, 2048   # c1 réel TP=2
    torch.manual_seed(7)  # poids/residual/router IDENTIQUES des deux côtés
    w1 = torch.randint(-127, 128, (E, 2 * I, K), device=dev,
                       dtype=torch.int8)
    w2 = torch.randint(-127, 128, (E, K, I), device=dev, dtype=torch.int8)
    w1s = (torch.rand(E, 2 * I, device=dev) * 0.02 + 0.001)
    w2s = (torch.rand(E, K, device=dev) * 0.02 + 0.001)
    res = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.2
    norm_w = torch.randn(K, device=dev).float() * 0.5 + 1.0
    router_w = torch.randn(E, K, device=dev, dtype=torch.bfloat16) * 0.02
    torch.manual_seed(100 + rank)  # x : shard PROPRE à chaque rang
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5

    n_chunks = (I + 7) // 8
    comm = MoeMegakernelComm(dist.group.WORLD, M, K, n_chunks)
    out = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    res_out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    dbg_q = torch.empty(M, K, device=dev, dtype=torch.int8)
    dbg_scale = torch.zeros(M, device=dev)
    dbg_ids = torch.zeros(M, top_k, device=dev, dtype=torch.int32)
    dbg_w = torch.zeros(M, top_k, device=dev)
    times = []
    for _ in range(n_calls):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused_moe_megakernel(
            out, x, w1, w2, None, None, None, w1s, w2s,
            naive=True, router_w=router_w, top_k_override=top_k,
            res=res, norm_w=norm_w, eps=1e-6, res_out=res_out,
            dbg_scale=dbg_scale, dbg_q=dbg_q,
            dbg_ids=dbg_ids, dbg_w=dbg_w, ar_comm=comm)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    n_flags = M * n_chunks * FLAG_ELEMS
    flags = torch.empty(n_flags, device=dev, dtype=torch.int32)
    _copy_flags_kernel[((n_flags + 1023) // 1024,)](
        comm.flag_ptr, flags, n_flags, BLOCK=1024)
    torch.cuda.synchronize()
    torch.save(
        {
            "rank": rank,
            "out": out.cpu(),
            "res_out": res_out.cpu(),
            "res": res.cpu(),
            "x": x.cpu(),
            "norm_w": norm_w.cpu(),
            "dbg_q": dbg_q.cpu(),
            "dbg_scale": dbg_scale.cpu(),
            "dbg_ids": dbg_ids.cpu(),
            "dbg_w": dbg_w.cpu(),
            "flags": flags.cpu(),
            "times": times,
        },
        f"{outdir}/rank{rank}.pt",
    )
    dist.destroy_process_group()


class TestIncrement3TwoRank:
    """INC 3 : AR 2 rangs in-kernel, méthode 2b — exact JUSQU'À la requant
    finale vs la somme fp32 exacte, accord inter-rangs, flags par slot.

    Opt-in (env `MOEMK_2GPU_TEST=1`) : exige une paire XGMI (sur une paire
    PCIe les atomiques entiers cross-device du spin ne sont pas garantis —
    le test pendrait)."""

    @pytest.mark.skipif(
        os.environ.get("MOEMK_2GPU_TEST") != "1",
        reason="set MOEMK_2GPU_TEST=1 on an XGMI pair",
    )
    def test_increment3_two_ranks(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("needs 2 GPUs")
        try:
            import vllm._custom_ops as _ops
            if not hasattr(_ops, "allocate_shared_buffer_and_handle"):
                pytest.skip("vllm._C lacks the shared-buffer IPC ops")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"vllm._custom_ops unavailable: {exc}")
        import torch.multiprocessing as mp

        n_calls = 8  # la séquence est dérivée par slot : n appels -> seq n
        ctx = mp.get_context("spawn")
        outdir = os.environ.get("MOEMK_TEST_OUT", "/tmp/moemk-2rank")
        os.makedirs(outdir, exist_ok=True)
        procs = [
            ctx.Process(target=_mk_two_rank_worker,
                        args=(r, 2, 29531, outdir, n_calls))
            for r in (0, 1)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=300)
        hung = [p for p in procs if p.is_alive()]
        for p in hung:
            p.kill()
        assert not hung, (
            "worker(s) en SPIN (deadlock de rendezvous ?) — kill après 300s")
        assert all(p.exitcode == 0 for p in procs), (
            f"worker exit codes: {[p.exitcode for p in procs]}")
        got = {r: torch.load(f"{outdir}/rank{r}.pt", weights_only=False)
               for r in (0, 1)}

        # --- flags par (rangée, chunk) : SÉQUENCE DÉRIVÉE PAR SLOT ---
        # un flag par rangée ferait xchg la séquence n_chunks× (corruption) :
        # chaque slot doit valoir EXACTEMENT n_calls (1 xchg/CTA/appel), et
        # le padding 256 B doit rester à zéro (aucun débordement).
        for rank in (0, 1):
            fl = got[rank]["flags"]
            n_chunks = fl.numel() // FLAG_ELEMS
            slots = fl.view(-1, FLAG_ELEMS)
            assert (slots[:, 0] == n_calls).all(), (
                f"rank {rank}: slots != {n_calls}: {slots[:, 0].tolist()}")
            assert (slots[:, 1:] == 0).all(), (
                f"rank {rank}: padding de flag non nul")

        res0, w0 = got[0]["res"], got[0]["norm_w"]
        for rank in (0, 1):
            x_self, x_peer = got[rank]["x"], got[1 - rank]["x"]
            q_ref, scale_ref, res_ref, yb_ref = reference_norm_chain(
                x_self, res0, w0, 1e-6, peer=x_peer)
            # 1. res_out BIT-EXACT (xf = bf16(fp32+fp32) identique)
            assert torch.equal(got[rank]["res_out"], res_ref), (
                f"rank {rank}: res_out != contrat")
            # 2. scale : classe-3 (ordre de réduction de var)
            assert torch.allclose(got[rank]["dbg_scale"], scale_ref[:, 0],
                                  rtol=1e-4, atol=1e-8), (
                f"rank {rank}: scale hors contrat")
            # 3. méthode 2b : q EXACT jusqu'à la requant finale (±1 LSB)
            flips = (got[rank]["dbg_q"].int() - q_ref.int()).abs()
            assert (flips <= 1).all(), (
                f"rank {rank}: q diverge de >1 LSB: {flips.max()}")
            print(f"rank {rank}: q +/-1 LSB flips: "
                  f"{int((flips > 0).sum())}/{flips.numel()}")
        # (le contrat travail d'expert est couvert par
        # test_increment2_norm_quant_contract ; ici : la chaîne
        # AR+norm+quant + accord inter-rangs, ci-dessous)
        # 5. ACCORD INTER-RANGS : les deux rangs calculent la même somme
        diff = (got[0]["out"].float() - got[1]["out"].float()).abs().max()
        scale = got[0]["out"].float().abs().max().clamp_min(1e-6)
        assert float(diff / scale) < 2e-3, (
            f"les rangs divergent: {float(diff / scale):.4f}")
        # 6. dbg_q identique entre rangs (même xf -> même y -> même q)
        assert torch.equal(got[0]["dbg_q"], got[1]["dbg_q"]), (
            "les rangs divergent sur q")
        # 7. timing (rapporté — borne, pas un gate).  Le wall d'un appel
        #    inclut le SKEW d'amorçage entre les processus rangs (le spin
        #    absorbe le skew en temps d'attente) : min sur N appels = le
        #    meilleur couplage observé, médiane = le régime courant.
        tmin = max(min(got[r]["times"]) for r in (0, 1))
        tmed = max(statistics.median(got[r]["times"]) for r in (0, 1))
        print(f"AR in-kernel (M=1 c1) : min {tmin:.1f} µs / méd {tmed:.1f} µs "
              f"(max des rangs, {n_calls} appels)")
        for r in (0, 1):
            print(f"  rank {r} times: "
                  f"{[round(t, 1) for t in got[r]['times']]}")
        assert got[0]["flags"].numel() == got[1]["flags"].numel()
