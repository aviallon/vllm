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
* INC-A (mandat MFMA) : le codegen de CHAQUE kernel contient v_mfma et
  AUCUN v_dot4 (test ASM — sans lui, le mandat MFMA n'est pas prouvé).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_megakernel import (
    MK_BM,
    MK_BN2,
    MK_CI,
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


@pytest.mark.parametrize("strategy", ["pair", "atomic", "tree"])
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


@pytest.mark.parametrize("strategy", ["pair", "atomic", "tree"])
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


@pytest.mark.parametrize("strategy", ["pair", "atomic", "tree"])
@pytest.mark.parametrize("M", [32, 128])
def test_grouped_contract_real_shapes(M, strategy):
    """Extension c32/c128 (mandat 2026-10-29) : régime grouped aux shapes
    locaux RÉELS TP=2 (E=256, I=256, K=2048, topk=8) — M=32 et M=128 tokens.

    C'est la zone d'éligibilité étendue (le plafond `MAX_TOKENS` était 8) :
    le launcher grouped (blocs de 16 paires d'un même expert) doit router
    (dispatch externe) / gemm / sommer à ces M avec le contrat livré.
    """
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_STRATEGY"] = strategy
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = "128"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = strategy
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 128

    top_k, E, I, K = 8, 256, 256, 2048   # shapes locaux réels TP=2
    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    naive = M * top_k * 4 <= E
    assert not naive, "c32/c128 = régime grouped — l'objet de ce test"
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
    )
    sorted_ids, expert_ids, _ = _prepare_expert_assignment(
        ids, {"BLOCK_SIZE_M": 16}, M, top_k, E, None)
    ref = reference_chain(x, w1, w2, w1s, w2s, ids, wts, naive=False)

    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    cache3 = None
    if strategy == "pair":
        cache3 = torch.empty(M, top_k, K, device=x.device,
                             dtype=torch.bfloat16)
    fused_moe_megakernel(
        out, x, w1, w2, wts, ids, None, w1s, w2s,
        naive=False, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        cache3=cache3)
    torch.cuda.synchronize()
    err = _rel_err(out, ref)
    assert err < 0.03, f"M={M} strategy={strategy} rel_err={err:.4f}"
    assert ref.abs().max() > 0


def test_eligibility_max_tokens_env(monkeypatch):
    """Le plafond M est lisible depuis l'ENV (chemin moteur réel : l'attribut
    `envs` n'est JAMAIS assigné, le getter lit `os.getenv`) : la campagne
    c32/c128 l'élève à 128 PAR L'ENV, sans rebuild.

    Ce test est le canari du branchement des runs c32/c128 : si la variable
    n'était pas enregistrée dans `environment_variables`, `getattr(envs, …)`
    la lirait comme toujours 8 et le bras resterait inéligible SILENCIEUSEMENT
    (leçon de la gate muette, 2026-10-28).  `monkeypatch.delattr` simule
    exactement l'état du processus moteur (aucune assignation d'attribut).
    """
    M, top_k, E, I, K = 128, 8, 256, 256, 2048
    x, w1, w2, w1s, w2s, ids, _ = make_problem(M, top_k, E, I, K)

    def elig():
        return moe_megakernel_eligible(
            hidden_states=x, w1=w1, w2=w2, topk_ids=ids,
            use_int8_w8a8=True, per_act_token_quant=True,
            block_shape=None, quantization_emulation=False,
            activation_is_silu=True, clamp_limit_is_none=True,
            has_lora=False, has_bias=False, expert_map=None,
            apply_router_weight_on_input=False,
            a1q_scale=torch.ones(M, 1, device=x.device),
        )

    from vllm import envs
    envs.disable_envs_cache()   # sinon la valeur est figée après init
    envs.VLLM_MOE_MEGAKERNEL = True
    # retire l'ombre d'éventuels tests précédents (assignation d'attribut) :
    # le processus moteur, lui, n'assigne jamais — il lit via le getter.
    monkeypatch.delattr(envs, "VLLM_MOE_MEGAKERNEL_MAX_TOKENS",
                        raising=False)
    # plafond historique 8 (défaut getter) : M=128 refusé
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = "8"
    assert elig() is not None
    # extension campagne : env 128 -> M=128 éligible (getter os.environ)
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = "128"
    assert elig() is None
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = "8"


def test_scale_layout_3d_contract():
    """Layout réel du checkpoint quark int8 MoE : échelles per-channel en
    3-D [E, N, 1] (« 2D -> 3D for the int8 MoE kernel », quark_moe.py).

    ⚠ Leçon 2026-10-29 (gate muette n°2) : le bras entier repliait
    silencieusement sur ce décalage — les tests utilisaient des échelles
    2-D synthétiques et le checkpoint réel n'avait JAMAIS été exercé.  Ce
    test exige la SORTIE IDENTIQUE aux deux layouts (la normalisation est
    un changement de vue, pas de valeur).
    """
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_STRATEGY"] = "atomic"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "atomic"
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    M, top_k, E, I, K = 8, 8, 256, 256, 2048   # shapes réels TP=2
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
    for scales3d in (False, True):
        w1s_i = w1s.unsqueeze(-1).contiguous() if scales3d else w1s
        w2s_i = w2s.unsqueeze(-1).contiguous() if scales3d else w2s
        assert (w1s_i.dim() == 3) == scales3d
        out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
        fused_moe_megakernel(
            out, x, w1, w2, wts, ids, None, w1s_i, w2s_i,
            naive=naive, sorted_token_ids=sorted_ids, expert_ids=expert_ids)
        torch.cuda.synchronize()
        outs.append(out)
    # normalisation = vue, pas valeur : sorties BIT-identiques
    diff = (outs[0].float() - outs[1].float()).abs().max()
    assert float(diff) == 0.0, f"3-D layout diverge de 2-D : {diff}"
    assert outs[0].abs().max() > 0


def test_scales_ok_predicate():
    """Le prédicat du site moteur accepte [E, N] et [E, N, 1], refuse le
    reste (fail loud) — y compris les 3-D non unitaires."""
    from vllm.model_executor.layers.fused_moe.moe_megakernel import (
        moe_megakernel_scales_ok,
        normalize_w_scale,
    )
    M, top_k, E, I, K = 2, 8, 32, 32, 256
    x, w1, w2, w1s, w2s, ids, _ = make_problem(M, top_k, E, I, K)
    assert moe_megakernel_scales_ok(w1, w2, w1s, w2s)            # 2-D
    assert moe_megakernel_scales_ok(w1, w2, w1s.unsqueeze(-1),
                                    w2s.unsqueeze(-1))            # 3-D [.,.,1]
    assert not moe_megakernel_scales_ok(w1, w2, None, w2s)        # None
    assert not moe_megakernel_scales_ok(w1, w2, w1s, None)
    bad3d = w1s.unsqueeze(-1).expand(E, 2 * I, 7).contiguous()   # non unitaire
    assert not moe_megakernel_scales_ok(w1, w2, bad3d, w2s)
    assert not moe_megakernel_scales_ok(w1, w2, w1s[:, :2], w2s)  # taille
    assert not moe_megakernel_scales_ok(w1, w2, w1s, w2s[:, :2])
    # normalize_w_scale : vue contiguë conservée (le kernel indexe [E, N])
    n = normalize_w_scale(w1s.unsqueeze(-1).contiguous())
    assert n.shape == w1s.shape and n.is_contiguous()


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
    """Le softmax+argmax DU PROLOGUE DU KERNEL reproduit topk_softmax —
    vérifié sur la sortie publiée du kernel (canary : dbg_ids écrits), pas
    seulement entre deux références torch (revue mimo M1 : un test qui passe
    dans les deux cas ne prouve rien).  La comparaison
    référence-vs-référence est conservée en sous-contrat."""
    from vllm import envs

    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_STRATEGY"] = "routing"
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = "routing"
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64

    M, top_k, E, I, K = 8, 8, 64, 64, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    router_w = torch.randn(E, K, device="cuda", dtype=torch.bfloat16) * 0.02
    w_a, i_a = _torch_routing(x, router_w, top_k)
    w_b, i_b = _torch_softmax_topk(x, router_w, top_k)
    # sous-contrat : les deux références torch concordent (pas de tirage)
    assert (i_a.to(torch.int64) == i_b).all(), (
        f"topk ids divergent: {i_a} vs {i_b}")
    assert (w_a - w_b).abs().max() < 2e-3, "poids topk divergents"

    # contrat réel : le prologue DU KERNEL doit publier la même sélection.
    # Poids experts nuls : seul le routage nous intéresse ici.
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    out = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    dbg_ids = torch.full((M, top_k), -1, device="cuda", dtype=torch.int32)
    dbg_w = torch.full((M, top_k), float("nan"), device="cuda")
    fused_moe_megakernel(
        out, x, w1, w2, None, None, None,
        torch.ones(E, 2 * I, device="cuda"),
        torch.ones(E, K, device="cuda"),
        naive=True, router_w=router_w, top_k_override=top_k,
        dbg_ids=dbg_ids, dbg_w=dbg_w)
    torch.cuda.synchronize()
    # canary (fail loud) : le kernel a ÉCRIT la sélection — sinon ce test
    # passerait aussi en repli silencieux (c'est exactement le trou M1).
    assert bool((dbg_ids >= 0).all()), (
        "dbg_ids non écrits — le prologue routing n'a pas tourné")
    assert bool(torch.isfinite(dbg_w).all()), "dbg_w non écrits"
    # ids exacts (données sans frontière d'égalité tirée au sort) ; poids
    # ≈ (renormalize du kernel vs topk_softmax, même écart que le contrat
    # routing, marge 2 %)
    assert (dbg_ids.to(torch.int64) == i_b).all(), (
        f"ids kernel {dbg_ids.tolist()} vs ref {i_b.tolist()}")
    w_err = float((dbg_w - w_b).abs().max() / w_b.abs().max())
    assert w_err < 2e-2, f"poids topk kernel err={w_err:.4f}"


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
    _ar_max_ctas,
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
    M, top_k, E, I, K = 13, 8, 256, 256, 512   # 13 × 8 = 104 CTAs > 96
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(E, 2 * I, K, device="cuda", dtype=torch.int8)
    w2 = torch.zeros(E, K, I, device="cuda", dtype=torch.int8)
    fake_comm = types.SimpleNamespace(
        rows=256, hidden=K, n_chunks=-(-I // MK_CI),
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
    assert M * (-(-I // MK_CI)) > _ar_max_ctas()  # précondition du test
    assert _ar_max_ctas() <= 104  # résidence mesurée (borne dure)


def test_engine_site_fallback_without_prologue_inputs(monkeypatch):
    """Le site d'appel moteur (TritonExperts.apply) n'a ni residual, ni
    poids de norm, ni poids de router, ni handles IPC : gates NORM/AR/
    ROUTING demandées => repli sur la chaîne livrée, ligne de log INCLUSE
    (chaque ligne expédiée exécutée — leçon 2026-10-25)."""
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = 64
    # m4 (revue GLM) : la ligne de log de repli est ASSERTÉE, pas seulement
    # supposée — la leçon 2026-10-25 est qu'une ligne jamais exécutée ne
    # prouve rien.
    import vllm.model_executor.layers.fused_moe.experts.triton_moe as tm
    seen: list[str] = []
    monkeypatch.setattr(
        tm.logger, "warning_once",
        lambda msg, *a: seen.append(msg % a if a else msg))
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
    # (raising=False : `vllm.envs` résout paresseusement — les gates sont
    # posées comme attributs de module par ce test)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_NORM", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_AR", False, raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_STRATEGY", "pair",
                        raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_ROUTING", False,
                        raising=False)
    assert call() is False
    assert any("NORM/AR" in s for s in seen), (
        f"ligne de repli NORM/AR non exécutée: {seen}")
    # ROUTING demandé (pas de router_w au site) -> repli + log
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_NORM", False,
                        raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_ROUTING", True,
                        raising=False)
    assert call() is False
    assert any("routage in-kernel" in s for s in seen), (
        f"ligne de repli ROUTING non exécutée: {seen}")
    assert moe_megakernel_flags() == (True, False, False)


@triton.jit
def _copy_flags_kernel(ptr, out_ptr, n, BLOCK: tl.constexpr):
    p = tl.cast(ptr.to(tl.int64), tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    for base in range(0, n, BLOCK):
        o = base + offs
        tl.store(out_ptr + o, tl.load(p + o, mask=o < n, other=0),
                 mask=o < n)


def _mk_two_rank_worker(rank, world, port, outdir, n_timed, n_burst):
    """Module-level pour spawn (picklable) ; résultats en FICHIERS (pas de
    mp.Queue — mesuré 2026-10-22 : EOFError sur le resource_sharer).

    Deux phases : (A) `n_timed` appels chronométrés (sync par appel) ;
    (B) `n_burst` appels ENCHAÎNÉS sans sync, x RE-TIRÉ à chaque appel.
    La phase B est ce qui peut falsifier la course de réutilisation du slot
    entre appels (revue GLM 2026-10-27 M1 : avec x constant + syncs, la
    batterie ne pouvait pas échouer sur B1 — la propriété qui tue en
    production était précisément celle que le test ne pouvait pas manquer).
    """
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

    n_chunks = -(-I // MK_CI)
    comm = MoeMegakernelComm(dist.group.WORLD, M, K, n_chunks)
    out = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    res_out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    dbg_q = torch.empty(M, K, device=dev, dtype=torch.int8)
    dbg_scale = torch.zeros(M, device=dev)
    dbg_ids = torch.zeros(M, top_k, device=dev, dtype=torch.int32)
    dbg_w = torch.zeros(M, top_k, device=dev)

    total = n_timed + n_burst
    times, recs = [], []
    for k in range(total):
        # x RE-TIRÉ à chaque appel (shard propre au rang) : une lecture
        # déchirée du slot entre appels mélange deux x différents et est
        # alors DÉTECTABLE bit-exact sur res_out.
        torch.manual_seed(1000 + 100 * rank + k)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5
        timed = k < n_timed
        if timed:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        fused_moe_megakernel(
            out, x, w1, w2, None, None, None, w1s, w2s,
            naive=True, router_w=router_w, top_k_override=top_k,
            res=res, norm_w=norm_w, eps=1e-6, res_out=res_out,
            dbg_scale=dbg_scale, dbg_q=dbg_q,
            dbg_ids=dbg_ids, dbg_w=dbg_w, ar_comm=comm)
        if timed:
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)
        # clones ASYNCHRONES sur le flux (aucun sync dans la rafale B)
        recs.append((x, res_out.clone(), dbg_q.clone(), dbg_scale.clone(),
                     dbg_ids.clone(), dbg_w.clone(), out.clone()))
    torch.cuda.synchronize()

    n_flags = M * n_chunks * FLAG_ELEMS
    flags = torch.empty(n_flags, device=dev, dtype=torch.int32)
    _copy_flags_kernel[((n_flags + 1023) // 1024,)](
        comm.flag_ptr, flags, n_flags, BLOCK=1024)
    torch.cuda.synchronize()
    save = {
        "rank": rank,
        "res": res.cpu(),
        "norm_w": norm_w.cpu(),
        "flags": flags.cpu(),
        "times": times,
        "recs": [tuple(t.cpu() for t in r) for r in recs],
    }
    if rank == 0:
        # poids (identiques des deux côtés, seed 7) pour la référence
        # d'experts côté parent (revue GLM M2 : sous AR, `out` doit avoir
        # une RÉFÉRENCE, pas seulement l'accord inter-rangs)
        save["w1"] = w1.cpu()
        save["w2"] = w2.cpu()
        save["w1s"] = w1s.cpu()
        save["w2s"] = w2s.cpu()
    torch.save(save, f"{outdir}/rank{rank}.pt")
    dist.destroy_process_group()


class TestIncrement3TwoRank:
    """INC 3 : AR 2 rangs in-kernel, méthode 2b — exact JUSQU'À la requant
    finale vs la somme fp32 exacte, accord inter-rangs, flags par slot
    (publication ET consommation), et réutilisation du slot entre appels
    falsifiée (x varié + rafale sans sync).

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

        n_timed, n_burst = 8, 16
        total = n_timed + n_burst
        ctx = mp.get_context("spawn")
        outdir = os.environ.get("MOEMK_TEST_OUT", "/tmp/moemk-2rank")
        os.makedirs(outdir, exist_ok=True)
        procs = [
            ctx.Process(target=_mk_two_rank_worker,
                        args=(r, 2, 29531, outdir, n_timed, n_burst))
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

        # --- flags par (rangée, chunk) : moitié PUBLICATION du handshake ---
        # élément 0 = flag de publication (séquence dérivée par slot —
        # exactement UN xchg par slot par appel : total, ni plus ni moins) ;
        # éléments 1+ = padding 256 B (vierge).  La moitié CONSOMMATION est
        # le double-tampon ping-pong (pas de compteur ici) — falsifiée par
        # la rafale à x varié ci-dessous.
        # Un flag par rangée ferait xchg la séquence n_chunks× (corruption).
        for rank in (0, 1):
            fl = got[rank]["flags"]
            slots = fl.view(-1, FLAG_ELEMS)
            assert (slots[:, 0] == total).all(), (
                f"rank {rank}: flag publication != {total}: "
                f"{slots[:, 0].tolist()}")
            assert (slots[:, 1:] == 0).all(), (
                f"rank {rank}: padding de flag non nul")

        res0, w0 = got[0]["res"], got[0]["norm_w"]
        # référence d'experts (M2) : poids sauvegardés par le rang 0, sur
        # GPU pour les ops de quant (même régime que les tests inc 2)
        w1 = got[0]["w1"].cuda()
        w2 = got[0]["w2"].cuda()
        w1s = got[0]["w1s"].cuda()
        w2s = got[0]["w2s"].cuda()

        flips_total = 0
        for k in range(total):
            for rank in (0, 1):
                x_self = got[rank]["recs"][k][0]
                x_peer = got[1 - rank]["recs"][k][0]
                q_ref, scale_ref, res_ref, _ = reference_norm_chain(
                    x_self, res0, w0, 1e-6, peer=x_peer)
                (_x, res_out_k, dbg_q_k, dbg_scale_k,
                 dbg_ids_k, dbg_w_k, out_k) = got[rank]["recs"][k]
                # 1. res_out BIT-EXACT — un slot déchiré entre appels
                #    (x_k mélangé à x_{k+1}) casse CETTE égalité
                assert torch.equal(res_out_k, res_ref), (
                    f"rank {rank} call {k}: res_out != contrat "
                    f"(slot déchiré entre appels ?)")
                # 2. scale : classe-3 (ordre de réduction de var)
                assert torch.allclose(dbg_scale_k, scale_ref[:, 0],
                                      rtol=1e-4, atol=1e-8), (
                    f"rank {rank} call {k}: scale hors contrat")
                # 3. méthode 2b : q EXACT jusqu'à la requant finale (±1 LSB)
                flips = (dbg_q_k.int() - q_ref.int()).abs()
                assert (flips <= 1).all(), (
                    f"rank {rank} call {k}: q diverge de >1 LSB: {flips.max()}")
                flips_total += int((flips > 0).sum())
                # 4. M2 : le travail d'expert a une RÉFÉRENCE sous AR
                ref_acc = reference_experts_from_q(
                    q_ref.cuda(), scale_ref.cuda(), w1, w2, w1s, w2s,
                    dbg_ids_k.cuda(), dbg_w_k.cuda())
                ref_out = _requant_roundtrip(ref_acc)
                err = _rel_err(out_k.cuda(), ref_out)
                assert err < 0.03, (
                    f"rank {rank} call {k}: expert-work rel_err={err:.4f}")
            # 5. ACCORD INTER-RANGS par appel
            out0 = got[0]["recs"][k][6].float()
            out1 = got[1]["recs"][k][6].float()
            diff = (out0 - out1).abs().max()
            scale = out0.abs().max().clamp_min(1e-6)
            assert float(diff / scale) < 2e-3, (
                f"call {k}: les rangs divergent: {float(diff / scale):.4f}")
            assert torch.equal(got[0]["recs"][k][2], got[1]["recs"][k][2]), (
                f"call {k}: les rangs divergent sur q")
        print(f"q +/-1 LSB flips (total {total} appels x 2 rangs) : "
              f"{flips_total}")
        # plafond de flips (revue mimo : « 0/2048 » est une OBSERVATION, le
        # gate doit être un plafond — une régression à des centaines de
        # flips ±1 LSB ne doit pas passer) : ≤ 1 % des éléments.
        n_elems = total * 2 * got[0]["recs"][0][2].numel()
        assert flips_total <= n_elems // 100, (
            f"trop de flips requant: {flips_total}/{n_elems}")

        # 6. timing (rapporté — borne, pas un gate).  Le wall d'un appel
        #    inclut le SKEW d'amorçage entre les processus rangs (le spin
        #    absorbe le skew en temps d'attente) : min sur N appels = le
        #    meilleur couplage observé, médiane = le régime courant.
        tmin = max(min(got[r]["times"]) for r in (0, 1))
        tmed = max(statistics.median(got[r]["times"]) for r in (0, 1))
        print(f"AR in-kernel (M=1 c1) : min {tmin:.1f} µs / méd {tmed:.1f} µs "
              f"(max des rangs, {n_timed} appels chronométrés)")
        for r in (0, 1):
            print(f"  rank {r} times: "
                  f"{[round(t, 1) for t in got[r]['times']]}")
        assert got[0]["flags"].numel() == got[1]["flags"].numel()


# ---------------------------------------------------------------------------
# INC-A (tour 3, mandat MFMA — TOUR3-FULL-FORWARD-DESIGN.md §3.4)
# ---------------------------------------------------------------------------
_MK_SIG = {
    "x_ptr": "*bf16", "x_scale_ptr": "*fp32",
    "w1_ptr": "*i8", "w1_scale_ptr": "*fp32",
    "w2_ptr": "*i8", "w2_scale_ptr": "*fp32",
    "topk_w_ptr": "*fp32", "topk_ids_ptr": "*i32",
    "sorted_token_ids_ptr": "*i32", "expert_ids_ptr": "*i32",
    "out_acc_ptr": "*fp32", "num_valid": "i32",
    "router_w_ptr": "*bf16", "dbg_ids_ptr": "*i32", "dbg_w_ptr": "*fp32",
    "stride_xm": "i32", "stride_w1e": "i32", "stride_w1n": "i32",
    "stride_w2e": "i32", "stride_w2k": "i32",
    "res_ptr": "*bf16", "res_out_ptr": "*bf16", "norm_w_ptr": "*fp32",
    "eps": "fp32", "dbg_scale_ptr": "*fp32", "dbg_q_ptr": "*i8",
    "slot_ptr": "i64", "peer_x_ptr": "i64",
    "flag_ptr": "i64", "peer_flag_ptr": "i64",
}


def _compile_asm(fn, sig, constexprs, num_warps, num_stages):
    """Compile pour gfx90a et retourne l'asm — méthode du smoke
    `moe_megakernel_smoke.py` (ASTSource + triton_compile ; compile-only,
    pas de GPU requis).  Piège tour 2 : ASTSource n'applique PAS les défauts
    des constexprs — tout est explicite."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as triton_compile

    names = list(fn.arg_names)
    sig = {k: v for k, v in sig.items() if k in names}
    src = ASTSource(fn=fn, signature=sig, constexprs=constexprs)
    c = triton_compile(src, target=GPUTarget("hip", "gfx90a", 32),
                       options={"num_warps": num_warps,
                                "num_stages": num_stages})
    return c.asm.get("amdgcn", "") or ""


def test_asm_mfma_no_vdot4():
    """Mandat opérateur (2026-10-27) : les dots des kernels du megakernel
    sont en v_mfma **K>=16 par instruction** — AUCUN v_dot4.  Sans ce test la
    revendication MFMA n'est pas prouvée : les dots tombent silencieusement
    en v_dot4 quand BM<16 ou contraction <16 (mesuré tour 1 : 6-19x plus
    lent ; smoke campagne : BM=1/CI=8 = 68-544 v_dot4, BM=16/CI=32 = 288-560
    v_mfma et 0 v_dot4)."""
    import re

    from vllm.model_executor.layers.fused_moe import moe_megakernel as mk

    base = {"top_k": 8, "K": 2048, "I": 256}
    cases = []

    # 1. routing + norm + ar (le kernel phare des incréments 1-3), config
    #    de lancement réelle (launcher : BM=MK_BM, CI=MK_CI, BN2=MK_BN2,
    #    num_warps=4, num_stages=1).
    cases.append((
        "routing+norm+ar", mk.qwen3_moe_megakernel, dict(_MK_SIG),
        {**base, "BM": MK_BM, "BKO": 128, "CI": MK_CI, "BN2": MK_BN2,
         "NAIVE": True, "ROUTING": True, "RENORM": True, "E": 256,
         "NORM": True, "AR": True, "FLAG_ELEMS_C": 64}, 4, 1))
    # 2. atomic tour 1 (chemin non-routing, x int8) — config du launcher.
    sig = dict(_MK_SIG)
    sig["x_ptr"] = "*i8"
    cases.append((
        "atomic-tour1", mk.qwen3_moe_megakernel, sig,
        {**base, "BM": MK_BM, "BKO": 128, "CI": MK_CI, "BN2": MK_BN2,
         "NAIVE": True, "ROUTING": False, "RENORM": True, "E": 256,
         "NORM": False, "AR": False, "FLAG_ELEMS_C": 64}, 4, 2))
    # 3. pair (v1) — config du launcher (BM=16, BN2=64, nw=8, ns=1).
    sig = dict(_MK_SIG)
    sig["x_ptr"] = "*i8"
    sig["c3_ptr"] = "*bf16"
    sig.pop("out_acc_ptr")
    sig["stride_c3m"] = "i32"
    cases.append((
        "pair-v1", mk.qwen3_moe_pair_fusion_kernel, sig,
        {**base, "BM": MK_BM, "BKO": 128, "BN2": MK_BN2, "NAIVE": True},
        8, 1))

    for name, fn, sig, cst, nw, ns in cases:
        asm = _compile_asm(fn, sig, cst, nw, ns)
        mnems = sorted(set(re.findall(r"v_mfma_[a-z0-9_]+", asm)))
        n_dot4 = asm.count("v_dot4")
        print(f"asm {name}: mfma={mnems} v_dot4 x{n_dot4}")
        assert mnems, (
            f"{name}: AUCUN v_mfma dans le codegen — les dots sont tombés "
            f"en micro-dots (BM<16 ou contraction <16 ?)")
        assert n_dot4 == 0, (
            f"{name}: v_dot4 présent {n_dot4}x — mandat MFMA violé")
        # K>=16 par instruction : le 3e nombre du mnémonique
        # (v_mfma_i32_16x16x16i8 -> K=16 ; v_mfma_i32_32x32x8i8 -> K=8).
        for m in mnems:
            mt = re.search(r"_(\d+)x(\d+)x(\d+)", m)
            assert mt, f"{name}: mnémonique MFMA illisible: {m}"
            assert int(mt.group(3)) >= 16, (
                f"{name}: {m} a K={mt.group(3)} < 16 par instruction — "
                f"mandat « v_mfma K>=16 » violé")

    # MUTATION (revue mimo M3) : le test doit pouvoir DISCORDER — la
    # configuration tour-1 BM=1/CI=8 doit être vue en v_dot4 (mesuré au
    # smoke : 68-544 v_dot4, 0 v_mfma).  Sans cette branche négative, un
    # test ASM qui passe dans les deux cas ne prouverait rien.
    sig = dict(_MK_SIG)
    sig["x_ptr"] = "*i8"
    asm_bad = _compile_asm(
        mk.qwen3_moe_megakernel, sig,
        {**base, "BM": 1, "BKO": 128, "CI": 8, "BN2": MK_BN2,
         "NAIVE": True, "ROUTING": False, "RENORM": True, "E": 256,
         "NORM": False, "AR": False, "FLAG_ELEMS_C": 64}, 4, 2)
    n_mfma_bad = asm_bad.count("v_mfma")
    n_dot4_bad = asm_bad.count("v_dot4")
    print(f"asm mutation BM=1/CI=8 (attendu v_dot4) : mfma x{n_mfma_bad} "
          f"v_dot4 x{n_dot4_bad}")
    assert n_dot4_bad > 0, (
        "mutation: BM=1/CI=8 n'émet plus de v_dot4 — le test ASM perd sa "
        "capacité à discorder (ou le codegen a changé : re-évaluer le mandat)")


# ===========================================================================
# stratégie `tree` (v3 — ANTIATOM-DESIGN.md) : zéro atomic.
# Contrat aux shapes RÉELLES avec les échelles au LAYOUT RÉEL du checkpoint
# quark [E, N, 1] 3-D (leçon de la gate muette n°2, commit 06d240d67e) ;
# déterminisme bit-exact ; mutations FALSIFIABLES isolées (chaque passe
# prouvée consommée) ; gate codegen « aucun atomic dans le codegen tree »
# avec contrôle NÉGATIF (le kernel `atomic` doit EN contenir — sinon le test
# ne peut pas discorder).
# ===========================================================================
from vllm.model_executor.layers.fused_moe.moe_megakernel import (
    MK_TREE_BKO,
    MK_TREE_CI,
    moe_megakernel_call_stats,
    moe_megakernel_record_call,
    moe_megakernel_reset_call_stats,
    moe_megakernel_reduce_kernel,
    normalize_w_scale,
    qwen3_moe_mk_tree_pass1_kernel,
    qwen3_moe_mk_tree_pass2_kernel,
)


def _tree_set_gates(strategy="tree", max_tokens="128"):
    # auto-portant : ÉCRASE les deux canaux (os.environ d'abord — il gagne)
    # y compris les gates de prologue qui fuient depuis les tests INC 1-3
    # (ROUTING/NORM/AR laissées à 1 par un test précédent = le site entre en
    # mode routing et demande router_w — pollution d'ordre de suite).
    os.environ["VLLM_MOE_MEGAKERNEL"] = "1"
    os.environ["VLLM_MOE_MEGAKERNEL_STRATEGY"] = strategy
    os.environ["VLLM_MOE_MEGAKERNEL_MAX_TOKENS"] = max_tokens
    for name in ("VLLM_MOE_MEGAKERNEL_ROUTING", "VLLM_MOE_MEGAKERNEL_NORM",
                 "VLLM_MOE_MEGAKERNEL_AR"):
        os.environ[name] = "0"
    from vllm import envs
    envs.VLLM_MOE_MEGAKERNEL = True
    envs.VLLM_MOE_MEGAKERNEL_STRATEGY = strategy
    envs.VLLM_MOE_MEGAKERNEL_MAX_TOKENS = int(max_tokens)
    envs.VLLM_MOE_MEGAKERNEL_ROUTING = False
    envs.VLLM_MOE_MEGAKERNEL_NORM = False
    envs.VLLM_MOE_MEGAKERNEL_AR = False


def _tree_problem(M, top_k, E, I, K, scale_3d=True):
    """Échelles poids au layout RÉEL quark [E, N, 1] (scale_3d=True) — la
    forme que le loader produit réellement ; sinon [E, N] 2-D (contrôle)."""
    x, w1, w2, w1s, w2s, ids, wts = make_problem(M, top_k, E, I, K)
    if scale_3d:
        w1s = w1s.unsqueeze(-1).contiguous()
        w2s = w2s.unsqueeze(-1).contiguous()
    return x, w1, w2, w1s, w2s, ids, wts


def _tree_pipeline(x, w1, w2, w1s, w2s, ids, wts, naive, sorted_ids,
                   expert_ids, *, pass2_fn=None, reduce_fn=None):
    """Les 3 passes `tree` avec kernels remplaçables (harnais de mutation).
    Reproduit EXACTEMENT le launcher — `test_tree_pipeline_matches_launcher`
    le prouve bit-exact ; sans cette preuve les mutants ne diraient rien."""
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )
    M, K = x.shape
    top_k = ids.size(1)
    I = w2.size(2)
    CI, BM, BKO = min(MK_TREE_CI, I), MK_BM, MK_TREE_BKO
    n_chunks = triton.cdiv(I, CI)
    num_pairs = M * top_k
    w1s2 = normalize_w_scale(w1s)
    w2s2 = normalize_w_scale(w2s)
    xq, xs = per_token_quant_int8(x)
    s1 = sorted_ids if sorted_ids is not None else ids
    e1 = expert_ids if expert_ids is not None else ids
    n_blocks = num_pairs if naive else triton.cdiv(sorted_ids.size(0), 16)
    mid_ws = torch.empty((num_pairs, I), device=x.device,
                         dtype=torch.float32)
    ws = torch.empty((num_pairs, K), device=x.device, dtype=torch.bfloat16)
    qwen3_moe_mk_tree_pass1_kernel[(n_blocks, n_chunks)](
        xq, xs, w1, w1s2, mid_ws, s1, e1, num_pairs,
        xq.stride(0), w1.stride(0), w1.stride(1),
        top_k=top_k, K=K, I=I, BM=BM, BKO=BKO, CI=CI,
        N_CHUNKS=n_chunks, NAIVE=naive, num_warps=2, num_stages=1)
    p2 = pass2_fn or qwen3_moe_mk_tree_pass2_kernel
    p2[(n_blocks, triton.cdiv(K, 64))](
        mid_ws, w2, w2s2, wts, ws, s1, e1, num_pairs,
        w2.stride(0), w2.stride(1),
        top_k=top_k, K=K, I=I, BM=BM, BN2=64, NAIVE=naive,
        num_warps=2, num_stages=1)
    rd = reduce_fn or moe_megakernel_reduce_kernel
    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    rd[(M, triton.cdiv(K, 64))](ws, out, top_k=top_k, K=K, BLOCK=64,
                                OUT_FP32=False, num_warps=4)
    torch.cuda.synchronize()
    return out


# --- mutants isolés (copies du kernel sain + UNE mutation nommée) ----------
@triton.jit
def _mut_pass2_drop_half_i(
    mid_ws_ptr, w2_ptr, w2_scale_ptr, topk_w_ptr, ws_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_valid,
    stride_w2e, stride_w2k,
    top_k: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN2: tl.constexpr, NAIVE: tl.constexpr,
):
    """MUTANT : la moitié haute de l'intermédiaire (I/2:I) est annulée avant
    le quant — si la sortie ne bouge pas, le dot ne consomme pas tout I."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)
    tokens = pairs // top_k
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
    e = tl.load(expert_ids_ptr + pid)
    e64 = e.to(tl.int64)
    offs_i_full = tl.arange(0, I)
    mid = tl.load(mid_ws_ptr + pairs_i64[:, None] * I + offs_i_full[None, :],
                  mask=valid[:, None], other=0.0)
    mid = tl.where(offs_i_full[None, :] >= I // 2, 0.0, mid)   # MUTATION
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    s_row = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])
    offs_o = tile * BN2 + tl.arange(0, BN2)
    m_o = offs_o < K
    s2 = tl.load(w2_scale_ptr + e64 * K + offs_o, mask=m_o, other=0.0)
    w2t = tl.load(w2_ptr + e64 * stride_w2e
                  + offs_i_full[:, None]
                  + offs_o[None, :] * stride_w2k)
    acc2 = tl.zeros((BM, BN2), dtype=tl.float32) + tl.dot(q, w2t)
    contrib = acc2 * (s_row * wt)[:, None] * s2[None, :]
    tl.store(ws_ptr + pairs_i64[:, None] * K + offs_o[None, :], contrib,
             mask=valid[:, None] & m_o[None, :])


@triton.jit
def _mut_pass2_scale_squared(
    mid_ws_ptr, w2_ptr, w2_scale_ptr, topk_w_ptr, ws_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_valid,
    stride_w2e, stride_w2k,
    top_k: tl.constexpr, K: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN2: tl.constexpr, NAIVE: tl.constexpr,
):
    """MUTANT : l'échelle per-token est appliquée DEUX fois (s_row²) — si la
    sortie ne bouge pas, l'échelle de quant n'est pas consommée."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    offs_pair = pid * BM + tl.arange(0, BM)
    if NAIVE:
        pairs = tl.where(offs_pair % BM == 0, offs_pair // BM, num_valid)
    else:
        pairs = tl.load(sorted_token_ids_ptr + offs_pair)
    valid = pairs < num_valid
    pairs_i64 = pairs.to(tl.int64)
    tokens = pairs // top_k
    wt = tl.load(topk_w_ptr + pairs_i64, mask=valid, other=0.0)
    e = tl.load(expert_ids_ptr + pid)
    e64 = e.to(tl.int64)
    offs_i_full = tl.arange(0, I)
    mid = tl.load(mid_ws_ptr + pairs_i64[:, None] * I + offs_i_full[None, :],
                  mask=valid[:, None], other=0.0)
    amax = tl.maximum(tl.max(tl.abs(mid), axis=1), 1e-10)
    s_row = amax / 127.0
    q = _round_i8(mid * (127.0 / amax)[:, None])
    offs_o = tile * BN2 + tl.arange(0, BN2)
    m_o = offs_o < K
    s2 = tl.load(w2_scale_ptr + e64 * K + offs_o, mask=m_o, other=0.0)
    w2t = tl.load(w2_ptr + e64 * stride_w2e
                  + offs_i_full[:, None]
                  + offs_o[None, :] * stride_w2k)
    acc2 = tl.zeros((BM, BN2), dtype=tl.float32) + tl.dot(q, w2t)
    contrib = acc2 * (s_row * s_row * wt)[:, None] * s2[None, :]  # MUTATION
    tl.store(ws_ptr + pairs_i64[:, None] * K + offs_o[None, :], contrib,
             mask=valid[:, None] & m_o[None, :])


@triton.jit
def _mut_reduce_drop_last_slot(
    ws_ptr, out_ptr,
    top_k: tl.constexpr, K: tl.constexpr,
    BLOCK: tl.constexpr, OUT_FP32: tl.constexpr,
):
    """MUTANT : la dernière paire du topk n'est pas sommée — si la sortie ne
    bouge pas, la boucle topk du reduce n'est pas consommée."""
    t = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    m = offs < K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, top_k - 1):
        p = (t * top_k + s) * K
        v = tl.load(ws_ptr + p + offs, mask=m, other=0.0)
        acc += v
    if OUT_FP32:
        tl.store(out_ptr + t * K + offs, acc, mask=m)
    else:
        tl.store(out_ptr + t * K + offs, acc.to(tl.bfloat16), mask=m)


def _tree_grouped_ids(M, top_k, E, I, K):
    x, w1, w2, w1s, w2s, ids, wts = _tree_problem(M, top_k, E, I, K, True)
    naive = M * top_k * 4 <= E
    sorted_ids = expert_ids = None
    if not naive:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
        )
        sorted_ids, expert_ids, _ = _prepare_expert_assignment(
            ids, {"BLOCK_SIZE_M": 16}, M, top_k, E, None)
    return x, w1, w2, w1s, w2s, ids, wts, naive, sorted_ids, expert_ids


def test_tree_pipeline_matches_launcher():
    """Le harnais de mutation reproduit le launcher au BIT près — sinon les
    mutants ne prouveraient rien (garde anti-harnais-faux)."""
    _tree_set_gates()
    M, top_k, E, I, K = 8, 8, 256, 256, 2048
    x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed = _tree_grouped_ids(
        M, top_k, E, I, K)
    out_launch = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    fused_moe_megakernel(
        out_launch, x, w1, w2, wts, ids, None, w1s, w2s,
        naive=naive, sorted_token_ids=sd, expert_ids=ed)
    torch.cuda.synchronize()
    out_pipe = _tree_pipeline(x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed)
    assert torch.equal(out_launch, out_pipe), (
        "le harnais de mutation diverge du launcher — les mutants sont hors "
        "jeu tant que ce n'est pas égal au bit près")


@pytest.mark.parametrize("M", [8, 32, 128])
def test_tree_contract_real_shapes_and_3d_scales(M):
    """Contrat `tree` aux shapes RÉELS TP=2 (E=256, I=256, K=2048, topk=8)
    avec les échelles poids au LAYOUT RÉEL du checkpoint quark **[E, N, 1]
    3-D** — la forme qui a rendu le bras muet en 2026-10-29.  Deux
    assertions : (1) contrat < 3 % vs la référence sémantique livrée ;
    (2) sortie BIT-IDENTIQUE 2-D vs 3-D (normalize_w_scale = vue, pas
    valeurs).  M=8 = régime naive, M=32/128 = grouped."""
    _tree_set_gates()
    top_k, E, I, K = 8, 256, 256, 2048
    x, w1, w2, w1s3, w2s3, ids, wts, naive, sd, ed = _tree_grouped_ids(
        M, top_k, E, I, K)
    if M == 8:
        assert naive
    else:
        assert not naive
    ref = reference_chain(x, w1, w2, normalize_w_scale(w1s3),
                          normalize_w_scale(w2s3), ids, wts, naive)

    out3 = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    fused_moe_megakernel(
        out3, x, w1, w2, wts, ids, None, w1s3, w2s3,
        naive=naive, sorted_token_ids=sd, expert_ids=ed)
    torch.cuda.synchronize()
    err = _rel_err(out3, ref)
    assert err < 0.03, f"M={M} tree rel_err={err:.4f}"
    assert ref.abs().max() > 0

    out2 = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    fused_moe_megakernel(
        out2, x, w1, w2, wts, ids, None,
        normalize_w_scale(w1s3), normalize_w_scale(w2s3),
        naive=naive, sorted_token_ids=sd, expert_ids=ed)
    torch.cuda.synchronize()
    assert torch.equal(out2, out3), (
        f"M={M} : 2-D vs 3-D non bit-identique — normalize_w_scale a "
        "changé des valeurs (ou le kernel indexe faux)")


@pytest.mark.parametrize("M", [8, 32])
def test_tree_determinism_bit_exact(M):
    """`tree` a un ordre de sommation FIXE : deux runs sont bit-identiques
    (strict — pas de tolérance ULP).  C'est un contrat NOMMÉ de `tree` que
    `atomic` ne peut pas tenir (somme flottante non-associative par
    atomiques)."""
    _tree_set_gates()
    top_k, E, I, K = 8, 256, 256, 2048
    x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed = _tree_grouped_ids(
        M, top_k, E, I, K)
    outs = []
    for _ in range(2):
        out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
        fused_moe_megakernel(
            out, x, w1, w2, wts, ids, None, w1s, w2s,
            naive=naive, sorted_token_ids=sd, expert_ids=ed)
        torch.cuda.synchronize()
        outs.append(out)
    assert torch.equal(outs[0], outs[1]), (
        f"M={M} : tree non déterministe bit-à-bit — l'ordre de réduction "
        "n'est pas fixe (ou il reste un atomic quelque part)")


@pytest.mark.parametrize("mut_name", [
    "pass2_drop_half_i", "pass2_scale_squared", "reduce_drop_last_slot"])
def test_tree_mutations_falsifiable(mut_name):
    """Mutations isolées : chaque passe/entrée doit être CONSOMMÉE.  Le
    mutant est une copie du kernel sain avec UNE mutation ; le noyau sain du
    MÊME harnais donne la référence.  Si le mutant ne diverge pas, la passe
    visée est décorative et le test l'annonce (silent-pass fermé)."""
    _tree_set_gates()
    M, top_k, E, I, K = 32, 8, 256, 256, 2048
    x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed = _tree_grouped_ids(
        M, top_k, E, I, K)
    ref = reference_chain(x, w1, w2, normalize_w_scale(w1s),
                          normalize_w_scale(w2s), ids, wts, naive)

    healthy = _tree_pipeline(x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed)
    err_h = _rel_err(healthy, ref)
    assert err_h < 0.03, f"noyau sain hors contrat: rel_err={err_h:.4f}"

    mut_fns = {
        "pass2_drop_half_i": dict(pass2_fn=_mut_pass2_drop_half_i),
        "pass2_scale_squared": dict(pass2_fn=_mut_pass2_scale_squared),
        "reduce_drop_last_slot": dict(reduce_fn=_mut_reduce_drop_last_slot),
    }
    mutant = _tree_pipeline(x, w1, w2, w1s, w2s, ids, wts, naive, sd, ed,
                            **mut_fns[mut_name])
    drift = _rel_err(mutant, healthy)
    assert drift > 0.02, (
        f"mutation {mut_name} INVISIBLE (drift={drift:.5f}) — la passe/entrée "
        "visée n'est pas consommée : le test est muet, la gate est décorative")


def test_tree_isa_no_atomics():
    """Gate CODEGEN (F10, M32-M128-EXTENSION §8.4 : « le lowering exact de
    tl.atomic_add n'était pas vérifié ») : les 3 kernels `tree` n'émettent
    AUCUNE instruction atomic (aucun addressing mode) — l'accumulateur
    atomique est structurellement IMPOSSIBLE dans `tree`.  Contrôle NÉGATIF
    : le kernel `atomic` en émet (sinon ce test ne peut pas discorder)."""
    import re as _re

    from vllm.model_executor.layers.fused_moe import moe_megakernel as mk

    cases = [
        ("tree-pass1", mk.qwen3_moe_mk_tree_pass1_kernel, {
            "x_ptr": "*i8", "x_scale_ptr": "*fp32", "w1_ptr": "*i8",
            "w1_scale_ptr": "*fp32", "mid_ws_ptr": "*fp32",
            "sorted_token_ids_ptr": "*i32", "expert_ids_ptr": "*i32",
            "num_valid": "i32", "stride_xm": "i32", "stride_w1e": "i32",
            "stride_w1n": "i32",
        }, {"top_k": 8, "K": 2048, "I": 256, "BM": MK_BM, "BKO": MK_TREE_BKO,
            "CI": MK_TREE_CI, "N_CHUNKS": 4, "NAIVE": True}, 2, 1),
        ("tree-pass2", mk.qwen3_moe_mk_tree_pass2_kernel, {
            "mid_ws_ptr": "*fp32", "w2_ptr": "*i8",
            "w2_scale_ptr": "*fp32", "topk_w_ptr": "*fp32",
            "ws_ptr": "*bf16", "sorted_token_ids_ptr": "*i32",
            "expert_ids_ptr": "*i32", "num_valid": "i32",
            "stride_w2e": "i32", "stride_w2k": "i32",
        }, {"top_k": 8, "K": 2048, "I": 256, "BM": MK_BM,
            "BN2": 64, "NAIVE": True}, 2, 1),
        ("tree-reduce", mk.moe_megakernel_reduce_kernel, {
            "ws_ptr": "*bf16", "out_ptr": "*bf16",
        }, {"top_k": 8, "K": 2048, "BLOCK": 64, "OUT_FP32": False},
            4, 1),
    ]
    for name, fn, sig, cst, nw, ns in cases:
        asm = _compile_asm(fn, sig, cst, nw, ns)
        atoms = _re.findall(r"\b[a-z_]*atomic[a-z0-9_]*", asm)
        print(f"asm {name}: atomics={len(atoms)} "
              f"({sorted(set(atoms))[:6]}) mfma x{asm.count('v_mfma')}")
        assert not atoms, (
            f"{name}: instructions atomic dans le codegen {sorted(set(atoms))}"
            " — l'accumulateur atomique n'est pas éliminé")
        if name in ("tree-pass1", "tree-pass2"):
            assert asm.count("v_mfma") > 0, (
                f"{name}: aucun v_mfma — les dots sont tombés en micro-dots")
            assert asm.count("v_dot4") == 0, (
                f"{name}: v_dot4 présent — mandat MFMA violé")

    # CONTRÔLE NÉGATIF : le kernel `atomic` doit contenir des atomics —
    # sinon la gate ci-dessus ne peut pas discorder (test-vide).
    sig = dict(_MK_SIG)
    sig["x_ptr"] = "*i8"
    asm_a = _compile_asm(
        mk.qwen3_moe_megakernel, sig,
        {"top_k": 8, "K": 2048, "I": 256, "BM": MK_BM, "BKO": 128,
         "CI": MK_CI, "BN2": MK_BN2, "NAIVE": True, "ROUTING": False,
         "RENORM": True, "E": 256, "NORM": False, "AR": False,
         "FLAG_ELEMS_C": 64}, 4, 2)
    atoms_a = _re.findall(r"\b[a-z_]*atomic[a-z0-9_]*", asm_a)
    print(f"asm atomic-controle: atomics x{len(atoms_a)} "
          f"({sorted(set(atoms_a))[:6]})")
    assert atoms_a, (
        "contrôle négatif: le kernel `atomic` n'émet AUCUN atomic — le "
        "codegen a changé (ou le test ne voit pas le bon kernel) : la gate "
        "« no atomics » est alors vacuante")


def test_call_counter_records_distribution(caplog):
    """Le compteur d'appels (travail nommé F1/W1) journalise la DISTRIBUTION
    (issue, M, régime) et émet sa ligne de résumé dès le 1er appel — le
    falsifiant qui remplace `warning_once` (muet après le 1er message)."""
    import logging

    moe_megakernel_reset_call_stats()
    before = moe_megakernel_call_stats()
    with caplog.at_level(logging.INFO, logger=(
            "vllm.model_executor.layers.fused_moe.moe_megakernel")):
        moe_megakernel_record_call("repli:gate-test", 1, None)
        moe_megakernel_record_call("bras:tree", 32, False)
        moe_megakernel_record_call("bras:tree", 32, False)
        moe_megakernel_record_call("repli:M=32 above threshold", 32, None)
    after = moe_megakernel_call_stats()
    assert after[("bras:tree", 32, "grouped")] == (
        before.get(("bras:tree", 32, "grouped"), 0) + 2)
    assert after[("repli:M=32 above threshold", 32, "-")] == (
        before.get(("repli:M=32 above threshold", 32, "-"), 0) + 1)
    # la ligne de résumé doit être ÉMISE (preuve de consommation du canal)
    assert any("appels-site" in r.message for r in caplog.records), (
        "aucune ligne appels-site loguée — le canal de preuve est muet")
