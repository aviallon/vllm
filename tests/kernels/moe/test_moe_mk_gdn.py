# SPDX-License-Identifier: Apache-2.0
"""Tests GPU de la chaîne GDN decode FUSIONNÉE (INC-B, campagne MEGAKERNEL).

Exécution réelle MI210 (gfx90a) exigée — skip sans GPU.  Auto-portant :
`python3 test_moe_mk_gdn.py` (runner __main__) ou pytest.

Contrat testé (docstring de `moe_megakernel_gdn.py`) : la fusion
conv+scan = composition EXACTE de `causal_conv1d_update` +
`fused_recurrent_gated_delta_rule_packed_decode`, **sur plusieurs pas
consécutifs** (les états évoluent — un test mono-pas ne verrait pas une
fenêtre d'état mal tournée) :

* (a) sorties + états scan (fp32) vs chaîne livrée, multi-pas ;
* (b) état conv **bit-exact** (fenêtre de x BRUT [s1,s2,x_t] — le contrat
  de stockage est entier, pas approché) ;
* (c) lignes nulles (state_idx<=0) : sortie zéro, états intacts ;
* (d) déterminisme run-to-run ; x NON modifié par le chemin fusionné
  (asymétrie NOMMÉE vs le livré qui écrase x en place) ;
* (e) mutations « break the fix » : single-buffer (DIVERGENCE attendue —
  la course CTA est le risque du design) et état silué au lieu de x brut
  (divergence attendue au pas suivant) ;
* gate `VLLM_MOE_MEGAKERNEL_GDN` : canaux env/envs, défaut OFF, repli =
  chaîne livrée verbatim (identité), fail loud sur entrées hors contrat.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_megakernel_gdn import (
    gdn_decode_fallback,
    gdn_decode_fusion,
    gdn_fusion_enabled,
)

torch.manual_seed(20261028)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU requis (MI210 gfx90a)"
)

# Shapes locales TP=2 réelles (config qwen3.6-35b-a3b, design §1.2) :
# H=8 têtes q/k, HV=16 têtes v, K=V=128, C=2*H*K+HV*V=4096.
REAL = dict(H=8, HV=16, K=128, V=128)


def _gate_on() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_GDN"] = "1"


def _gate_off() -> None:
    os.environ["VLLM_MOE_MEGAKERNEL_GDN"] = "0"


def make_problem(
    B: int = 4,
    num_cache: int = 32,
    H: int = 8,
    HV: int = 16,
    K: int = 128,
    V: int = 128,
    device: str = "cuda",
):
    """Problème GDN decode : x [B,C] bf16 (brut), états, poids conv."""
    C = 2 * H * K + HV * V
    x = torch.randn(B, C, device=device, dtype=torch.bfloat16) * 0.5
    b = torch.randn(B, HV, device=device, dtype=torch.bfloat16)
    a = torch.randn(B, HV, device=device, dtype=torch.bfloat16) * 0.3
    a_log = torch.randn(HV, device=device) * 0.1 - 1.0
    dt_bias = torch.randn(HV, device=device) * 0.1
    conv_w = torch.randn(C, 4, device=device, dtype=torch.bfloat16) * 0.25
    conv_bias = torch.randn(C, device=device, dtype=torch.bfloat16) * 0.1
    cs0 = torch.randn(num_cache, C, 3, device=device, dtype=torch.bfloat16) * 0.3
    cs1 = torch.zeros_like(cs0)
    st = torch.randn(num_cache, HV, V, K, device=device) * 0.05
    idx = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    out = torch.zeros(B, HV, V, device=device, dtype=torch.bfloat16)
    scale = K**-0.5
    return {
        "x": x,
        "b": b,
        "a": a,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "conv_w": conv_w,
        "conv_bias": conv_bias,
        "cs0": cs0,
        "cs1": cs1,
        "st": st,
        "idx": idx,
        "out": out,
        "scale": scale,
    }


def shipped_chain_step(
    x: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_w: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    idx: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Chaîne livrée APPELÉE DIRECTEMENT (indépendant du repli du module).

    ⚠ `causal_conv1d_update` écrase `x` en place (contrat livré) : le
    caller doit passer un clone par pas.
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    mixed = causal_conv1d_update(
        x, conv_state, conv_w, conv_bias, "silu",
        conv_state_indices=idx, validate_data=False,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=ssm_state,
        out=out.unsqueeze(1),
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    return out


# ===========================================================================
# (a)+(b) contrat multi-pas vs chaîne livrée
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize("B", [1, 4, 8])
def test_fusion_multi_step_vs_shipped(B):
    """5 pas consécutifs : sorties, état conv (bit-exact) et état scan.

    L'état conv est comparé **bit-à-bit** (`torch.equal`) : les deux
    chemins stockent la fenêtre de x BRUT — toute différence trahit le
    contrat de stockage (silué, fenêtre mal tournée, canal non écrit).
    L'état scan (fp32, accumulations) et les sorties (bf16) portent une
    tolérance documentée : la conv du fusionné somme ses 4 taps en
    réduction d'arbre (`tl.sum`) alors que le livré accumule en séquence
    — différences fp32 de dernier ULP possibles, amplifiées par le scan.
    Tolérance mesurée : voir `assert_close` (0.5 % relatif) — au-delà,
    c'est un écart de contrat, pas du bruit.
    """
    _gate_on()
    assert gdn_fusion_enabled(), "gate doit être ON dans ce test"
    p = make_problem(B=B, **REAL)
    steps = 5
    # flux d'entrées VARIÉ par pas (même flux pour les deux chemins) —
    # avec un x constant la fenêtre d'état converge vers [x,x,x] au pas 3
    # et le harnais ne verrait plus rien (piège de sensibilité).
    x_steps = [p["x"] + 0.01 * s * torch.randn_like(p["x"]) for s in range(steps)]
    rows = p["idx"].long()

    # référence : chaîne livrée, état conv UNIQUE (contrat de prod),
    # snapshot de l'état À CHAQUE PAS (comparer au final comparerait
    # l'étape s à l'étape finale — piège de calage temporel).
    ref_cs = p["cs0"].clone()
    ref_st = p["st"].clone()
    ref_outs, ref_cs_snaps = [], []
    for s in range(steps):
        o = torch.zeros_like(p["out"])
        shipped_chain_step(
            x_steps[s].clone(),  # le livré écrase x — clone par pas
            p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], ref_cs, p["idx"], ref_st, o,
            p["scale"],
        )
        ref_outs.append(o.clone())
        ref_cs_snaps.append(ref_cs[rows].clone())

    # fusionné : double-tampon, parité alternée par pas
    fu_cs0 = p["cs0"].clone()
    fu_cs1 = p["cs1"].clone()
    fu_st = p["st"].clone()
    x_kept = x_steps[0].clone()
    for s in range(steps):
        o = torch.zeros_like(p["out"])
        gdn_decode_fusion(
            x_steps[s], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], fu_cs0, fu_cs1, s % 2,
            p["idx"], fu_st, o, p["scale"],
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            o.float(), ref_outs[s].float(), rtol=5e-3, atol=5e-3,
            msg=f"sortie pas {s} : fusionné != chaîne livrée",
        )
        # état conv courant = le slot qui vient d'être écrit (1-parity) ;
        # seules les lignes ACTIVES sont écrites (les autres portent leur
        # init respective : zéro côté double-tampon, bruit côté livré).
        written = fu_cs1 if s % 2 == 0 else fu_cs0
        assert torch.equal(written[rows], ref_cs_snaps[s]), (
            f"état conv pas {s} : la fenêtre de x BRUT n'est pas bit-exact "
            f"(max diff "
            f"{(written[rows].float() - ref_cs_snaps[s].float()).abs().max().item()})"
        )
    # état scan final — tolérance fp32 : le fusionné somme ses taps en
    # réduction d'arbre, le livré en séquence ; sur 5 pas les écarts de
    # dernier ULP s'amplifient légèrement (mesuré : 9 éléments/8,4 M à
    # ≤2,4e-3).  Un vrai écart de contrat (fenêtre, canal non écrit) est
    # O(1), pas 2e-3.
    torch.testing.assert_close(fu_st, ref_st, rtol=5e-3, atol=5e-3)
    # x NON modifié par le chemin fusionné (asymétrie NOMMÉE vs livré,
    # qui écrase x en place — vérifié sur le pas 0)
    assert torch.equal(x_steps[0], x_kept), "le fusionné a modifié x (contrat : non)"


@needs_gpu
def test_null_row_zero_and_states_untouched():
    """state_idx<=0 : sortie zéro, états intacts — contrat livré."""
    _gate_on()
    p = make_problem(B=4, **REAL)
    p["idx"][:] = torch.tensor([0, 2, 0, 3], device="cuda", dtype=torch.int32)
    cs0 = p["cs0"].clone()
    cs1 = torch.full_like(p["cs0"], 7.0)  # SENTINELLE : toute écriture se voit
    st = p["st"].clone()
    o = torch.full_like(p["out"], 7.0)
    gdn_decode_fusion(
        p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], cs0, cs1, 0, p["idx"], st, o, p["scale"],
    )
    torch.cuda.synchronize()
    for i in (0, 2):
        assert bool((o[i] == 0).all()), f"ligne nulle {i} : sortie non nulle"
    for i in (1, 3):
        assert bool((o[i] != 0).any()), f"ligne valide {i} : sortie nulle"
    # ⚠ les rangées d'état sont indexées par state_idx (idx[token]), pas
    # par token : token 1 -> rangée 2, token 3 -> rangée 3 ; les tokens
    # nuls (idx=0) n'écrivent AUCUNE rangée.
    touched = {2, 3}
    for r in range(cs1.shape[0]):
        if r in touched:
            assert bool((cs1[r] != 7.0).any()), (
                f"état conv non écrit sur la rangée valide {r}"
            )
        else:
            assert torch.equal(cs1[r], torch.full_like(cs1[r], 7.0)), (
                f"état conv écrit sur la rangée {r} (non allouée à un token valide)"
            )
    # état scan : aussi indexé par state_idx — token 1 -> rangée 2,
    # token 3 -> rangée 3 ; les autres rangées doivent être intactes.
    for r in range(st.shape[0]):
        if r in touched:
            assert bool((st[r] != 0).any()), f"état scan non écrit rangée {r}"
        else:
            assert torch.equal(st[r], p["st"][r]), (
                f"état scan écrit sur la rangée {r} (non allouée)"
            )


@needs_gpu
def test_determinism_run_to_run():
    _gate_on()
    p = make_problem(B=4, **REAL)
    outs = []
    for _ in range(2):
        cs0, cs1, st = p["cs0"].clone(), p["cs1"].clone(), p["st"].clone()
        o = torch.zeros_like(p["out"])
        gdn_decode_fusion(
            p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], cs0, cs1, 0, p["idx"], st, o,
            p["scale"],
        )
        torch.cuda.synchronize()
        outs.append(o.clone())
    assert torch.equal(outs[0], outs[1]), "non-déterminisme run-to-run"


# ===========================================================================
# (e) mutations — « break the fix on purpose »
# ===========================================================================
@needs_gpu
def test_mutation_single_buffer_diverges():
    """Séparation des slots de l'état conv — propriété DÉTERMINISTE.

    ⚠ Correction revue deepseek F1 (silent-pass) : la version précédente
    comparait 3 pas de référence à 1 pas de mutant (différence « par
    construction ») et la divergence par course CTA est **non observable**
    dans ce kernel standalone (chaque programme lit tôt et écrit tard — la
    fenêtre de course est quasi nulle, mesuré : 0 divergence sur 5 reps).
    La revendication « mutation-prouvé pour la course » est donc
    RETIRÉE ; ce qui est prouvé, et ce qui compte pour INC-D :

    * le chemin sain n'écit JAMAIS dans le slot LU (`cs0` intact) ;
    * la mutation single-buffer écrit DANS le slot lu (observable,
      déterministe) ;
    * le launcher refuse l'alias des deux slots (assert data_ptr, revue
      F3).

    ⚠ Et la VRAIE dépendance est nommée (revue F4) : inlinée dans la
    boucle persistante, l'écriture de la couche L et la lecture de la
    couche L+1 partagent le même lancement — le double-tampon ne suffit
    plus, une barrière de phase est requise (c'est l'infra INC-D).
    """
    _gate_on()
    p = make_problem(B=8, **REAL)

    # chemin sain : le slot LU reste intact
    cs0 = p["cs0"].clone()
    cs1 = p["cs1"].clone()
    gdn_decode_fusion(
        p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], cs0, cs1, 0, p["idx"],
        p["st"].clone(), torch.zeros_like(p["out"]), p["scale"],
    )
    torch.cuda.synchronize()
    assert torch.equal(cs0, p["cs0"]), (
        "le chemin sain écrit dans le slot LU — la séparation des slots "
        "n'est pas respectée (c'est le contrat du double-tampon)"
    )
    assert bool((cs1 != p["cs1"]).any()), (
        "le chemin sain n'écrit pas dans le slot d'écriture — harnais aveugle"
    )

    # mutation single-buffer : l'écriture tombe DANS le slot lu (visible)
    cs0m = p["cs0"].clone()
    cs1m = p["cs1"].clone()
    gdn_decode_fusion(
        p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], cs0m, cs1m, 0, p["idx"],
        p["st"].clone(), torch.zeros_like(p["out"]), p["scale"],
        _mut_single_buffer=True,
    )
    torch.cuda.synchronize()
    assert not torch.equal(cs0m, p["cs0"]), (
        "MUT_SINGLE_BUFFER n'écrit pas dans le slot lu — la mutation est "
        "invisible, le test ne prouve rien"
    )


@needs_gpu
def test_mutation_silu_state_diverges():
    """Mutation STORE_SILU_STATE (état = x silué au lieu de x brut) →
    divergence DÉTERMINISTE au pas suivant (la conv du pas n+1 lit une
    fenêtre fausse)."""
    _gate_on()
    p = make_problem(B=4, **REAL)
    ref_cs = p["cs0"].clone()
    ref_st = p["st"].clone()
    ref_o = torch.zeros_like(p["out"])
    shipped_chain_step(
        p["x"].clone(), p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], ref_cs, p["idx"], ref_st, ref_o,
        p["scale"],
    )
    # pas 2 : référence vs monde muté
    xs = p["x"].clone()
    ref_o2 = torch.zeros_like(p["out"])
    shipped_chain_step(
        xs, p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], ref_cs, p["idx"], ref_st, ref_o2,
        p["scale"],
    )

    mut_cs0 = p["cs0"].clone()
    mut_cs1 = p["cs1"].clone()
    mut_st = p["st"].clone()
    o1 = torch.zeros_like(p["out"])
    gdn_decode_fusion(
        p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], mut_cs0, mut_cs1, 0, p["idx"],
        mut_st, o1, p["scale"], _mut_store_silu_state=True,
    )
    o2 = torch.zeros_like(p["out"])
    gdn_decode_fusion(
        p["x"], p["b"], p["a"], p["a_log"],
        p["dt_bias"], p["conv_w"], p["conv_bias"], mut_cs0, mut_cs1, 1,
        p["idx"], mut_st, o2, p["scale"], _mut_store_silu_state=True,
    )
    torch.cuda.synchronize()
    assert not torch.equal(o2, ref_o2), (
        "état silué stocké : sortie du pas 2 IDENTIQUE au référentiel — "
        "la mutation n'est pas observable (harnais aveugle)"
    )
    # le pas 1 lui-même doit être identique (la mutation ne vaut que pour
    # la fenêtre future) — garde de sensibilité du harnais
    torch.testing.assert_close(
        o1.float(), ref_o.float(), rtol=5e-3, atol=5e-3,
        msg="le pas 1 diverge : la mutation ne devrait PAS l'affecter",
    )


# ===========================================================================
# gate / repli / fail loud
# ===========================================================================
def test_gate_channels_and_default_off(monkeypatch):
    """Gate `VLLM_MOE_MEGAKERNEL_GDN`, défaut OFF.  CANARI declared-input :
    le canal `os.environ` doit allumer la gate SANS enregistrement dans
    `vllm.envs` (interdit en INC-B) — sinon la gate existe sans jamais
    s'allumer (piège « déclaré mais jamais consommé »)."""
    from vllm import envs

    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_GDN", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_GDN", False, raising=False)
    assert gdn_fusion_enabled() is False, "défaut doit être OFF"
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_GDN", "1")
    assert gdn_fusion_enabled() is True, "VLLM_MOE_MEGAKERNEL_GDN=1 ignoré"
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_GDN", "true")
    assert gdn_fusion_enabled() is True
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_GDN", "0")
    assert gdn_fusion_enabled() is False
    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_GDN", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_GDN", True, raising=False)
    assert gdn_fusion_enabled() is True, "canal envs.VLLM_MOE_MEGAKERNEL_GDN ignoré"


@needs_gpu
def test_fallback_gate_off_is_shipped_path(monkeypatch):
    """Gate OFF : repli = chaîne livrée VERBATIM — le kernel fusionné ne
    doit PAS tourner (compteur) et le résultat doit être celui de la
    chaîne livrée appelée directement (identité de repli)."""
    _gate_off()
    import vllm.model_executor.layers.fused_moe.moe_megakernel_gdn as gdnm

    calls = {"n": 0}

    class _CountingKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                calls["n"] += 1

            return _launch

    monkeypatch.setattr(gdnm, "qwen3_gdn_decode_fusion_kernel", _CountingKernel())

    p = make_problem(B=2, **REAL)
    x0 = p["x"].clone()  # le livré écrase x en place — garder l'original
    cs0_orig = p["cs0"].clone()  # le repli mute conv_state en place aussi
    fu_cs1 = p["cs1"].clone()
    o = torch.zeros_like(p["out"])
    gdn_decode_fusion(
        x0.clone(), p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], p["cs0"], fu_cs1, 0, p["idx"],
        p["st"].clone(), o, p["scale"],
    )
    torch.cuda.synchronize()
    assert calls["n"] == 0, (
        "le kernel fusionné a tourné malgré la gate OFF — le repli n'est "
        "pas la chaîne livrée"
    )
    # identité : même sortie ET même état conv que la chaîne livrée
    # appelée directement (même x d'entrée, mêmes états initiaux)
    st_check = p["st"].clone()
    ref_o2 = torch.zeros_like(p["out"])
    ref_cs2 = cs0_orig.clone()
    shipped_chain_step(
        x0.clone(), p["b"], p["a"], p["a_log"], p["dt_bias"],
        p["conv_w"], p["conv_bias"], ref_cs2, p["idx"], st_check, ref_o2,
        p["scale"],
    )
    rows = p["idx"].long()
    assert torch.equal(o, ref_o2), "repli : sortie != chaîne livrée"
    assert torch.equal(p["cs0"][rows], ref_cs2[rows]), (
        "repli : état conv != chaîne livrée"
    )
    assert torch.equal(fu_cs1[rows], p["cs1"][rows]), (
        "repli : le second slot (double-tampon) ne doit pas être touché — "
        "le chemin livré n'a qu'un état conv"
    )


@needs_gpu
def test_launcher_fail_loud():
    """Entrées hors contrat : asserts nommés, pas de comportement silencieux."""
    _gate_on()
    p = make_problem(B=2, **REAL)
    with pytest.raises(AssertionError, match="parity"):
        gdn_decode_fusion(
            p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], p["cs0"], p["cs1"], 2, p["idx"],
            p["st"], p["out"], p["scale"],
        )
    with pytest.raises(AssertionError, match="state_indices"):
        gdn_decode_fusion(
            p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], p["cs0"], p["cs1"], 0,
            p["idx"].to(torch.int64), p["st"], p["out"], p["scale"],
        )
    bad_cs = p["cs0"][:, :, :2].contiguous()
    with pytest.raises(AssertionError, match="conv_state_0"):
        gdn_decode_fusion(
            p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], bad_cs, p["cs1"], 0, p["idx"],
            p["st"], p["out"], p["scale"],
        )
    with pytest.raises(AssertionError, match="num_valid"):
        gdn_decode_fusion(
            p["x"], p["b"], p["a"], p["a_log"], p["dt_bias"],
            p["conv_w"], p["conv_bias"], p["cs0"], p["cs1"], 0, p["idx"],
            p["st"], p["out"], p["scale"], num_valid=99,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
