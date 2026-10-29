# SPDX-License-Identifier: Apache-2.0
"""Tests GPU de INC-C — lm_head fp32 + SAMPLING shard-aware (campagne
MEGAKERNEL, MI210 gfx90a).

Exécution réelle MI210 exigée — les tests numériques sont skip sans GPU.
Auto-portant : ``python3 test_moe_mk_sampling.py`` (runner __main__) ou pytest.

Contrat testé (TOUR3-FULL-FORWARD-DESIGN.md §3.5 ; docstring de
``moe_megakernel_sampling.py`` pour les écarts NOMMÉS) :

* lm_head fp32 : contrat vs référence torch fp64 (≤ 1 LSB de quant) ET vs
  ``triton_scaled_mm`` (le livré, arrondi bf16 — écart NOMMÉ : le gate-ON
  garde le fp32 de l'acc) ;
* greedy identique vs chemin livré (argmax vocab entier) sur plusieurs
  seeds, ex aequo au plus petit id inclus ;
* **test statistique de distribution** : milliers de tirages (Philox
  déterministe) sur une petite distribution fixe vs la loi attendue calculée
  analytiquement (proportions par bin à 5σ + chi-carré seuillé) — la
  comparaison inter-impls de logprobs est INTERDITE comme critère et n'est
  PAS utilisée ici ;
* **test NÉGATIF OOB** : reproduit le comportement de l'ancien processeur
  non-shard-aware (``index_put_`` ids globaux sur tenseur de largeur shard :
  ``eos=248044`` → écriture à la colonne 123884 de la rangée SUIVANTE, hors
  allocation pour la dernière rangée — ETAPE-B2 §3) et prouve que le
  nouveau mappe correctement (gid → gid − vocab_start_index) avec store
  borné ; **test de MUTATION** : neutraliser le mapping fait ÉCHOUER la
  vérification ;
* replis comptés + loggés (one-shot) : num_logprobs == -1, ids exigés hors
  shard, processeurs exotiques, top-k/top-p non couverts, sans troncature,
  gate OFF ; chaque ligne de log est un CANARI asserté ;
* 2 rangs : même token greedy et même tirage des deux côtés (Philox
  synchronisé) — pattern ``TestIncrement3TwoRank`` (processus spawn +
  fichiers) et fake_comm in-process ;
* gate OFF = repli verbatim (identité D'OBJET du ``delivered_fn``) ; gate
  canari : les 2 canaux ``os.environ`` / ``envs``.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

import vllm.model_executor.layers.fused_moe.moe_megakernel_sampling as mks
from vllm.model_executor.layers.fused_moe.moe_megakernel_sampling import (
    ShardContractError,
    ShardProcessorState,
    ShardRequestParams,
    apply_shard_aware_processors,
    default_gather_fn,
    global_to_local,
    mk_lm_head_logits,
    mk_sampling_enabled,
    philox_uniforms,
    shard_sample,
    shard_topk,
)

torch.manual_seed(20261028)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU requis (MI210 gfx90a)"
)


# ===========================================================================
# fixtures : reset des one-shot logs / compteurs + capture des logs
# ===========================================================================
@pytest.fixture(autouse=True)
def _reset_oneshot(monkeypatch):
    """Un motif de log one-shot ne peut être canari qu'une fois par processus :
    chaque test repart d'un registre vierge (le one-shot lui-même est testé
    explicitement par test_one_shot_logging).  Hygiène mémoire : le pod porte
    l'engine chargée (~2 GiB libres, fluctuants) — le cache de l'allocateur
    HIP est rendu entre tests (sinon OOM/« invalid device function » d'un
    test à l'autre, observé 2026-10-29)."""
    monkeypatch.setattr(mks, "_LOGGED", set())
    monkeypatch.setattr(mks, "FALLBACK_COUNTERS", {})
    yield
    torch.cuda.empty_cache()


@pytest.fixture
def logs(monkeypatch):
    """Capture les lignes de log du module (canary : une ligne de log
    expédiée doit être exécutée par un test — doctrine 2026-10-25)."""
    seen: list[str] = []

    def _info(msg, *args, **kwargs):
        seen.append(msg % args if args else msg)

    monkeypatch.setattr(mks.logger, "info", _info)
    return seen


def _gate_on(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_SAMPLING", "1")


def _gate_off(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_SAMPLING", "0")


def _spy_mfma_kernel(monkeypatch) -> dict[str, int]:
    """Compteur de lancements du kernel MFMA (compte ET exécute).

    ⚠ Sans cette épingle + la gate DENSE posée, les tests « lm_head fp32
    MFMA » passeraient trivialement sur la chaîne livrée bf16 (repli interne
    de dense_mfma_gemm quand sa propre gate est OFF) — silent-pass sur le
    headline même d'INC-C (revue GLM 2026-10-28, warning 1).
    """
    import vllm.model_executor.layers.fused_moe.moe_megakernel_dense as mdd

    calls = {"n": 0}
    real = mdd.qwen3_dense_mfma_gemm_kernel

    class _Spy:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                calls["n"] += 1
                return real[grid](*args, **kwargs)

            return _launch

    monkeypatch.setattr(mdd, "qwen3_dense_mfma_gemm_kernel", _Spy())
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "1")
    return calls


# ===========================================================================
# helpers : référence INDÉPENDANTE du chemin livré (vocab entier, ids globaux)
# ===========================================================================
def reference_processed_full_logits(
    full: torch.Tensor, params: ShardRequestParams
) -> torch.Tensor:
    """Processeurs livrés sur le vocab ENTIER — verbatim la sémantique de
    ``MinTokensLogitsProcessor.apply`` / ``apply_bad_words`` / logit_bias :
    ``index_put_`` avec ids GLOBAUX (le mécanisme du crash ETAPE-B2)."""
    x = full.clone()
    M = x.shape[0]
    rows: list[int] = []
    gids: list[int] = []
    for row in range(M):
        for gid in (*params.stop_token_ids, *params.banned_token_ids):
            rows.append(row)
            gids.append(gid)
    if rows:
        x.index_put_(
            (
                torch.tensor(rows, device=x.device),
                torch.tensor(gids, device=x.device),
            ),
            torch.tensor(float("-inf"), device=x.device),
        )
    brows, bgids, bdeltas = [], [], []
    for row in range(M):
        for gid, delta in params.bias_token_ids:
            brows.append(row)
            bgids.append(gid)
            bdeltas.append(delta)
    if brows:
        x.index_put_(
            (
                torch.tensor(brows, device=x.device),
                torch.tensor(bgids, device=x.device),
            ),
            torch.tensor(bdeltas, dtype=x.dtype, device=x.device),
            accumulate=True,
        )
    if params.allowed_token_ids is not None:
        for row in range(M):
            for gid in range(x.shape[1]):
                if gid not in params.allowed_token_ids:
                    x[row, gid] = float("-inf")
    return x


def reference_greedy_token(full_processed: torch.Tensor) -> int:
    """Token greedy du chemin livré (``Sampler.greedy_sample`` = argmax) ;
    ex aequo : la première colonne (le plus petit id) — le contrat de
    ``_argmax_lowest_id``."""
    x = full_processed[0]
    mx = x.max()
    return int(torch.nonzero(x == mx, as_tuple=False)[0].item())


def reference_law(
    v: torch.Tensor,
    temp: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Loi attendue du tirage — implémentation INDÉPENDANTE (torch pur, vocab
    entier) de la spec écrite : température ; masque top-k par VALEUR (>= la
    k-ième plus grande = conservé, ex aequo inclus) ; top-p sur softmax
    renormalisé sur les survivants top-k (ou softmax du vocab entier sans
    top-k) : gardé iff la masse des candidats de rang strictement supérieur <
    p, au moins le max conservé ; loi finale = softmax des survivants.

    ⚠ AUCUNE comparaison de logprobs entre implémentations ici (interdite
    comme critère) : la loi est calculée analytiquement, puis confrontée aux
    FRÉQUENCES de tirage observées.
    """
    x = v / temp
    keep = torch.ones_like(v, dtype=torch.bool)
    if top_k is not None:
        v_k = x.sort(descending=True).values[top_k - 1]
        keep = x >= v_k
    if top_p is not None and top_p < 1.0:
        neg = torch.where(keep, x, torch.full_like(x, float("-inf")))
        if top_k is not None:
            q = torch.softmax(neg, dim=-1)
        else:
            q = torch.softmax(x, dim=-1)
        q = q * keep
        order = _ref_sort_order(x)
        q_s = q[order]
        prefix = torch.cumsum(q_s, dim=-1) - q_s
        # survivant = (top-k) ET (masse strictement meilleure < p) ; au moins
        # le max conservé (le max passe toujours le top-k : v_k <= max)
        keep_s = keep[order] & (prefix < top_p)
        keep_s[0] = True
        keep = torch.zeros_like(keep)
        keep[order] = keep_s
    neg = torch.where(keep, x, torch.full_like(x, float("-inf")))
    return torch.softmax(neg, dim=-1)


def _ref_sort_order(x: torch.Tensor) -> torch.Tensor:
    """Ordre du contrat : valeur décroissante, puis id global croissant
    (tri composite stable — même définition que le module)."""
    id_sorted = torch.arange(x.numel(), device=x.device)  # ids = colonnes
    vals_by_id = x[id_sorted]
    val_desc = torch.argsort(-vals_by_id, stable=True)
    return id_sorted[val_desc]


class FakeTwoRankGather:
    """fake_comm : mini gather in-process entre deux « rangs » (specs des
    tests 2 rangs sans NCCL).  Les payloads pairs sont figés à la
    construction (shards statiques, pas de processeurs dans ces tests)."""

    def __init__(
        self,
        peer_vals: torch.Tensor,
        peer_ids: torch.Tensor,
        peer_lse: torch.Tensor,
    ):
        self.peer_vals = peer_vals
        self.peer_ids = peer_ids
        self.peer_lse = peer_lse

    def __call__(self, vals, ids, lse):
        return (
            torch.cat([vals, self.peer_vals], dim=1),
            torch.cat([ids, self.peer_ids], dim=1),
            torch.cat([lse, self.peer_lse], dim=1),
        )


def two_rank_setup(
    full: torch.Tensor,
    k_loc: int,
    temp: torch.Tensor,
    params: ShardRequestParams | None = None,
):
    """Retourne (gather pour rang0, gather pour rang1) — les deux rangs
    voient le merged set complet via le peer pré-calculé.  Les payloads du
    peer sont calculés sur le shard peer APRÈS processeurs (les top-K' et lse
    doivent refléter le masquage, sinon le merged set diverge du contrat)."""
    half = full.shape[1] // 2
    sh0 = full[:, :half].clone()
    sh1 = full[:, half:].clone()
    if params is not None:
        apply_shard_aware_processors(
            sh0, 0, full.shape[1], params, ShardProcessorState()
        )
        apply_shard_aware_processors(
            sh1, half, full.shape[1], params, ShardProcessorState()
        )
    v0, i0 = shard_topk(sh0, k_loc, 0)
    v1, i1 = shard_topk(sh1, k_loc, half)
    lse0 = torch.logsumexp(sh0 / temp[:, None], dim=1).unsqueeze(1)
    lse1 = torch.logsumexp(sh1 / temp[:, None], dim=1).unsqueeze(1)
    return (
        FakeTwoRankGather(v1, i1, lse1),
        FakeTwoRankGather(v0, i0, lse0),
        sh0,
        sh1,
    )


# ===========================================================================
# gate : 2 canaux (os.environ prioritaire, puis attribut envs), défaut OFF
# ===========================================================================
def test_gate_channels_and_default_off(monkeypatch):
    """CANARI declared-input : `os.environ` doit allumer la gate même sans
    enregistrement dans `vllm.envs` — sinon la variable existerait sans
    jamais rien activer (piège « déclaré mais jamais consommé »)."""
    from vllm import envs

    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_SAMPLING", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_SAMPLING", False, raising=False)
    assert mk_sampling_enabled() is False, "défaut doit être OFF"

    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_SAMPLING", "1")
    assert mk_sampling_enabled() is True, (
        "VLLM_MOE_MEGAKERNEL_SAMPLING=1 ignoré — la gate ne s'allume jamais"
    )
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_SAMPLING", "true")
    assert mk_sampling_enabled() is True
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_SAMPLING", "0")
    assert mk_sampling_enabled() is False

    monkeypatch.delenv("VLLM_MOE_MEGAKERNEL_SAMPLING", raising=False)
    monkeypatch.setattr(envs, "VLLM_MOE_MEGAKERNEL_SAMPLING", True, raising=False)
    assert mk_sampling_enabled() is True, (
        "canal envs.VLLM_MOE_MEGAKERNEL_SAMPLING ignoré"
    )


# ===========================================================================
# lm_head fp32
# ===========================================================================
def _quant_contract_torch(a: torch.Tensor):
    """Quant per-token du contrat scaled_mm (expressions verbatim, torch)."""
    a_f = a.float()
    amax = a_f.abs().amax(dim=1, keepdim=True)
    scale = amax / 127.0
    inv_s = torch.where(amax == 0.0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(a_f * inv_s).clamp(-128.0, 127.0).to(torch.int8)
    return q, scale.view(-1)


@needs_gpu
@pytest.mark.parametrize(
    "M,K,V",
    [
        (2, 256, 32),  # petit shard
        (1, 2048, 124160),  # lm_head local TP=2 réel, c1
        (4, 512, 100),  # N non multiple de la tuile
    ],
)
def test_lm_head_fp32_contract_vs_torch_and_scaled_mm(M, K, V, monkeypatch):
    """Contrat lm_head fp32 : (a) vs référence torch fp64 — écart ≤ 1 LSB de
    quant (même borne que test_mfma_dense_gemm) ; (b) vs ``triton_scaled_mm``
    (le livré) cast fp32 — la seule différence est l'arrondi bf16 du livré
    (écart NOMMÉ : gate-ON garde le fp32 de l'acc, ≤ 0.5 ulp bf16).  Le
    kernel MFMA est ÉPINGLÉ (compteur de lancements + gate dense posée) :
    sans ça, le repli interne bf16 rendrait le test trivialement vert
    (revue GLM 2026-10-28, warning 1)."""
    calls = _spy_mfma_kernel(monkeypatch)
    _gate_on(monkeypatch)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randint(-127, 128, (K, V), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(V, device="cuda") * 0.02 + 0.001

    out = mk_lm_head_logits(x, w, w_scale)
    torch.cuda.synchronize()
    assert calls["n"] >= 1, (
        "le kernel MFMA n'a pas tourné — le test comparerait le repli bf16 "
        "à lui-même (silent-pass)"
    )
    assert out.shape == (M, V) and out.dtype == torch.float32, (
        f"sortie {tuple(out.shape)} {out.dtype} != [{M},{V}] fp32"
    )

    # (a) borne de quant vs fp64 — référence par CHUNKS de colonnes : le pod
    # porte l'engine chargée (~2 GiB libres, mesuré 2026-10-29), le fp64
    # plein [K,V] = 2,03 GiB fait OOM (observé).  Même contrat, même borne.
    _, a_scale = _quant_contract_torch(x)
    a_s = a_scale.double().view(-1, 1)
    bad = 0
    diff_max = 0.0
    tol_max = 0.0
    bound_max = 0.0
    chk = 2048  # chunks frugaux : ~32 MiB fp64 par tranche (budget pod serré)
    for c0 in range(0, V, chk):
        c1 = min(c0 + chk, V)
        b_dq = w[:, c0:c1].double() * w_scale[c0:c1].double().view(1, -1)
        ref_c = x.double() @ b_dq
        out_c = out[:, c0:c1].double()
        bound_c = 0.5 * a_s * b_dq.abs().sum(0).view(1, -1)
        tol_c = bound_c + ref_c.abs() * 2.0**-20 + out_c.abs() * 2.0**-9 + 1e-9
        diff_c = (out_c - ref_c).abs()
        bad += int((diff_c > tol_c).sum())
        diff_max = max(diff_max, float(diff_c.max()))
        tol_max = max(tol_max, float(tol_c.max()))
        bound_max = max(bound_max, float(bound_c.max()))
        del b_dq, ref_c, out_c, bound_c, tol_c, diff_c
        torch.cuda.empty_cache()
    assert bad == 0, (
        f"[{M},{K},{V}] : {bad} éléments au-delà de 1 LSB de quant — "
        f"max diff={diff_max:.6g}, max tol={tol_max:.6g}"
    )
    assert bound_max > 0, "borne de quant nulle : test vide"
    assert diff_max > 0, "diff nulle : la référence n'a pas mordu"

    # (b) vs le livré : écart = arrondi bf16 seul
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    ref_bf16, _ = triton_scaled_mm_fused_quant(x, w, w_scale, torch.bfloat16)
    torch.cuda.synchronize()
    d = (out - ref_bf16.float()).abs()
    ulp = ref_bf16.float().abs() * 2.0**-8 + 1e-6
    assert bool((d <= ulp).all()), (
        f"[{M},{K},{V}] : écart fp32-vs-bf16 hors ulp bf16 : "
        f"max {float(d.max()):.6g} vs ulp {float(ulp.max()):.6g}"
    )
    # discriminateur (revue GLM 2026-10-28, warning 2) : l'épilogue fp32 doit
    # se DISTINGUER de l'arrondi bf16 sur valeurs génériques — sinon le test
    # resterait vert même si l'épilogue régressait en stockage bf16.
    assert float(d.max()) > 0.0, (
        f"[{M},{K},{V}] : sortie bit-identique à bf16 cast — l'épilogue "
        f"fp32 n'est pas discriminé (données ou cast régressé ?)"
    )


@needs_gpu
def test_lm_head_fallback_gate_off_is_shipped_path(monkeypatch, logs):
    """Gate OFF : repli = chaîne livrée VERBATIM (lm_head bf16 + cast fp32) :
    identité BIT-EXACTE avec ``triton_scaled_mm_fused_quant(...).float()`` ;
    le kernel MFMA ne tourne PAS (compteur) ; la ligne de log de repli est
    un CANARI asserté."""
    _gate_off(monkeypatch)
    calls = {"n": 0}
    real_gemm = mks.dense_mfma_gemm

    def _counting_gemm(*a, **k):
        calls["n"] += 1
        return real_gemm(*a, **k)

    monkeypatch.setattr(mks, "dense_mfma_gemm", _counting_gemm)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    M, K, V = 2, 256, 64
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randint(-127, 128, (K, V), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(V, device="cuda") * 0.02 + 0.001

    out = mk_lm_head_logits(x, w, w_scale)
    ref, _ = triton_scaled_mm_fused_quant(x, w, w_scale, torch.bfloat16)
    torch.cuda.synchronize()

    assert calls["n"] == 0, (
        "dense_mfma_gemm appelé malgré la gate OFF — le repli n'est pas la "
        "chaîne livrée"
    )
    assert torch.equal(out, ref.float()), (
        "repli lm_head != triton_scaled_mm_fused_quant cast fp32 (identité)"
    )
    assert mks.FALLBACK_COUNTERS.get("lm_head_gate_off") == 1, (
        f"compteur de repli lm_head_gate_off absent : {mks.FALLBACK_COUNTERS}"
    )
    assert any("lm_head livré" in s for s in logs), (
        f"ligne de log de repli lm_head non exécutée : {logs}"
    )


@needs_gpu
def test_lm_head_gate_on_active_log(monkeypatch, logs):
    """Gate ON : le launcher MK tourne (sortie fp32 cohérente) et la ligne
    one-shot ACTIVE est un CANARI asserté.  Le kernel MFMA est épinglé
    (compteur de lancements) : le test doit POUVOIR échouer si le repli
    bf16 prenait le relais silencieusement (revue GLM 2026-10-28)."""
    calls = _spy_mfma_kernel(monkeypatch)
    _gate_on(monkeypatch)
    M, K, V = 2, 256, 64
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randint(-127, 128, (K, V), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(V, device="cuda") * 0.02 + 0.001
    out = mk_lm_head_logits(x, w, w_scale)
    out2 = mk_lm_head_logits(x, w, w_scale)
    torch.cuda.synchronize()
    assert calls["n"] == 2, (
        f"lancements MFMA : {calls['n']} != 2 — le chemin gate-ON n'a pas "
        f"utilisé le kernel (repli silencieux ?)"
    )
    assert torch.equal(out, out2), "lm_head MFMA non déterministe"
    n_active = sum("lm_head fp32 MFMA ACTIF" in s for s in logs)
    assert n_active == 1, (
        f"ligne one-shot active lm_head : {n_active} occurrences (attendu 1)"
    )
    assert mks.FALLBACK_COUNTERS.get("lm_head_gate_off") is None, (
        "compteur de repli lm_head_gate_off incrémenté en gate ON"
    )


# ===========================================================================
# greedy identique vs chemin livré (plusieurs seeds) + ex aequo
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_greedy_matches_shipped_across_seeds(seed, monkeypatch):
    """Greedy identique au chemin livré (argmax vocab entier après
    processeurs) sur 5 seeds, avec processeurs min_tokens actifs (ids
    globaux, certains hors shard de rang 0)."""
    _gate_on(monkeypatch)
    torch.manual_seed(seed)
    M, V_global = 3, 256
    full = torch.randn(M, V_global, device="cuda")
    params = ShardRequestParams(stop_token_ids=(250, 7, 130))
    ref_full = reference_processed_full_logits(full, params)
    expected = [reference_greedy_token(ref_full[i : i + 1]) for i in range(M)]

    half = V_global // 2
    g0, g1, _, _ = two_rank_setup(
        full,
        min(mks.DEFAULT_K_PRIME, half),
        torch.ones(M, device="cuda"),
        params=params,
    )
    for rank in (0, 1):
        sh = full[:, rank * half : (rank + 1) * half].clone()
        state = ShardProcessorState()
        res = shard_sample(
            sh,
            rank * half,
            V_global,
            params,
            state,
            gather_fn=(g0 if rank == 0 else g1),
        )
        assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
        got = res.tokens.tolist()
        assert got == expected, (
            f"seed {seed} rang {rank} : greedy {got} != livré {expected}"
        )
    assert mks.FALLBACK_COUNTERS == {}, (
        f"replis comptés en chemin couvert : {mks.FALLBACK_COUNTERS}"
    )


@needs_gpu
def test_greedy_tie_breaks_lowest_id(monkeypatch):
    """Ex aequo en tête : le shard path choisit le plus petit id global,
    identique au ``argmax`` livré (première colonne = plus petit id)."""
    _gate_on(monkeypatch)
    M, V_global = 1, 64
    full = torch.zeros(M, V_global, device="cuda")
    full[0, 50] = 10.0
    full[0, 5] = 10.0  # ex aequo avec l'id 50
    half = V_global // 2
    g0, g1, _, _ = two_rank_setup(
        full, min(mks.DEFAULT_K_PRIME, half), torch.ones(M, device="cuda")
    )
    res = shard_sample(
        full[:, :half].clone(),
        0,
        V_global,
        ShardRequestParams(),
        ShardProcessorState(),
        gather_fn=g0,
    )
    expected = reference_greedy_token(full)
    assert expected == 5, f"référence livrée : ex aequo 5/50 → {expected}"
    assert res.tokens.tolist() == [expected], (
        f"ex aequo 5/50 : shard {res.tokens.tolist()} != livré [{expected}]"
    )


# ===========================================================================
# processeurs shard-aware : mapping, union exacte, store borné
# ===========================================================================
@needs_gpu
def test_processor_union_equals_global_mask():
    """L'union des processeurs appliqués par shard = le masque global livré,
    BIT-EXACT (stop/banned masqués, biais additifs, liste blanche) — chaque
    field de ShardRequestParams ARRIVE (canary declared-input)."""
    torch.manual_seed(11)
    M, V_global = 2, 64
    full = torch.randn(M, V_global, device="cuda")
    params = ShardRequestParams(
        stop_token_ids=(7, 60),
        banned_token_ids=(13,),
        bias_token_ids=((20, 0.5), (40, -1.25)),
        allowed_token_ids=frozenset(range(0, 64)) - {3, 55, 63},
    )
    ref = reference_processed_full_logits(full, params)

    half = V_global // 2
    state = ShardProcessorState()
    sh0 = full[:, :half].clone()
    sh1 = full[:, half:].clone()
    out0 = apply_shard_aware_processors(sh0, 0, V_global, params, state)
    out1 = apply_shard_aware_processors(sh1, half, V_global, params, state)
    merged = torch.cat([sh0, sh1], dim=1)

    assert torch.equal(merged, ref), (
        "union des shards != masque global livré :\n"
        f"diff {(merged - ref).abs().sum()} "
        f"({int((merged != ref).sum())} éléments)"
    )
    # observations nombrées (M=2 lignes ⇒ chaque id produit 2 écritures) :
    # rang0 [0,32) : stop 7 + banned 13 + complément de liste blanche {3}
    # = 3 ids × 2 lignes = 6 écritures ; rang1 [32,64) : stop 60 +
    # complément {55,63} = 3 ids × 2 lignes = 6 écritures.
    assert out0.masked == 6, f"rang0 masked={out0.masked} (attendu 6)"
    assert out1.masked == 6, f"rang1 masked={out1.masked} (attendu 6)"
    # skips hors shard (× 2 lignes) : rang0 voit stop 60 (1×2) ; rang1 voit
    # stop 7 + banned 13 (2×2) — gérés par l'AUTRE rang, aucun repli
    assert out0.skipped_out_of_shard == 2, (
        f"rang0 skipped={out0.skipped_out_of_shard} (attendu 2 : stop 60 ×2)"
    )
    assert out1.skipped_out_of_shard == 4, (
        f"rang1 skipped={out1.skipped_out_of_shard} (attendu 4 : stop 7 + "
        f"banned 13, ×2 lignes)"
    )
    assert not out0.needs_full_gather and not out1.needs_full_gather, (
        "les ids masquants hors shard ne doivent pas déclencher de repli"
    )


@needs_gpu
def test_store_bounded_guard_and_mask_counts(monkeypatch):
    """La garde de store borné `_assert_store_bounded` refusse toute colonne
    hors shard (fail loud) — une garde jamais exécutée ne prouve rien :
    elle est testée directement dans ses deux branches."""
    cols_ok = torch.tensor([0, 5, 31], device="cuda")
    mks._assert_store_bounded(cols_ok, 32, "masque")  # ne doit pas lever
    with pytest.raises(AssertionError, match="hors shard"):
        mks._assert_store_bounded(torch.tensor([0, 32], device="cuda"), 32, "masque")
    with pytest.raises(AssertionError, match="hors shard"):
        mks._assert_store_bounded(torch.tensor([-1], device="cuda"), 32, "biais")
    # et la garde ACTIVE : un gid hors shard ne produit AUCUNE écriture
    logits = torch.zeros(1, 32, device="cuda")
    state = ShardProcessorState()
    out = apply_shard_aware_processors(
        logits,
        0,
        64,
        ShardRequestParams(stop_token_ids=(60,)),
        state,
    )
    assert out.skipped_out_of_shard == 1 and out.masked == 0, (
        f"gid hors shard : skipped={out.skipped_out_of_shard} "
        f"masked={out.masked} (attendu 1/0)"
    )
    assert bool((logits == 0).all()), "écriture détectée sur un gid hors shard"
    assert out.needs_full_gather is False, (
        "un id MASQUANT hors shard ne doit pas déclencher de repli — "
        "c'est l'autre rang qui le masque"
    )


# ===========================================================================
# test NÉGATIF OOB : reproduction du crash historique + mapping correct
# ===========================================================================
@needs_gpu
def test_oob_historical_crash_reproduced_and_fixed():
    """Reproduit ETAPE-B2 §3 (le crash G1-c128) puis prouve le correctif.

    ANCIEN (MinTokensLogitsProcessor sur logits de largeur SHARD) :
    ``logits.index_put_((req_idx, stop_token_ids), -inf)`` avec des ids
    GLOBAUX.  Sur un tenseur contigu de largeur 124160, l'adresse calculée
    ``row*124160 + 248044`` tombe à la colonne ``248044 - 124160 = 123884``
    de la rangée SUIVANTE (corruption en-borne des logits du voisin) et,
    pour la dernière rangée, 123884 éléments ≈ 484 Kio HORS allocation
    (UB CUDA → SIGABRT des deux rangs observé).

    NOUVEAU : ``global_to_local(gid, vocab_start_index)`` + store borné —
    le gid n'est écrit que s'il est dans CE shard, jamais ailleurs.
    """
    M, V_loc = 3, 124160
    V_global = 248320
    eos = 248044  # eos_token_id réel du checkpoint (ETAPE-B2 §3)
    assert eos - V_loc == 123884, "arithmétique historique changée ?"

    # --- ANCIEN comportement, reproduit SÛREMENT : l'indexation linéaire
    # row*V_loc + gid sur un tampon [M+1, V_loc] (la rangée M = zone de
    # garde = l'allocation en aval dans le crash réel).
    big = torch.zeros(M + 1, V_loc, device="cuda")
    flat = big.view(-1)
    rows = torch.arange(M, device="cuda") * V_loc
    flat.index_put_((rows + eos,), torch.tensor(float("-inf"), device="cuda"))
    torch.cuda.synchronize()
    obs = (big == float("-inf")).nonzero().tolist()
    expected_oob = [[r + 1, 123884] for r in range(M)]
    assert obs == expected_oob, (
        f"reproduction OOB : observé {obs} != attendu {expected_oob}"
    )
    # la dernière écriture (rangée 2) atterrit dans la zone de garde = HORS
    # de la région logits [M, V_loc] — l'OOB historique, en chiffres
    assert obs[-1][0] == M, f"la dernière écriture ne sort pas de [M] : {obs[-1]}"
    # la ligne 0 n'a RIEN reçu : ses -inf sont partis chez le voisin
    assert bool((big[0] == 0).all()), "ligne 0 écrite (attendu : rien)"

    # --- NOUVEAU comportement : mapping gid→col local, store borné
    for rank, start in ((0, 0), (1, V_loc)):
        logits = torch.zeros(M + 1, V_loc, device="cuda")  # rangée garde
        state = ShardProcessorState()
        out = apply_shard_aware_processors(
            logits[:M],
            start,
            V_global,
            ShardRequestParams(stop_token_ids=(eos,)),
            state,
        )
        torch.cuda.synchronize()
        if rank == 1:
            # gid 248044 dans le shard [124160, 248320) → col 123884 de la
            # MÊME rangée (pas de décalage de rangée)
            col = global_to_local(eos, V_loc)
            assert col == 123884, f"mapping : {col} != 123884"
            exp = [[r, 123884] for r in range(M)]
            obs2 = (logits[:M] == float("-inf")).nonzero().tolist()
            assert obs2 == exp, f"rang1 : écritures {obs2} != {exp}"
            assert out.masked == M, f"rang1 masked={out.masked} != {M}"
        else:
            obs2 = (logits[:M] == float("-inf")).nonzero().tolist()
            assert obs2 == [], f"rang0 a écrit hors shard : {obs2}"
            assert out.skipped_out_of_shard == M, (
                f"rang0 skipped={out.skipped_out_of_shard} != {M}"
            )
        # le crash OOB est IMPOSSIBLE : la zone de garde est intacte
        assert bool((logits[M] == 0).all()), (
            f"rang{rank} : écriture dans la zone de garde = OOB encore possible"
        )


@needs_gpu
def test_mapping_mutation_detected(monkeypatch):
    """MUTATION (mandat) : neutraliser ``gid - vocab_start_index`` (mapping
    identité = l'ancien processeur non-shard-aware) doit faire ÉCHOUER la
    vérification de mapping.  Un test qui passe aussi avec la mutation ne
    prouve rien."""
    M, V_loc = 2, 124160
    eos = 248044

    def _verify(logits: torch.Tensor) -> None:
        """Vérification commune (test positif + mutation) : le stop id doit
        être masqué à la colonne LOCALE du rang 1, et rien ailleurs."""
        obs = (logits == float("-inf")).nonzero().tolist()
        assert obs == [[0, 123884], [1, 123884]], (
            f"stop id non masqué à sa colonne locale : {obs}"
        )

    # POSITIF : mapping réel → la vérification passe
    logits = torch.zeros(M, V_loc, device="cuda")
    apply_shard_aware_processors(
        logits,
        V_loc,
        2 * V_loc,
        ShardRequestParams(stop_token_ids=(eos,)),
        ShardProcessorState(),
    )
    _verify(logits)

    # MUTATION : mapping neutralisé (= ancien comportement) → la
    # vérification DOIT échouer (le gid tombe hors shard et est rejeté par le
    # store borné : le stop id n'est plus masqué du tout)
    monkeypatch.setattr(mks, "global_to_local", lambda gid, start: gid)
    logits_mut = torch.zeros(M, V_loc, device="cuda")
    apply_shard_aware_processors(
        logits_mut,
        V_loc,
        2 * V_loc,
        ShardRequestParams(stop_token_ids=(eos,)),
        ShardProcessorState(),
    )
    with pytest.raises(AssertionError, match="non masqué"):
        _verify(logits_mut)


# ===========================================================================
# logprobs : décomposition logsumexp exacte (contrat « identiques »)
# ===========================================================================
@needs_gpu
def test_logprobs_decomposition_matches_reference(monkeypatch):
    """Logprobs = log_softmax des logits traités (post-température,
    post-processeurs, PRÉ-troncature) : le shard path (décomposition
    logsumexp exacte) doit coïncider avec la référence vocab-entier à l'ordre
    de réduction fp32 près (tolérance mesurée dans le message d'échec).
    ⚠ Ce test porte le CONTRAT d'identité greedy/logprobs — il n'est PAS un
    critère de distribution (interdiction inter-impls sur la distribution)."""
    _gate_on(monkeypatch)
    torch.manual_seed(23)
    M, V_global = 2, 256
    full = torch.randn(M, V_global, device="cuda")
    params = ShardRequestParams(
        stop_token_ids=(11,),
        required_logprob_token_ids=(70, 100),  # couverts par le shard 0
        num_logprobs=3,
    )
    temp = torch.tensor([1.0, 0.7], device="cuda")
    top_k = torch.tensor([4, 6], device="cuda")
    top_p = torch.tensor([0.95, 0.95], device="cuda")

    ref_full = reference_processed_full_logits(full, params)
    # température livrée : division puis log_softmax (PRÉ-troncature)
    ref_logprobs = (ref_full / temp[:, None]).log_softmax(dim=-1)

    half = V_global // 2
    g0, g1, _, _ = two_rank_setup(
        full, min(mks.DEFAULT_K_PRIME, half), temp, params=params
    )
    state = ShardProcessorState()
    res = shard_sample(
        full[:, :half].clone(),
        0,
        V_global,
        params,
        state,
        gather_fn=g0,
        request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
        step=7,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
    )
    assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    lp = res.logprobs
    assert lp is not None

    # logprob du token tiré
    rows = torch.arange(M, device="cuda")
    ref_chosen = ref_logprobs[rows, res.tokens]
    d = (lp.chosen - ref_chosen).abs().max()
    assert float(d) < 1e-4, (
        f"logprob tiré : max |shard - livré| = {float(d):.3e} ≥ 1e-4"
    )

    # top-3 : ids et logprobs identiques au tri (valeur décroissante, id
    # croissant) de la référence
    for i in range(M):
        ref_row = ref_logprobs[i]
        order = _ref_sort_order(ref_row)
        exp_ids = order[:3].tolist()
        exp_lp = ref_row[order[:3]]
        assert lp.top_ids[i].tolist() == exp_ids, (
            f"ligne {i} : top-3 ids {lp.top_ids[i].tolist()} != {exp_ids}"
        )
        dd = (lp.top_logprobs[i] - exp_lp).abs().max()
        assert float(dd) < 1e-4, f"ligne {i} : top-3 logprobs max diff {float(dd):.3e}"

    # required_logprob_token_ids : le champ déclaré ARRIVE (canary) et sa
    # valeur est exacte — ids couverts par CE shard (start=0, local=[0,128))
    assert lp.required_ids == ((0, 70), (0, 100), (1, 70), (1, 100)), (
        f"required_ids = {lp.required_ids}"
    )
    assert lp.required is not None and lp.required.shape == (4,), (
        f"required shape {None if lp.required is None else tuple(lp.required.shape)}"
    )
    req_ref = torch.stack(
        [ref_logprobs[r, g] for r, g in ((0, 70), (0, 100), (1, 70), (1, 100))]
    )
    dr = (lp.required - req_ref).abs().max()
    assert float(dr) < 1e-4, (
        f"required logprobs : max |shard - livré| = {float(dr):.3e}"
    )


# ===========================================================================
# test STATISTIQUE de distribution (la loi, pas les logprobs inter-impls)
# ===========================================================================
@needs_gpu
@pytest.mark.parametrize(
    "top_k,top_p",
    [(None, 0.9), (4, None), (6, 0.95)],
    ids=["top_p_0.9", "top_k_4", "top_k6_top_p0.95"],
)
def test_statistical_distribution_vs_expected_law(top_k, top_p, monkeypatch):
    """Milliers de tirages sur une petite distribution FIXE vs la loi
    attendue (calculée analytiquement par ``reference_law``, vocab entier).

    Tolérance DOCUMENTÉE : pour chaque bin, |observé − attendu| ≤ 5σ binomial
    (σ = sqrt(N·p·(1−p))) ; global : chi-carré Σ(obs−exp)²/exp ≤ critique
    (ddl = bins−1, α = 1e-6) — table intégrée ci-dessous.  Le tirage Philox
    étant déterministe, le test est reproductible bit-à-bit : un échec est
    une régression, jamais du bruit.  AUCUNE comparaison de logprobs entre
    implémentations (INTERDITE comme critère de distribution)."""
    _gate_on(monkeypatch)
    N = 8192  # milliers de draws : une ligne = un tirage
    V = 16
    # petite distribution fixe (mêmes logits pour chaque « requête »)
    base = torch.tensor(
        [
            3.0,
            2.2,
            1.7,
            1.1,
            0.6,
            0.2,
            -0.3,
            -0.9,
            -1.4,
            -2.0,
            -2.7,
            -3.5,
            -4.4,
            -5.4,
            -6.5,
            -7.7,
        ],
        device="cuda",
    )
    law = reference_law(base, temp=1.0, top_k=top_k, top_p=top_p)
    bins = int((law > 0).sum())
    assert bins >= 2, f"loi dégénérée ({bins} bins) : test sans puissance"
    assert law.sum().item() == pytest.approx(1.0, abs=1e-5), (
        f"loi non normalisée : {float(law.sum())}"
    )

    full = base.unsqueeze(0).expand(N, V).contiguous()
    hashes = (torch.arange(N, device="cuda", dtype=torch.int64) * 2654435761) % (2**62)
    state = ShardProcessorState()
    res = shard_sample(
        full,
        0,
        V,
        ShardRequestParams(),
        state,
        k_prime=V,
        gather_fn=None,  # mono-rang : gather_fn par défaut, repli compté
        request_id_hashes=hashes,
        step=42,
        temperature=torch.ones(N, device="cuda"),
        top_k=(
            torch.full((N,), top_k, device="cuda", dtype=torch.long)
            if top_k is not None
            else None
        ),
        top_p=(torch.full((N,), top_p, device="cuda") if top_p is not None else None),
    )
    assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    assert res.tokens.shape == (N,), f"tokens {tuple(res.tokens.shape)} != ({N},)"

    obs = torch.bincount(res.tokens, minlength=V).double()
    exp = law.double() * N
    chi2_crit = {
        1: 30.0,
        2: 32.0,
        3: 34.0,
        4: 36.0,
        5: 37.0,
        6: 39.0,
        7: 40.0,
        8: 41.0,
        15: 47.0,
    }
    df = bins - 1
    tol_chi2 = chi2_crit.get(df, 47.0)
    sigma = (N * law.double() * (1 - law.double())).clamp_min(1e-9).sqrt()
    chi2 = ((obs - exp) ** 2 / exp.clamp_min(1e-9))[law > 0].sum()
    max_z = float(((obs - exp).abs() / sigma)[law > 0].max())
    counts = {i: int(obs[i]) for i in range(V) if float(law[i]) > 0}
    expected_p = {i: round(float(law[i]), 5) for i in range(V) if float(law[i]) > 0}
    print(
        f"\n[stat] top_k={top_k} top_p={top_p} N={N} bins={bins} "
        f"chi2={float(chi2):.3f} (crit ddl {df} : {tol_chi2}) "
        f"max|z|={max_z:.3f} (seuil 5)\n  obs={counts}\n  loi={expected_p}"
    )
    assert max_z <= 5.0, (
        f"bin hors 5σ : max|z|={max_z:.3f} — obs={counts} loi={expected_p}"
    )
    assert float(chi2) <= tol_chi2, (
        f"chi-carré {float(chi2):.3f} > {tol_chi2} (ddl {df}, α=1e-6) — "
        f"obs={counts} loi={expected_p}"
    )
    # garde anti-test-vide : chaque bin attendu a une masse exploitable
    assert float(exp[law > 0].min()) >= 100.0, (
        f"bin trop faible pour discriminer : min exp={float(exp[law > 0].min())}"
    )


# ===========================================================================
# Philox : structure, déterminisme, séparation de streams, uniformité
# ===========================================================================
def test_philox_mul_lo_hi_exact_vs_python_ints():
    """`_mul_lo_hi` (la seule arithmétique du stream) = produit 32×32→64
    EXACT, oraculaire en entiers Python sur 1000 paires."""
    import random as _random

    rng = _random.Random(20261028)
    for _ in range(1000):
        a = rng.randrange(0, 2**32)
        b = rng.randrange(0, 2**32)
        lo, hi = mks._mul_lo_hi(a, b)
        prod = a * b
        assert lo == (prod & 0xFFFFFFFF) and hi == (prod >> 32) & 0xFFFFFFFF, (
            f"_mul_lo_hi({a:#x},{b:#x}) = ({lo:#x},{hi:#x}) != "
            f"({prod & 0xFFFFFFFF:#x},{(prod >> 32) & 0xFFFFFFFF:#x})"
        )


@needs_gpu
def test_philox_determinism_and_stream_separation(monkeypatch):
    """Philox par (request_id_hash, step) : même clé ⇒ même stream (identique
    sur les deux rangs, par construction) ; pas de collision entre steps ni
    entre hashes ; uniforms dans [0,1)."""
    dev = "cuda"
    h = torch.tensor([123456789, 987654321], device=dev, dtype=torch.int64)
    u1 = philox_uniforms(h, step=3, n=8, device=dev)
    u2 = philox_uniforms(h, step=3, n=8, device=dev)
    assert torch.equal(u1, u2), "stream non déterministe pour (hash, step) fixé"
    u_step = philox_uniforms(h, step=4, n=8, device=dev)
    assert not torch.equal(u1, u_step), "le step n'avance pas le stream"
    u_hash = philox_uniforms(h + 1, step=3, n=8, device=dev)
    assert not torch.equal(u1, u_hash), "les hashes partagent le même stream"
    assert bool((u1 >= 0).all() and (u1 < 1).all()), (
        f"uniforms hors [0,1) : min {float(u1.min())} max {float(u1.max())}"
    )
    # uniformité : moyenne sur 100k draws à 5σ (σ ≈ 1/(2·sqrt(3N)))
    big = philox_uniforms(
        torch.arange(100_000, device=dev, dtype=torch.int64),
        step=0,
        n=1,
        device=dev,
    )
    mean = float(big.mean())
    assert abs(mean - 0.5) < 5 * 0.2887 / (100_000**0.5), (
        f"moyenne {mean:.6f} hors 5σ de 0.5 sur 100k draws"
    )
    # fail loud : hash négatif / step hors contrat
    with pytest.raises(AssertionError, match="0"):
        philox_uniforms(torch.tensor([-1], device=dev, dtype=torch.int64), 0, 1, dev)
    with pytest.raises(AssertionError, match="step"):
        philox_uniforms(torch.tensor([1], device=dev, dtype=torch.int64), 2**32, 1, dev)


# ===========================================================================
# replis comptés + loggés (one-shot) : chaque motif a son canari
# ===========================================================================
def _mk_small_logits(M=2, V=64, seed=5):
    torch.manual_seed(seed)
    return torch.randn(M, V, device="cuda")


def _run_with_sentinel(logits, params, k_prime=8, **kw):
    """shard_sample avec un delivered_fn sentinelle : le repli doit être
    l'identité d'objet du chemin livré."""
    sentinel = types.SimpleNamespace(kind="delivered", tokens="SENTINEL")
    res = shard_sample(
        logits,
        0,
        logits.shape[1],
        params,
        ShardProcessorState(),
        k_prime=k_prime,
        delivered_fn=lambda: sentinel,
        **kw,
    )
    return res, sentinel


@needs_gpu
def test_fallback_gate_off_identity(monkeypatch, logs):
    """Gate OFF = repli verbatim : identité D'OBJET du delivered_fn, compteur
    gate_off, ligne de log one-shot."""
    _gate_off(monkeypatch)
    res, sentinel = _run_with_sentinel(_mk_small_logits(), ShardRequestParams())
    assert res is sentinel, "gate OFF : repli non verbatim (pas l'objet livré)"
    assert mks.FALLBACK_COUNTERS.get("gate_off") == 1, (
        f"compteur gate_off : {mks.FALLBACK_COUNTERS}"
    )
    assert any("motif gate_off" in s for s in logs), (
        f"log de repli gate_off absent : {logs}"
    )


@needs_gpu
def test_fallback_num_logprobs_all(monkeypatch, logs):
    """num_logprobs == -1 (logprobs pleine-vocab) → gather complet."""
    _gate_on(monkeypatch)
    res, sentinel = _run_with_sentinel(
        _mk_small_logits(),
        ShardRequestParams(num_logprobs=-1),
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("num_logprobs_all") == 1
    assert any("motif num_logprobs_all" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_fallback_logprobs_topn_uncovered(monkeypatch, logs):
    """num_logprobs > K' : le merged set ne couvre pas le top-n global."""
    _gate_on(monkeypatch)
    res, sentinel = _run_with_sentinel(
        _mk_small_logits(),
        ShardRequestParams(num_logprobs=16),  # > k_prime=8 du helper
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("logprobs_topn_uncovered") == 1
    assert any("motif logprobs_topn_uncovered" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_fallback_logprob_token_ids_oob(monkeypatch, logs):
    """logprob_token_ids exigé hors shard (mais DANS le vocab global) →
    gather complet (spec §3.5-5).  Le shard doit être STRICTEMENT plus petit
    que le vocab global — sinon « hors shard » = id invalide (autre garde)."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits()
    V = logits.shape[1]  # shard local
    V_global = 2 * V  # l'id exigé (V+5) est valide globalement, hors shard
    state = ShardProcessorState()
    sentinel = types.SimpleNamespace(kind="delivered", tokens="SENTINEL")
    res = shard_sample(
        logits,
        0,
        V_global,
        ShardRequestParams(required_logprob_token_ids=(V + 5,)),
        state,
        k_prime=8,
        delivered_fn=lambda: sentinel,
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("logprob_token_ids_oob") == 1
    assert any("motif logprob_token_ids_oob" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_required_oob_strict_fail_loud(monkeypatch):
    """Mode strict : le contrat exige un id hors shard ⇒ fail loud
    (ShardContractError), jamais un repli muet."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits()
    V = logits.shape[1]
    with pytest.raises(ShardContractError, match="hors du shard"):
        apply_shard_aware_processors(
            logits,
            0,
            V * 2,
            ShardRequestParams(required_logprob_token_ids=(V + 5,)),
            ShardProcessorState(),
            strict=True,
        )


@needs_gpu
def test_fallback_processor_exotic(monkeypatch, logs):
    """Processeurs non couverts (MinP, spec-decode, structured-output…) →
    repli compté+loggé."""
    _gate_on(monkeypatch)
    res, sentinel = _run_with_sentinel(
        _mk_small_logits(),
        ShardRequestParams(exotic=("MinPLogitsProcessor",)),
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("processor_exotic") == 1
    assert any("motif processor_exotic" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_fallback_topk_uncovered(monkeypatch, logs):
    """top_k > K' : le merged set ne couvre pas le top-k global."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits()
    M = logits.shape[0]
    res, sentinel = _run_with_sentinel(
        logits,
        ShardRequestParams(),
        request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
        step=1,
        temperature=torch.ones(M, device="cuda"),
        top_k=torch.full((M,), 16, device="cuda", dtype=torch.long),  # > 8
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("top_k_uncovered") == 1
    assert any("motif top_k_uncovered" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_fallback_topp_uncovered(monkeypatch, logs):
    """Noyau top-p qui s'étend au-delà du merged set : distribution PLATE sur
    16 tokens, top_p=0.5, K'=2 → masse du top-2 global = 2/16 = 0,125 < 0,5
    ⇒ repli compté+loggé."""
    _gate_on(monkeypatch)
    M = 2
    logits = torch.zeros(M, 16, device="cuda")  # plate : loi uniforme
    res, sentinel = _run_with_sentinel(
        logits,
        ShardRequestParams(),
        k_prime=2,
        request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
        step=1,
        temperature=torch.ones(M, device="cuda"),
        top_p=torch.full((M,), 0.5, device="cuda"),
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("top_p_uncovered") == 1
    assert any("motif top_p_uncovered" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_fallback_no_truncation(monkeypatch, logs):
    """Ni top-k ni top-p : le support de la loi livrée est le vocab entier —
    irréductible à 2K' candidats ⇒ repli nommé."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits()
    M = logits.shape[0]
    res, sentinel = _run_with_sentinel(
        logits,
        ShardRequestParams(),
        request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
        step=1,
        temperature=torch.ones(M, device="cuda"),
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("no_truncation") == 1
    assert any("motif no_truncation" in s for s in logs), f"logs={logs}"


def test_fallback_gather_unavailable_counted(logs):
    """Sans torch.distributed : le mini gather replie en mono-rang, COMPÉTÉ
    + loggé (one-shot) — le repli est visible, pas silencieux."""
    import torch.distributed as dist

    assert not (dist.is_available() and dist.is_initialized()), (
        "précondition : ce test exige un process group non initialisé"
    )
    v = torch.randn(2, 4)
    ids = torch.arange(4).unsqueeze(0).expand(2, 4).contiguous()
    lse = torch.zeros(2, 1)
    mv, mi, ml = default_gather_fn(v, ids, lse)
    assert torch.equal(mv, v) and torch.equal(mi, ids) and torch.equal(ml, lse), (
        "repli mono-rang : les tenseurs doivent passer inchangés"
    )
    assert mks.FALLBACK_COUNTERS.get("gather_unavailable") == 1, (
        f"compteur gather_unavailable : {mks.FALLBACK_COUNTERS}"
    )
    assert any("motif gather_unavailable" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_gather_world_size_1(tmp_path):
    """torch.distributed initialisé en world_size=1 : retour identique, SANS
    compter de repli (c'est un cas légitime, pas un repli)."""
    import torch.distributed as dist

    init_file = str(tmp_path / "pg")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        v = torch.randn(2, 4, device="cuda")
        ids = torch.arange(4, device="cuda").unsqueeze(0).expand(2, 4).contiguous()
        lse = torch.zeros(2, 1, device="cuda")
        before = dict(mks.FALLBACK_COUNTERS)
        mv, mi, ml = default_gather_fn(v, ids, lse)
        assert torch.equal(mv, v) and torch.equal(mi, ids) and torch.equal(ml, lse)
        assert dict(mks.FALLBACK_COUNTERS) == before, (
            f"world_size=1 ne doit pas compter de repli : {mks.FALLBACK_COUNTERS}"
        )
    finally:
        dist.destroy_process_group()


@needs_gpu
def test_one_shot_logging(monkeypatch, logs):
    """Le log de repli est ONE-SHOT : deux replis identiques → UNE ligne,
    compteur à 2.  (Doctrine : la ligne de log doit être exécutée — ici elle
    l'est par le test ET son one-shot est vérifié.)"""
    _gate_on(monkeypatch)
    for _ in range(2):
        _run_with_sentinel(
            _mk_small_logits(),
            ShardRequestParams(num_logprobs=-1),
        )
    n = sum("motif num_logprobs_all" in s for s in logs)
    assert n == 1, f"log one-shot : {n} lignes (attendu 1) : {logs}"
    assert mks.FALLBACK_COUNTERS.get("num_logprobs_all") == 2, (
        f"compteur : {mks.FALLBACK_COUNTERS} (attendu 2)"
    )


@needs_gpu
def test_gate_off_calls_delivered_but_never_shard_path(monkeypatch):
    """Gate OFF : delivered_fn rappelé verbatim, AUCUN compteur de chemin
    shard (state.counters vide de shard_path_calls)."""
    _gate_off(monkeypatch)
    state = ShardProcessorState()
    sentinel = object()
    res = shard_sample(
        _mk_small_logits(),
        0,
        64,
        ShardRequestParams(),
        state,
        delivered_fn=lambda: sentinel,
    )
    assert res is sentinel
    assert "shard_path_calls" not in state.counters, (
        f"chemin shard exécuté en gate OFF : {state.counters}"
    )


# ===========================================================================
# 2 rangs — même greedy et même tirage (Philox synchronisé)
# ===========================================================================
@needs_gpu
def test_two_rank_same_token_greedy_and_draws_fake_comm(monkeypatch):
    """fake_comm in-process : les deux « rangs » produisent le MÊME token
    greedy (vs référence livrée) et le MÊME tirage aléatoire (Philox par
    (request_id_hash, step) avancé identiquement) — même distribution,
    mapeage RNG différent assumé (écart NOMMÉ)."""
    _gate_on(monkeypatch)
    torch.manual_seed(77)
    M, V_global = 8, 256
    full = torch.randn(M, V_global, device="cuda")
    half = V_global // 2
    temp = torch.ones(M, device="cuda")
    top_k = torch.full((M,), 8, device="cuda", dtype=torch.long)
    top_p = torch.full((M,), 0.95, device="cuda")
    hashes = torch.arange(M, device="cuda", dtype=torch.int64) * 7919

    # greedy (temp < eps) d'abord : référence livrée
    ref_full = reference_processed_full_logits(full, ShardRequestParams())
    ref_greedy = torch.tensor(
        [reference_greedy_token(ref_full[i : i + 1]) for i in range(M)],
        device="cuda",
    )
    results = {}
    for rank in (0, 1):
        g0, g1, _, _ = two_rank_setup(
            full, min(mks.DEFAULT_K_PRIME, half), torch.ones(M, device="cuda")
        )
        res_g = shard_sample(
            full[:, rank * half : (rank + 1) * half].clone(),
            rank * half,
            V_global,
            ShardRequestParams(),
            ShardProcessorState(),
            gather_fn=(g0 if rank == 0 else g1),
            temperature=torch.zeros(M, device="cuda"),  # < eps ⇒ greedy
        )
        res_r = shard_sample(
            full[:, rank * half : (rank + 1) * half].clone(),
            rank * half,
            V_global,
            ShardRequestParams(),
            ShardProcessorState(),
            gather_fn=(g0 if rank == 0 else g1),
            request_id_hashes=hashes,
            step=1234,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
        )
        assert res_g.used_shard_path and res_r.used_shard_path, (
            f"rang {rank} : repli inattendu "
            f"{res_g.fallback_reason}/{res_r.fallback_reason}"
        )
        results[rank] = (res_g.tokens, res_r.tokens)

    # 1. greedy identique des deux rangs ET = au chemin livré
    assert torch.equal(results[0][0], results[1][0]), (
        f"greedy diverge entre rangs : {results[0][0]} vs {results[1][0]}"
    )
    assert torch.equal(results[0][0], ref_greedy), (
        f"greedy shard != livré : {results[0][0]} vs {ref_greedy}"
    )
    # 2. MÊME tirage des deux côtés (Philox synchronisé par construction)
    assert torch.equal(results[0][1], results[1][1]), (
        f"tirages divergents entre rangs : {results[0][1]} vs {results[1][1]}"
    )
    # 3. garde anti-test-vide : le tirage n'est pas degénéré (les rangs
    # aléatoires ne retombent pas tous sur le greedy)
    n_same = int((results[0][1] == results[0][0]).sum())
    assert n_same < M, f"tirage dégénéré : {n_same}/{M} tirages = tokens greedy"


# --- worker 2 processus (pattern TestIncrement3TwoRank : spawn + fichiers)
def _mk_sampling_two_rank_worker(rank: int, world: int, port: int, outdir: str):
    """Module-level pour spawn (picklable) ; résultats en FICHIERS (pattern
    ``_mk_two_rank_worker`` : pas de mp.Queue).  Les deux rangs partagent les
    MÊMES poids/logits (même seed) mais ne voient que LEUR shard ; le mini
    gather passe par ``default_gather_fn`` (torch.distributed)."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    n_dev = torch.cuda.device_count()
    dev_id = rank % max(n_dev, 1)
    if n_dev > 0:
        torch.cuda.set_device(dev_id)
        dev = f"cuda:{dev_id}"
    else:
        dev = "cpu"
    os.environ["VLLM_MOE_MEGAKERNEL_SAMPLING"] = "1"

    M, V_global = 8, 256
    half = V_global // 2
    torch.manual_seed(4242)  # IDENTIQUE des deux côtés
    full = torch.randn(M, V_global, device=dev)
    shard = full[:, rank * half : (rank + 1) * half].clone()
    hashes = torch.arange(M, device=dev, dtype=torch.int64) * 104729
    temp = torch.ones(M, device=dev)
    top_k = torch.full((M,), 8, device=dev, dtype=torch.long)

    res_g = shard_sample(
        shard,
        rank * half,
        V_global,
        ShardRequestParams(),
        ShardProcessorState(),
        temperature=torch.zeros(M, device=dev),  # greedy
    )
    res_r = shard_sample(
        shard,
        rank * half,
        V_global,
        ShardRequestParams(),
        ShardProcessorState(),
        request_id_hashes=hashes,
        step=99,
        temperature=temp,
        top_k=top_k,
        top_p=torch.full((M,), 0.95, device=dev),
    )
    torch.save(
        {
            "rank": rank,
            "greedy": res_g.tokens.cpu(),
            "drawn": res_r.tokens.cpu(),
            "full_row0": full[0].cpu(),  # témoin : mêmes logits des 2 côtés
        },
        f"{outdir}/rank{rank}.pt",
    )
    dist.destroy_process_group()


@needs_gpu
def test_two_rank_processes_same_token_and_draw(tmp_path):
    """2 processus (pattern ``TestIncrement3TwoRank``) + mini gather
    torch.distributed : les deux rangs produisent le MÊME token greedy et le
    MÊME tirage.  Même sans XGMI (gloo + tenseurs CPU pour le petit
    gather) — pas de spin IPC ici, donc pas d'opt-in requis."""
    import torch.multiprocessing as mp

    outdir = str(tmp_path)
    # port libre dynamique (pas de collision sous runs parallèles — revue GLM
    # 2026-10-28 : un port codé en dur hang 300 s avant d'échouer)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_mk_sampling_two_rank_worker, args=(r, 2, port, outdir))
        for r in (0, 1)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.kill()
    assert not hung, "worker(s) en attente — kill après 300s"
    assert all(p.exitcode == 0 for p in procs), (
        f"worker exit codes: {[p.exitcode for p in procs]}"
    )
    got = {r: torch.load(f"{outdir}/rank{r}.pt", weights_only=False) for r in (0, 1)}
    assert torch.equal(got[0]["full_row0"], got[1]["full_row0"]), (
        "les rangs n'ont pas les mêmes logits : le test ne prouve rien"
    )
    assert torch.equal(got[0]["greedy"], got[1]["greedy"]), (
        f"greedy diverge : {got[0]['greedy']} vs {got[1]['greedy']}"
    )
    assert torch.equal(got[0]["drawn"], got[1]["drawn"]), (
        f"tirages divergents : {got[0]['drawn']} vs {got[1]['drawn']}"
    )


@needs_gpu
def test_active_log_one_shot(monkeypatch, logs):
    """La ligne one-shot ACTIVE du chemin shard est un CANARI asserté :
    deux échantillonnages réussis → UNE seule ligne (et le compteur
    d'appels avance bien)."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits()
    M = logits.shape[0]
    for step in (0, 1):
        res = shard_sample(
            logits,
            0,
            logits.shape[1],
            ShardRequestParams(),
            ShardProcessorState(),
            k_prime=8,
            request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
            step=step,
            temperature=torch.ones(M, device="cuda"),
            top_k=torch.full((M,), 4, device="cuda", dtype=torch.long),
        )
        assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    n = sum("SAMPLING shard-aware ACTIF" in s for s in logs)
    assert n == 1, f"ligne one-shot ACTIVE : {n} occurrences (attendu 1)"


# ===========================================================================
# fail loud (entrées hors contrat)
# ===========================================================================
@needs_gpu
def test_fail_loud_inputs(monkeypatch):
    """Entrées hors contrat : asserts nommés, aucune sortie silencieuse."""
    _gate_on(monkeypatch)
    logits = _mk_small_logits(M=2, V=64)
    st = ShardProcessorState()

    # shard hors du vocab global
    with pytest.raises(AssertionError, match="hors"):
        shard_sample(logits, 32, 64, ShardRequestParams(), st)
    # nombre de requêtes != M
    with pytest.raises(AssertionError, match="requêtes"):
        shard_sample(logits, 0, 64, [ShardRequestParams()], st)
    # allowed_token_ids vide = contradiction
    with pytest.raises(AssertionError, match="allowed_token_ids"):
        apply_shard_aware_processors(
            logits,
            0,
            64,
            ShardRequestParams(allowed_token_ids=frozenset()),
            st,
        )
    # allowed_token_ids hors du vocab global (cohérence fail-loud, revue GLM)
    with pytest.raises(AssertionError, match="allowed_token_ids"):
        apply_shard_aware_processors(
            logits,
            0,
            64,
            ShardRequestParams(allowed_token_ids=frozenset({5, 64})),
            st,
        )
    # gid hors du vocab global
    with pytest.raises(AssertionError, match="hors"):
        apply_shard_aware_processors(
            logits, 0, 64, ShardRequestParams(stop_token_ids=(64,)), st
        )
    # tirage sans request_id_hashes
    with pytest.raises(AssertionError, match="request_id_hashes"):
        shard_sample(
            logits,
            0,
            64,
            ShardRequestParams(),
            st,
            temperature=torch.ones(2, device="cuda"),
            top_k=torch.full((2,), 4, device="cuda", dtype=torch.long),
        )
    # gather_fn de forme incohérente
    with pytest.raises(AssertionError, match="gather_fn"):
        shard_sample(
            logits,
            0,
            64,
            ShardRequestParams(),
            st,
            gather_fn=lambda v, i, lz: (v[:, :2], i, lz),
        )
    # k_prime hors contrat
    with pytest.raises(AssertionError, match="k_prime"):
        shard_sample(logits, 0, 64, ShardRequestParams(), st, k_prime=0)
    # hashes dupliqués = tirages identiques pour toujours (contrat d'unicité,
    # revue GLM 2026-10-28, warning 3)
    with pytest.raises(AssertionError, match="unique"):
        shard_sample(
            logits,
            0,
            64,
            ShardRequestParams(),
            st,
            request_id_hashes=torch.zeros(2, device="cuda", dtype=torch.int64),
            temperature=torch.ones(2, device="cuda"),
            top_k=torch.full((2,), 4, device="cuda", dtype=torch.long),
        )
    # shape top_k != [M]
    with pytest.raises(AssertionError, match="top_k"):
        shard_sample(
            logits,
            0,
            64,
            ShardRequestParams(),
            st,
            request_id_hashes=torch.arange(2, device="cuda", dtype=torch.int64),
            temperature=torch.ones(2, device="cuda"),
            top_k=torch.tensor([4], device="cuda"),
        )


@needs_gpu
def test_shard_topk_global_ids(monkeypatch):
    """shard_topk renvoie des ids GLOBAUX (colonne + vocab_start_index) et se
    borne au shard ; la réunion des top-K' locaux couvre le top-K' global."""
    _gate_on(monkeypatch)
    torch.manual_seed(9)
    M, V_global = 2, 64
    full = torch.randn(M, V_global, device="cuda")
    half = V_global // 2
    k_loc = 8
    v0, i0 = shard_topk(full[:, :half], k_loc, 0)
    v1, i1 = shard_topk(full[:, half:], k_loc, half)
    assert i0.min() >= 0 and i0.max() < half, f"ids rang0 hors shard : {i0}"
    assert i1.min() >= half and i1.max() < V_global, f"ids rang1 hors shard : {i1}"
    merged_vals = torch.cat([v0, v1], dim=1)
    merged_ids = torch.cat([i0, i1], dim=1)
    for i in range(M):
        ref_top = full[i].topk(k_loc)
        got_vals, got_ids = merged_vals[i], merged_ids[i]
        # le top-K' global est couvert : ses valeurs existent dans le merged
        for val, gid in zip(ref_top.values.tolist(), ref_top.indices.tolist()):
            hit = (got_ids == gid).nonzero()
            assert len(hit) > 0 and abs(float(got_vals[hit[0]]) - val) < 1e-6, (
                f"ligne {i} : id global {gid} (val {val}) absent du merged set"
            )


# ===========================================================================
# dense_mfma_gemm : sortie fp32 (la SEULE modification de fichier existant)
# ===========================================================================
@needs_gpu
def test_dense_mfma_gemm_fp32_out_and_fallback(monkeypatch):
    """Le launcher étendu accepte un tampon ``out`` fp32 : le gate-ON écrit le
    fp32 de l'acc (cast absorbé), le repli reproduit la chaîne livrée
    (bf16 puis cast = bit-exact).  Gate dense via monkeypatch (pas de fuite
    d'env entre tests) + compteur de lancements MFMA."""
    calls = _spy_mfma_kernel(monkeypatch)
    from vllm.model_executor.layers.fused_moe.moe_megakernel_dense import (
        dense_mfma_gemm,
    )

    M, N, K = 2, 64, 256
    torch.manual_seed(3)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randint(-127, 128, (K, N), device="cuda", dtype=torch.int8)
    b_scale = torch.rand(N, device="cuda") * 0.02 + 0.001

    # gate dense ON : sortie fp32 directe
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "1")
    buf = torch.full((M, N), -7.0, device="cuda", dtype=torch.float32)
    out32 = dense_mfma_gemm(a, b, b_scale, out=buf)
    ref16 = dense_mfma_gemm(a, b, b_scale)  # bf16
    torch.cuda.synchronize()
    assert calls["n"] == 2, f"lancements MFMA : {calls['n']} != 2"
    assert out32 is buf, "le tampon out fp32 n'est pas retourné tel quel"
    assert buf.dtype == torch.float32, f"dtype {buf.dtype}"
    d = (buf - ref16.float()).abs()
    ulp = ref16.float().abs() * 2.0**-8 + 1e-6
    assert bool((d <= ulp).all()), (
        f"fp32 vs bf16 hors ulp bf16 : max {float(d.max()):.6g} "
        f"(ulp {float(ulp.max()):.6g})"
    )
    # DISCRIMINATEUR (revue GLM 2026-10-28, warning 2) : sur les valeurs
    # génériques (seed 3), l'épilogue fp32 doit se distinguer de l'arrondi
    # bf16 — sinon le test resterait vert avec un épilogue régressé en
    # stockage bf16.  Ce n'est PAS un critère de bit-exactitude : c'est la
    # preuve que le « cast absorbé » est réellement exercé.
    assert float(ref16.abs().max()) > 0
    assert float(d.max()) > 0.0, (
        "sortie fp32 bit-identique à bf16 cast (seed 3) : l'épilogue "
        "direct n'est pas discriminé — cast régressé ou données ?"
    )
    print(f"écart fp32-épilogue vs arrondi bf16 (max) : {float(d.max()):.6g}")

    # gate dense OFF : repli = triton_scaled_mm bf16 puis cast (bit-exact)
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "0")
    buf2 = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    out32b = dense_mfma_gemm(a, b, b_scale, out=buf2)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm_fused_quant,
    )

    ref_bf16, _ = triton_scaled_mm_fused_quant(a, b, b_scale, torch.bfloat16)
    torch.cuda.synchronize()
    assert out32b is buf2
    assert torch.equal(buf2, ref_bf16.float()), (
        "repli fp32 != chaîne livrée (bf16 puis cast)"
    )
    # fail loud : dtype de sortie hors contrat (ni bf16 ni fp32)
    monkeypatch.setenv("VLLM_MOE_MEGAKERNEL_DENSE", "1")
    with pytest.raises(AssertionError, match="out"):
        dense_mfma_gemm(
            a,
            b,
            b_scale,
            out=torch.zeros(M, N, device="cuda", dtype=torch.float16),
        )


# ===========================================================================
# lm_head int8 pré-quantifié (mode A int8 du launcher)
# ===========================================================================
@needs_gpu
def test_lm_head_int8_prequant(monkeypatch):
    """X int8 pré-quantifié : a_scale per-token requis (fail loud sinon) et
    la sortie fp32 suit le même contrat.  Kernel MFMA épinglé (compteur)."""
    calls = _spy_mfma_kernel(monkeypatch)
    _gate_on(monkeypatch)
    M, K, V = 2, 256, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randint(-127, 128, (K, V), device="cuda", dtype=torch.int8)
    w_scale = torch.rand(V, device="cuda") * 0.02 + 0.001
    xq, a_scale = _quant_contract_torch(x)

    with pytest.raises(AssertionError, match="a_scale"):
        mk_lm_head_logits(xq, w, w_scale)
    out = mk_lm_head_logits(xq, w, w_scale, a_scale=a_scale)
    torch.cuda.synchronize()
    assert calls["n"] == 1, f"lancements MFMA : {calls['n']} != 1"
    assert out.dtype == torch.float32 and out.shape == (M, V)
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    ref = triton_scaled_mm(xq, w, a_scale, w_scale, torch.bfloat16)
    torch.cuda.synchronize()
    d = (out - ref.float()).abs()
    ulp = ref.float().abs() * 2.0**-8 + 1e-6
    assert bool((d <= ulp).all()), (
        f"int8 pré-quantifié : écart vs livré hors ulp bf16 : {float(d.max()):.6g}"
    )


# ===========================================================================
# revue GLM 2026-10-28 — couverture des findings contractuels
# ===========================================================================
@needs_gpu
def test_fallback_logprobs_mixed(monkeypatch, logs):
    """num_logprobs hétérogènes dans le batch = cas réel ORDINAIRE (batch
    mixte logprobs/non) : repli compté+loggé, PAS un crash (revue GLM
    2026-10-28, critique 2)."""
    _gate_on(monkeypatch)
    res, sentinel = _run_with_sentinel(
        _mk_small_logits(),
        [ShardRequestParams(num_logprobs=0), ShardRequestParams(num_logprobs=3)],
    )
    assert res is sentinel
    assert mks.FALLBACK_COUNTERS.get("logprobs_mixed") == 1, (
        f"compteur logprobs_mixed : {mks.FALLBACK_COUNTERS}"
    )
    assert any("motif logprobs_mixed" in s for s in logs), f"logs={logs}"


@needs_gpu
def test_logprob_oob_fallback_symmetric_across_ranks(monkeypatch):
    """SYMÉTRIE du repli `logprob_token_ids_oob` entre rangs (revue GLM
    2026-10-28, critique 1) : la décision est portée par le shard du rang de
    SORTIE (``output_vocab_start``) — les deux rangs tombent d'accord SANS
    collectif.  Un repli d'un seul rang désynchroniserait tokens ET collectifs
    (le rang en repli émettrait un all-gather que l'autre n'émet jamais).

    Observé (M=1, vocab global 256, shards [0,128)/[128,256)) :
    * id exigé 200 (shard du rang 1), rang de sortie = rang 0 → les DEUX
      rangs replient (compteur logprob_token_ids_oob à 1 de chaque côté) ;
    * id exigé 70 (shard du rang 0 = rang de sortie) → les DEUX rangs
      continuent, le rang 0 calcule la valeur, le rang 1 la laisse au rang
      de sortie (required_elsewhere) ;
    * id exigé 200 avec rang de sortie = rang 1 → les DEUX rangs continuent
      (valeur calculée par le rang 1, laissée par le rang 0).
    """
    _gate_on(monkeypatch)
    M, V_global = 1, 256
    half = V_global // 2
    torch.manual_seed(4)
    full = torch.randn(M, V_global, device="cuda")
    temp = torch.ones(M, device="cuda")
    k_loc = 8

    def run(rank: int, params: ShardRequestParams, out_start: int):
        g0, g1, _, _ = two_rank_setup(full, k_loc, temp, params=params)
        # le helper de setup rejoue les processeurs (et compte dans les
        # compteurs globaux) : on vide pour ne mesurer que l'appel réel
        mks.FALLBACK_COUNTERS.clear()
        sh = full[:, rank * half : (rank + 1) * half].clone()
        state = ShardProcessorState()
        sentinel = types.SimpleNamespace(kind="delivered")
        res = shard_sample(
            sh,
            rank * half,
            V_global,
            params,
            state,
            gather_fn=(g0 if rank == 0 else g1),
            delivered_fn=lambda: sentinel,
            output_vocab_start=out_start,
        )
        return res, sentinel, state

    # (a) id exigé hors shard du rang de sortie → les DEUX rangs replient
    for out_start in (0, half):
        for rank in (0, 1):
            res, sentinel, state = run(
                rank,
                ShardRequestParams(required_logprob_token_ids=(200,)),
                out_start,
            )
            if out_start == 0:
                assert res is sentinel, (
                    f"rang {rank} n'a pas replié alors que 200 est hors du "
                    f"shard de sortie [0,{half})"
                )
                assert state.counters.get("logprob_token_ids_oob") == 1, (
                    f"rang {rank} : compteurs {state.counters}"
                )
            else:
                assert res.used_shard_path, (
                    f"rang {rank} a replié à tort : 200 est couvert par le "
                    f"shard de sortie [{half},{V_global})"
                )
                assert mks.FALLBACK_COUNTERS == {}, (
                    f"aucun repli ne doit être compté quand le shard de "
                    f"sortie couvre l'id : {mks.FALLBACK_COUNTERS}"
                )

    # (b) id exigé dans le shard du rang de sortie : les deux rangs
    # continuent ; le rang de sortie calcule la valeur, l'autre la lui laisse
    for rank in (0, 1):
        res, sentinel, state = run(
            rank,
            ShardRequestParams(required_logprob_token_ids=(70,)),
            0,
        )
        assert res.used_shard_path, (
            f"rang {rank} : repli inattendu {res.fallback_reason}"
        )
        if rank == 0:
            assert res.logprobs is not None and res.logprobs.required.shape == (1,), (
                f"rang 0 (de sortie) doit calculer la valeur requise : {res.logprobs}"
            )
        else:
            assert res.logprobs is not None
            assert res.logprobs.required_ids == (), (
                f"rang 1 : la valeur doit être laissée au rang de sortie, "
                f"pas calculée ici : {res.logprobs.required_ids}"
            )
        assert mks.FALLBACK_COUNTERS == {}, (
            f"partie (b) : aucun repli attendu : {mks.FALLBACK_COUNTERS}"
        )


@needs_gpu
def test_mixed_greedy_random_batch(monkeypatch):
    """Batch MIXTE greedy/aléatoire : la rangée greedy (temp < eps) reste
    identique au livré SANS être écrasée par le tirage — le masque
    `is_random_row` (depuis la température D'ORIGINE, avant le remplacement
    < eps → 1.0) est le mécanisme ; une régression qui le recalculerait
    depuis la température remplacée ferait tirer la rangée greedy."""
    _gate_on(monkeypatch)
    torch.manual_seed(31)
    M, V = 3, 128
    full = torch.randn(M, V, device="cuda")
    half = V // 2
    temp = torch.tensor([0.0, 1.0, 0.8], device="cuda")  # ligne 0 = greedy
    hashes = torch.tensor([11, 22, 33], device="cuda", dtype=torch.int64)
    g0, g1, _, _ = two_rank_setup(full, 32, torch.ones(M, device="cuda"))
    ref_full = reference_processed_full_logits(full, ShardRequestParams())
    expected_greedy = reference_greedy_token(ref_full[0:1])

    res = shard_sample(
        full[:, :half].clone(),
        0,
        V,
        ShardRequestParams(),
        ShardProcessorState(),
        gather_fn=g0,
        request_id_hashes=hashes,
        step=5,
        temperature=temp,
        top_k=torch.full((M,), 4, device="cuda", dtype=torch.long),
        top_p=torch.full((M,), 0.95, device="cuda"),
    )
    assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    got = res.tokens.tolist()
    assert got[0] == expected_greedy, (
        f"rangée greedy écrasée par le tirage : {got[0]} != {expected_greedy}"
    )
    # la rangée greedy est INVARIANTE au step (elle ne dépend pas du tirage) ;
    # les rangées aléatoires doivent rester dans leurs survivants top-4
    res2 = shard_sample(
        full[:, :half].clone(),
        0,
        V,
        ShardRequestParams(),
        ShardProcessorState(),
        gather_fn=g0,
        request_id_hashes=hashes,
        step=999,  # autre stream pour les rangées aléatoires
        temperature=temp,
        top_k=torch.full((M,), 4, device="cuda", dtype=torch.long),
        top_p=torch.full((M,), 0.95, device="cuda"),
    )
    assert res2.tokens.tolist()[0] == expected_greedy, (
        f"rangée greedy dépend du step : {res2.tokens.tolist()[0]}"
    )
    for i in (1, 2):
        top4 = ref_full[i].topk(4)
        assert got[i] in top4.indices.tolist(), (
            f"rangée {i} : token {got[i]} hors des survivants {top4.indices.tolist()}"
        )


@needs_gpu
def test_statistical_distribution_two_shard(monkeypatch):
    """Distribution statistique en configuration SHARDED (TP=2, fake_comm) :
    2 shards de 8, K'=8 par shard (merged = 16 = vocab complet, garde de
    couverture top-p en jeu), milliers de tirages vs la MÊME loi analytique
    pleine-vocab.  Un bug de fusion (logZ décomposé, tri composite, masque
    top-p sur merged) produirait une loi fausse — testé ici, pas seulement
    en mono-rang (revue GLM 2026-10-28, warning 5)."""
    _gate_on(monkeypatch)
    N, V = 8192, 16
    half = V // 2
    base = torch.tensor(
        [
            3.0,
            2.2,
            1.7,
            1.1,
            0.6,
            0.2,
            -0.3,
            -0.9,
            -1.4,
            -2.0,
            -2.7,
            -3.5,
            -4.4,
            -5.4,
            -6.5,
            -7.7,
        ],
        device="cuda",
    )
    top_k, top_p = None, 0.9
    law = reference_law(base, temp=1.0, top_k=top_k, top_p=top_p)
    bins = int((law > 0).sum())
    full = base.unsqueeze(0).expand(N, V).contiguous()
    hashes = (torch.arange(N, device="cuda", dtype=torch.int64) * 2654435761) % (2**62)

    # payloads du « rang 1 » pré-calculés comme le ferait son propre top-K'
    peer_vals, peer_ids = shard_topk(full[:, half:], half, half)
    peer_lse = torch.logsumexp(full[:, half:], dim=1).unsqueeze(1)
    gather = FakeTwoRankGather(peer_vals, peer_ids, peer_lse)

    res = shard_sample(
        full[:, :half].contiguous(),
        0,
        V,
        ShardRequestParams(),
        ShardProcessorState(),
        k_prime=half,
        gather_fn=gather,
        request_id_hashes=hashes,
        step=43,
        temperature=torch.ones(N, device="cuda"),
        top_p=torch.full((N,), top_p, device="cuda"),
    )
    assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    obs = torch.bincount(res.tokens, minlength=V).double()
    exp = law.double() * N
    sigma = (N * law.double() * (1 - law.double())).clamp_min(1e-9).sqrt()
    chi2 = float(((obs - exp) ** 2 / exp.clamp_min(1e-9))[law > 0].sum())
    max_z = float(((obs - exp).abs() / sigma)[law > 0].max())
    counts = {i: int(obs[i]) for i in range(V) if float(law[i]) > 0}
    expected_p = {i: round(float(law[i]), 5) for i in range(V) if float(law[i]) > 0}
    print(
        f"\n[stat-2shards] N={N} bins={bins} chi2={chi2:.3f} "
        f"(crit ddl {bins - 1} : 34.0) max|z|={max_z:.3f} (seuil 5)\n"
        f"  obs={counts}\n  loi={expected_p}"
    )
    assert max_z <= 5.0, f"bin hors 5σ : max|z|={max_z:.3f} — obs={counts}"
    assert chi2 <= 34.0, (
        f"chi-carré {chi2:.3f} > 34.0 (ddl {bins - 1}, α=1e-6) — obs={counts}"
    )
    assert float(exp[law > 0].min()) >= 100.0


# --- revue mimo 2026-10-29 (C3 + S4 + S5) ---------------------------------
@needs_gpu
def test_logprob_no_nan_with_masked_candidates(monkeypatch):
    """C3 (revue mimo) : un candidat MASQUÉ à -inf dans le merged set ne doit
    pas contaminer le logprob choisi par 0 × -inf = NaN.  Atteint par
    allowed_token_ids (petite liste blanche → le merged set contient des
    -inf) + num_logprobs > 0.  Observé : le logprob choisi est FINI et égal
    à la référence log_softmax — l'ancienne réduction arithmétique
    `(vals * onehot).sum()` rendait NaN ici (le token restait correct, le
    logprob silencieusement pourri)."""
    _gate_on(monkeypatch)
    torch.manual_seed(57)
    M, V_global = 2, 128
    full = torch.randn(M, V_global, device="cuda")
    # liste blanche resserrée : ~6 ids vivants sur 128 → le top-K' local
    # ramène des -inf dans le merged set
    params = ShardRequestParams(
        allowed_token_ids=frozenset({3, 17, 42, 66, 90, 120}),
        num_logprobs=2,
    )
    temp = torch.tensor([1.0, 0.8], device="cuda")
    ref_full = reference_processed_full_logits(full, params)
    ref_lp = (ref_full / temp[:, None]).log_softmax(dim=-1)

    half = V_global // 2
    g0, g1, _, _ = two_rank_setup(full, 64, temp, params=params)
    res = shard_sample(
        full[:, :half].clone(),
        0,
        V_global,
        params,
        ShardProcessorState(),
        gather_fn=g0,
        request_id_hashes=torch.arange(M, device="cuda", dtype=torch.int64),
        step=3,
        temperature=temp,
        top_k=torch.full((M,), 3, device="cuda", dtype=torch.long),
    )
    assert res.used_shard_path, f"repli inattendu : {res.fallback_reason}"
    lp = res.logprobs
    assert lp is not None
    assert bool(torch.isfinite(lp.chosen).all()), (
        f"NaN/inf dans les logprobs choisis : {lp.chosen} — le merged set "
        f"contient des -inf masqués (bug C3 : 0 × -inf)"
    )
    rows = torch.arange(M, device="cuda")
    ref_chosen = ref_lp[rows, res.tokens]
    d = (lp.chosen - ref_chosen).abs().max()
    assert float(d) < 1e-4, f"logprob choisi : max diff {float(d):.3e}"
    assert bool(torch.isfinite(lp.top_logprobs).all()), (
        f"NaN dans le top-n : {lp.top_logprobs}"
    )
    # garde : le merged set contient BIEN des -inf (sinon le test ne prouve
    # rien — l'ancien code n'aurait pas pu y trouver de NaN).  Le shard doit
    # être PROCÉSSÉ avant le top-K' (c'est la liste blanche qui masque).
    sh_chk = full[:, :half].clone()
    apply_shard_aware_processors(sh_chk, 0, V_global, params, ShardProcessorState())
    v0, _ = shard_topk(sh_chk, 64, 0)
    assert bool((v0 == -float("inf")).any()), (
        f"précondition : le top-64 local doit contenir des -inf (min={float(v0.min())})"
    )


@needs_gpu
def test_bias_out_of_shard_counted(monkeypatch):
    """S4 (revue mimo) : les biais hors shard sont laissés à l'autre rang —
    COMPTÉS (observabilité), pas silencieusement avalés.  Observé : rank0
    (shard [0,32)) avec biais sur 40 → skipped_bias_out_of_shard == 1."""
    _gate_on(monkeypatch)
    logits = torch.zeros(1, 32, device="cuda")
    out = apply_shard_aware_processors(
        logits,
        0,
        64,
        ShardRequestParams(bias_token_ids=((40, 0.5), (10, 1.0))),
        ShardProcessorState(),
    )
    assert out.skipped_bias_out_of_shard == 1, (
        f"biais hors shard non compté : {out.skipped_bias_out_of_shard}"
    )
    assert float(logits[0, 10]) == 1.0, f"biais in-shard non appliqué : {logits}"
    assert float(logits[0, 5]) == 0.0, "écriture parasite hors cible"


@needs_gpu
def test_gid_accepts_numpy_and_torch_ints(monkeypatch):
    """S5 (revue mimo) : les ids numpy/torch sont des entiers « réels » —
    `operator.index` les accepte ; floats/strings sont REFUSÉS (pas de
    coercion muette)."""
    _gate_on(monkeypatch)
    import numpy as np

    logits = torch.zeros(1, 32, device="cuda")
    out = apply_shard_aware_processors(
        logits,
        0,
        64,
        ShardRequestParams(stop_token_ids=(np.int64(5),)),
        ShardProcessorState(),
    )
    assert out.masked == 1 and float(logits[0, 5]) == -float("inf"), (
        f"id numpy ignoré : masked={out.masked}"
    )
    with pytest.raises(AssertionError, match="entier"):
        apply_shard_aware_processors(
            logits,
            0,
            64,
            ShardRequestParams(stop_token_ids=(5.0,)),
            ShardProcessorState(),
        )
    with pytest.raises(AssertionError, match="entier"):
        apply_shard_aware_processors(
            logits,
            0,
            64,
            ShardRequestParams(stop_token_ids=("5",)),
            ShardProcessorState(),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
