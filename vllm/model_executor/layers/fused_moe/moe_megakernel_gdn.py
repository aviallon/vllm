# SPDX-License-Identifier: Apache-2.0
"""INC-B — chaîne GDN decode FUSIONNÉE (conv + scan) en un seul lancement.

Campagne MEGAKERNEL full-forward 2026-10-27/28 (MI210 gfx90a, Triton 3.8 HIP) ;
mandat INC-B « core attention + GDN in-kernel » :
`research/moe-megakernel-campaign-2026-10-27/TOUR3-FULL-FORWARD-DESIGN.md`
§3.5-3.6 / §5.2.  Ce module fusionne la chaîne GDN **decode** livrée :

``causal_conv1d_update`` (conv causale depthwise w=4, silu, mise à jour de
l'état conv) → ``fused_recurrent_gated_delta_rule_packed_decode`` (l2norm
q/k in-kernel, règle delta récurrente, état fp32) —

en **UN lancement** (`qwen3_gdn_decode_fusion_kernel`), un programme CTA par
(bloc de V × token × tête-v).  Au decode la masse GDN est dans les
projections (INC-A) et les lancements : le scan est déjà à son plancher
bande (2,5 µs/couche, gdn-profile 2026-10-17 — le « scan 79 % » du mandat
est un fait PREFILL, §0.5 du design).  Le gain propre de cette fusion =
1 lancement/couche (L) + l'aller-retour HBM de ``mixed_qkv`` (O) +
l'absorption de la passe de stockage d'état (K) — plafond honnête §5.2.

Contrat numérique = composition des deux kernels livrés, EXACTEMENT :

* conv : ``y[c] = silu(bias[c] + Σ_{j<W} w[c,j] · u_j)`` avec
  ``u = [état[0], état[1], état[2], x_t]`` (fenêtre glissante, x_t en
  dernier — contrat `_causal_conv1d_update_kernel` STEP 4-5) ; sortie
  arrondie en **bf16** avant le scan (le livré stocke la conv en bf16) ;
  l'état conv conserve le x **BRUT** (pré-silu) : ``[s1, s2, x_t]``
  (contrat `new_conv_state = where(mask, conv_state, loaded_x)`) ;
* scan : identique à `fused_recurrent_gated_delta_rule_packed_decode`
  (l2norm ``q/√(Σq²+1e-6)`` puis × scale, ``g = −exp(A_log)·softplus(a +
  dt_bias)`` (seuil 20), ``β = σ(b)``, règle delta, sortie bf16, état fp32
  lu/écrit en place) ; bloc de V BV=32 comme au livré (le scan a besoin
  des vecteurs q/k TÊTE ENTIÈRE ; nw=2 = mesure du livré).

⚠ **Écart de structure NOMMÉ : état conv DOUBLE-TAMPONNÉ (ping-pong).**
La fusion met dans le MÊME lancement des programmes qui L'ANCIEN état conv
(les GROUP×NV consommateurs d'une tête q/k) et un programme qui ÉCRIT le
nouvel état (unique écrivain par canal : le propriétaire).  Sans
séparation, un lecteur peut voir la fenêtre déjà tournée (``[s1,s2,x_t]``)
au lieu de ``[s0,s1,s2]`` — course entre CTAs, aucune barrière de grille
disponible en Triton.  Le double-tampon est le modèle du tour 2 (« pas
d'état entre appels sans double-tampon ») : le lancement lit le slot
``parity`` et écrit le slot ``1-parity`` ; la parité par pas est un
argument (INC-D la dérive in-kernel).  Coût : 3×C×2 o de plus par séq —
négligeable.  **Mutation-testé** : forcer le single-buffer fait diverger
le contrat multi-pas (test `test_mutation_single_buffer_diverges`).

Gate ``VLLM_MOE_MEGAKERNEL_GDN`` (défaut OFF) — **AUCUN câblage moteur**
(celui-ci est INC-D).  Gate OFF = repli = chaîne livrée VERBATIM
(`causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`).
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
    "qwen3_gdn_conv_row",
    "qwen3_gdn_store_state",
    "qwen3_gdn_decode_fusion_kernel",
    "gdn_decode_fusion",
    "gdn_decode_fallback",
    "gdn_fusion_enabled",
]

# BV = bloc de V (unité de l'état scan), comme le livré :
# BV=min(next_pow2(V), 32) — le commentaire occupancy du kernel livré
# (gfx90a, 512 VGPR/lane) mesure nw=2 gagnant partout.  BK = K (128,
# une seule itération — le livré n'accepte qu'NK=1).
GDN_BV = 32
GDN_NUM_WARPS = 2
GDN_NUM_STAGES = 2
GDN_CONV_WIDTH = 4
GDN_CONV_STATE_LEN = GDN_CONV_WIDTH - 1  # 3 emplacements d'historique


@triton.jit
def qwen3_gdn_conv_row(
    chan_off,
    x_base,
    cs_base,
    convw_ptr,
    convb_ptr,
    stride_cs_dim,
    stride_cs_tok,
    stride_x_c,
    LEN: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
):
    """Conv causale depthwise sur les canaux ``chan_off`` — fenêtre glissante.

    Retourne ``(y_bf16, x_t)`` : la fenêtre ``u`` [LEN, W] est interne —
    ``u[:, j<W-1]`` = état ancien (chargé masqué), ``u[:, W-1]`` = x_t BRUT
    (pré-silu) ; ``y`` = silu(acc) arrondi en bf16 (le livré stocke la
    sortie conv en bf16 avant le scan).  Expressions VERBATIM
    `_causal_conv1d_update_kernel` STEP 3-5 : ``acc = bias + Σ u_j·w_j``
    en fp32, ``silu = acc/(1+exp(-acc))``.
    """
    tok = tl.arange(0, W)  # W puissance de 2 (4)
    # état ancien [LEN, W] : slots 0..W-2 valides, slot W-1 rempli après
    u_hist = tl.load(
        cs_base + chan_off[:, None] * stride_cs_dim + tok[None, :] * stride_cs_tok,
        mask=(chan_off[:, None] < C) & (tok[None, :] < W - 1),
        other=0.0,
    ).to(tl.float32)
    x_t = tl.load(x_base + chan_off * stride_x_c, mask=chan_off < C, other=0.0).to(
        tl.float32
    )
    u = tl.where(tok[None, :] == W - 1, x_t[:, None], u_hist)
    # poids [chan, W] (contigu W — contrat weight.stride(1)==1 du livré)
    w = tl.load(
        convw_ptr + chan_off[:, None] * W + tok[None, :],
        mask=chan_off[:, None] < C,
        other=0.0,
    ).to(tl.float32)
    acc = tl.load(convb_ptr + chan_off, mask=chan_off < C, other=0.0).to(tl.float32)
    acc = acc + tl.sum(u * w, axis=1)
    y = acc / (1.0 + tl.exp(-acc))
    return y.to(tl.bfloat16), x_t


@triton.jit
def qwen3_gdn_store_state(
    chan_off,
    x_t,
    y,
    cs_old,
    cs_tgt,
    stride_cs_dim,
    stride_cs_tok,
    LEN: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    MUT_STORE_SILU_STATE: tl.constexpr,
):
    """Écrit le nouvel état conv : ``[ancien[1], ancien[2], x_t]`` (x BRUT).

    Contrat livré : ``new_conv_state = where(mask, old[1:], x)`` — l'état
    conserve le x PRÉ-SILU.  Le shift se fait par re-lecture de l'ancien
    slot décalé (+1) — les deux buffers sont distincts (double-tampon),
    donc cette lecture ne peut pas voir l'écriture (voir docstring module).
    Hook de mutation ``MUT_STORE_SILU_STATE`` : stocke ``y`` (silué) à la
    place de x_t — le contrat multi-pas DOIT diverger (test dédié).
    """
    slots = tl.arange(0, W)  # cibles 0..W-2 (W puissance de 2)
    win2 = tl.where(MUT_STORE_SILU_STATE, y.to(tl.float32), x_t)
    # slots j < W-2  <- ancien[j+1] ; slot W-2 <- x_t (ou y en mutation)
    vals = tl.load(
        cs_old
        + chan_off[:, None] * stride_cs_dim
        + (slots[None, :] + 1) * stride_cs_tok,
        mask=(chan_off[:, None] < C) & (slots[None, :] < W - 2),
        other=0.0,
    ).to(tl.float32)
    vals = tl.where(slots[None, :] == W - 2, win2[:, None], vals)
    tl.store(
        cs_tgt + chan_off[:, None] * stride_cs_dim + slots[None, :] * stride_cs_tok,
        vals.to(tl.bfloat16),
        mask=(chan_off[:, None] < C) & (slots[None, :] < W - 1),
    )


@triton.jit
def qwen3_gdn_decode_fusion_kernel(
    # tenseurs
    x_ptr,  # [B, C] bf16 — entrée conv BRUTE (sortie in_proj, pré-silu)
    b_ptr,  # [B, HV] bf16 (β logit)
    a_ptr,  # [B, HV] bf16 (δ logit)
    alog_ptr,  # [HV] fp32 — A_log
    dtb_ptr,  # [HV] fp32 — dt_bias
    convw_ptr,  # [C, W] bf16 — poids conv (stride (W,1))
    convb_ptr,  # [C] bf16 — biais conv
    cs_old_ptr,  # [num_cache, C, W-1] bf16 — état conv LU (slot parity)
    cs_new_ptr,  # [num_cache, C, W-1] bf16 — état conv ÉCRIT (slot 1-parity)
    st_ptr,  # [num_cache, HV, V, K] fp32 — état scan (lu/écrit en place)
    out_ptr,  # [B, HV, V] bf16
    idx_ptr,  # [B] int32 — state_indices (<=0 = ligne nulle)
    # scalaires (jamais de copie H2D par lancement)
    scale,  # head_k_dim ** -0.5
    num_valid,  # lignes actives (B logique)
    # strides
    stride_x_tok,
    stride_x_c,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_st_seq,
    stride_out_tok,
    # formes / constexpr
    C: tl.constexpr,  # conv_dim = 2*H*K + HV*V
    H: tl.constexpr,  # têtes q/k locales
    HV: tl.constexpr,  # têtes v locales
    K: tl.constexpr,  # head_k_dim (128)
    V: tl.constexpr,  # head_v_dim (128)
    W: tl.constexpr,  # largeur conv (4)
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr = 20.0,
    # hooks de mutation (tests « break the fix » — jamais True en prod) :
    MUT_SINGLE_BUFFER: tl.constexpr = False,  # lit ET écrit slot 0
    MUT_STORE_SILU_STATE: tl.constexpr = False,  # état = x silué (faux)
):
    """Un programme = (bloc de V, token, tête-v) — cf. docstring module.

    Propriété d'écriture (single-writer par canal d'état conv) :
    * canaux v du bloc : ce programme (toujours) ;
    * canaux q/k de la tête i_h : UNIQUEMENT si ``i_v == 0`` et
      ``i_hv % GROUP == 0`` (le propriétaire de la tête) — les autres
      programmes recalculent la conv en lecture seule (déterministe) et
      n'écrivent rien : sans cela 8 écritures concurrentes par canal.
    """
    i_v = tl.program_id(axis=0)
    i_nh = tl.program_id(axis=1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    if i_n >= num_valid:
        return
    GROUP = HV // H
    i_h = i_hv // GROUP

    state_idx = tl.load(idx_ptr + i_n).to(tl.int64)
    # Ligne nulle (NULL_BLOCK_ID=0) : sorties nulles, aucun état touché —
    # contrat du livré (`fused_recurrent_..._packed_decode` : store zeros ;
    # `causal_conv1d_update` : return sans écrire).
    if state_idx <= 0:
        o_vz = i_v * BV + tl.arange(0, BV)
        o_offz = (i_n * HV + i_hv) * V + o_vz
        tl.store(
            out_ptr + o_offz,
            tl.zeros([BV], dtype=tl.bfloat16),
            mask=o_vz < V,
        )
        return

    # ------------------------------------------------------------------
    # 1. Conv causale — canaux consommés : tête q i_h (K), tête k i_h (K),
    #    bloc v (BV).  y bf16 AVANT le scan (le livré stocke la conv).
    # ------------------------------------------------------------------
    x_base = x_ptr + i_n * stride_x_tok
    cs_base = cs_old_ptr + state_idx * stride_cs_seq

    q_off = i_h * K + tl.arange(0, K)  # canaux q de la tête
    k_off = H * K + i_h * K + tl.arange(0, K)  # canaux k de la tête
    v_off = 2 * H * K + i_hv * V + i_v * BV + tl.arange(0, BV)  # canaux v

    y_q, x_q = qwen3_gdn_conv_row(
        q_off, x_base, cs_base, convw_ptr, convb_ptr,
        stride_cs_dim, stride_cs_tok, stride_x_c, LEN=K, C=C, W=W,
    )
    y_k, x_k = qwen3_gdn_conv_row(
        k_off, x_base, cs_base, convw_ptr, convb_ptr,
        stride_cs_dim, stride_cs_tok, stride_x_c, LEN=K, C=C, W=W,
    )
    y_v, x_v = qwen3_gdn_conv_row(
        v_off, x_base, cs_base, convw_ptr, convb_ptr,
        stride_cs_dim, stride_cs_tok, stride_x_c, LEN=BV, C=C, W=W,
    )

    # ------------------------------------------------------------------
    # 2. Scan — règle delta, verbatim du livré.
    # ------------------------------------------------------------------
    st_base = st_ptr + state_idx * stride_st_seq + i_hv * V * K
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_h = (o_v < V)[:, None] & (o_k < K)[None, :]
    b_h = tl.load(
        st_base + o_v[:, None] * K + o_k[None, :], mask=mask_h, other=0.0
    ).to(tl.float32)

    b_q = y_q.to(tl.float32)
    b_k = y_k.to(tl.float32)
    b_vv = y_v.to(tl.float32)
    # l2norm in-kernel (contrat USE_QK_L2NORM_IN_KERNEL=True du livré)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a_ptr + i_n * HV + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * HV + i_hv).to(tl.float32)
    A_log_val = tl.load(alog_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dtb_ptr + i_hv).to(tl.float32)
    xg = a_val + dt_bias_val
    softplus_x = tl.where(xg <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(xg)), xg)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)

    b_h = b_h * tl.exp(g_val)
    b_vv = b_vv - tl.sum(b_h * b_k[None, :], 1)
    b_vv = b_vv * beta_val
    b_h = b_h + b_vv[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)

    o_off = (i_n * HV + i_hv) * V + o_v
    tl.store(out_ptr + o_off, b_o.to(tl.bfloat16), mask=o_v < V)

    # état scan en place (single-writer par (i_hv, bloc) — pas de course)
    tl.store(st_base + o_v[:, None] * K + o_k[None, :], b_h, mask=mask_h)

    # ------------------------------------------------------------------
    # 3. État conv — slot 1-parity (ou 0 en mutation), single-writer.
    #    Valeurs : fenêtre glissante [u1, u2, x_t BRUT] (pré-silu).
    # ------------------------------------------------------------------
    if MUT_SINGLE_BUFFER:
        cs_tgt = cs_old_ptr + state_idx * stride_cs_seq
    else:
        cs_tgt = cs_new_ptr + state_idx * stride_cs_seq

    owner_qk = (i_v == 0) & ((i_hv % GROUP) == 0)
    qwen3_gdn_store_state(
        v_off, x_v, y_v, cs_base, cs_tgt, stride_cs_dim, stride_cs_tok,
        LEN=BV, C=C, W=W, MUT_STORE_SILU_STATE=MUT_STORE_SILU_STATE,
    )
    if owner_qk:
        qwen3_gdn_store_state(
            q_off, x_q, y_q, cs_base, cs_tgt, stride_cs_dim, stride_cs_tok,
            LEN=K, C=C, W=W, MUT_STORE_SILU_STATE=MUT_STORE_SILU_STATE,
        )
        qwen3_gdn_store_state(
            k_off, x_k, y_k, cs_base, cs_tgt, stride_cs_dim, stride_cs_tok,
            LEN=K, C=C, W=W, MUT_STORE_SILU_STATE=MUT_STORE_SILU_STATE,
        )


# ---------------------------------------------------------------------------
# API python
# ---------------------------------------------------------------------------
_LOGGED: set[str] = set()


def _log_once(tag: str, msg: str, *args: Any) -> None:
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        logger.info(msg, *args)


def gdn_fusion_enabled() -> bool:
    """Gate INC-B ``VLLM_MOE_MEGAKERNEL_GDN`` (défaut OFF = repli livré).

    ⚠ La variable n'est PAS enregistrée dans ``vllm.envs`` (pas de câblage
    moteur en INC-B) : le ``getattr(envs, ...)`` seul la lirait comme
    TOUJOURS False — la gate existerait mais ne s'allumerait jamais.
    Deux canaux, dans cet ordre : ``os.environ`` (canal opérateur,
    prépondérant) puis l'attribut ``envs`` (canal de test / futur
    enregistrement INC-D).  Même pattern que ``dense_mfma_enabled()``.
    """
    raw = os.environ.get("VLLM_MOE_MEGAKERNEL_GDN")
    if raw is not None:
        return raw.strip().lower() in ("1", "true")
    return bool(getattr(envs, "VLLM_MOE_MEGAKERNEL_GDN", False))


def gdn_decode_fallback(
    x: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_w: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Repli = chaîne livrée VERBATIM (gate OFF) — aucune réécriture.

    ``causal_conv1d_update`` (sur ``x`` en place, contrat livré) puis
    ``fused_recurrent_gated_delta_rule_packed_decode`` (état fp32 en place).
    ``conv_state`` est le slot UNIQUE livré (pas de double-tampon côté
    repli : c'est le contrat du chemin de production).
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    mixed_qkv = causal_conv1d_update(
        x,
        conv_state,
        conv_w,
        conv_bias,
        activation,
        conv_state_indices=state_indices,
        validate_data=False,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=ssm_state,
        out=out.unsqueeze(1),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out


def gdn_decode_fusion(
    x: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_w: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_0: torch.Tensor,
    conv_state_1: torch.Tensor,
    parity: int,
    state_indices: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    num_valid: int | None = None,
    # hooks de mutation (tests uniquement)
    _mut_single_buffer: bool = False,
    _mut_store_silu_state: bool = False,
) -> torch.Tensor:
    """Chaîne GDN decode FUSIONNÉE — un lancement (INC-B).

    * ``x`` [B, C] bf16 : entrée conv BRUTE (sortie in_proj, pré-silu) —
      ⚠ non modifiée (contrairement au livré qui écrase ``x`` en place) ;
    * ``conv_state_0``/``conv_state_1`` [num_cache, C, 3] bf16 : double-
      tampon de l'état conv (voir docstring module — écart NOMMÉ) ;
      ``parity`` = le slot qui contient l'état COURANT ; le nouvel état va
      dans ``1-parity``.  Le pas suivant doit appeler avec ``parity ^ 1`` ;
    * ``state_indices`` [B] int32 : ≤0 = ligne nulle (sortie zéro, états
      intacts) ;
    * ``ssm_state`` [num_cache, HV, V, K] fp32 : état scan lu/écrit EN
      PLACE (comme le livré) ;
    * ``out`` [B, HV, V] bf16.

    Gate ``VLLM_MOE_MEGAKERNEL_GDN`` OFF → repli verbatim sur la chaîne
    livrée (uniquement avec un état conv unique : ``conv_state_0``).
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16, (
        f"x doit être bf16 2D : {x.dtype} {tuple(x.shape)}"
    )
    B, C = x.shape
    assert ssm_state.dim() == 4, (
        f"ssm_state doit être [num_cache, HV, V, K] : {tuple(ssm_state.shape)}"
    )
    HV, V, K = ssm_state.shape[-3:]
    # H déduit de C = 2*H*K + HV*V (invariant du layout mixed_qkv q|k|v)
    assert (C - HV * V) % (2 * K) == 0, (
        f"C={C} incompatible avec HV={HV} V={V} K={K} (layout q|k|v)"
    )
    H = (C - HV * V) // (2 * K)
    assert H > 0 and HV % H == 0, f"H={H} HV={HV} : HV%H != 0 (contrat livré)"
    assert parity in (0, 1), f"parity doit être 0/1 : {parity}"
    if num_valid is None:
        num_valid = B
    assert 0 <= num_valid <= B, f"num_valid={num_valid} hors [0,{B}]"
    assert out.shape == (B, HV, V) and out.dtype == torch.bfloat16, (
        f"out doit être bf16 [{B},{HV},{V}] : {out.dtype} {tuple(out.shape)}"
    )
    assert state_indices.dtype == torch.int32 and state_indices.shape == (B,), (
        f"state_indices doit être int32 [{B}] : "
        f"{state_indices.dtype} {tuple(state_indices.shape)}"
    )
    assert conv_w.shape == (C, GDN_CONV_WIDTH) and conv_w.stride(1) == 1, (
        f"conv_w doit être [{C},{GDN_CONV_WIDTH}] contigu en W : "
        f"{tuple(conv_w.shape)} strides {conv_w.stride()}"
    )
    for name, cs in (("conv_state_0", conv_state_0), ("conv_state_1", conv_state_1)):
        assert cs.shape == (cs.shape[0], C, GDN_CONV_STATE_LEN), (
            f"{name} doit être [num_cache,{C},{GDN_CONV_STATE_LEN}] : "
            f"{tuple(cs.shape)}"
        )
        assert cs.stride(2) == 1, f"{name} doit être contigu en slot : {cs.stride()}"
    # Revue deepseek F3 : toute la protection intra-lancement repose sur
    # cs_old en lecture seule pendant que l'autre slot est écrit — si les
    # deux slots ALIASENT, la protection disparaît silencieusement (exact.
    # MUT_SINGLE_BUFFER sans le signal).  Précondition load-bearing.
    assert conv_state_0.data_ptr() != conv_state_1.data_ptr(), (
        "conv_state_0 et conv_state_1 doivent être des tampons DISTINCTS — "
        "le double-tampon protège uniquement si lecture et écriture visent "
        "des adresses différentes (sinon : course, comme MUT_SINGLE_BUFFER)"
    )
    assert x.stride(1) == 1 and out.stride(2) == 1, (
        f"x/out doivent être contigus en dernière dim : {x.stride()} {out.stride()}"
    )
    assert b.stride(1) == 1 and a.stride(1) == 1, (
        f"b/a doivent être contigus en HV : {b.stride()} {a.stride()}"
    )
    assert a_log.stride(0) == 1 and dt_bias.stride(0) == 1, (
        f"a_log/dt_bias doivent être contigus : {a_log.stride()} {dt_bias.stride()}"
    )
    assert state_indices.stride(0) == 1, (
        f"state_indices doit être contigu : {state_indices.stride()}"
    )
    assert ssm_state.stride(3) == 1, (
        f"ssm_state doit être contigu en K : {ssm_state.stride()}"
    )
    for name, t in (
        ("x", x),
        ("b", b),
        ("a", a),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
        ("conv_w", conv_w),
        ("conv_bias", conv_bias),
        ("conv_state_0", conv_state_0),
        ("conv_state_1", conv_state_1),
        ("state_indices", state_indices),
        ("ssm_state", ssm_state),
        ("out", out),
    ):
        assert t.device == x.device, f"{name} sur {t.device} != x sur {x.device}"

    if not gdn_fusion_enabled():
        _log_once(
            "gdn-fusion-fallback",
            "[gdn-fusion] gate VLLM_MOE_MEGAKERNEL_GDN OFF — repli sur la "
            "chaîne livrée (causal_conv1d_update + packed_decode)",
        )
        return gdn_decode_fallback(
            x,
            b,
            a,
            a_log,
            dt_bias,
            conv_w,
            conv_bias,
            conv_state_0,
            state_indices,
            ssm_state,
            out,
            scale,
        )

    _log_once(
        f"gdn-fusion-{B}x{C}",
        "[gdn-fusion] chaîne GDN decode FUSIONNÉE ACTIVE "
        "(VLLM_MOE_MEGAKERNEL_GDN=1) : B=%d C=%d HV=%d V=%d K=%d "
        "BV=%d nw=%d — état conv double-tampon (parity=%d)",
        B,
        C,
        HV,
        V,
        K,
        GDN_BV,
        GDN_NUM_WARPS,
        parity,
    )

    cs_old = conv_state_0 if parity == 0 else conv_state_1
    cs_new = conv_state_1 if parity == 0 else conv_state_0
    NV = triton.cdiv(V, GDN_BV)
    grid = (NV, int(num_valid) * HV)
    qwen3_gdn_decode_fusion_kernel[grid](
        x,
        b,
        a,
        a_log,
        dt_bias,
        conv_w,
        conv_bias,
        cs_old,
        cs_new,
        ssm_state,
        out,
        state_indices,
        scale,
        int(num_valid),
        x.stride(0),
        x.stride(1),
        cs_old.stride(0),
        cs_old.stride(1),
        cs_old.stride(2),
        ssm_state.stride(0),
        out.stride(0),
        C=C,
        H=H,
        HV=HV,
        K=K,
        V=V,
        W=GDN_CONV_WIDTH,
        BV=GDN_BV,
        num_warps=GDN_NUM_WARPS,
        num_stages=GDN_NUM_STAGES,
        MUT_SINGLE_BUFFER=bool(_mut_single_buffer),
        MUT_STORE_SILU_STATE=bool(_mut_store_silu_state),
    )
    return out
