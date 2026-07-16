# SPDX-License-Identifier: Apache-2.0
"""3-step fused int8 W8A8 MoE kernel for gfx90a (MI210/CDNA2).

Uses int8 MFMA hardware (v_mfma_i32_16x16x16i8) via Triton's
tl.dot(a, b, out_dtype=tl.int32) with int32 accumulator.

Pipeline (3 launches per MoE layer, down from 7):
  1. moe_sort (reuse AITER) — sort token-expert pairs
  2. pertoken_quant (reuse AITER) — bf16 input → int8 + scale
  3. fused_fc1_act_requant — int8 GEMM + SwiGLU + requant → int8 intermediate
  4. fused_fc2_scatter — int8 GEMM + dequant + weighted atomicAdd scatter

Steps 3 and 4 are new Triton kernels. Steps 1-2 reuse existing AITER ops.

Model shapes (Qwen3.6-35B-A3B, TP=2):
  hidden=2048, inter_dim=256, N1=512 (g1u1), E=256, topk=8, 40 layers
"""
import torch
import triton
import triton.language as tl


# ============================================================================
# Step 3: Fused FC1 (int8 GEMM) + SwiGLU/SiLU + per-token requant → int8
# ============================================================================
@triton.jit
def fused_fc1_act_requant_kernel(
    # Input (pre-quantized int8)
    a_ptr,               # [M, hidden] int8 (sorted/gathered input)
    a_scale_ptr,         # [M] float32 (per-token input scale)
    # Weights
    w1_ptr,              # [E, N1, hidden] int8 (gate+up fused)
    w1_scale_ptr,        # [E, N1] float32 (per-channel weight scale)
    # Output
    inter_int8_ptr,      # [M_padded, inter_dim] int8
    inter_scale_ptr,     # [M_padded] float32
    # MoE metadata
    sorted_token_ids_ptr,  # [M_padded] int32
    expert_ids_ptr,        # [num_m_blocks] int32
    # Dimensions
    hidden, N1, inter_dim, M, num_tokens,
    # Strides
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn,
    stride_im, stride_in,
    # Constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_G1U1: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(inter_dim, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Expert and token indices
    # AITER moe_sorting packs sorted_ids as (token_id & 0x00FFFFFF) | (topk_id << 24)
    expert_id = tl.load(expert_ids_ptr + pid_m)
    # Skip blocks with invalid expert (padding blocks)
    if expert_id < 0:
        return
    expert_id = expert_id.to(tl.int64)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = offs_m < M
    packed_ids = tl.load(sorted_token_ids_ptr + offs_m, mask=m_mask, other=num_tokens)
    offs_token = (packed_ids & 0x00FFFFFF).to(tl.int64)
    offs_token = tl.where(offs_token < num_tokens, offs_token, 0)
    token_mask = offs_token < num_tokens

    # Output column indices
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < inter_dim

    # Load per-token input scale (clamped to valid range)
    safe_token = tl.where(token_mask, offs_token, 0)
    a_scale = tl.load(a_scale_ptr + safe_token, mask=token_mask, other=0.0)  # [BM]

    # Load weight scale for this expert and N columns
    # w1_scale is [E, N1], we need [BN] for the gate columns
    w_scale_gate = tl.load(w1_scale_ptr + expert_id * stride_wse + offs_n * stride_wsn,
                           mask=n_mask, other=0.0)  # [BN]

    # For g1u1: also load up column scales
    if IS_G1U1:
        w_scale_up = tl.load(w1_scale_ptr + expert_id * stride_wse +
                             (offs_n + inter_dim) * stride_wsn,
                             mask=n_mask, other=0.0)  # [BN]

    # FC1 GEMM: int8 × int8 → int32 accumulation
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Gate accumulator
    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    # Up accumulator (g1u1 only)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(hidden, BLOCK_SIZE_K)):
        k_mask = offs_k + k * BLOCK_SIZE_K < hidden
        # Load input int8 [BM, BK]
        a = tl.load(a_ptr + safe_token[:, None] * stride_am + (offs_k[None, :] + k * BLOCK_SIZE_K) * stride_ak,
                     mask=token_mask[:, None] & k_mask[None, :], other=0)
        # Load gate weight int8 [BK, BN] — w1[expert, n, k] layout
        b_gate = tl.load(w1_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn +
                         (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_wk,
                         mask=k_mask[:, None] & n_mask[None, :], other=0)
        # int8 MFMA: int8 × int8 → int32
        acc_gate = tl.dot(a, b_gate, acc=acc_gate, out_dtype=tl.int32)

        if IS_G1U1:
            # Load up weight int8 [BK, BN]
            b_up = tl.load(w1_ptr + expert_id * stride_we + (offs_n[None, :] + inter_dim) * stride_wn +
                           (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_wk,
                           mask=k_mask[:, None] & n_mask[None, :], other=0)
            acc_up = tl.dot(a, b_up, acc=acc_up, out_dtype=tl.int32)

    # Dequant: int32 → float32
    gate_f = acc_gate.to(tl.float32) * a_scale[:, None] * w_scale_gate[None, :]

    if IS_G1U1:
        up_f = acc_up.to(tl.float32) * a_scale[:, None] * w_scale_up[None, :]
        # SwiGLU: silu(gate) * up
        act_f = (gate_f / (1.0 + tl.exp(-gate_f))) * up_f
    else:
        # SiLU
        act_f = gate_f / (1.0 + tl.exp(-gate_f))

    # Per-token requant: find max-abs per row, scale, quantize to int8
    act_abs = tl.abs(act_f)
    # Per-row max reduction
    max_abs = tl.max(act_abs, axis=1)  # [BM]
    max_abs = tl.where(max_abs < 1e-8, 1e-8, max_abs)
    inter_scale = max_abs / 127.0  # [BM]

    # Quantize
    act_q = (act_f / inter_scale[:, None]).to(tl.int8)

    # Store int8 intermediate and scale
    tl.store(inter_int8_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in,
             act_q, mask=token_mask[:, None] & n_mask[None, :])
    tl.store(inter_scale_ptr + offs_m, inter_scale, mask=token_mask)


# ============================================================================
# Step 4: Fused FC2 (int8 GEMM) + dequant + weighted atomicAdd scatter
# ============================================================================
@triton.jit
def fused_fc2_scatter_kernel(
    # Input (int8 intermediate from step 3)
    inter_int8_ptr,      # [M_padded, inter_dim] int8
    inter_scale_ptr,     # [M_padded] float32
    # Weights
    w2_ptr,              # [E, hidden, inter_dim] int8
    w2_scale_ptr,        # [E, hidden] float32
    # Output
    out_ptr,             # [num_tokens, hidden] float32 (atomicAdd target)
    # MoE metadata
    sorted_token_ids_ptr,  # [M_padded] int32
    sorted_weights_ptr,    # [M_padded] float32 (router weights)
    expert_ids_ptr,        # [num_m_blocks] int32
    # Dimensions
    hidden, inter_dim, M, num_tokens,
    # Strides
    stride_im, stride_ik,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn,
    stride_om, stride_on,
    # Constexprs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(hidden, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert_id = tl.load(expert_ids_ptr + pid_m)
    if expert_id < 0:
        return
    expert_id = expert_id.to(tl.int64)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = offs_m < M
    packed_ids = tl.load(sorted_token_ids_ptr + offs_m, mask=m_mask, other=num_tokens)
    offs_token = (packed_ids & 0x00FFFFFF).to(tl.int64)
    offs_token = tl.where(offs_token < num_tokens, offs_token, 0)
    token_mask = offs_token < num_tokens

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < hidden

    # Load scales
    safe_m = tl.where(m_mask, offs_m, 0)
    inter_scale = tl.load(inter_scale_ptr + safe_m, mask=m_mask, other=0.0)  # [BM]
    w_scale = tl.load(w2_scale_ptr + expert_id * stride_wse + offs_n * stride_wsn,
                      mask=n_mask, other=0.0)  # [BN]
    router_weight = tl.load(sorted_weights_ptr + safe_m, mask=m_mask, other=0.0)  # [BM]

    # FC2 GEMM: int8 × int8 → int32
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(inter_dim, BLOCK_SIZE_K)):
        k_mask = offs_k + k * BLOCK_SIZE_K < inter_dim
        # Load int8 intermediate [BM, BK]
        a = tl.load(inter_int8_ptr + offs_m[:, None] * stride_im +
                    (offs_k[None, :] + k * BLOCK_SIZE_K) * stride_ik,
                    mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
        # Load int8 weight [BK, BN] — w2[expert, n, k] layout
        b = tl.load(w2_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn +
                    (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_wk,
                    mask=k_mask[:, None] & n_mask[None, :], other=0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)

    # Dequant: int32 → float32, apply scales + router weight
    result = acc.to(tl.float32) * inter_scale[:, None] * w_scale[None, :] * router_weight[:, None]

    # Scatter via atomicAdd (multiple experts contribute to same token)
    tl.atomic_add(out_ptr + offs_token[:, None] * stride_om + offs_n[None, :] * stride_on,
                  result, mask=token_mask[:, None] & n_mask[None, :])


# ============================================================================
# Python launchers
# ============================================================================
def invoke_fused_fc1_act_requant(
    a_int8, a_scale, w1, w1_scale,
    inter_int8, inter_scale,
    sorted_token_ids, expert_ids,
    hidden, N1, inter_dim, M, num_tokens,
    block_size_m=32, block_size_n=16, block_size_k=16, group_size_m=8,
):
    is_g1u1 = (N1 == 2 * inter_dim)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(inter_dim, META['BLOCK_SIZE_N']),
    )
    fused_fc1_act_requant_kernel[grid](
        a_int8, a_scale, w1, w1_scale,
        inter_int8, inter_scale,
        sorted_token_ids, expert_ids,
        hidden, N1, inter_dim, M, num_tokens,
        a_int8.stride(0), a_int8.stride(1),
        w1.stride(0), w1.stride(1), w1.stride(2),
        w1_scale.stride(0), w1_scale.stride(1) if w1_scale.ndim >= 2 else 1,
        inter_int8.stride(0), inter_int8.stride(1),
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k, GROUP_SIZE_M=group_size_m,
        IS_G1U1=is_g1u1,
    )


def invoke_fused_fc2_scatter(
    inter_int8, inter_scale, w2, w2_scale, out,
    sorted_token_ids, sorted_weights, expert_ids,
    hidden, inter_dim, M, num_tokens,
    block_size_m=32, block_size_n=16, block_size_k=16, group_size_m=8,
):
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(hidden, META['BLOCK_SIZE_N']),
    )
    fused_fc2_scatter_kernel[grid](
        inter_int8, inter_scale, w2, w2_scale, out,
        sorted_token_ids, sorted_weights, expert_ids,
        hidden, inter_dim, M, num_tokens,
        inter_int8.stride(0), inter_int8.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        w2_scale.stride(0), w2_scale.stride(1) if w2_scale.ndim >= 2 else 1,
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k, GROUP_SIZE_M=group_size_m,
    )


# ============================================================================
# Orchestration: sort + quant + FC1+act+requant + FC2+scatter
# ============================================================================
def fused_experts_int8_gfx90a(
    hidden_states,  # [M, hidden] bf16
    w1,             # [E, N1, hidden] int8
    w2,             # [E, hidden, inter_dim] int8
    w1_scale,       # [E, N1] or [E, N1, 1] float32
    w2_scale,       # [E, hidden] or [E, hidden, 1] float32
    topk_weights,   # [M, topk] float32
    topk_ids,       # [M, topk] int32
    num_experts,
    block_size_m=32,
):
    M = hidden_states.shape[0]
    hidden = w1.shape[2]
    N1 = w1.shape[1]
    inter_dim = w2.shape[2]
    device = hidden_states.device

    # Flatten scales if needed
    if w1_scale.ndim == 3:
        w1_scale = w1_scale.squeeze(-1)
    if w2_scale.ndim == 3:
        w2_scale = w2_scale.squeeze(-1)

    # Step 1: moe_sorting (reuse AITER)
    from aiter.fused_moe import moe_sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        num_experts, hidden, hidden_states.dtype)

    num_tokens = M
    # Clamp invalid expert_ids to 0 (padding blocks have garbage values)
    sorted_expert_ids = torch.where(sorted_expert_ids < num_experts, sorted_expert_ids, 0)
    M_padded = sorted_ids.shape[0]
    # Pad to multiple of BLOCK_SIZE_M to avoid OOB in kernel
    BM = 32
    M_padded_padded = ((M_padded + BM - 1) // BM) * BM
    if M_padded_padded > M_padded:
        sorted_ids = torch.cat([sorted_ids, torch.full((M_padded_padded - M_padded,),
            num_tokens, dtype=torch.int32, device=device)])
        sorted_weights = torch.cat([sorted_weights, torch.zeros(M_padded_padded - M_padded,
            dtype=torch.float32, device=device)])

    # Step 2: per-token quant of input (reuse AITER)
    from aiter import pertoken_quant
    a_int8, a_scale = pertoken_quant(hidden_states, quant_dtype=torch.int8)
    a_int8 = a_int8.view(M, hidden)
    a_scale = a_scale.view(M)

    # Step 3: FC1 + activation + requant → int8 intermediate
    inter_int8 = torch.empty(M_padded_padded, inter_dim, dtype=torch.int8, device=device)
    inter_scale = torch.empty(M_padded_padded, dtype=torch.float32, device=device)

    invoke_fused_fc1_act_requant(
        a_int8, a_scale, w1, w1_scale,
        inter_int8, inter_scale,
        sorted_ids, sorted_expert_ids,
        hidden, N1, inter_dim, M_padded_padded, M,
        block_size_m=block_size_m,
    )

    # Step 4: FC2 + dequant + weighted scatter
    out_f32 = torch.zeros(M, hidden, dtype=torch.float32, device=device)

    invoke_fused_fc2_scatter(
        inter_int8, inter_scale, w2, w2_scale, out_f32,
        sorted_ids, sorted_weights, sorted_expert_ids,
        hidden, inter_dim, M_padded_padded, M,
        block_size_m=block_size_m,
    )

    return out_f32.to(torch.bfloat16)
