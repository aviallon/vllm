# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.triton_utils import tl, triton, use_tensor_descriptor
from vllm.triton_utils.allocation import set_triton_allocator

_TD_ALLOCATOR_DEVICES: set[torch.device] = set()


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    USE_TD: tl.constexpr = False,
    B_T: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    if USE_TD:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        if B_T:
            # Checkpoint weights arrive as a transposed [N, K] view, so describe
            # the underlying N-major buffer and transpose each tile.
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[N, K],
                strides=[stride_bn, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
        else:
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[K, N],
                strides=[stride_bk, 1],
                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if USE_TD:
            a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
            if B_T:
                b = tl.trans(b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]))
            else:
                b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
        else:
            masks_k = offsets_k < K
            masks_a = masks_am[:, None] & masks_k[None, :]
            a = tl.load(a_ptrs, mask=masks_a)

            masks_b = masks_k[:, None] & masks_bn[None, :]
            b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def scaled_mm_fused_quant_kernel(
    a_ptr,
    b_ptr,
    scale_a_out_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    AMAX_SCAN: tl.constexpr = True,
):
    """W8A8 GEMM with the dynamic per-token int8 quantization of A fused into
    the prologue (megakernel campaign 2026-10-23, pair (a)).

    Replaces the two-kernel chain
    ``_C::dynamic_scaled_int8_quant`` -> ``triton_scaled_mm`` : A is read ONCE
    as its original float dtype, quantized per K chunk in registers, and fed to
    the same int8 ``tl.dot``.  The int8 activation is never written to HBM.

    NUMERICS: bit-exact vs the two-kernel chain by construction — keep every
    expression below in lockstep with
    ``csrc/quantization/w8a8/int8/scaled_quant.cu::dynamic_scaled_int8_quant_kernel``
    and with ``scaled_mm_kernel``'s epilogue (the tests assert torch.equal):

      * absmax = fp32 max of |x| per row (exact, reduction-order independent);
      * scale   = absmax / 127.0   (what the quant kernel writes to its
        ``scale_out``, and what the GEMM epilogue consumes as scale_a);
      * inv_s   = 127.0 / absmax if absmax != 0 else 0.0 (the quant kernel's
        quantization multiplier);
      * quantization = clamp(nearbyint(x * inv_s), -128, 127).to(int8)
        (``float_to_int8_rn`` on ROCm = std::nearbyint + saturate);
      * the epilogue below is the SAME expressions as ``scaled_mm_kernel``
        (scale_a * acc.to(fp32)) then (scale_b.T * acc), bias after the output
        conversion — same order, same rounding.

    The int8 dot accumulates in int32 (exact), so the K-sum order is free.
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    # ------------------------------------------------------------------
    # Pass 1 — per-row absmax over the whole K (the dynamic per-token scale
    # needs the full row before any quantized value can be formed).
    # Masked K tail loads contribute |0| = 0, like the quant kernel which
    # only walks the valid elements from a 0.0 initial max.
    # AMAX_SCAN=False (probe/ablation only) skips the scan and reads the
    # per-row scales from `scale_a_out_ptr` instead (single pass).
    # ------------------------------------------------------------------
    if AMAX_SCAN:
        amax = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        k_off = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        a_scan = a_ptr + stride_am * offsets_am[:, None] + stride_ak * k_off[None, :]
        # num_stages pipelines the loads across iterations: without it the
        # K-chunk loads are serialised and their latency is exposed (measured
        # +5.6 us at decode shapes in session mk-quant-gemm-b2).  Budget:
        # each staged buffer is (A tile + B tile) = 24 KiB at the decode tile,
        # and the LDS window is 64 KiB minus the dot's own staging (16 KiB) —
        # num_stages=2 fits (48 KiB), 4 overflows (73728 B > 65536, measured
        # in mk-quant-gemm-b4).
        for _k1 in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=2):
            masks_k = k_off < K
            a_f = tl.load(
                a_scan, mask=masks_am[:, None] & masks_k[None, :], other=0.0
            )
            amax = tl.maximum(amax, tl.max(tl.abs(a_f.to(tl.float32)), axis=1))
            k_off = k_off + BLOCK_SIZE_K
            a_scan = a_scan + BLOCK_SIZE_K * stride_ak

        inv_s = tl.where(amax == 0.0, 0.0, 127.0 / amax)
        scale_a = amax / 127.0
    else:
        scale_a = tl.load(scale_a_out_ptr + offsets_am, mask=masks_am, other=0.0)
        inv_s = tl.where(scale_a == 0.0, 0.0, 1.0 / scale_a)

    # ------------------------------------------------------------------
    # Pass 2 — quantize each A chunk in registers (VALU) and accumulate with
    # the int8 weights (MFMA).  tl.range(num_stages=) software-pipelines the
    # loads across iterations: without it the load->quant->dot chain exposes
    # the memory latency of EVERY chunk (measured +4.5 us at M<=32 and +39 us
    # at M=256 vs the shipped chain in mk-quant-gemm-b2/b3).  With it, the
    # VALU work of chunk k+1 sits in the MFMA/load bubbles of chunk k — the
    # VALU/MFMA overlap the campaign asked for.  num_stages=2 only (LDS
    # budget: see the scan loop's note; 4 measured to overflow at 73728 B).
    # ------------------------------------------------------------------
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=2):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a_f = tl.load(a_ptrs, mask=masks_a, other=0.0)
        a_r = tl.extra.hip.libdevice.nearbyint(a_f.to(tl.float32) * inv_s[:, None])
        a_q = tl.maximum(tl.minimum(a_r, 127.0), -128.0).to(tl.int8)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)

        # Accumulate results.
        accumulator = tl.dot(a_q, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    # KEEP IN SYNC with scaled_mm_kernel (bit-exactness contract).
    scale_a_t = scale_a[:, None].to(tl.float32)
    accumulator = scale_a_t * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)

    # Publish the per-row scales — same values the two-kernel chain would
    # have written via dynamic_scaled_int8_quant's scale_out ([M, 1] fp32).
    if AMAX_SCAN:
        offs_scale_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        tl.store(
            scale_a_out_ptr + offs_scale_m,
            scale_a,
            mask=offs_scale_m < M,
        )
# weight - [K, N]
def _choose_tile_shape(
    M: int,
    N: int,
    use_heuristic: bool,
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Tile selection shared by ``scaled_mm_kernel`` and the fused-quant
    variant, so both paths run the SAME tiles (the fused prologue is then the
    only difference between them).

    BM=16 for decode (M<=32): -19..-37% vs BM=64 on gfx90a INT8 GEMM,
    bit-exact, scoped here so prefill (M>64) keeps BM=64.
    """
    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            return (16, 64, 256) if is_small_N else (16, 128, 256)
        elif next_power_of_2_M <= 64:
            return (64, 64, 256)
        elif next_power_of_2_M <= 128:
            return (64, 128, 128)
        else:
            return (128, 128, 128)
    return fallback


def _triton_scaled_mm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
    use_td: bool | None = None,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    tile_shape = _choose_tile_shape(
        M, N, use_heuristic, (block_size_m, block_size_n, block_size_k)
    )

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # TD operand loads; gated on inner-dim contiguity and 16-byte tile alignment.
    # Checkpoint weights are a transposed [N, K] view, so accept either
    # orientation and let the kernel transpose per tile.
    b_t = weight.stride(1) != 1 and weight.stride(0) == 1
    b_inner = K if b_t else N
    use_td = (
        use_tensor_descriptor(use_td)
        and input.stride(1) == 1
        and (weight.stride(1) == 1 or b_t)
        and (K * input.element_size()) % 16 == 0
        and (b_inner * weight.element_size()) % 16 == 0
        and (block_size_m & (block_size_m - 1)) == 0
        and (block_size_n & (block_size_n - 1)) == 0
        and (block_size_k & (block_size_k - 1)) == 0
    )
    if use_td and input.device not in _TD_ALLOCATOR_DEVICES:
        set_triton_allocator(input.device)
        _TD_ALLOCATOR_DEVICES.add(input.device)

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        USE_TD=use_td,
        B_T=b_t,
    )

    return result.to(out_dtype)


# ---------------------------------------------------------------------------
# torch.compile-safe wrapper.
#
# `scaled_mm_kernel` is a raw `@triton.jit` kernel. When the int8 W8A8 linear
# path is traced by torch.compile/inductor (e.g. on ROCm, where
# cutlass_scaled_mm dispatches here), torch tries to introspect the raw kernel
# via `identify_mutated_tensors` -> `generate_ttir`, which raises
# `IndexError('Function argument index out of range')`. Torch then falls back
# to "assuming every input is mutated", which poisons inductor's buffer
# aliasing/functionalization and produces all-zero outputs (garbage tokens).
#
# Registering an opaque custom op (which mutates nothing and returns a fresh
# tensor) prevents torch from introspecting the raw Triton kernel, fixing the
# miscompilation while keeping eager behaviour identical.
# ---------------------------------------------------------------------------
def _triton_scaled_mm_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _triton_scaled_mm_impl(input, weight, scale_a, scale_b, out_dtype, bias)


def _triton_scaled_mm_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        (input.shape[0], weight.shape[1]), dtype=out_dtype, device=input.device
    )


try:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="triton_scaled_mm",
        op_func=_triton_scaled_mm_op,
        mutates_args=[],
        fake_impl=_triton_scaled_mm_op_fake,
    )
    _HAS_TRITON_SCALED_MM_OP = True
except Exception:
    # If registration fails (e.g. op already registered), fall back to the
    # direct implementation. Eager correctness is unaffected either way.
    _HAS_TRITON_SCALED_MM_OP = False


def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
) -> torch.Tensor:
    # Route through the opaque custom op so torch.compile does not introspect
    # the raw Triton kernel (see note above). Only the default tuning params
    # are used by the kernel, so non-default block sizes are not forwarded.
    if _HAS_TRITON_SCALED_MM_OP:
        return torch.ops.vllm.triton_scaled_mm(
            input, weight, scale_a, scale_b, out_dtype, bias
        )
    return _triton_scaled_mm_impl(
        input,
        weight,
        scale_a,
        scale_b,
        out_dtype,
        bias,
        block_size_m,
        block_size_n,
        block_size_k,
        use_heuristic,
    )


# ---------------------------------------------------------------------------
# Fused-quant variant (megakernel campaign 2026-10-23, pair (a)).
#
# input   - [M, K] float (bf16/fp16/fp32), quantized per row inside the kernel
# weight  - [K, N] int8
# returns (result [M, N] out_dtype, scale_a [M, 1] fp32) — scale_a is exactly
# what `dynamic_scaled_int8_quant` would have written to its scale_out.
#
# Same torch.compile treatment as `triton_scaled_mm` above: routed through an
# opaque custom op so inductor never introspects the raw Triton kernel.  Unlike
# the plain variant, a registration failure here falls back to the TWO-KERNEL
# CHAIN (never to the raw fused kernel), so torch.compile can never see it.
# ---------------------------------------------------------------------------
def _triton_scaled_mm_fused_quant_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
    amax_scan: bool = True,
    scale_a_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype != weight.dtype, "input must be unquantized (float)"
    assert input.is_floating_point()
    assert weight.dtype == torch.int8

    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_b.is_floating_point()
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)
    scale_a_out = torch.empty((M, 1), dtype=torch.float32, device=input.device)
    if not amax_scan:
        # probe/ablation path: the per-row scales come from outside
        assert scale_a_in is not None
        scale_a_out = scale_a_in.contiguous().view(M, 1).to(torch.float32)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    tile_shape = _choose_tile_shape(
        M, N, use_heuristic, (block_size_m, block_size_n, block_size_k)
    )
    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    # int8 x int8 -> int32 accumulation, like the quantized `scaled_mm_kernel`.
    accumulator_dtype = tl.int32

    scaled_mm_fused_quant_kernel[grid](
        input,
        weight,
        scale_a_out,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        AMAX_SCAN=amax_scan,
    )

    return result.to(out_dtype), scale_a_out


def _triton_scaled_mm_fused_quant_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_scaled_mm_fused_quant_impl(input, weight, scale_b, out_dtype, bias)


def _triton_scaled_mm_fused_quant_op_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (input.shape[0], weight.shape[1]), dtype=out_dtype, device=input.device
        ),
        torch.empty((input.shape[0], 1), dtype=torch.float32, device=input.device),
    )


try:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="triton_scaled_mm_fused_quant",
        op_func=_triton_scaled_mm_fused_quant_op,
        mutates_args=[],
        fake_impl=_triton_scaled_mm_fused_quant_op_fake,
    )
    _HAS_TRITON_SCALED_MM_FUSED_QUANT_OP = True
except Exception:
    # Registration failure must NOT fall back to the raw fused kernel: under
    # torch.compile the kernel would be introspected and miscompiled (see the
    # note above).  Callers check this flag and keep the two-kernel chain.
    _HAS_TRITON_SCALED_MM_FUSED_QUANT_OP = False


def has_triton_scaled_mm_fused_quant_op() -> bool:
    return _HAS_TRITON_SCALED_MM_FUSED_QUANT_OP


def triton_scaled_mm_fused_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `input` per row (dynamic per-token symmetric int8) and run the
    W8A8 GEMM in ONE kernel.  See `scaled_mm_fused_quant_kernel` for the
    bit-exactness contract.
    """
    if _HAS_TRITON_SCALED_MM_FUSED_QUANT_OP:
        return torch.ops.vllm.triton_scaled_mm_fused_quant(
            input, weight, scale_b, out_dtype, bias
        )
    return _triton_scaled_mm_fused_quant_impl(
        input, weight, scale_b, out_dtype, bias
    )
