# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the JustAI project
"""Contract tests for the INCREMENT-2a requant epilogue of the W8A8 GEMMs
(research/ar-vol-count-2026-10-25, ETAPE-E section 4, gate of 2a).

The epilogue writes the GEMM output as INT8 + per-(row, output-tile) scales
instead of bf16 (the reference value is the one the bf16 chain would STORE).
Contracts asserted here:

1. LOCKSTEP (the bit-exact contract named in ETAPE-E section 5): the two
   kernels' epilogues (``scaled_mm_kernel`` and ``scaled_mm_fused_quant_kernel``)
   produce ``torch.equal`` (q, scale_c) for the same effective dot inputs.
2. TOLERANCE vs the bf16 chain (the 2a gate wording "vs chaine bf16,
   tolerance requant"): the dequantized requant output stays within half an
   LSB of the chain's bf16 value, per element, LSB = its tile scale.
3. The quant expressions are the ``dynamic_scaled_int8_quant`` contract
   (scale = amax/127, inv_s = 127/amax, nearbyint half-to-even + saturate):
   the half-even guard below FAILS if the rounding mode drifts.
4. Zero tiles: scale 0, q 0 (the absmax==0 special case).
5. The call counters count launches, INCLUDING CUDA-graph replayed ones (a
   host-side counter cannot see replays), split in the 2 classes of the
   census (class 1 = in-kernel prologue, class 2 = shipped chain).
6. Env gate: the requant entry points fail loud with
   VLLM_SCALED_MM_REQUANT_OUT off, and the SHIPPED entry points keep the
   bf16 path byte-identical (the "repli bf16" of the gate).
"""

import pytest
import torch

from vllm._custom_ops import scaled_int8_quant
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    _choose_tile_shape,
    _triton_scaled_mm_fused_quant_impl as fused_quant_impl,
    _triton_scaled_mm_impl as bare_impl,
    read_vol_counts,
    triton_scaled_mm,
    triton_scaled_mm_fused_quant,
    triton_scaled_mm_fused_quant_requant,
    triton_scaled_mm_requant,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)

# decode shapes first (the campaign's target regime), then off-tile shapes.
SHAPES = [
    (1, 2048, 1024),
    (4, 2048, 2048),
    (32, 2048, 512),
    (33, 1024, 1024),
    (128, 2048, 1024),
]


def _make_inputs(M, K, N, dtype=torch.bfloat16, device="cuda", seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.float32, device=device).to(dtype)
    w_q = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
    scale_b = torch.rand(N, 1, dtype=torch.float32, device=device) * 0.01 + 0.001
    return x, w_q, scale_b


def _n_tiles(M, N):
    _bm, bn, _bk = _choose_tile_shape(M, N, True, (32, 32, 32))
    return (N + bn - 1) // bn


def _chain_bf16(x, w_q, scale_b, out_dtype=torch.bfloat16, bias=None):
    """The shipped two-kernel chain: quantize A, plain GEMM, bf16 out."""
    x_q, x_s, _ = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    out = triton_scaled_mm(
        x_q, w_q, scale_a=x_s, scale_b=scale_b, out_dtype=out_dtype, bias=bias
    )
    return out, x_q, x_s


@requires_cuda
def test_requant_lockstep_two_epilogues():
    """THE bit-exact contract (ETAPE-E 5): both kernels' requant epilogues see
    the same int8 dot inputs (the fused prologue's quantization is bit-exact
    with scaled_int8_quant -- asserted by the megakernel tests), so (q,
    scale_c) must be torch.equal."""
    for M, K, N in SHAPES:
        x, w_q, scale_b = _make_inputs(M, K, N)
        bias = torch.randn(N, dtype=torch.bfloat16, device=x.device)
        # bare side: feed the chain's own quantization of A
        x_q, x_s, _ = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
        q_bare, s_bare = bare_impl(
            x_q, w_q, x_s, scale_b, torch.bfloat16, bias=bias, requant_out=True
        )
        # fused side: quantize A inside the kernel prologue
        q_fused, s_fused = fused_quant_impl(
            x, w_q, scale_b, torch.bfloat16, bias=bias, requant_out=True
        )
        assert q_bare.dtype == torch.int8 and q_fused.dtype == torch.int8
        assert s_bare.shape == (M, _n_tiles(M, N))
        assert s_fused.shape == s_bare.shape
        assert torch.equal(s_bare, s_fused), f"scale_c differs ({M},{K},{N})"
        assert torch.equal(q_bare, q_fused), f"requant q differs ({M},{K},{N})"


@requires_cuda
def test_requant_within_half_lsb_of_chain():
    """The 2a numeric gate: the dequantized requant output stays within half
    an LSB (= scale_c of the element's output tile) of the value the bf16
    chain stores.  This is a tolerance contract, not bit-exactness: the
    requant IS a quantization."""
    for M, K, N in SHAPES:
        x, w_q, scale_b = _make_inputs(M, K, N, seed=1)
        bias = torch.randn(N, dtype=torch.bfloat16, device=x.device)
        ref_out, x_q, x_s = _chain_bf16(x, w_q, scale_b, bias=bias)
        q_out, scale_c = bare_impl(
            x_q, w_q, x_s, scale_b, torch.bfloat16, bias=bias, requant_out=True
        )
        n_tiles = scale_c.shape[1]
        # per-element LSB: its (row, output-tile) scale
        bn = (N + n_tiles - 1) // n_tiles
        tile_idx = torch.arange(N, device=x.device) // bn
        s_elem = scale_c[:, tile_idx]  # [M, N]
        dequant = q_out.to(torch.float32) * s_elem
        ref = ref_out.to(torch.float32)
        lsb = dequant - ref
        # |q*s - v| <= s/2 exactly (nearbyint), plus the fp32 rounding of the
        # test-side q*s multiply (~eps*|q*s|, measured 1.2e-6 at these
        # magnitudes) and of the kernel's v*inv_s product.
        eps = torch.finfo(torch.float32).eps
        tol = 0.5 * s_elem + 3.0 * eps * dequant.abs() + 1e-9
        bad = lsb.abs() > tol
        assert not bad.any(), (
            f"requant error beyond half an LSB at ({M},{K},{N}): "
            f"{int(bad.sum())} elements, worst "
            f"{(lsb.abs() - tol).max().item():.3g}"
        )


@requires_cuda
def test_requant_half_even_rounding_guard():
    """Mutation guard for the rounding mode: values landing on exact .5 must
    round half-to-EVEN (nearbyint / float_to_int8_rn).  Crafted so the tile
    amax is exactly 127 -> inv_s = 1 -> q = nearbyint(v).

    v = acc*scale_a*scale_b + bias: bias=0.5 on the .5 columns and a 127
    column (bias 0) pins the tile amax at exactly 127.
    """
    M, K, N = 1, 256, 10
    device = "cuda"
    w_q = torch.zeros(K, N, dtype=torch.int8, device=device)
    w_q[0, :] = 1  # acc[row, c] = x_q[row, 0]
    x_q = torch.zeros(M, K, dtype=torch.int8, device=device)
    x_q[0, 0] = 1  # acc = 1 everywhere
    scale_a = torch.ones(M, 1, dtype=torch.float32, device=device)
    scale_b = torch.ones(N, 1, dtype=torch.float32, device=device)
    bias = torch.full((N,), 0.5, dtype=torch.bfloat16, device=device)
    halves = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]
    # columns 0..7 carry the .5 values via bias (acc=1 -> v = 1 + bias - 1?)
    # simplest: acc = 0 on those columns and bias = the .5 value itself.
    for c, h in enumerate(halves):
        w_q[0, c] = 0  # acc = 0 -> v = bias = h
        bias[c] = h
    # column 8: v = 127 (pins amax); column 9: v = 0
    w_q[0, 8] = 1
    x_q[0, 0] = 127  # acc[0, 8] = 127 * 1 = 127
    bias[8] = 0.0
    w_q[0, 9] = 0
    bias[9] = 0.0

    q_out, scale_c = bare_impl(
        x_q, w_q, scale_a, scale_b, torch.bfloat16, bias=bias, requant_out=True
    )
    assert scale_c[0, 0].item() == pytest.approx(127.0 / 127.0), (
        "the crafted tile amax must be exactly 127 for inv_s to be 1"
    )
    expected = torch.tensor([0, 2, 2, 4, 0, -2, -2, -4], dtype=torch.int8)
    got = q_out[0, :8].cpu()
    assert torch.equal(got, expected), (
        f"rounding mode drifted: got {got.tolist()} want {expected.tolist()}"
    )


@requires_cuda
def test_requant_zero_tile():
    """absmax == 0 -> inv_s = 0 -> q = 0, scale = 0 (the special case of
    dynamic_scaled_int8_quant)."""
    M, K, N = 3, 512, 128
    x, w_q, scale_b = _make_inputs(M, K, N, seed=2)
    x[1] = 0.0
    x_q, x_s, _ = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    q_out, scale_c = bare_impl(
        x_q, w_q, x_s, scale_b, torch.bfloat16, requant_out=True
    )
    assert (scale_c[1] == 0.0).all(), "zero row must give zero scales"
    assert (q_out[1] == 0).all(), "zero row must quantize to zero"


@requires_cuda
def test_requant_counter_counts_graph_replays():
    """The census counter counts LAUNCHES, including CUDA-graph replays (the
    reason it is a device atomic, not a host counter).  Class 1 = in-kernel
    prologue (fused impl); class 2 = shipped chain (M > 32 regime)."""
    x, w_q, scale_b = _make_inputs(4, 2048, 256, seed=3)
    before = read_vol_counts()
    for _ in range(3):
        fused_quant_impl(x, w_q, scale_b, torch.bfloat16)
    mid = read_vol_counts()
    assert mid[0] - before[0] == 3, "class-1 counter must count fused launches"
    assert mid[1] - before[1] == 0

    xc, w_qc, scale_bc = _make_inputs(128, 2048, 256, seed=4)
    # M=128 > FUSED_QUANT_MAX_ROWS -> the shipped chain (class 2)
    before = read_vol_counts()
    out = triton_scaled_mm_fused_quant(xc, w_qc, scale_bc, torch.bfloat16)
    assert out[0].dtype == torch.bfloat16
    mid = read_vol_counts()
    assert mid[1] - before[1] == 1, "class-2 counter must count chain launches"
    assert mid[0] - before[0] == 0

    # graph replays must be counted too (a host-side counter sees only the
    # capture-time call)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fused_quant_impl(x, w_q, scale_b, torch.bfloat16)  # warmup/allocs
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    before = read_vol_counts()
    with torch.cuda.graph(g):
        fused_quant_impl(x, w_q, scale_b, torch.bfloat16)
    for _ in range(7):
        g.replay()
    torch.cuda.synchronize()
    after = read_vol_counts()
    counted = after[0] - before[0]
    assert 7 <= counted <= 8, (
        f"graph replays must be counted (got {counted} for 7 replays; a host "
        "counter would report at most 1)"
    )


@requires_cuda
def test_requant_env_gate_fails_loud_and_shipped_fallback_is_bf16(monkeypatch):
    """Gate off = fail loud at the requant entry points (never a silent bf16
    answer in a caller expecting int8+scales), and the SHIPPED entry points
    keep byte-identical bf16 outputs (the 'repli bf16')."""
    x, w_q, scale_b = _make_inputs(4, 2048, 512, seed=5)
    x_q, x_s, _ = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)

    monkeypatch.setenv("VLLM_SCALED_MM_REQUANT_OUT", "0")
    with pytest.raises(RuntimeError):
        triton_scaled_mm_requant(
            x_q, w_q, x_s, scale_b, torch.bfloat16
        )
    with pytest.raises(RuntimeError):
        triton_scaled_mm_fused_quant_requant(x, w_q, scale_b, torch.bfloat16)

    monkeypatch.setenv("VLLM_SCALED_MM_REQUANT_OUT", "1")
    q1, s1 = triton_scaled_mm_requant(x_q, w_q, x_s, scale_b, torch.bfloat16)
    assert q1.dtype == torch.int8 and s1.dtype == torch.float32
    q2, s2 = triton_scaled_mm_fused_quant_requant(x, w_q, scale_b, torch.bfloat16)
    assert torch.equal(q1, q2) and torch.equal(s1, s2)

    # shipped path unchanged (bf16, bit-exact vs its own chain)
    monkeypatch.setenv("VLLM_SCALED_MM_REQUANT_OUT", "0")
    ref_out, _, _ = _chain_bf16(x, w_q, scale_b)
    out, xs = fused_quant_impl(x, w_q, scale_b, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, ref_out), "shipped fused path drifted"


@requires_cuda
def test_requant_transposed_weight_layout():
    """Production weight layout: [N, K] transposed view (process_weights_after_loading
    stores w_q.t()) must requant identically to the contiguous layout."""
    M, K, N = 8, 1024, 512
    x, w_q, scale_b = _make_inputs(M, K, N, seed=6)
    x_q, x_s, _ = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    q_a, s_a = bare_impl(
        x_q, w_q, x_s, scale_b, torch.bfloat16, requant_out=True
    )
    w_t = w_q.t().contiguous().t()  # same values, [N, K]-major stride layout
    q_b, s_b = bare_impl(
        x_q, w_t, x_s, scale_b, torch.bfloat16, requant_out=True
    )
    assert torch.equal(s_a, s_b) and torch.equal(q_a, q_b)
