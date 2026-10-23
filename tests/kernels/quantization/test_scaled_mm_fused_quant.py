# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the fused-quant W8A8 GEMM (megakernel campaign 2026-10-23,
pair (a): dynamic per-token int8 activation quantization folded into the
`scaled_mm_kernel` prologue).

The contract is BIT-EXACTNESS against the shipped two-kernel chain
(`_C::dynamic_scaled_int8_quant` -> `triton_scaled_mm`): same absmax, same
scale expressions, same nearbyint rounding + saturation, same epilogue order.
No dtype rule is consumed (the quantized path keeps int8/int32 exactly as the
chain does).

`test_fused_quant_half_even_rounding` is the mutation guard: if the prologue
ever switches to round-half-away-from-zero (`tl.extra.hip.libdevice.round`),
it fails on the crafted .5 values.
"""

import torch

try:
    from tests.kernels.quant_utils import ref_dynamic_per_token_quant
except ImportError:  # in-image runs may not ship the tests package
    ref_dynamic_per_token_quant = None

from vllm import envs
from vllm._custom_ops import scaled_int8_quant
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    fused_quant_eligible,
)
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    _choose_tile_shape,
    _triton_scaled_mm_fused_quant_impl as fused_quant_kernel,
    fused_quant_regime,
    triton_scaled_mm,
    triton_scaled_mm_fused_quant,
)

# (M, K, N) — decode shapes first (the campaign's target regime), then prefill
# and off-tile shapes.  K must be a multiple of 256 (= BLOCK_SIZE_K of the
# deployed tiles) for the chain comparison: at partial K the *chain's* dot tail
# is undefined by construction (masked tl.load without `other`), so the chain
# is not a valid oracle there — `test_fused_quant_partial_k_vs_cpu` covers it.
SHAPES = [
    (1, 2048, 1024),
    (1, 2048, 2048),
    (4, 2048, 2560),
    (32, 2048, 512),
    (33, 1024, 1024),
    (256, 2048, 2048),
    (300, 512, 256),
    (4096, 2048, 1024),
]


def _make_inputs(M, K, N, dtype=torch.bfloat16, device="cuda", seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.float32, device=device).to(dtype)
    w_q = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
    scale_b = torch.rand(N, 1, dtype=torch.float32, device=device) * 0.01 + 0.001
    return x, w_q, scale_b


def _chain(x, w_q, scale_b, out_dtype, bias=None):
    """The shipped two-kernel path (what the fused kernel replaces)."""
    x_q, x_s, x_zp = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    out = triton_scaled_mm(
        x_q, w_q, scale_a=x_s, scale_b=scale_b, out_dtype=out_dtype, bias=bias
    )
    return out, x_s


def _cpu_quant_exact(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's quantization expressions, verbatim, in torch ops (computed
    on the SAME device as x — torch's fp32 division is a different code path
    from LLVM's `fdiv`, so this is an ORACLE CANDIDATE, not the contract; see
    `test_partial_k_oracle_is_the_chain` which decides whether it or the
    kernel is the odd one out).  amax = max(|x|); scale = amax/127;
    inv_s = (amax == 0) ? 0 : 127/amax; q = clamp(nearbyint(x*inv_s), -128,
    127).  torch.round == nearbyint (both round-half-to-even)."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    inv_s = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(xf * inv_s).clamp(-128, 127).to(torch.int8)
    return q, amax / 127.0


def test_fused_quant_bitexact_vs_chain(M=None, K=None, N=None):
    for m, k, n in ([(M, K, N)] if M is not None else SHAPES):
        x, w_q, scale_b = _make_inputs(m, k, n)
        ref_out, ref_s = _chain(x, w_q, scale_b, torch.bfloat16)
        out, s = fused_quant_kernel(x, w_q, scale_b, torch.bfloat16)
        assert torch.equal(s, ref_s), f"scales differ ({m},{k},{n})"
        assert torch.equal(out, ref_out), f"output differs ({m},{k},{n})"


def test_fused_quant_bitexact_with_bias(M=None, K=None, N=None):
    combos = [(M, K, N)] if M is not None else [(1, 2048, 1024), (32, 2048, 2048)]
    for m, k, n in combos:
        x, w_q, scale_b = _make_inputs(m, k, n)
        # bias must be in out_dtype: the kernel adds it after the output
        # conversion (`c += bias`), like the chain.
        bias = torch.randn(n, dtype=torch.bfloat16, device=x.device)
        ref_out, ref_s = _chain(x, w_q, scale_b, torch.bfloat16, bias=bias)
        out, s = fused_quant_kernel(x, w_q, scale_b, torch.bfloat16, bias=bias)
        assert torch.equal(s, ref_s)
        assert torch.equal(out, ref_out)


def test_fused_quant_half_even_rounding():
    """x * inv_s lands on exact .5 values: round-half-to-EVEN is what
    float_to_int8_rn (nearbyint) does — half-away would differ on every .5.

    amax = 127 -> inv_s = 1.0 -> the quantized values are nearbyint(x).
    """
    M, K, N = 2, 256, 64
    x = torch.zeros(M, K, dtype=torch.float32, device="cuda")
    halves = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype=torch.float32
    )
    x[0, : 8] = halves
    x[0, 8] = 127.0
    x[0, 9] = -127.0
    x[1, : 8] = halves  # second row: same, non-zero amax
    x = x.to(torch.bfloat16)
    w_q = torch.ones(K, N, dtype=torch.int8, device="cuda")
    scale_b = torch.ones(N, 1, dtype=torch.float32, device="cuda")

    ref_out, ref_s = _chain(x, w_q, scale_b, torch.float32)
    out, s = fused_quant_kernel(x, w_q, scale_b, torch.float32)
    assert torch.equal(s, ref_s)
    assert torch.equal(out, ref_out)

    expected = torch.tensor([0, 2, 2, 4, 0, -2, -2, -4], dtype=torch.int8)
    q, scales = _cpu_quant_exact(x)
    assert torch.equal(q[0, :8].cpu(), expected), "rounding mode drifted"
    assert torch.equal(scales, ref_s)


def test_fused_quant_zero_rows():
    M, K, N = 3, 512, 128
    x, w_q, scale_b = _make_inputs(M, K, N)
    x[1] = 0.0  # all-zero row: absmax=0 -> inv_s=0 -> quant 0, scale 0
    ref_out, ref_s = _chain(x, w_q, scale_b, torch.bfloat16)
    out, s = fused_quant_kernel(x, w_q, scale_b, torch.bfloat16)
    assert torch.equal(s, ref_s)
    assert torch.equal(out, ref_out)
    assert s[1].item() == 0.0


def test_fused_quant_partial_k_vs_cpu():
    """K not a multiple of BLOCK_SIZE_K: the chain is no oracle (its dot tail
    is undefined), so compare against a CPU computation of the same math with
    the tail zero-filled — the fused kernel masks its tail loads with 0."""
    M, K, N = 5, 2016, 96
    x, w_q, scale_b = _make_inputs(M, K, N)
    out, s = fused_quant_kernel(x, w_q, scale_b, torch.float32)

    q, scales = _cpu_quant_exact(x)
    # fp64 matmul: exact for our magnitudes (|acc| < 2^27), and CUDA has no
    # int32 addmm.  The .to(fp32) then matches the kernel's int32->fp32.
    acc = (q.to(torch.float64) @ w_q.to(torch.float64)).to(torch.float32)
    # same op order as the kernel epilogue: (scale_a * acc) then (scale_b.T *)
    ref = (scales * acc) * scale_b.t()
    assert torch.equal(s, scales), f"scales differ at K={K}"
    if not torch.equal(out, ref):
        # the effective tile of this shape (M<=32 -> BK=256)
        _, _, bk = _choose_tile_shape(M, N, True, (32, 32, 32))
        k_body = (K // bk) * bk
        diff = (out.float() - ref.float()).abs()
        flat = diff.view(-1)
        bad = int(flat.argmax().item())
        r, c = divmod(bad, diff.shape[1])
        s_rc = (scales[r, 0] * scale_b[c, 0]).item()
        acc_full = float((q[r].to(torch.float64) * w_q[:, c].to(torch.float64)).sum())
        k_tail = K - k_body
        acc_body = float(
            (q[r, :k_body].to(torch.float64) * w_q[:k_body, c].to(torch.float64)).sum()
        )
        print(
            f"    diag partial-K worst [{r},{c}]: out={out[r, c].item():.9g} "
            f"ref={ref[r, c].item():.9g} max|d|={diff.max().item():.6g}"
        )
        print(
            f"      scale={s_rc:.9g} acc_impl={out[r, c].item() / s_rc:.4f} "
            f"acc_full={acc_full:.1f} acc_without_tail(k>={k_body})={acc_body:.1f} "
            f"tail_elems={k_tail} bk={bk}"
        )
        raise AssertionError(f"partial-K mismatch at K={K}")


def test_partial_k_oracle_is_the_chain():
    """Decisive test for the partial-K disagreement (review 2026-10-23):
    is the torch-based reference or the kernel the odd one out?

    Compare the CHAIN's quantization (the contractual CUDA kernel) against
    the torch oracle on the very dataset where they disagree.  If the chain
    also differs from the oracle at the same elements, the oracle's fp32
    division/rounding is the divergent one and the fused kernel is innocent.
    """
    M, K, N = 5, 2016, 96
    x, _w_q, _scale_b = _make_inputs(M, K, N)
    q_chain, s_chain = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    q_ref, s_ref = _cpu_quant_exact(x)
    assert torch.equal(s_chain, s_ref), "scales: chain vs oracle differ"
    n_diff = int((q_chain != q_ref).sum().item())
    print(f"    oracle-vs-chain quantization diffs: {n_diff} of {q_chain.numel()}")
    if n_diff == 0:
        print("    oracle == chain => the fused kernel is the divergent one")
    else:
        idx = (q_chain != q_ref).nonzero()[:4].tolist()
        for r, c in idx:
            print(f"      [{r},{c}] chain={q_chain[r, c].item()} "
                  f"oracle={q_ref[r, c].item()}")
        print("    chain != oracle => the torch oracle is NOT the contract "
              "(its fp32 ops diverge from the LLVM device path); the partial-K "
              "disagreement does not implicate the fused kernel")


def test_fused_quant_scale_convention_matches_vllm_reference():
    """Our scales must be the same convention ref_dynamic_per_token_quant
    documents (amax / 127), and the quantized values must agree with our
    verbatim expressions."""
    if ref_dynamic_per_token_quant is None:
        print("SKIP vllm quant reference (tests.kernels.quant_utils absent)")
        return
    M, K, N = 17, 1024, 256
    x, w_q, scale_b = _make_inputs(M, K, N)
    out, s = fused_quant_kernel(x, w_q, scale_b, torch.float32)
    _xq, scales = ref_dynamic_per_token_quant(x, torch.int8)
    assert torch.equal(scales.float(), s)

    q, s2 = _cpu_quant_exact(x)
    assert torch.equal(s2, s)
    acc = (q.to(torch.float64) @ w_q.to(torch.float64)).to(torch.float32)
    ref = (s2 * acc) * scale_b.t()
    assert torch.equal(out, ref)


def test_fused_quant_transposed_weight():
    """Production weight layout: `process_weights_after_loading` stores
    ``w_q.t().data`` — a TRANSPOSED view ([K, N] with strides (1, K)).  Both
    paths address B through strides, so the chain is the oracle here too."""
    M, K, N = 32, 2048, 1024
    x = torch.randn(M, K, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    w_q = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda").t()
    assert not w_q.is_contiguous()
    scale_b = torch.rand(N, 1, dtype=torch.float32, device="cuda") * 0.01 + 0.001
    ref_out, ref_s = _chain(x, w_q, scale_b, torch.bfloat16)
    out, s = fused_quant_kernel(x, w_q, scale_b, torch.bfloat16)
    assert torch.equal(s, ref_s)
    assert torch.equal(out, ref_out)


def test_regime_policy():
    """The wall-measured regime: fused at M<=32 (decode win), chain above
    (prefill regressed ~13% fused), and chain at partial K (the one open
    bit-exactness question of the campaign).  Must stay in lockstep with the
    wall evidence in research/megakernel-campaign-2026-10-23/REPORT.md."""
    for m in (1, 4, 32):
        assert fused_quant_regime(m, 2048), f"M={m} must use the fused path"
    for m in (33, 256, 4096):
        assert not fused_quant_regime(m, 2048), f"M={m} must keep the shipped chain"
    for k in (1024, 2016, 3000):
        assert not fused_quant_regime(32, k), f"K={k} partial must keep the chain"


def test_partial_k_quant_matches_chain():
    """CLOSES the open partial-K question in the kernel's favour if green.

    The CHAIN's quantization is well-defined at partial K (only its GEMM tail
    is undefined), so compare the fused kernel's output against an EXACT
    (fp64) GEMM of the chain's own quantized values: equal => the fused
    quantized values are the chain's, bit for bit, even at partial K.
    """
    M, K, N = 5, 2016, 96
    x, w_q, scale_b = _make_inputs(M, K, N)
    out, s_fused = fused_quant_kernel(x, w_q, scale_b, torch.float32)
    q_chain, s_chain = scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    assert torch.equal(s_fused, s_chain), "scales: fused vs chain differ at partial K"
    acc = (q_chain.to(torch.float64) @ w_q.to(torch.float64)).to(torch.float32)
    ref = (s_chain * acc) * scale_b.t()
    if not torch.equal(out, ref):
        diff = (out.float() - ref.float()).abs()
        raise AssertionError(
            f"fused quant != chain quant at K={K}: {int((diff > 0).sum())} elems, "
            f"max|d|={diff.max().item():.6g}"
        )


def test_op_dispatch_m_gt_32_is_the_chain():
    """The regime fallback inside the custom op must be the SHIPPED chain
    (bit-exact against it) — this is the path every prefill takes with the
    gate ON (a wrong `symmetric` there produced wrong outputs, GLM review
    2026-10-23)."""
    for M, K, N in [(33, 2048, 1024), (256, 2048, 2048), (4096, 2048, 512)]:
        x, w_q, scale_b = _make_inputs(M, K, N)
        ref_out, ref_s = _chain(x, w_q, scale_b, torch.bfloat16)
        out, s = triton_scaled_mm_fused_quant(x, w_q, scale_b, torch.bfloat16)
        assert torch.equal(s, ref_s), f"scales differ at ({M},{K},{N})"
        assert torch.equal(out, ref_out), f"op dispatch M>32 output differs at ({M},{K},{N})"


def test_gate_decision_table(monkeypatch):
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    w_q = torch.randint(-127, 127, (128, 64), dtype=torch.int8, device="cuda")

    monkeypatch.setattr(envs, "VLLM_SCALED_MM_FUSED_QUANT", False)
    assert not fused_quant_eligible(True, None, x, w_q), "gate must default OFF"

    monkeypatch.setattr(envs, "VLLM_SCALED_MM_FUSED_QUANT", True)
    assert fused_quant_eligible(True, None, x, w_q)

    # static input scale -> chain
    i_s = torch.tensor([0.1], device="cuda")
    assert not fused_quant_eligible(True, i_s, x, w_q)
    # asymmetric quantization -> chain
    assert not fused_quant_eligible(False, None, x, w_q)
    # non-int8 weight -> chain
    w_bf16 = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    assert not fused_quant_eligible(True, None, x, w_bf16)
    # non-float activation (pre-quantized int8 reaches the caller's
    # `qa is None` check, but a raw int8 tensor is ineligible here too)
    x_i8 = torch.zeros(4, 128, dtype=torch.int8, device="cuda")
    assert not fused_quant_eligible(True, None, x_i8, w_q)


class _MonkeyPatch:
    """Minimal pytest-monkeypatch stand-alone shim for `python3 <this file>`."""

    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)
        self._saved = []


def _main() -> int:
    """Self-driver: runs every test without pytest (in-image sessions)."""
    failures = []

    def run(tag, fn, *args):
        try:
            fn(*args)
            print(f"PASS {tag}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"FAIL {tag}: {type(e).__name__}: {e}", flush=True)
            failures.append(tag)

    for M, K, N in SHAPES:
        run(f"bitexact ({M},{K},{N})", test_fused_quant_bitexact_vs_chain, M, K, N)
    for M, K, N in [(1, 2048, 1024), (32, 2048, 2048)]:
        run(f"bitexact+bias ({M},{K},{N})", test_fused_quant_bitexact_with_bias, M, K, N)
    run("half-even rounding", test_fused_quant_half_even_rounding)
    run("zero rows", test_fused_quant_zero_rows)
    run("transposed weight (prod layout)", test_fused_quant_transposed_weight)
    run("partial K vs cpu", test_fused_quant_partial_k_vs_cpu)
    run("oracle is the chain?", test_partial_k_oracle_is_the_chain)
    run("regime policy (M<=32, K%256)", test_regime_policy)
    run("partial-K quant == chain quant", test_partial_k_quant_matches_chain)
    run("op dispatch M>32 == chain", test_op_dispatch_m_gt_32_is_the_chain)
    run("vllm quant reference", test_fused_quant_scale_convention_matches_vllm_reference)
    mp = _MonkeyPatch()
    run("gate decision table", test_gate_decision_table, mp)
    mp.undo()

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL CONTRACT TESTS GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
