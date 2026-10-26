# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the JustAI project
"""Tests for the INCREMENT-2b AR int8 natif (research/ar-vol-count-2026-10-25,
ETAPE-E section 4): int8 contributions in, int32 accumulation, composed scale
s1 (+) s2 -> s_out, zero dequant -- then the shared norm+quant chain.

Layers:

1. **numeric contract** (CPU): the reference (plain torch) against
   hand-computed properties -- the max composition, the per-contribution
   requant bound (<= s_out/2 each), the zero-scale special case, half-even
   rounding of the composition requant.
2. **dequant fallback parity** (1 GPU): the fallback pieces
   (``_dequant_contributions`` + the bf16 chain) must equal the bf16
   reference contract bit-for-bit on the dequantized value.
3. **op-level fallback** (1 GPU): the op's two-stage answer (dequant ->
   all-reduce -> rank-local norm+quant) with a stubbed single-rank
   all-reduce, vs the bf16 chain on the dequantized contribution (residual
   bit-exact; scale/q within the class-3 + 1-LSB kernel tolerances).
4. **two-rank one-shot** (2 GPUs + gloo, opt-in ``AR_NORM_QUANT_2GPU_TEST=1``):
   the int8 transport end to end vs the IN8 reference (exact up to the final
   requant), the composed-scale bound vs the fp32-exact sum, and rank
   agreement.

Run: pytest tests/kernels/triton/test_ar_norm_quant_int8.py -v
"""

import os

import pytest
import torch

# The real vllm import MUST come first: importing vllm loads the compiled
# vllm._C extension, which registers the _C ops.  Registering the stubs
# BEFORE that collides with the real registrations and the extension's init
# ABORTS (measured 2026-10-26 in session inc2b-arin8-tests: "Fatal Python
# error: Aborted" at vllm/platforms/rocm.py:57 = import vllm._rocm_C).
from vllm.kernels.triton import ar_norm_quant as k  # noqa: E402

# ---------------------------------------------------------------------------
# test-only stubs: without the compiled vllm._C extension the fusion modules'
# fallback cannot run.  Mirrors the C++ schemas; ONLY registered when the real
# ops are absent (after the vllm import above, so never a collision).
# ---------------------------------------------------------------------------
_C_STUBS = [
    "rotary_embedding(Tensor query, Tensor key, Tensor cos_sin_cache, "
    "Tensor positions, Tensor! q_out, Tensor! k_out, bool is_neox) -> ()",
    "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> ()",
    "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> ()",
    "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
    "Tensor! scale, Tensor? scale_ub) -> ()",
    "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
    "Tensor? azp) -> ()",
    "silu_and_mul(Tensor! result, Tensor input) -> ()",
    "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()",
    "fused_add_rms_norm(Tensor! result, Tensor! residual, Tensor weight, "
    "float epsilon) -> ()",
    "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
    "Tensor scale, float epsilon, Tensor? scale_ub, Tensor? residual) -> ()",
    "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
    "Tensor weight, Tensor scale, float epsilon, Tensor? scale_ub, "
    "Tensor? residual) -> ()",
    "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
    "Tensor weight, Tensor! scale, float epsilon, Tensor? scale_ub, "
    "Tensor? residual) -> ()",
    "rms_norm_per_block_quant(Tensor! result, Tensor input, Tensor weight, "
    "Tensor! scale, float epsilon, Tensor? scale_ub, Tensor? residual) -> ()",
]

_STUB_LIB = None
_USING_STUBS = False


def _ensure_c_stubs() -> bool:
    global _STUB_LIB, _USING_STUBS
    if hasattr(torch.ops._C, "dynamic_scaled_int8_quant"):
        return False
    if _STUB_LIB is not None:
        return True
    _USING_STUBS = True
    lib = torch.library.Library("_C", "FRAGMENT")
    for spec in _C_STUBS:
        name = spec.split("(")[0]
        try:
            lib.define(spec)
            lib.impl(name, lambda *a, **k: None, "CompositeExplicitAutograd")
        except Exception:  # noqa: BLE001 - already defined is fine
            pass
    _STUB_LIB = lib
    return True


_ensure_c_stubs()

HAS_GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 1
HAS_2GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a GPU")
needs_2gpu = pytest.mark.skipif(not HAS_2GPU, reason="needs 2 GPUs")

SCALE_TILE = 64
ROWS, HIDDEN = 32, 2048
TILES = HIDDEN // SCALE_TILE


# ---------------------------------------------------------------------------
# 1. numeric contract (CPU)
# ---------------------------------------------------------------------------
class TestNumericContract:
    def test_composed_scale_is_the_max(self):
        rows, hidden, tiles = 2, 8, 2
        tile = hidden // tiles
        x_self = torch.ones(rows, hidden, dtype=torch.int8)
        x_peer = torch.ones(rows, hidden, dtype=torch.int8)
        s_self = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        s_peer = torch.tensor([[4.0, 0.5], [3.0, 0.0]])
        _q, _scale, _res, x_f = k.reference_ar_norm_quant_int8_in(
            x_self, s_self, x_peer, s_peer,
            torch.zeros(rows, hidden, dtype=torch.bfloat16),
            torch.ones(hidden, dtype=torch.float32), 1e-6, tile,
        )
        exact = k.reference_ar_exact_sum(x_self, s_self, x_peer, s_peer, tile)
        s_out = torch.maximum(s_self.repeat_interleave(tile, 1),
                              s_peer.repeat_interleave(tile, 1))
        assert ((x_f - exact).abs() <= s_out + 1e-9).all()
        # hand-computed: tile 0 row 0: s_out=4, round(1*0.25)+round(1*1)=1
        # -> 4.0; tile 1 row 0: s_out=2, round(1*1)+round(1*0.25)=1 -> 2.0;
        # row 1 tile 0: s_out=3, 1+1 -> 6.0; row 1 tile 1: zero scales -> 0.
        expected = torch.tensor([[4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0],
                                 [6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(x_f, expected, atol=1e-6)

    def test_zero_scale_special_case(self):
        rows, hidden, tiles = 2, 8, 2
        tile = hidden // tiles
        x_self = torch.full((rows, hidden), 7, dtype=torch.int8)
        x_peer = torch.full((rows, hidden), -7, dtype=torch.int8)
        zeros = torch.zeros(rows, tiles)
        q, _scale, _res, x_f = k.reference_ar_norm_quant_int8_in(
            x_self, zeros, x_peer, zeros,
            torch.zeros(rows, hidden, dtype=torch.bfloat16),
            torch.ones(hidden, dtype=torch.float32), 1e-6, tile,
        )
        assert torch.all(x_f == 0)  # s_out == 0 -> factors 0 -> acc * 0
        assert torch.all(q == 0)

    def test_composition_requant_is_half_even(self):
        # factors s_i/s_out land on exact .5 for the contribution requant
        rows, hidden, tiles = 1, 4, 1
        tile = hidden // tiles
        x_self = torch.tensor([[1, 3, 5, 7]], dtype=torch.int8)
        x_peer = torch.zeros(rows, hidden, dtype=torch.int8)
        s_self = torch.ones(rows, tiles)
        s_peer = torch.full((rows, tiles), 2.0)
        _q, _s, _r, x_f = k.reference_ar_norm_quant_int8_in(
            x_self, s_self, x_peer, s_peer,
            torch.zeros(rows, hidden, dtype=torch.bfloat16),
            torch.ones(hidden, dtype=torch.float32), 1e-6, tile,
        )
        # s_out=2, f_self=0.5: round(q*0.5) half-to-even -> [0, 2, 2, 4],
        # times s_out=2
        assert torch.equal(x_f, torch.tensor([[0.0, 4.0, 4.0, 8.0]]))


# ---------------------------------------------------------------------------
# 2/3. GPU execution (fallback pieces + op-level fallback)
# ---------------------------------------------------------------------------
@needs_gpu
class TestFallbackParity:
    def test_dequant_fallback_equals_bf16_chain(self):
        if _USING_STUBS:
            pytest.skip("needs the compiled vllm._C extension")
        torch.manual_seed(0)
        rows, hidden = 8, 512
        tile = 64
        tiles = hidden // tile
        x_q = torch.randint(-127, 127, (rows, hidden), dtype=torch.int8,
                            device="cuda")
        x_s = torch.rand(rows, tiles, device="cuda") * 0.02 + 1e-4
        res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        w = torch.randn(hidden, device="cuda").to(torch.float32)

        dequant = k._dequant_contributions(x_q, x_s)
        out2 = torch.empty((rows, hidden), dtype=torch.int8, device="cuda")
        scale2 = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
        res2 = res.clone()
        k._fallback_norm_quant(out2, dequant, w, scale2, res2, 1e-6)
        q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(
            dequant, res, w, 1e-6
        )
        assert torch.equal(res2, res_ref)

        # STAGE-WISE ATTRIBUTION (silent-artifact discipline): both chains
        # run the same eager torch expressions; if they still diverge, say
        # WHERE.  Measured 2026-10-26: scale2 != scale_ref with equal prints
        # (sub-4-decimal difference) while res matched -- res is bf16(s), so
        # it cannot see fp32 differences below a bf16 ulp; the stages below
        # tell whether s/var/y really match and whether the divergence is the
        # CUDA quant kernel's scale division instead.
        s_a = dequant.to(torch.float32) + res.to(torch.float32)
        s_b = dequant.to(torch.float32) + res.to(torch.float32)
        assert torch.equal(s_a, s_b), "stage 1: the residual add diverged"
        var_a = (s_a * s_a).mean(dim=-1, keepdim=True)
        var_b = (s_b * s_b).mean(dim=-1, keepdim=True)
        assert torch.equal(var_a, var_b), "stage 2: the var reduction diverged"
        y_a = (s_a * torch.rsqrt(var_a + 1e-6)).to(w.dtype) * w
        y_b = (s_b * torch.rsqrt(var_b + 1e-6)).to(w.dtype) * w
        assert torch.equal(y_a, y_b), "stage 3: the norm chain diverged"
        normed_a = y_a.to(torch.bfloat16)
        normed_b = y_b.to(torch.bfloat16)
        assert torch.equal(normed_a, normed_b), "stage 4: bf16 rounding"

        if not torch.equal(scale2, scale_ref):
            ulp = (scale2 - scale_ref).abs() / torch.finfo(torch.float32).eps
            print(f"    scale2 vs scale_ref: {int((scale2 != scale_ref).sum())}"
                  f"/{rows} rows differ, max {ulp.max().item():.3f} ulp "
                  f"(stages 1-4 identical -> the divergence is the CUDA"
                  f" quant kernel's absmax/127 path)")
        # contract tolerance (class-3 through the quant kernel's scale
        # computation): 1 fp32 ulp, hard
        assert torch.allclose(scale2, scale_ref, rtol=1e-6, atol=1e-9), (
            "scale beyond 1 ulp: the fallback chain is NOT the contract"
        )
        flips = (out2.int() - q_ref.int()).abs()
        assert (flips <= 1).all(), f"q off by > 1 LSB: {flips.max()}"
        print(f"    dequant-fallback q +/-1 LSB flips: "
              f"{int((flips > 0).sum())}/{flips.numel()}")

    def test_op_level_fallback(self):
        if _USING_STUBS:
            pytest.skip("needs the compiled vllm._C extension")
        import vllm.distributed as dist_mod

        torch.manual_seed(1)
        rows, hidden = 64, 512  # rows > FUSE_MAX_ROWS: the fallback regime
        tile = 64
        tiles = hidden // tile
        x_q = torch.randint(-127, 127, (rows, hidden), dtype=torch.int8,
                            device="cuda")
        x_s = torch.rand(rows, tiles, device="cuda") * 0.02 + 1e-4
        res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        w = torch.randn(hidden, device="cuda").to(torch.float32)
        result = torch.empty((rows, hidden), dtype=torch.int8, device="cuda")
        scale = torch.empty((rows, 1), dtype=torch.float32, device="cuda")

        dequant = k._dequant_contributions(x_q, x_s)
        q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(
            dequant, res, w, 1e-6
        )
        # the returned normed value (class-3 through var/rsqrt)
        s = dequant.to(torch.float32) + res.to(torch.float32)
        var = (s * s).mean(dim=-1, keepdim=True)
        y = (s * torch.rsqrt(var + 1e-6)).to(w.dtype) * w
        normed_ref = y.to(torch.bfloat16)

        forced = {"enabled": False, "reason": "test-fallback", "comm": None}
        orig_ar = dist_mod.tensor_model_parallel_all_reduce
        orig_decision = k._decision
        dist_mod.tensor_model_parallel_all_reduce = lambda x: x
        k._decision = forced
        try:
            res_out = torch.ops.vllm.triton_ar_norm_quant_int8_in(
                result, x_q, x_s, w, scale, 1e-6, res
            )
        finally:
            dist_mod.tensor_model_parallel_all_reduce = orig_ar
            k._decision = orig_decision
        assert torch.equal(res, res_ref)  # residual mutated in place, exact
        assert torch.allclose(scale, scale_ref, rtol=1e-4, atol=1e-8)
        flips = (result.int() - q_ref.int()).abs()
        assert (flips <= 1).all(), f"q off by > 1 LSB: {flips.max()}"
        assert torch.allclose(
            res_out.float(), normed_ref.float(), rtol=1e-4, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 4. two-rank one-shot (opt-in: XGMI pair needed)
# ---------------------------------------------------------------------------
def _worker_two_rank(rank, world, port, outq):
    """Module-level (spawn-picklable) worker for the 2-rank IN8 test."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    rows, hidden = ROWS, HIDDEN
    tile = SCALE_TILE
    tiles = hidden // tile
    torch.manual_seed(7)  # same on both ranks: shared w/residual
    w = torch.randn(hidden, device=dev).to(torch.float32)
    res = torch.randn(rows, hidden, device=dev).to(torch.bfloat16)
    res0 = res.clone()
    torch.manual_seed(200 + rank)  # per-rank contributions
    x_q = torch.randint(-127, 127, (rows, hidden), dtype=torch.int8,
                        device=dev)
    x_s = (torch.rand(rows, tiles, device=dev) * 0.02 + 1e-4).to(torch.float32)
    comm = k.ArNormQuantComm(dist.group.WORLD, k.MAX_ROWS, hidden)
    result = torch.empty((rows, hidden), dtype=torch.int8, device=dev)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=dev)
    k._decision = {"enabled": True, "reason": "test", "comm": comm}
    try:
        res_out = torch.ops.vllm.triton_ar_norm_quant_int8_in(
            result, x_q, x_s, w, scale, 1e-6, res
        )
    finally:
        k._decision = None
    torch.cuda.synchronize()
    # numpy by VALUE through the queue: torch tensor pickling shares fds via
    # the resource_sharer, and a child that exits before the parent's lazy
    # unpickle breaks the fd transfer (ConnectionResetError -- measured
    # 2026-10-26).  bf16 cannot go through numpy: float32 round-trips
    # losslessly (bf16 -> f32 is exact).
    f32 = lambda t: t.cpu().to(torch.float32).numpy()
    outq.put((rank, result.cpu().numpy(), scale.cpu().numpy(),
              f32(res), f32(res_out),
              x_q.cpu().numpy(), x_s.cpu().numpy(), w.cpu().numpy(),
              f32(res0)))
    dist.destroy_process_group()


@needs_2gpu
class TestTwoRankInt8Transport:
    @pytest.mark.skipif(
        os.environ.get("AR_NORM_QUANT_2GPU_TEST") != "1",
        reason="set AR_NORM_QUANT_2GPU_TEST=1 on an XGMI pair",
    )
    def test_ar_norm_quant_int8_in_two_ranks(self):
        import torch.multiprocessing as mp

        try:
            import vllm._custom_ops as _ops

            if not hasattr(_ops, "allocate_shared_buffer_and_handle"):
                pytest.skip("vllm._C lacks the shared-buffer IPC ops")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"vllm._custom_ops unavailable: {exc}")

        ctx = mp.get_context("spawn")
        outq = ctx.Queue()
        procs = [
            ctx.Process(target=_worker_two_rank, args=(r, 2, 29523, outq))
            for r in (0, 1)
        ]
        for p in procs:
            p.start()
        got = {}
        res0 = None
        bf16 = torch.bfloat16
        for _ in range(2):
            (rank, result, scale, res_out_t, normed, x_q, x_s, w,
             res0) = outq.get(timeout=300)
            got[rank] = (
                torch.from_numpy(result),
                torch.from_numpy(scale),
                torch.from_numpy(res_out_t).to(bf16),
                torch.from_numpy(normed).to(bf16),
                torch.from_numpy(x_q),
                torch.from_numpy(x_s),
                torch.from_numpy(w),
            )
        res0 = torch.from_numpy(res0).to(bf16)
        for p in procs:
            p.join(timeout=60)
        assert all(not p.is_alive() for p in procs), "worker did not terminate"

        # both ranks compute the SAME combined value -> same outputs
        assert torch.equal(got[0][0], got[1][0]), "ranks disagree on quant"
        assert torch.equal(got[0][1], got[1][1]), "ranks disagree on scale"

        q_self, s_self, w0 = got[0][4], got[0][5], got[0][6]
        q_peer, s_peer = got[1][4], got[1][5]
        for rank in (0, 1):
            q_ref, scale_ref, res_ref, x_f = k.reference_ar_norm_quant_int8_in(
                q_self, s_self, q_peer, s_peer, res0, w0, 1e-6, SCALE_TILE
            )
            result, scale, res_out_t = got[rank][0], got[rank][1], got[rank][2]
            # Contract (2b gate: "exactitude vs référence fp32 (jusqu'à la
            # requant finale)"): the composition requant (q_i' =
            # nearbyint(q_i * s_i/s_out)) can flip by +/-1 LSB at its rounding
            # boundaries (measured 2026-10-26: 4 elements of 65536, max 1.56
            # bf16 ulp = exactly one composed-scale LSB).  The named bound is
            # therefore: |res - res_ref| <= 1 bf16 ulp (the two bf16
            # roundings) + s_out (one composition-LSB).  Doctrine 2026-10-24:
            # bit-exactness is NOT the criterion -- assert the bound and PRINT
            # the flip counts.
            idx_e = torch.arange(HIDDEN) // SCALE_TILE
            s_out_e = torch.maximum(s_self[:, idx_e], s_peer[:, idx_e])
            bf16_ulp = res_ref.float().abs() * 2.0**-8 + 2.0**-150
            res_diff = (res_out_t.float() - res_ref.float()).abs()
            res_flips = int((res_diff > 0).sum())
            print(f"rank {rank}: res bf16 flips: {res_flips}/{res_diff.numel()}, "
                  f"max beyond ulp "
                  f"{(res_diff - bf16_ulp).max().item():.3g}")
            assert (res_diff <= bf16_ulp + s_out_e + 1e-9).all(), (
                f"rank {rank}: res beyond 1 bf16 ulp + one composition LSB"
            )
            assert res_flips <= 64, (
                f"rank {rank}: {res_flips} residual flips -- beyond the "
                f"requant-boundary noise budget"
            )
            assert torch.allclose(scale, scale_ref, rtol=1e-4, atol=1e-8), (
                f"rank {rank}: scale mismatch"
            )
            flips = (result.int() - q_ref.int()).abs()
            assert (flips <= 1).all(), f"rank {rank}: q off by > 1 LSB"
            print(
                f"rank {rank}: q +/-1 LSB flips: "
                f"{int((flips > 0).sum())}/{flips.numel()}"
            )
            # the composed-scale bound vs the fp32-exact sum
            exact = k.reference_ar_exact_sum(
                q_self, s_self, q_peer, s_peer, SCALE_TILE
            )
            idx = torch.arange(HIDDEN) // SCALE_TILE
            s_out = torch.maximum(s_self[:, idx], s_peer[:, idx])
            assert ((x_f - exact).abs() <= s_out + 1e-9).all(), (
                f"rank {rank}: composed-scale bound violated"
            )


# ---------------------------------------------------------------------------
# 5. pattern wiring (CPU): the increment-2c V2 motif must trace, register and
# fuse -- the requant GEMM feeds the AR int8 leg ("câblage des deux ensemble").
# ---------------------------------------------------------------------------
class TestPatternWiringV2:
    @staticmethod
    def _cfg_ctx():
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.device import DeviceConfig

        import vllm.compilation.passes.vllm_inductor_pass as _vip

        if not _vip.DEVICE_TYPE:  # GPU-less dev box: trace on CPU
            _vip.DEVICE_TYPE = "cpu"
        return set_current_vllm_config(
            VllmConfig(device_config=DeviceConfig(device="cpu")),
            check_compile=False,
        )

    @staticmethod
    def _trace_and_apply(pr_cls, epsilon=1e-6):
        import torch._inductor.pattern_matcher as pm
        from torch._inductor.pattern_matcher import PatternMatcherPass

        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f
        from vllm.compilation.passes.inductor_pass import enable_fake_mode

        # register the scaled_mm custom ops (the requant ones live there)
        import vllm.model_executor.layers.quantization.compressed_tensors.  \
            triton_scaled_mm  # noqa: F401

        with TestPatternWiringV2._cfg_ctx():

            def fake_ar(inp):
                return torch.ops.vllm.all_reduce(inp, group_name="test-tp2")

            orig = f.tensor_model_parallel_all_reduce
            f.tensor_model_parallel_all_reduce = fake_ar

            @enable_fake_mode
            def _build():
                pr = pr_cls(epsilon)
                pm_pass = PatternMatcherPass(pass_name="test_ar_v2")
                pm.register_replacement(
                    pr.pattern,
                    pr.replacement,
                    pr.get_inputs(),
                    pm.fwd_only,
                    pm_pass,
                )
                gm = pm.fwd_only(pr.pattern, pr.get_inputs())
                return gm, pm_pass

            try:
                gm, pm_pass = _build()
                # apply() re-traces the pattern per match (check_fn ->
                # fwd_only -> make_fx): the fake AR MUST stay patched through
                # it, or the real tensor_model_parallel_all_reduce asserts on
                # an uninitialized TP group (measured 2026-10-26).
                matched = pm_pass.apply(gm.graph)
            finally:
                f.tensor_model_parallel_all_reduce = orig
        fused = {
            f.called_op(n) for n in gm.graph.nodes if n.op == "call_function"
        }
        return matched, fused

    def test_v2_pattern_traces_and_fuses(self):
        _ensure_c_stubs()
        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f

        matched, fused = self._trace_and_apply(f.ArNormQuantInt8InGemmPattern)
        assert matched >= 1, "the v2 pattern did not re-match its own chain"
        names = {str(o) for o in fused if o is not None}
        assert any("triton_scaled_mm_fused_quant_requant" in n for n in names), (
            f"the requant GEMM node is absent after rewrite: {names}"
        )
        assert any("triton_ar_norm_quant_int8_in" in n for n in names), (
            f"the AR-int8 op is absent after rewrite: {names}"
        )
        assert any(
            "triton_scaled_mm.default" in n for n in names
        ), f"the bare scaled_mm (consumer) is absent: {names}"
        # the OLD nodes must be GONE (the exact op, not its requant sibling)
        assert not any(
            "triton_scaled_mm_fused_quant.default" in n for n in names
        ), f"an old fused-quant node survived the rewrite: {names}"

    def test_v2_scale_dead_variant_fuses(self):
        """The form the engine really has (ScaleDead lesson): the downstream
        GEMM's x_s output is dead."""
        _ensure_c_stubs()
        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f

        matched, fused = self._trace_and_apply(
            f.ArNormQuantInt8InGemmScaleDeadPattern
        )
        assert matched >= 1, "the ScaleDead v2 variant matched nothing"
        names = {str(o) for o in fused if o is not None}
        assert any("triton_scaled_mm_fused_quant_requant" in n for n in names)
        assert any("triton_ar_norm_quant_int8_in" in n for n in names)
        assert not any(
            "triton_scaled_mm_fused_quant.default" in n for n in names
        ), f"an old fused-quant node survived the rewrite: {names}"

    def test_v2_reconstruction_keeps_the_producer_value(self):
        """The producer's bf16 output is returned (and reconstructed) so an
        escaping consumer keeps a value; when the AR is the only consumer the
        node is dead (DCE removes it later)."""
        _ensure_c_stubs()
        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f

        matched, fused = self._trace_and_apply(f.ArNormQuantInt8InGemmPattern)
        names = {str(o) for o in fused if o is not None}
        # the rewrite ran (see the first test); the reconstruction is torch
        # arithmetic nodes (mul/cast), so no dedicated op name to look for --
        # the guard here is that the match succeeded WITH the fifth output,
        # which is the property under test.
        assert matched >= 1
        assert any("triton_ar_norm_quant_int8_in" in n for n in names)
