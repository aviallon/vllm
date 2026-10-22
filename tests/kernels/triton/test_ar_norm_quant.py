# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton AR+norm+quant / norm+quant kernel (v2).

Layers, in order of how much machinery they need:

1. **numeric contract** (CPU, no GPU): the reference implementation of the
   executed-baseline contract (research/ar-rms-quant-v2-spec-2026-10-22.md
   section 3.2) against hand-computed values -- scale = absmax/127 without
   clamp, the absmax==0 special case, half-to-even ``rint`` (never
   half-away).
2. **gfx90a codegen** (needs triton only): a real ``triton_compile`` of the
   kernel for ``GPUTarget("hip", "gfx90a", 32)`` -- the check v1's
   ``-fsyntax-only`` faked. Catches "needs target feature" and similar
   codegen-time failures without a GPU.
3. **kernel execution** (needs 1 GPU): kernel vs reference. The quant path
   must be bit-exact on inputs where the fp32 variance sum is exact (integer
   rows); on random data only the reduction order of ``var`` may differ
   (class-3, ~1e-7) and at most a few ``q`` values may flip by +/-1 LSB --
   the flip COUNT is asserted, not just "close".
4. **fallback parity** (needs 1 GPU): the op's in-node fallback chain must
   equal the unfused native chain bit-for-bit.
5. **two-rank one-shot AR** (needs 2 GPUs + gloo): the barrier + peer-load
   data path of the 3-way, end to end, vs the peer-sum reference.
6. **pattern wiring** (CPU): the Inductor patterns trace, register and fuse a
   small chain graph -- the S1 "kernel passed through the motif" sub-test.

Run: pytest tests/kernels/triton/test_ar_norm_quant.py -v
"""

import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# test-only stubs: without the compiled vllm._C extension (CPU-only dev
# boxes, CI without a ROCm build) the fusion modules cannot import. The
# stubs mirror the C++ schemas and are ONLY registered when the real ops are
# absent -- on a real image the compiled ops are used.
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
    """Returns True if stubs were installed (no real vllm._C in this env)."""
    global _STUB_LIB, _USING_STUBS
    if hasattr(torch.ops._C, "dynamic_scaled_int8_quant"):
        return False  # real extension: nothing to stub
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
    # keep the Library object alive for the process lifetime
    _STUB_LIB = lib
    return True

from vllm.kernels.triton import ar_norm_quant as k  # noqa: E402

HAS_GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 1
HAS_2GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a GPU")
needs_2gpu = pytest.mark.skipif(not HAS_2GPU, reason="needs 2 GPUs")


# ---------------------------------------------------------------------------
# 1. numeric contract (CPU)
# ---------------------------------------------------------------------------
class TestNumericContract:
    def test_scale_is_absmax_over_127_without_clamp(self):
        # a row with a tiny absmax: the OTHER kernel's contract would clamp
        # scale to min_scaling_factor; ours must NOT.
        x = torch.full((1, 4), 1e-6, dtype=torch.bfloat16)
        res = torch.zeros_like(x)
        w = torch.ones(4, dtype=torch.float32)
        q, scale, _ = k.reference_norm_quant_int8(x, res, w, 0.0)
        # constant row -> y = w = 1 exactly -> amax = 1 -> scale = 1/127
        assert scale.item() == pytest.approx(1.0 / 127.0, rel=1e-6)
        assert q.abs().max().item() == 127

    def test_zero_row_special_case(self):
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        res = torch.zeros_like(x)
        w = torch.ones(8, dtype=torch.float32)
        q, scale, res_out = k.reference_norm_quant_int8(x, res, w, 1e-6)
        assert scale.item() == 0.0  # absmax / 127, no clamp, no epsilon floor
        assert torch.all(q == 0)  # inv_s = 0 -> rint(0 * 0) = 0
        assert torch.all(res_out == 0)

    def test_rint_is_half_to_even(self):
        # Build rows whose y_bf16 * inv_s lands exactly on x.5: x=0,
        # residual=[[1,1]] -> s=[1,1], var=1, rstd=1 -> y = w exactly.
        # With w=[v, 127] the amax is 127 and inv_s = 1 exactly, so
        # q[0] = rint(v) -- and rint is half-to-EVEN: 2.5->2, 3.5->4,
        # -2.5->-2. A half-away or half-up implementation fails these.
        for value, want in ((2.5, 2), (3.5, 4), (-2.5, -2), (0.5, 0), (1.5, 2)):
            x = torch.zeros(1, 2, dtype=torch.bfloat16)
            res = torch.ones(1, 2, dtype=torch.bfloat16)
            w = torch.tensor([value, 127.0], dtype=torch.float32)
            q, scale, _ = k.reference_norm_quant_int8(x, res, w, 0.0)
            assert q[0, 1].item() == 127
            got = q[0, 0].item()
            assert got == want, f"rint({value}) = {got}, expected {want}"

    def test_quant_saturates(self):
        # y_bf16 * inv_s is bounded by 127 by construction of inv_s, so the
        # saturating cast only guards fp32 rounding of the product.
        w = torch.tensor([127.0, 127.0], dtype=torch.float32)
        x = torch.zeros(1, 2, dtype=torch.bfloat16)
        res = torch.ones(1, 2, dtype=torch.bfloat16)
        q, scale, _ = k.reference_norm_quant_int8(x, res, w, 0.0)
        assert q.abs().max().item() <= 127
        assert q[0, 0].item() == 127


# ---------------------------------------------------------------------------
# 2. gfx90a codegen (the v1 "-fsyntax-only" lesson: real compilation)
# ---------------------------------------------------------------------------
class TestGfx90aCodegen:
    def test_kernel_compiles_for_gfx90a(self):
        triton = pytest.importorskip("triton")
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile

        sig = {
            "x_ptr": "*bf16",
            "res_ptr": "*bf16",
            "weight_ptr": "*fp32",
            "out_ptr": "*i8",
            "scale_ptr": "*fp32",
            "slot_ptr": "i64",
            "peer_ptr": "i64",
            "flag_ptr": "i64",
            "peer_flag_ptr": "i64",
            "eps": "fp32",
            "stride_x": "i32",
            "stride_res": "i32",
            "stride_out": "i32",
            "stride_slot": "i32",
        }
        for peer in (0, 1):
            src = ASTSource(
                fn=k._norm_quant_int8_kernel,
                signature=sig,
                constexprs={
                    "H": 2048,
                    "BLOCK_H": 2048,
                    "PEER": peer,
                    "FLAG_STRIDE": 32,
                },
            )
            compiled = triton_compile(
                src, target=GPUTarget("hip", "gfx90a", 32),
                options={"num_warps": 8},
            )
            asm = compiled.asm.get("amdgcn", "")
            assert len(asm) > 1000
            # the contract's rounding primitive: rint -> v_rndne_f32
            assert "v_rndne_f32" in asm
            if peer:
                # the barrier primitives must survive lowering
                assert "global_atomic" in asm


# ---------------------------------------------------------------------------
# 3/4/5. GPU execution
# ---------------------------------------------------------------------------
def _run_op(op, x, res, w, eps):
    rows, hidden = x.shape
    result = torch.empty((rows, hidden), dtype=torch.int8, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    op(result, x, w, scale, eps, res)
    return result, scale, res


@needs_gpu
class TestKernelExecution:
    def test_unit_variance_rows_are_bit_exact(self):
        # rows with |s| = 1 everywhere -> sum(s*s) = 2048 -> var = 1 exactly
        # in ANY reduction order, rstd = 1 exactly even on approximate
        # hardware rsqrt. So y, y_bf16, amax, scale, inv_s and q are exact:
        # the whole quant path must be BIT-EXACT vs the reference. This is
        # the S1 gate (i): "quant bit-exact given y identical".
        torch.manual_seed(0)
        rows, hidden = 80, 2048
        s = torch.randint(0, 2, (rows, hidden), device="cuda") * 2 - 1  # +-1
        res = s.to(torch.bfloat16)
        x = torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, device="cuda").to(torch.float32)
        out, scale, res_out = _run_op(k._norm_quant_int8_impl, x, res, w, 0.0)
        q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(x, res, w, 0.0)
        assert torch.equal(res_out.cpu(), res_ref.cpu()), "res_out not bit-exact"
        assert torch.equal(scale.cpu(), scale_ref.cpu()), "scale not bit-exact"
        assert torch.equal(out.cpu(), q_ref.cpu()), "quant not bit-exact"

    def test_random_rows_class3_and_flip_count(self):
        torch.manual_seed(1)
        rows, hidden = 16, 2048
        x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        w = torch.randn(hidden, device="cuda").to(torch.float32)
        out, scale, res_out = _run_op(k._norm_quant_int8_impl, x, res, w, 1e-6)
        q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(x, res, w, 1e-6)
        assert torch.equal(res_out.cpu(), res_ref.cpu())
        # scale: class-3 only (var reduction order may move y by ~1e-7 rel)
        assert torch.allclose(scale.cpu(), scale_ref.cpu(), rtol=1e-4, atol=1e-8)
        diff = (out.cpu().int() - q_ref.cpu().int()).abs()
        flips = int((diff > 0).sum())
        assert (diff <= 1).all(), f"q diverged by more than 1 LSB: {diff.max()}"
        # measured 2026-10-22 in the S1 gate spec: flips are rare; assert a
        # generous ceiling and archive the real count in the test output
        print(f"q +/-1 LSB flips: {flips}/{diff.numel()}")
        assert flips <= diff.numel() // 100

    def test_zero_row(self):
        x = torch.zeros(4, 256, dtype=torch.bfloat16, device="cuda")
        res = torch.zeros_like(x)
        w = torch.ones(256, dtype=torch.float32, device="cuda")
        out, scale, res_out = _run_op(k._norm_quant_int8_impl, x, res, w, 1e-6)
        assert torch.all(scale == 0)
        assert torch.all(out == 0)

    def test_residual_in_place_is_aliasing_safe(self):
        torch.manual_seed(2)
        rows, hidden = 4, 256
        x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        w = torch.randn(hidden, device="cuda").to(torch.float32)
        res_clone = res.clone()
        _, _, res_out = _run_op(k._norm_quant_int8_impl, x, res, w, 1e-6)
        _, _, res_ref = k.reference_norm_quant_int8(x, res_clone, w, 1e-6)
        assert res_out.data_ptr() == res.data_ptr()  # mutated in place
        assert torch.equal(res_out.cpu(), res_ref.cpu())

    def test_fallback_chain_equals_reference_contract(self):
        # the in-node fallback must reproduce the unfused chain EXACTLY
        # (same expressions as vllm/ir/ops/layernorm.py + the _C quant kernel).
        # Needs the real _C quant kernel (a stub would no-op the write).
        _ensure_c_stubs()
        if _USING_STUBS:
            pytest.skip("needs the compiled vllm._C extension")
        torch.manual_seed(3)
        rows, hidden = 4, 512
        x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
        w = torch.randn(hidden, device="cuda").to(torch.float32)
        out2 = torch.empty((rows, hidden), dtype=torch.int8, device="cuda")
        scale2 = torch.empty((rows, 1), dtype=torch.float32, device="cuda")
        res2 = res.clone()
        k._fallback_norm_quant(out2, x, w, scale2, res2, 1e-6)
        q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(x, res, w, 1e-6)
        assert torch.equal(res2.cpu(), res_ref.cpu())
        assert torch.equal(scale2.cpu(), scale_ref.cpu())
        assert torch.equal(out2.cpu(), q_ref.cpu())


@needs_2gpu
class TestTwoRankOneShot:
    """The 3-way barrier + peer-load data path, end to end, on 2 ranks.

    Opt-in (env ``AR_NORM_QUANT_2GPU_TEST=1``) because it needs an XGMI (or
    otherwise peer-atomic-capable) pair: on a PCIe pair the MI200/XGMI
    integer-atomic guarantee does not hold and the test would spin forever.
    """

    @pytest.mark.skipif(
        os.environ.get("AR_NORM_QUANT_2GPU_TEST") != "1",
        reason="set AR_NORM_QUANT_2GPU_TEST=1 on an XGMI pair",
    )
    def test_ar_norm_quant_two_ranks(self):
        import torch.multiprocessing as mp

        def worker(rank, world, port, outq):
            import torch.distributed as dist

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group("gloo", rank=rank, world_size=world)
            torch.cuda.set_device(rank)
            dev = f"cuda:{rank}"
            rows, hidden = 32, 2048
            torch.manual_seed(7)
            # SAME weight and residual on both ranks, different x shards
            w = torch.randn(hidden, device=dev).to(torch.float32)
            res = torch.randn(rows, hidden, device=dev).to(torch.bfloat16)
            torch.manual_seed(100 + rank)
            x = torch.randn(rows, hidden, device=dev).to(torch.bfloat16)
            comm = k.ArNormQuantComm(dist.group.WORLD, k.MAX_ROWS, hidden)
            result = torch.empty((rows, hidden), dtype=torch.int8, device=dev)
            scale = torch.empty((rows, 1), dtype=torch.float32, device=dev)
            k._decision = {"enabled": True, "reason": "test", "comm": comm}
            try:
                k._ar_norm_quant_int8_impl(result, x, w, scale, 1e-6, res)
            finally:
                k._decision = None
            torch.cuda.synchronize()
            outq.put(
                (rank, result.cpu(), scale.cpu(), res.cpu(), x.cpu(), w.cpu())
            )
            dist.destroy_process_group()

        ctx = mp.get_context("spawn")
        outq = ctx.Queue()
        procs = [
            ctx.Process(target=worker, args=(r, 2, 29517, outq)) for r in (0, 1)
        ]
        for p in procs:
            p.start()
        got = {}
        for _ in range(2):
            rank, result, scale, res, x, w = outq.get(timeout=300)
            got[rank] = (result, scale, res, x, w)
        for p in procs:
            p.join(timeout=60)
        assert all(not p.is_alive() for p in procs), "worker did not terminate"

        x0, res0, w0 = got[0][3], got[0][2], got[0][4]
        x1 = got[1][3]
        for rank in (0, 1):
            x_self = got[rank][3]
            x_peer = got[1 - rank][3]
            q_ref, scale_ref, res_ref = k.reference_norm_quant_int8(
                x_self, res0, w0, 1e-6, peer=x_peer
            )
            result, scale, res_out = got[rank][0], got[rank][1], got[rank][2]
            assert torch.equal(scale, scale_ref), f"rank {rank}: scale mismatch"
            assert torch.equal(res_out, res_ref), f"rank {rank}: res mismatch"
            flips = (result.int() - q_ref.int()).abs()
            assert (flips <= 1).all(), f"rank {rank}: q off by >1 LSB"
            print(
                f"rank {rank}: q +/-1 LSB flips: "
                f"{int((flips > 0).sum())}/{flips.numel()}"
            )
        # both ranks must have the SAME outputs (the AR sums both shards)
        assert torch.equal(got[0][0], got[1][0]), "ranks disagree on quant"
        assert torch.equal(got[0][1], got[1][1]), "ranks disagree on scale"


# ---------------------------------------------------------------------------
# 6. pattern wiring (CPU): the motif must trace and fuse
# ---------------------------------------------------------------------------
class TestPatternWiring:
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

    def test_norm_quant_pattern_registers_and_fuses(self):
        _ensure_c_stubs()
        import torch._inductor.pattern_matcher as pm
        from torch._inductor.pattern_matcher import PatternMatcherPass

        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f
        from vllm.compilation.passes.inductor_pass import enable_fake_mode

        with self._cfg_ctx():

            @enable_fake_mode
            def _build():
                pr = f.NormQuantInt8Pattern(1e-6)
                pm_pass = PatternMatcherPass(pass_name="test_ar_norm_quant")
                pm.register_replacement(
                    pr.pattern,
                    pr.replacement,
                    pr.get_inputs(),
                    pm.fwd_only,
                    pm_pass,
                )
                # the same chain, traced as the engine's graph carries it
                gm = pm.fwd_only(pr.pattern, pr.get_inputs())
                return gm, pm_pass

            gm, pm_pass = _build()
        ops = {
            f.called_op(n)
            for n in gm.graph.nodes
            if n.op == "call_function"
        }
        assert f._FUSED_ADD_RMS_NORM_OP in ops
        assert any(
            o is not None and "dynamic_scaled_int8_quant" in str(o)
            for o in ops
        ), f"quant node missing: {ops}"
        matched = pm_pass.apply(gm.graph)
        assert matched >= 1, "pattern did not re-match its own traced chain"
        fused_ops = {
            f.called_op(n) for n in gm.graph.nodes if n.op == "call_function"
        }
        assert any(
            o is not None and "triton_norm_quant_int8" in str(o)
            for o in fused_ops
        ), f"fused op absent after rewrite: {fused_ops}"

    def test_ar_pattern_traces_the_ar_node(self):
        _ensure_c_stubs()
        import torch._inductor.pattern_matcher as pm

        import vllm.compilation.passes.fusion.ar_norm_quant_fusion as f
        from vllm.compilation.passes.inductor_pass import enable_fake_mode

        with self._cfg_ctx():
            # tp==1 (no distributed here) traces the collective to a no-op;
            # emit the exact node form a TP=2 engine graph carries instead.
            def fake_ar(inp):
                return torch.ops.vllm.all_reduce(inp, group_name="test-tp2")

            orig = f.tensor_model_parallel_all_reduce
            f.tensor_model_parallel_all_reduce = fake_ar

            @enable_fake_mode
            def _build():
                pr = f.ArNormQuantInt8Pattern(1e-6)
                return pm.fwd_only(pr.pattern, pr.get_inputs())

            try:
                gm = _build()
            finally:
                f.tensor_model_parallel_all_reduce = orig
        ops = {
            f.called_op(n)
            for n in gm.graph.nodes
            if n.op == "call_function"
        }
        assert f._AR_OP in ops, f"AR node missing from 3-way pattern: {ops}"
        assert f._FUSED_ADD_RMS_NORM_OP in ops
        assert any(
            o is not None and "dynamic_scaled_int8_quant" in str(o)
            for o in ops
        ), f"quant node missing: {ops}"

    def test_pass_config_flags_exist(self):
        from vllm.config.compilation import PassConfig

        cfg = PassConfig()
        assert hasattr(cfg, "fuse_ar_norm_quant")
        assert hasattr(cfg, "fuse_norm_quant_triton")
