# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fusion pass: Triton norm+quant and one-shot AR+norm+quant (the "3-way").

This is the *standard* vLLM integration point for this kind of work (the
decision is recorded in
``research/ar-rms-quant-v2-impl-2026-10-22/REPORT.md`` section 0.5):

* the fusion lives in an Inductor post-grad pass (``VllmPatternMatcherPass``),
  registered in ``PostGradPassManager.configure`` exactly like
  ``fuse_norm_quant`` / ``fuse_act_quant`` / ``fuse_attn_quant``;
* it is toggled by a ``PassConfig`` flag (``fuse_ar_norm_quant`` /
  ``fuse_norm_quant_triton``), i.e. ``-cc.pass_config.fuse_ar_norm_quant=True``
  per role, default OFF;
* the fused node is a plain vLLM custom op (``direct_register_custom_op``)
  whose mutating schema mirrors ``_C.rms_norm_dynamic_per_token_quant``,
  implemented by a stock-Triton kernel
  (``vllm/kernels/triton/ar_norm_quant.py``);
* the all-reduce half is consumed *inside* the pattern
  (``tensor_model_parallel_all_reduce``), the same way the existing
  ``fuse_allreduce_rms`` family replaces the AR node -- no monkey-patching of
  the collective, no private dispatch path.

Pattern priority (v1 lesson: "Inductor dispatche par cible de noeud"): the
3-way patterns are registered BEFORE the norm+quant ones inside this pass, the
pass runs before ``fuse_allreduce_rms`` / ``fuse_norm_quant`` in
``PostGradPassManager``, and the 2-way AR+RMS patterns of
``allreduce_rms_fusion.py`` decline int8-quant sites whenever a 3-way owner is
registered (``_norm_output_not_int8_quant``).
"""

from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
import vllm.ir.ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.kernels.triton import ar_norm_quant as _ar_norm_quant  # noqa: F401 - registers the custom ops
from vllm.kernels.triton.ar_norm_quant import MAX_ROWS
from vllm.logger import init_logger

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternReplacement,
    log_pattern_match_summary,
)
from .matcher_utils import (
    FusionSiteCensus,
    MatcherQuantInt8,
    called_op,
    functionalized_mutable_args,
    op_call_kwargs,
)
from .rms_quant_fusion import (
    _int8_alloc_conform,
    _rms_input_weight_dtype_match,
    kInt8DynamicTokenSym,
)

logger = init_logger(__name__)

_AR_OP = torch.ops.vllm.all_reduce.default
_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default

NORM_QUANT_OP = torch.ops.vllm.triton_norm_quant_int8.default
AR_NORM_QUANT_OP = torch.ops.vllm.triton_ar_norm_quant_int8.default

# The megakernel's fused quant+GEMM (campaign megakernel-2026-10-23, pair (a)).
# Resolved lazily at call time: the ops register with the quantization layer,
# which may load after this module. Both schemas are 5/6-arg positional
# (input, weight, [scale_a,] scale_b, out_dtype, bias).
def _scaled_mm_fused_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Any,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = torch.ops.vllm.triton_scaled_mm_fused_quant.default
    return op(input, weight, scale_b, out_dtype, bias)


def _scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Any,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    op = torch.ops.vllm.triton_scaled_mm.default
    return op(input, weight, scale_a, scale_b, out_dtype, bias)


def _decode_regime(match: pm.Match) -> bool:
    """Decode-only regime gate, resolved at match time (section 3.4).

    The motif is only taken for rank-2 tensors with at most ``MAX_ROWS`` rows
    (the one-stage custom-AR bound): prefill and large batches never match, so
    the QR-INT8 transport regime and every large-batch path stay untouched.
    Shapes are static per captured graph, so this is equivalent to a runtime
    gate for the graph replay -- and it is rank-symmetric (TP ranks compile
    the same graphs), which the fused-vs-fallback decision requires.

    Within that bound the *measured* regime is narrower: the AR op takes its
    fused one-stage kernel at rows <= ``FUSE_MAX_ROWS`` only (ETAPE-B3 row
    gate, 2026-10-25) and answers with the two-stage chain above, so the
    rewrite is safe at every shape and no site inherits the c128 kernel
    regression.
    """
    for node in match.nodes:
        if called_op(node) is _AR_OP:
            # schema: vllm::all_reduce(Tensor tensor, str group_name)
            kwargs = op_call_kwargs(node, _AR_OP)
            inp = kwargs.get("tensor", kwargs.get("input"))
        elif node.target is _FUSED_ADD_RMS_NORM_OP:
            inp = node.args[0]
        else:
            continue
        if inp is None:
            continue
        try:
            val = inp.meta["val"]
        except (AttributeError, KeyError, TypeError):
            return False
        if len(val.shape) != 2:
            return False
        rows = val.shape[0]
        if isinstance(rows, int) and rows > MAX_ROWS:
            # concretely out of the fused regime: leave the site on its chain
            return False
        # Symbolic rows are ACCEPTED (measured 2026-10-22: the prod engine
        # compiles backed-dynamic -- one graph serves rows 1..4096 -- and
        # refusing symbolic dims silently cut the reach to 10/68 sites). The
        # per-call bound in get_decision/_norm_gate enforces the regime at
        # runtime (fused at rows <= comm.rows and <= FUSE_MAX_ROWS, the
        # two-stage chain above), so the node is safe at every shape and every
        # decode size is served.
        if val.dtype != torch.bfloat16:
            return False
        return True
    return True


def _weight_fp32(match: pm.Match) -> bool:
    """The contract is the *executed* baseline: fp32 Gemma ``(1+w)`` gamma
    multiplied into the fp32 normed value (no bf16 rounding of the weight).
    A site with a non-fp32 gamma would need a different contract -- leave it
    on the unfused chain instead of diverging silently."""
    for node in match.nodes:
        if node.target is _FUSED_ADD_RMS_NORM_OP:
            # fused_add_rms_norm(x, residual, weight, epsilon[, variance_size])
            weight = node.args[2]
            try:
                w_val = weight.meta["val"]
            except (AttributeError, KeyError, TypeError):
                return False
            return w_val.dtype == torch.float32
    return True


class _TritonNormQuantPatternBase(VllmPatternReplacement):
    """Shared machinery: MatcherQuantInt8 with the allocs as pattern inputs."""

    def __init__(self, epsilon: float, quant_op: Any, motif: str) -> None:
        self.epsilon = epsilon
        self.quant_op = quant_op
        self.motif = motif
        self.quant_matcher = MatcherQuantInt8(kInt8DynamicTokenSym)
        self.extra_check: Any = None

    def buffers(self) -> list[torch.Tensor]:
        return self.quant_matcher.buffer_examples(5, 16)

    def _replacement_call(
        self,
        result_buf: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_buf: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        at = auto_functionalized(
            self.quant_op,
            result=result_buf,
            input=input,
            weight=weight,
            scale=scale_buf,
            epsilon=self.epsilon,
            residual=residual,
        )
        outs = functionalized_mutable_args(self.quant_op, at)
        # at[0] is the op's return value = the normed value (see the op
        # docstring: the sites whose norm output has OTHER consumers need it
        # kept alive as a graph value).
        values = at if isinstance(at, tuple) else (at,)
        normed = values[0]
        return outs["result"], outs["residual"], outs["scale"], normed


class ArNormQuantFusedGemmPattern(_TritonNormQuantPatternBase):
    """``all_reduce -> fused_add_rms_norm -> fused quant+GEMM`` -> 3-way int8 +
    plain scaled_mm.

    Why this motif exists (research/ar-vol-count-2026-10-25, ETAPE-B1): the
    megakernel's fused-quant GEMM absorbs the per-token quant *node* that the
    ``ArNormQuantInt8Pattern`` motif needs, so the 3-way matches ZERO site
    whenever the fused-quant path is on (``matched_this_run=0`` in prod logs).
    This motif reaches exactly those sites: the fused quant+GEMM node is the
    consumer of the normed value, and the rewrite emits
    ``[3-way: AR+norm+quant int8] -> [scaled_mm with the 3-way's per-token
    scale]``. The GEMM's amax prologue then disappears (its input is already
    quantified -- ``fused_quant_eligible`` declines pre-quantized activations),
    so the two fusions coexist instead of fighting over one node.

    Numeric contract: the 3-way reproduces the executed chain expression by
    expression (spec ar-rms-quant-v2 §3.2); the GEMM computes the same
    ``scale_a * scale_b * (A_q @ B_q)`` with the quant produced one stage
    earlier. ``x_s`` (the fused op's scale output) is answered by the 3-way's
    own per-token scale -- the same quantity the fused kernel computed.

    ⚠ The ``x_s`` output is only part of the motif where the engine keeps it:
    on symmetric sites the azp correction that reads ``x_s`` never runs
    (``fused_quant_eligible`` requires ``i_zp is None``), so dead-code
    elimination removes ``getitem(op, 1)`` from the real graph and a motif
    referencing it matches NOTHING (measured 2026-10-26 on the I1 arm:
    ``matched_this_run=0`` with every site present). The ``ScaleDead``
    variants below cover exactly that -- the common -- form.
    """

    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, AR_NORM_QUANT_OP, "ArNormQuantFusedGemmPattern")

    @property
    def pattern(self) -> Any:
        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            gemm_out, x_s = _scaled_mm_fused_quant(
                rms, w_q, w_s, torch.bfloat16, None
            )
            # x_s/rms kept alive: x_s answers the fused op's scale output, rms
            # the sites whose normed value has OTHER consumers.
            return gemm_out, x_s, residual_out, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # The matched node allocates its outputs internally (no alloc
            # buffers in the motif), so the 3-way's mutable buffers are created
            # here -- the canonical MatcherQuantInt8 alloc shapes.
            rows, hidden = input.shape[0], input.shape[1]
            result_buf = torch.empty(
                (rows, hidden), dtype=torch.int8, device=input.device
            )
            scale_buf = torch.empty(
                (rows, 1), dtype=torch.float32, device=input.device
            )
            result, residual_out, scale, rms = self._replacement_call(
                result_buf, input, weight, scale_buf, residual
            )
            gemm_out = _scaled_mm(
                result, w_q, scale, w_s, torch.bfloat16, None
            )
            return gemm_out, scale, residual_out, rms

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        # residual, input, weight (fp32 Gemma gamma), w_q (int8), w_s
        return [
            self.empty_bf16(5, 16),
            self.empty_bf16(5, 16),
            self.empty_fp32(16),
            self.empty(16, 8, dtype=torch.int8),
            self.empty_fp32(8, 1),
        ]


class ArNormQuantFusedGemmOutputOnlyPattern(ArNormQuantFusedGemmPattern):
    """fused-GEMM variant for sites whose residual output is dead."""
    @property
    def pattern(self) -> Any:
        inner = super().pattern

        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            gemm_out, x_s, _, rms = inner(residual, input, weight, w_q, w_s)
            return gemm_out, x_s, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        inner = super().replacement

        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            gemm_out, x_s, _, rms = inner(residual, input, weight, w_q, w_s)
            return gemm_out, x_s, rms

        return _replacement


class ArNormQuantFusedGemmScaleDeadPattern(ArNormQuantFusedGemmPattern):
    """fused-GEMM variant for the form the engine REALLY has: the fused op's
    ``x_s`` output is DEAD.

    Measured root cause (2026-10-26, arm I1 of
    research/ar-vol-count-2026-10-25): on symmetric sites the azp correction
    that reads ``x_s`` never runs (``fused_quant_eligible`` requires
    ``i_zp is None``), so dead-code elimination removes ``getitem(op, 1)``
    from the graph and a motif that references it matches NOTHING -- all 80
    real sites present, ``matched_this_run=0``. The motif must not ask for a
    node the graph does not have; the scale is consumed INSIDE the
    replacement (the bare ``scaled_mm`` reads it) and needs no output.
    """

    @property
    def pattern(self) -> Any:
        inner = super().pattern

        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            gemm_out, _x_s, residual_out, rms = inner(
                residual, input, weight, w_q, w_s
            )
            return gemm_out, residual_out, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        inner = super().replacement

        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            gemm_out, _scale, residual_out, rms = inner(
                residual, input, weight, w_q, w_s
            )
            return gemm_out, residual_out, rms

        return _replacement


class ArNormQuantFusedGemmScaleDeadOutputOnlyPattern(
    ArNormQuantFusedGemmScaleDeadPattern
):
    """x_s-dead AND residual-dead variant (the two dead outputs together)."""

    @property
    def pattern(self) -> Any:
        inner = super().pattern

        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gemm_out, _res, rms = inner(residual, input, weight, w_q, w_s)
            return gemm_out, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        inner = super().replacement

        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            w_q: torch.Tensor,
            w_s: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gemm_out, _res, rms = inner(residual, input, weight, w_q, w_s)
            return gemm_out, rms

        return _replacement


class ArNormQuantInt8Pattern(_TritonNormQuantPatternBase):
    """``all_reduce -> fused_add_rms_norm -> per-token int8 quant`` -> one
    Triton op (the 3-way: AR absorbed into the kernel)."""

    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, AR_NORM_QUANT_OP, "ArNormQuantInt8Pattern")

    @property
    def pattern(self) -> Any:
        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            result, scale = self.quant_matcher(
                rms, result_buf=result_buf, scale_buf=scale_buf
            )
            # rms is returned too: the sites whose norm output feeds OTHER
            # consumers (a second quant, the shared-expert alias) must keep
            # receiving it -- that is what makes those sites matchable.
            return result, residual_out, scale, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self._replacement_call(
                result_buf, input, weight, scale_buf, residual
            )

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        # residual, input, weight (fp32 Gemma gamma), result, scale
        return [
            self.empty_bf16(5, 16),
            self.empty_bf16(5, 16),
            self.empty_fp32(16),
            *self.buffers(),
        ]


class ArNormQuantInt8OutputOnlyPattern(ArNormQuantInt8Pattern):
    """3-way variant for sites whose residual output is dead."""

    @property
    def pattern(self) -> Any:
        inner = super().pattern

        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result, _, scale, rms = inner(
                residual, input, weight, result_buf, scale_buf
            )
            return result, scale, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        inner = super().replacement

        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result, _, scale, rms = inner(
                residual, input, weight, result_buf, scale_buf
            )
            return result, scale, rms

        return _replacement


class NormQuantInt8Pattern(_TritonNormQuantPatternBase):
    """``fused_add_rms_norm -> per-token int8 quant`` -> one Triton op.

    The S1 / "norm+quant only" scope: no all-reduce in the motif (the AR stays
    its own node), same numeric contract, same mutation contract. Registered
    AFTER the 3-way patterns so the AR sites go to the 3-way.
    """

    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, NORM_QUANT_OP, "NormQuantInt8Pattern")

    @property
    def pattern(self) -> Any:
        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result, scale = self.quant_matcher(
                rms, result_buf=result_buf, scale_buf=scale_buf
            )
            # rms kept alive for the site's other consumers (see the 3-way)
            return result, residual_out, scale, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self._replacement_call(
                result_buf, input, weight, scale_buf, residual
            )

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_bf16(5, 16),
            self.empty_bf16(5, 16),
            self.empty_fp32(16),
            *self.buffers(),
        ]


class NormQuantInt8OutputOnlyPattern(NormQuantInt8Pattern):
    """norm+quant variant for sites whose residual output is dead."""

    @property
    def pattern(self) -> Any:
        inner = super().pattern

        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result, _, scale, rms = inner(
                residual, input, weight, result_buf, scale_buf
            )
            return result, scale, rms

        return _pattern

    @property
    def replacement(self) -> Any:
        inner = super().replacement

        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            result_buf: torch.Tensor,
            scale_buf: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result, _, scale, rms = inner(
                residual, input, weight, result_buf, scale_buf
            )
            return result, scale, rms

        return _replacement


def _composite_check(quant_op: Any) -> Any:
    def check(match: pm.Match) -> bool:
        return (
            _decode_regime(match)
            and _weight_fp32(match)
            and _rms_input_weight_dtype_match(match)
            and _int8_alloc_conform(quant_op, match)
        )

    return check


class ArNormQuantFusionPass(VllmFusionPatternMatcherPass):
    """Fuse AR + add-RMSNorm + per-token int8 quant (and norm+quant alone)
    into stock-Triton custom ops. Decode-only (see ``_decode_regime``)."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, pass_name="ar_norm_quant_fusion_pass")
        pass_config = config.compilation_config.pass_config
        want_ar = bool(pass_config.fuse_ar_norm_quant)
        want_norm = bool(pass_config.fuse_norm_quant_triton)
        # The 3-way needs TP=2 (the kernel's scope): at other world sizes
        # ``tensor_model_parallel_all_reduce`` traces to no op at all
        # (world_size == 1) or to a shape the kernel does not cover, so its
        # patterns are simply not registered there (its flag then matches
        # nothing -- documented, predictable).
        tp2 = get_tensor_model_parallel_world_size() == 2
        for epsilon in (1e-5, 1e-6):
            # 3-way FIRST: within one PatternMatcherPass, patterns keyed at
            # the same output nodes are tried in registration order.
            classes: list[type] = []
            if want_ar and tp2:
                classes += [
                    # fused-GEMM motif FIRST: today's graphs carry the quant
                    # inside the GEMM node (ETAPE-B1), and within one
                    # PatternMatcherPass patterns keyed at the same output
                    # nodes are tried in registration order. The x_s-alive
                    # forms come before the ScaleDead ones: they cannot match
                    # the dead form (a missing getitem has no counterpart) and
                    # they keep the scale output available where it is alive.
                    ArNormQuantFusedGemmPattern,
                    ArNormQuantFusedGemmOutputOnlyPattern,
                    ArNormQuantFusedGemmScaleDeadPattern,
                    ArNormQuantFusedGemmScaleDeadOutputOnlyPattern,
                    ArNormQuantInt8Pattern,
                    ArNormQuantInt8OutputOnlyPattern,
                ]
            if want_norm:
                classes += [
                    NormQuantInt8Pattern,
                    NormQuantInt8OutputOnlyPattern,
                ]
            for cls in classes:
                pr = cls(epsilon)
                pr.extra_check = _composite_check(pr.quant_op)
                self.register(pr)

        # Same trick as RMSNormQuantFusionPass / RocmAiterAllReduceFusionPass:
        # register_replacement caches traced patterns globally, and multiple
        # epsilon values need distinct pattern graphs.
        torch._inductor.pattern_matcher._seen_patterns.clear()

        self.dump_patterns(config, self.pm_pass)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # One-shot census before rewriting: candidate sites and their shapes
        # are the ground truth the per-motif match counts are read against.
        FusionSiteCensus.log(graph)
        self.matched_count = self.pm_pass.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        log_pattern_match_summary(self.pm_pass.pass_name, self.matched_count)
