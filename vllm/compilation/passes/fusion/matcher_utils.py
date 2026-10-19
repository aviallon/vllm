# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ClassVar

import torch
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    _normalize_quant_group_shape,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kInt8DynamicTokenSym,
    kNvfp4Dynamic,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

ROTARY_OP = torch.ops._C.rotary_embedding.default
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding.default
MROPE_OP = torch.ops.vllm.mrope.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if hasattr(torch.ops._C, "per_token_group_fp8_quant"):
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501

if hasattr(torch.ops._C, "dynamic_scaled_int8_quant"):
    QUANT_OPS[kInt8DynamicTokenSym] = torch.ops._C.dynamic_scaled_int8_quant.default  # noqa: E501

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.out  # noqa: E501


SILU_MUL_OP = torch.ops._C.silu_and_mul.default


def _mutable_arg_names(op: OpOverload) -> list[str]:
    """Names of the op's mutable arguments, in schema order.

    Optional mutable arguments (``Tensor!? azp``) count too: an argument is
    mutable because the SCHEMA says so, regardless of whether the call passed a
    value for it.
    """
    return [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]


def functionalized_mutable_args(op: OpOverload, out: Any) -> dict[str, Any]:
    """Map ``auto_functionalized``'s return tuple onto the op's mutable args.

    ``auto_functionalized`` returns ``(op_return, *mutated_in_schema_order)``:
    the leading entry is the op's return value (``None`` for ``-> ()`` ops) and
    the mutated entries include optional mutable arguments even when the call
    passed ``None`` for them.  The tuple's arity is therefore a property of the
    SCHEMA, not of the call: ``dynamic_scaled_int8_quant``
    (``Tensor! result, Tensor input, Tensor! scale, Tensor!? azp) -> ()``
    returns 4 entries ``(None, result, scale, azp)``, which a fixed
    ``_, result, scale = ...`` unpack cannot read (measured under
    torch 2.13.0+git733fca1, aragorn
    research/w8a8-pdtest-2026-10-18/evidence/arity-probe-torch213.log).  Index
    by schema instead of unpacking a fixed arity.
    """
    names = _mutable_arg_names(op)
    values = out if isinstance(out, tuple) else (out,)
    if len(values) < len(names):
        raise RuntimeError(
            f"auto_functionalized({op}) returned {len(values)} value(s), fewer "
            f"than the {len(names)} mutable argument(s) {names} of its schema"
        )
    return dict(zip(names, values[len(values) - len(names) :]))


class MatcherCustomOp(ABC):
    def __init__(self, enabled: bool) -> None:
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        self.device = config.device_config.device if config.device_config else None

        self.enabled = enabled
        self.forward = self.forward_custom if enabled else self.forward_native

    @abstractmethod
    def forward_custom(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def forward_native(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.model_dtype, device=self.device, **kwargs)

    def empty_int64(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.int64, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)

    def inputs(self) -> list[torch.Tensor]:
        """Utility for inputs to the pattern."""
        raise NotImplementedError


class MatcherRotaryEmbedding(MatcherCustomOp):
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_flashinfer: bool = False,
        match_rocm_aiter: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = RotaryEmbedding.enabled()
        if match_rocm_aiter is None:
            match_rocm_aiter = rocm_aiter_ops.is_triton_rotary_embed_enabled()

        super().__init__(enabled)
        self.is_neox = is_neox
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.rotary_dim = head_size
        if use_flashinfer:
            self.rotary_op = FLASHINFER_ROTARY_OP
        elif match_rocm_aiter:
            self.rotary_op = rocm_aiter_ops.get_triton_rotary_embedding_op()
        else:
            self.rotary_op = ROTARY_OP

    def inputs(self) -> list[torch.Tensor]:
        positions = self.empty_int64(5)
        query = self.empty(5, self.q_size)
        key = self.empty(5, self.kv_size)
        cos_sin_cache = self.empty(4096, self.rotary_dim)
        return [positions, query, key, cos_sin_cache]

    def forward_custom(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        result = auto_functionalized(
            self.rotary_op,
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
        )
        query_out = result[1]
        key_out = result[2] if len(result) > 2 else None
        return query_out, key_out

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        result: tuple[torch.Tensor, torch.Tensor | None] = (
            RotaryEmbedding.forward_static(
                positions,
                query,
                key,
                self.head_size,
                self.rotary_dim,
                cos_sin_cache,
                self.is_neox,
            )
        )
        return result


class MatcherMRotaryEmbedding(MatcherCustomOp):
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        rotary_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mrope_section: tuple[int, int, int],
        mrope_interleaved: bool,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = RotaryEmbedding.enabled()

        super().__init__(enabled)
        self.is_neox = is_neox
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

    def inputs(self) -> list[torch.Tensor]:
        positions = self.empty_int64(3, 5)
        query = self.empty(5, self.q_size)
        key = self.empty(5, self.kv_size)
        cos_sin_cache = self.empty(4096, self.rotary_dim)
        return [positions, query, key, cos_sin_cache]

    def forward_custom(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = auto_functionalized(
            MROPE_OP,
            positions=positions,
            query=query,
            key=key,
            cos_sin_cache=cos_sin_cache,
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            mrope_section_t=self.mrope_section[0],
            mrope_section_h=self.mrope_section[1],
            mrope_section_w=self.mrope_section[2],
            mrope_interleaved=self.mrope_interleaved,
            is_neox_style=self.is_neox,
        )
        return result[1], result[2]

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "MRoPE fusion requires the rotary_embedding custom op"
        )


class MatcherRMSNormGated(MatcherCustomOp):
    """Matches RMSNormGated with norm_before_gate=True and group_size=None."""

    def __init__(
        self,
        epsilon: float,
        enabled: bool | None = None,
        norm_before_gate: bool = True,
        group_size: int | None = None,
    ) -> None:
        if enabled is None:
            enabled = RMSNormGated.enabled()

        super().__init__(enabled)
        self.epsilon = epsilon
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size

    def inputs(self) -> list[torch.Tensor]:
        x = self.empty(5, 16)
        z = self.empty(5, 16)
        weight = self.empty(16)
        return [x, z, weight]

    def forward_custom(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            rmsnorm_fn,
        )

        return rmsnorm_fn(
            x,
            weight,
            bias=None,
            z=z,
            eps=self.epsilon,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return RMSNormGated.forward_static(
            x,
            z,
            weight,
            self.epsilon,
            self.model_dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


class MatcherDeepseekScalingRotaryEmbedding(MatcherCustomOp):
    def __init__(
        self,
        is_neox: bool,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_flashinfer: bool = False,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = DeepseekScalingRotaryEmbedding.enabled()

        super().__init__(enabled)
        self.is_neox = is_neox
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.rotary_dim = head_size
        self.use_flashinfer = use_flashinfer

    def inputs(self) -> list[torch.Tensor]:
        positions = self.empty_int64(5)
        query = self.empty(5, self.num_heads, self.head_size)
        key = self.empty(5, self.num_kv_heads, self.head_size)
        cos_sin_cache = self.empty(4096, self.rotary_dim)
        return [positions, query, key, cos_sin_cache]

    def forward_custom(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_flashinfer:
            torch.ops.vllm.flashinfer_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox,
            )
            return query, key
        result: tuple[torch.Tensor, torch.Tensor | None] = (
            DeepseekScalingRotaryEmbedding.forward_static(
                positions,
                query,
                key,
                self.head_size,
                self.rotary_dim,
                cos_sin_cache,
                self.is_neox,
            )
        )
        return result

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        result: tuple[torch.Tensor, torch.Tensor | None] = (
            DeepseekScalingRotaryEmbedding.forward_static(
                positions,
                query,
                key,
                self.head_size,
                self.rotary_dim,
                cos_sin_cache,
                self.is_neox,
            )
        )
        return result


class MatcherQuantFP8(MatcherCustomOp):
    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        match_rocm_aiter: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
        self.quant_key = quant_key
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        self.match_rocm_aiter = match_rocm_aiter
        self.is_tma_aligned = is_tma_aligned

        if match_rocm_aiter:
            assert not quant_key.scale.group_shape.is_per_tensor(), (
                "ROCm aiter fusion pass does not support per tensor quantization"
            )
            if quant_key.scale.group_shape.is_per_token():
                self.QUANT_OP = rocm_aiter_ops.get_per_token_quant_op()
            else:
                assert quant_key.scale.group_shape.col == 128, (
                    "ROCm aiter fusion pass currently supports "
                    "quantization operation with group_size 128"
                )
                self.QUANT_OP = rocm_aiter_ops.get_group_quant_op()

        else:
            assert quant_key in QUANT_OPS, (
                f"unsupported quantization scheme {quant_key}"
            )
            self.QUANT_OP = QUANT_OPS[quant_key]

            assert quant_key.dtype == current_platform.fp8_dtype(), (
                "Only QuantFP8 supported by"
            )
            assert quant_key.scale2 is None

        self.quant_fp8 = QuantFP8(
            quant_key.scale.static,
            quant_key.scale.group_shape,
            column_major_scales=has_col_major_scales,
            use_ue8m0=is_e8m0,
            tma_aligned_scales=self.is_tma_aligned,
            compile_native=False,
        )

    def forward_rocm_aiter(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quant_key_group_shape = self.quant_key.scale.group_shape
        if quant_key_group_shape == GroupShape.PER_TOKEN:
            return self.QUANT_OP(  # type: ignore[no-any-return]
                x=input,
                quant_dtype=self.quant_key.dtype,
                scale=scale,
            )
        else:
            return self.QUANT_OP(input, quant_key_group_shape.col)  # type: ignore[no-any-return]

    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.match_rocm_aiter:
            return self.forward_rocm_aiter(input, scale)

        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.group_shape.is_per_group():
            # for tma_aligned, the scale must be passed to forward_custom
            # tma_aligned fusion then matches by custom op arguments
            if not self.is_tma_aligned:
                assert scale is None
                scale = self.make_scale(input, transposed=self.has_col_major_scales)

            finfo = torch.finfo(self.quant_key.dtype)
            fp8_min = finfo.min
            fp8_max = finfo.max

            ret = auto_functionalized(
                self.QUANT_OP,
                input=input,
                output_q=result,
                output_s=scale,
                group_size=self.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )
            outs = functionalized_mutable_args(self.QUANT_OP, ret)
            return outs["output_q"], outs["output_s"]

        if self.quant_key.scale.static:
            assert scale is not None
            ret = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale
            )
            # NOTE: the old `_, result = ...` unpack read a 2-tuple, but this
            # op has TWO mutable args (result, scale), so auto_functionalized
            # returns 3 entries and the unpack raised "too many values".  The
            # scale the op wrote is the one to return, not the caller's input.
            outs = functionalized_mutable_args(self.QUANT_OP, ret)
            return outs["result"], outs["scale"]
        else:
            assert scale is None
            scale = self.make_scale(input)
            ret = auto_functionalized(
                self.QUANT_OP, result=result, input=input, scale=scale, scale_ub=None
            )
            outs = functionalized_mutable_args(self.QUANT_OP, ret)
            return outs["result"], outs["scale"]

    def forward_native(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.quant_fp8(input, scale)  # type: ignore[no-any-return]

    def make_scale(self, input: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        normalized_group_shape = _normalize_quant_group_shape(
            input, self.quant_key.scale.group_shape
        )
        scale_shape = (
            input.shape[0] // normalized_group_shape[0],
            input.shape[1] // normalized_group_shape[1],
        )
        if transposed:
            scale_shape = tuple(reversed(scale_shape))
            return torch.empty(
                scale_shape, device=input.device, dtype=torch.float32
            ).permute(-1, -2)

        return torch.empty(scale_shape, device=input.device, dtype=torch.float32)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        if self.quant_key.scale.static:
            return [input, self.empty_f32(1, 1)]

        return [input]


class MatcherQuantInt8(MatcherCustomOp):
    """Matcher for dynamic per-token symmetric int8 quantization.

    Mirrors MatcherQuantFP8's dynamic per-token branch, but traces the int8
    op's own call convention (``azp`` argument instead of ``scale_ub``). The
    scale convention is the consumer's: one float32 scale per token,
    absmax / 127 (see ops.scaled_int8_quant / per_token_quant_int8).
    """

    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = (
                not current_platform.is_xpu()
                and hasattr(torch.ops._C, "dynamic_scaled_int8_quant")
            )

        super().__init__(enabled)
        self.quant_key = quant_key

        assert quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[quant_key]

        assert quant_key.dtype == torch.int8, "Only int8 supported"
        assert quant_key.scale2 is None
        assert not quant_key.scale.static
        assert quant_key.scale.group_shape.is_per_token()

    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        result_buf: torch.Tensor | None = None,
        scale_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert scale is None
        # Two allocation regimes, deliberately:
        #
        # * ``result_buf``/``scale_buf`` given (the norm->quant patterns pass
        #   the allocs as PATTERN INPUTS): the buffers are plain pattern
        #   placeholders, so nothing about their shape is baked into the
        #   pattern. This is what makes ``register_replacement``'s exact
        #   re-match pass on every site: with computed allocs the re-traced
        #   pattern tied both alloc dims to ONE shared symbol placeholder,
        #   while the engine graph carries the token dim as separate
        #   (semantically equal) sym-int placeholder args per use site --
        #   a shared pattern placeholder must bind one node, so every site
        #   whose two allocs referenced different placeholders was silently
        #   rejected by check_fn (measured 2026-10-19/20).
        #
        # * computed (no buffers): mirrors the shipping
        #   ops.scaled_int8_quant call site's allocation in the form the
        #   ENGINE's graph contains after standalone_compile's pre-grad
        #   decompositions (empty_like -> empty(size, layout=strided,
        #   pin_memory=False)). Kept for the fp8 path and for callers that
        #   have no buffers to pass.
        if result_buf is None:
            result_buf = torch.empty(
                (input.numel() // input.shape[-1], input.shape[-1]),
                dtype=self.quant_key.dtype,
                layout=torch.strided,
                device=input.device,
                pin_memory=False,
            )
        if scale_buf is None:
            scale_buf = self.make_scale(input)
        ret = auto_functionalized(
            self.QUANT_OP,
            result=result_buf,
            input=input,
            scale=scale_buf,
            azp=None,
        )
        # Schema-indexed: dynamic_scaled_int8_quant has THREE mutable args
        # (result, scale, azp - `Tensor!? azp` is optional AND mutable), so the
        # return is a 4-tuple (None, result, scale, azp) and the old fixed
        # `_, result, scale =` unpack raised "too many values to unpack".
        outs = functionalized_mutable_args(self.QUANT_OP, ret)
        return outs["result"], outs["scale"]

    def forward_native(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.int8_utils import (
            per_token_quant_int8,
        )

        return per_token_quant_int8(input)  # type: ignore[no-any-return]

    def make_scale(self, input: torch.Tensor, transposed: bool = False) -> torch.Tensor:
        assert not transposed
        # One float32 scale per token, flattened like the shipping
        # ops.scaled_int8_quant call site.
        num_tokens = input.numel() // input.shape[-1]
        return torch.empty(
            (num_tokens, 1), device=input.device, dtype=torch.float32
        )

    def inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 16)]

    def buffer_examples(self, rows: int, hidden: int) -> list[torch.Tensor]:
        """Example alloc buffers, for patterns that take the quant call's
        ``result``/``scale`` allocations as pattern inputs (see
        :meth:`forward_custom`)."""
        return [
            torch.empty(
                (rows, hidden),
                dtype=self.quant_key.dtype,
                layout=torch.strided,
                device=self.device,
                pin_memory=False,
            ),
            torch.empty(
                (rows, 1),
                dtype=torch.float32,
                layout=torch.strided,
                device=self.device,
                pin_memory=False,
            ),
        ]


class MatcherSiluAndMul(MatcherCustomOp):
    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = SiluAndMul.enabled()
        super().__init__(enabled)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 4)
        return [input]

    def forward_custom(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ret = auto_functionalized(SILU_MUL_OP, result=out, input=x)
        return functionalized_mutable_args(SILU_MUL_OP, ret)["result"]

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return SiluAndMul.forward_native(x)


# ---------------------------------------------------------------------------
# Graph site census -- one-shot proof-of-branching diagnostic.
#
# Answers, on the real compiled graph and at match time: how many nodes of the
# fusion-relevant ops exist, how many candidate producer->quant sites there
# are, and through what chain the producer reaches the quant. "0 candidates"
# and "N candidates" are both proof; silence is not.
# ---------------------------------------------------------------------------

_AUTO_FUNCTIONALIZED_NAMES = ("auto_functionalized", "auto_functionalized_v2")


def called_op(node: fx.Node) -> OpOverload | None:
    """The :class:`OpOverload` a node calls, unwrapping ``auto_functionalized``.

    At post-grad match time mutating custom ops appear as
    ``auto_functionalized(op, **kwargs)`` HOP calls (``auto_functionalized_v2``
    on newer torch); both forms are recognised so the census sees the same
    nodes the fusion patterns match.
    """
    if node.op != "call_function":
        return None
    target = node.target
    if isinstance(target, OpOverload):
        return target
    if getattr(target, "__name__", "") in _AUTO_FUNCTIONALIZED_NAMES and (
        node.args and isinstance(node.args[0], OpOverload)
    ):
        return node.args[0]
    return None


def op_call_kwargs(node: fx.Node, op: OpOverload) -> dict[str, Any]:
    """Schema-name -> value for a (possibly functionalized) op call node."""
    out: dict[str, Any] = {}
    schema_args = list(op._schema.arguments)
    if isinstance(node.target, OpOverload):
        for name, value in zip((a.name for a in schema_args), node.args):
            out[name] = value
    out.update(node.kwargs)
    return out


def op_base_name(target: Any) -> str:
    """'contiguous' for aten.contiguous.default, 'silu' for aten.silu, ..."""
    if isinstance(target, OpOverload):
        return target.name().split("::")[-1].split(".")[0]
    return getattr(target, "__name__", "")


def walk_views(node: Any) -> tuple[Any, int]:
    """Follow no-op-ish view/cast nodes to the real producer.

    Returns (producer, chain length). Mutating-op outputs (getitem of
    auto_functionalized) count as producers.
    """
    chain = 0
    while isinstance(node, fx.Node) and node.op == "call_function":
        target = node.target
        name = op_base_name(target)
        if name in ("contiguous", "clone", "view", "reshape", "flatten", "to",
                    "permute", "transpose", "unsqueeze", "squeeze", "_unsafe_view"):
            node = node.args[0]
            chain += 1
            continue
        if (
            target is operator.getitem
            and node.args
            and isinstance(node.args[0], fx.Node)
        ):
            producer = called_op(node.args[0])
            if producer is not None:
                # the getitem reads a mutable output of an op call: the real
                # producer is that op
                return node.args[0], chain + 1
            node = node.args[0]
            chain += 1
            continue
        break
    return node, chain


class FusionSiteCensus:
    """One-shot log of fusion-relevant op counts and candidate sites."""

    _logged: ClassVar[bool] = False

    @classmethod
    def log(cls, graph: fx.Graph) -> None:
        if cls._logged:
            return
        cls._logged = True

        counts: dict[str, int] = defaultdict(int)
        candidates: dict[str, int] = defaultdict(int)
        quant_forms: dict[str, int] = defaultdict(int)
        quant_inputs: dict[Any, Any] = {}

        try:
            rms_op = torch.ops.vllm_ir.rms_norm.default
        except AttributeError:
            rms_op = None
        try:
            fused_add_rms_op = torch.ops.vllm_ir.fused_add_rms_norm.default
        except AttributeError:
            fused_add_rms_op = None

        tracked: dict[Any, str] = {SILU_MUL_OP: "silu_and_mul"}
        quant_op = QUANT_OPS.get(kInt8DynamicTokenSym)
        if quant_op is not None:
            tracked[quant_op] = "quant_int8_dynamic_token"
        if rms_op is not None:
            tracked[rms_op] = "rms_norm"
        if fused_add_rms_op is not None:
            tracked[fused_add_rms_op] = "fused_add_rms_norm"

        for node in graph.nodes:
            op = called_op(node)
            if op is None:
                if node.op == "call_function" and op_base_name(node.target) == "silu":
                    counts["aten_silu"] += 1
                continue
            if op in tracked:
                counts[tracked[op]] += 1
            elif op_base_name(op) == "silu":
                counts["aten_silu"] += 1
            elif op is torch.ops.aten.contiguous.default:
                counts["aten_contiguous"] += 1
            if quant_op is not None and op is quant_op:
                kwargs = op_call_kwargs(node, op)
                quant_inputs[node] = kwargs.get("input")
                if isinstance(node.target, OpOverload):
                    quant_forms["raw_call"] += 1
                else:
                    quant_forms[op_base_name(node.target)] += 1

        first_site: list[str] = []
        for qnode, qinput in quant_inputs.items():
            producer, chain = walk_views(qinput)
            pop = called_op(producer) if isinstance(producer, fx.Node) else None
            label = tracked.get(pop, "other")
            candidates[f"{label}->{'quant_int8'}(view_chain={chain})"] += 1
            if not first_site and label in ("rms_norm", "fused_add_rms_norm"):
                # one-shot site dump: the exact nodes between the producer and
                # the quant (the form the matchers must reproduce)
                cursor = qinput
                first_site.append(qnode.format_node())
                seen = 0
                while isinstance(cursor, fx.Node) and seen < 6:
                    first_site.append(cursor.format_node())
                    if called_op(cursor) is not None or not cursor.args:
                        break
                    cursor = cursor.args[0]
                    seen += 1
                # and the allocs feeding the quant call (result/scale)
                for key, val in op_call_kwargs(qnode, quant_op).items():
                    if isinstance(val, fx.Node) and val is not qinput:
                        first_site.append(f"{key}={val.format_node()}")

        logger.info(
            "fusion site census (real graph, one-shot): ops=%s "
            "quant_call_forms=%s int8_quant_candidates=%s",
            dict(counts),
            dict(quant_forms),
            dict(candidates),
        )
        if first_site:
            logger.info(
                "fusion site census: first norm->quant site nodes=%s", first_site
            )
