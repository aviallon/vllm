# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    has_triton_scaled_mm_fused_quant_op,
    triton_scaled_mm,
    triton_scaled_mm_fused_quant,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)

logger = logging.getLogger(__name__)

# Branch proof + shape probe for the fused-quant path (megakernel campaign):
# every (M, K, N) actually served by the fused kernel is logged once, so "the
# optimisation is not deployed" can never be mistaken for "it does not work",
# and the intermediate-traffic accounting gets its real shapes.
#
# ⚠ The dedup key MUST be the STRING form of the shapes: under torch.compile
# (backed dynamic shapes) the dims are SymInts, and putting them in a tuple/set
# compares them -> guard creation -> ConstraintViolationError -> the engine
# crash-loops at init (measured: arm A of the wall, 2026-10-23).  str() is
# guard-free; comparisons are not.
_FUSED_QUANT_LOGGED_SHAPES: set[str] = set()


def _log_fused_quant_shape(x_shape: torch.Size, w_shape: torch.Size) -> None:
    key = f"x{tuple(x_shape)}-w{tuple(w_shape)}"
    if key in _FUSED_QUANT_LOGGED_SHAPES or len(_FUSED_QUANT_LOGGED_SHAPES) >= 64:
        return
    _FUSED_QUANT_LOGGED_SHAPES.add(key)
    logger.info(
        "scaled_mm fused-quant ACTIVE (VLLM_SCALED_MM_FUSED_QUANT=1): "
        "dynamic per-token int8 quant in the GEMM prologue, %s",
        key,
    )


def fused_quant_eligible(
    symmetric: bool,
    i_s: torch.Tensor | None,
    x: object,
    w_q: torch.Tensor,
) -> bool:
    """Decision table for the fused-quant path (megakernel campaign, pair (a)).

    Everything the fused kernel does not cover keeps the shipped two-kernel
    chain verbatim: pre-quantized input (`as_quantized_activation`), static
    input scale, asymmetric quantization (azp), non-float or non-int8 dtypes.
    """
    return (
        envs.VLLM_SCALED_MM_FUSED_QUANT
        and has_triton_scaled_mm_fused_quant_op()
        and symmetric
        and i_s is None
        and isinstance(x, torch.Tensor)
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and w_q.dtype == torch.int8
    )


class TritonInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            return True, None
        return False, "requires ROCm, CUDA or XPU."

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, _, i_s, _, _ = self._get_layer_params(layer)
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Triton kernel supports only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            assert i_s is not None

            if self.config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(i_s.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # Reconstruct the ranges to find a single scale and azp
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (i_s * (int8_traits.max - azps)).max()
                range_min = (i_s * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(scale, requires_grad=False),
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer,
                    i_zp_name,
                    torch.nn.Parameter(azp, requires_grad=False),
                )
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # See csrc/quantization/w8a8/cutlass/Epilogues.md for the math.
        if not self.config.input_symmetric:
            weight = getattr(layer, w_q_name)
            # weight is already transposed to [K, N], sum over K (dim=0)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.config.is_static_input_scheme:
                # Fold azp into azp_adj for the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )
        else:
            setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        symmetric = azp_adj is None
        qa = as_quantized_activation(x, self.input_quant_key())
        # Pair (a) of the megakernel campaign: fold the dynamic per-token int8
        # quantization of the activation into the GEMM prologue (bit-exact, see
        # scaled_mm_fused_quant_kernel).  Gated OFF by default; every case the
        # fused path does not cover (pre-quantized input, static scale,
        # asymmetric quant) keeps the shipped two-kernel chain verbatim.
        use_fused_quant = (
            qa is None and fused_quant_eligible(symmetric, i_s, x, w_q)
        )
        if use_fused_quant:
            assert isinstance(x, torch.Tensor)
            orig_dtype = x.dtype
            out, x_s = triton_scaled_mm_fused_quant(
                x.contiguous(), w_q, scale_b=w_s, out_dtype=orig_dtype, bias=bias
            )
            x_zp = None
            _log_fused_quant_shape(x.shape, w_q.shape)
        else:
            if qa is not None:
                x_q, x_s, x_zp = qa.data, qa.scale, None
                orig_dtype = qa.orig_dtype
            else:
                assert isinstance(x, torch.Tensor)
                x_q, x_s, x_zp = ops.scaled_int8_quant(
                    x.contiguous(), i_s, i_zp, symmetric=symmetric
                )
                orig_dtype = x.dtype

            out = triton_scaled_mm(
                x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=orig_dtype, bias=bias
            )

        if azp_adj is not None:
            # Asymmetric quantization: subtract the zero-point correction.
            # D = scale_a * scale_b * (A_q @ B_q - azp * azp_adj) + bias
            # triton_scaled_mm already computed scale_a * scale_b * (A_q @ B_q) + bias
            # so we subtract scale_a * scale_b * azp * azp_adj
            #
            # x_s: [M, 1] or scalar, w_s: [N, 1] or scalar, azp_adj: [1, N]
            # Reshape w_s from [N, 1] to [1, N] for proper broadcasting.
            w_s_row = w_s.view(1, -1) if w_s.dim() > 0 else w_s
            static = i_zp is not None
            if not static and x_zp is not None:
                # Dynamic per-token: azp is per-token, azp_adj is per-channel
                # x_zp: [M, 1], azp_adj: [1, N]
                out -= x_s * w_s_row * (x_zp * azp_adj).to(orig_dtype)
            else:
                # Static per-tensor: azp already folded into azp_adj
                out -= (x_s * w_s_row * azp_adj).to(orig_dtype)

        return out


class TritonFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "only CUDA-alike and XPU devices are supported."
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            A,
            B,
            As,
            Bs,
            list(self.weight_group_shape),
            self.config.out_dtype,
        )


# TODO we should be able to change the type of block_size to GroupShape
# after we resolve GroupShape compilation issue
# https://github.com/vllm-project/vllm/issues/25270
def _w8a8_triton_block_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    return w8a8_triton_block_scaled_mm(
        qx, weight, x_scale, weight_scale, block_size, output_dtype
    )


def _w8a8_triton_block_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "w8a8_triton_block_scaled_mm_func",
    _w8a8_triton_block_scaled_mm_func,
    fake_impl=_w8a8_triton_block_scaled_mm_fake,
)
