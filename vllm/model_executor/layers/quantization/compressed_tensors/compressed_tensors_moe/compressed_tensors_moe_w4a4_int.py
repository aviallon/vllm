# SPDX-License-Identifier: MIT
"""CompressedTensors W4A4 int MoE method for gfx90a.

Integrates with vLLM's modular kernel infrastructure. Weight layout
matches the W8A8 int8 checkpoint (int8 weights + per-channel scales).
During process_weights_after_loading, int8 weights are re-quantized
to int4 packed (torch.bits4x2) and the modular kernel is built with
Gfx90aW4A4MoEExperts.
"""
import torch
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationStrategy,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4Int8MoEMethod"]


class CompressedTensorsW4A4Int8MoEMethod(CompressedTensorsMoEMethod):
    """W4A4 int MoE method for gfx90a.

    Checkpoint is W8A8 int8 (compressed-tensors). At load time,
    weights are re-quantized to int4 packed (torch.bits4x2).
    Activations are dynamically quantized to int4 inside the custom op.
    """

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.layer_name = layer_name
        self.experts_cls = None  # set lazily to avoid import at module load

        per_channel = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL
            and input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                f"W4A4 int MoE requires channel weights + token activations, "
                f"got {weight_quant.strategy}, {input_quant.strategy}"
            )
        self.static_input_scales = not input_quant.dynamic
        if self.static_input_scales:
            raise ValueError("W4A4 int MoE requires dynamic activation quant.")
        self.moe_kernel = None

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        # Same layout as W8A8 int8 — checkpoint is int8
        params_dtype = torch.int8
        w13_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_shards * intermediate_size_per_partition,
                        hidden_size, dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size,
                        intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.ones(num_experts, w13_shards * intermediate_size_per_partition,
                       1, dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        # Re-quantize int8 → int4 packed (torch.bits4x2)
        for w_name, s_name in [("w13_weight", "w13_weight_scale"),
                                ("w2_weight", "w2_weight_scale")]:
            w = getattr(layer, w_name)  # [E, N, K] int8
            s = getattr(layer, s_name)  # [E, N, 1] float32

            # Dequantize to float, then re-quantize to int4
            w_float = w.data.float() * s.data  # [E, N, K]
            max_val = w_float.abs().amax(dim=-1, keepdim=True)  # [E, N, 1]
            new_scale = (max_val / 7.0).clamp(min=1e-8)
            q = (w_float / new_scale).round().clamp(-8, 7).to(torch.int8)

            # Pack: [E, N, K] int8 → [E, N, K/2] uint8 → bits4x2
            E, N, K = q.shape
            flat = q.view(-1).to(torch.int16) & 0xF
            packed = (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8)
            packed = packed.view(E, N, K // 2).contiguous()
            packed_b4x2 = packed.view(torch.bits4x2)

            # Replace parameter
            from vllm.model_executor.layers.quantization.utils import (
                replace_parameter,
            )
            replace_parameter(layer, w_name,
                              torch.nn.Parameter(packed_b4x2, requires_grad=False))
            replace_parameter(layer, s_name,
                              torch.nn.Parameter(new_scale.squeeze(-1).float(),
                                                 requires_grad=False))

            logger.debug("W4A4: re-quantized %s int8 [%d,%d,%d] → int4 [%d,%d,%d]",
                         w_name, E, N, K, E, N, K // 2)

        # Build modular kernel
        self._build_modular_kernel(layer)

        # Pre-load HIP module (before torch.compile captures)
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.w4a4_op import (
            ensure_moe_loaded,
        )
        ensure_moe_loaded()

    def _build_modular_kernel(self, layer: FusedMoE):
        from vllm.model_executor.layers.fused_moe.experts.gfx90a_w4a4_moe import (
            Gfx90aW4A4MoEExperts,
        )
        self.experts_cls = Gfx90aW4A4MoEExperts

        # Quant config — signals W4A4 to the experts
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            allow_new_interface=True,
            use_monolithic=False,
        )
        assert prepare_finalize is not None

        if prepare_finalize.activation_format == \
                mk.FusedMoEActivationFormat.BatchedExperts:
            max_tokens = prepare_finalize.max_num_tokens_per_rank()
            assert max_tokens is not None
            experts = self.experts_cls(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=max_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers())
        else:
            experts = self.experts_cls(
                moe_config=self.moe, quant_config=self.moe_quant_config)

        self.moe_kernel = mk.FusedMoEKernel(
            prepare_finalize, experts,
            shared_experts=layer.shared_experts,
            inplace=not self.moe.disable_inplace)

    def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig:
        # Unquantized config — our experts handle quantization internally.
        # The prepare step passes unquantized BF16 inputs (expects_unquantized_inputs=True).
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise ValueError(
            f"{self.__class__.__name__} uses modular kernel init. "
            "This function should not be called.")

    @property
    def is_monolithic(self):
        return False

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )
