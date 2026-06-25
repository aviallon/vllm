# SPDX-License-Identifier: MIT
"""W4A4 MoE experts for gfx90a — uses fused W4A4 custom op.

Calls torch.ops.vllm.w4a4_gfx90a_moe_gemm twice (w1 + w2) with SiLU
activation in between. Token dispatch is handled inside the custom op
via moe_align_block_size (same pattern as Triton's fused_moe_kernel).
"""
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["Gfx90aW4A4MoEExperts"]


class Gfx90aW4A4MoEExperts(mk.FusedMoEExpertsModular):
    """MoE experts using W4A4 integer GEMM via fused custom op."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_monolithic() -> bool:
        return False

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU, MoEActivation.GELU, MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI, MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL, MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(cfg: FusedMoEParallelConfig) -> bool:
        return not (cfg.use_fi_nvl_two_sided_kernels
                    or cfg.use_fi_nvl_one_sided_kernels)

    @staticmethod
    def _supports_batch_invariance():
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(self, M, N, K, topk, global_num_experts,
                         local_num_experts, expert_tokens_meta, activation):
        act_out = self.adjust_N_for_activation(N, activation)
        ws1 = (M, topk, max(act_out, K))
        ws2 = (M, topk, max(N, K))
        out = (M, K)
        return (ws1, ws2, out)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        """W1 → activation → W2, both via W4A4 custom op."""
        E, num_tokens, N, K, topk = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E

        # ── W1 (gate_up): [M, K] @ [E, N_w1, K/2] → [M*topk, N_w1] ──
        cache1_flat = torch.ops.vllm.w4a4_gfx90a_moe_gemm(
            hidden_states, w1, self.w1_scale.squeeze(-1), topk_ids,
        )  # [M*topk, N_w1] BF16

        N_w1 = cache1_flat.shape[-1]
        intermediate_cache1 = _resize_cache(
            workspace2, (num_tokens, topk, N_w1))
        intermediate_cache1.copy_(cache1_flat.view(num_tokens, topk, N_w1))

        # ── Activation (SiLU gate * up) ──────────────────────────────
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * topk, cache2_dim))
        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N_w1))

        # ── W2 (down): [M*topk, inter] @ [E, K, inter/2] → [M*topk, K] ──
        # For W2, each "token" is already a (token, expert) pair.
        # Build topk_ids where each row has topk=1 and the expert is
        # determined by the original topk_ids.
        w2_topk_ids = topk_ids.view(-1, 1).to(torch.int32)  # [M*topk, 1]

        cache3_flat = torch.ops.vllm.w4a4_gfx90a_moe_gemm(
            intermediate_cache2, w2, self.w2_scale.squeeze(-1), w2_topk_ids,
        )  # [M*topk, K] BF16

        intermediate_cache3 = _resize_cache(
            workspace2, (num_tokens, topk, K))
        intermediate_cache3.copy_(cache3_flat.view(num_tokens, topk, K))

        # ── Reduce topk → output ────────────────────────────────────
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input, output):
        ops.moe_sum(input, output)
