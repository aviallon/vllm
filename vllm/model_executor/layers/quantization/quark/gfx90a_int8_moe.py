"""gfx90a int8 W8A8 fully-fused MoE dispatcher (v4 — single kernel + torch custom op).

Small-batch (<=32 tokens) MoE on gfx90a runs through ONE custom kernel launch
per layer (fused_moe_int8_kernel): FC1 + activation + per-token requant + FC2,
keeping all intermediates in LDS. Large batches fall back to Triton fused_experts.

The kernel is wrapped in a torch.library custom op so vLLM's CUDA graph / inductor
piecewise graph captures it (a bare ctypes call is invisible to the graph tracer and
incurs ~5x eager dispatch overhead).

Kernel grid: one block (256 threads) per (token, expert) pair. Uses
__builtin_amdgcn_mfma_i32_16x16x16i8 on CDNA2 (MI210).
"""
import ctypes
import os
import torch

GFX90A_INT8_MOE_MAX_TOKENS = 32
_lib = None


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    so_paths = [
        os.path.join(os.environ.get("AITER_ROOT_DIR", "/root/aiter"), "fused_moe_int8.so"),
        "/root/fused_moe_int8.so",
        "/opt/aiter/fused_moe_int8.so",
    ]
    for p in so_paths:
        if os.path.exists(p):
            _lib = ctypes.CDLL(p)
            break
    else:
        raise RuntimeError("fused_moe_int8.so not found")
    _lib.fused_moe_int8_launch.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int]*6 + [ctypes.c_void_p]
    _lib.fused_moe_int8_launch.restype = None
    return _lib


@torch.library.custom_op("gfx90a::fused_int8_moe", mutates_args=())
def _fused_int8_moe_op(
    x_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids_per_token: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_tokens: int,
    is_g1u1: bool,
) -> torch.Tensor:
    lib = _get_lib()
    device = x_int8.device
    hidden = w13_weight.shape[2]
    N1 = w13_weight.shape[1]
    inter_dim = w2_weight.shape[2]
    M = sorted_token_ids.shape[0]
    stream = torch.cuda.current_stream().cuda_stream

    out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=device)
    lib.fused_moe_int8_launch(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(x_int8.data_ptr()),
        ctypes.c_void_p(w13_weight.data_ptr()),
        ctypes.c_void_p(w2_weight.data_ptr()),
        ctypes.c_void_p(sorted_token_ids.data_ptr()),
        ctypes.c_void_p(expert_ids_per_token.data_ptr()),
        ctypes.c_void_p(x_scale.data_ptr()),
        ctypes.c_void_p(w13_weight_scale.data_ptr()),
        ctypes.c_void_p(w2_weight_scale.data_ptr()),
        ctypes.c_void_p(sorted_weights.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(N1),
        ctypes.c_int(inter_dim), ctypes.c_int(num_tokens), ctypes.c_int(int(is_g1u1)),
        ctypes.c_void_p(stream))
    return out.to(torch.bfloat16)


@_fused_int8_moe_op.register_fake
def _fused_int8_moe_op_fake(
    x_int8, x_scale, w13_weight, w13_weight_scale, w2_weight, w2_weight_scale,
    sorted_token_ids, expert_ids_per_token, sorted_weights, num_tokens, is_g1u1,
):
    hidden = w13_weight.shape[2]
    return torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=x_int8.device)


def apply_gfx90a_int8_moe(layer, x, topk_weights, topk_ids, moe_quant_config):
    """Fully-fused int8 W8A8 MoE for gfx90a (batch <= 32).

    Builds per-(token,expert) flat index arrays and calls the single fused kernel.
    """
    M = x.shape[0]
    hidden = layer.w13_weight.shape[2]
    N1 = layer.w13_weight.shape[1]
    inter_dim = layer.w2_weight.shape[2]
    is_g1u1 = (N1 == 2 * inter_dim)
    device = x.device
    topk = topk_ids.shape[1]

    from aiter import pertoken_quant

    x_int8, x_scale = pertoken_quant(x, quant_dtype=torch.int8)
    x_int8 = x_int8.view(M, hidden)
    x_scale = x_scale.view(M)

    # Flat (token, expert) pairs: one per topk slot. No sorting needed — the
    # fused kernel is one block per pair and atomicAdds into the output.
    num_pairs = M * topk
    sorted_token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(topk)
    expert_ids_per_token = topk_ids.reshape(-1).to(torch.int32)
    sorted_weights = topk_weights.reshape(-1).to(torch.float32)

    out = torch.ops.gfx90a.fused_int8_moe(
        x_int8, x_scale,
        layer.w13_weight, layer.w13_weight_scale,
        layer.w2_weight, layer.w2_weight_scale,
        sorted_token_ids, expert_ids_per_token, sorted_weights,
        M, is_g1u1,
    )
    return out
