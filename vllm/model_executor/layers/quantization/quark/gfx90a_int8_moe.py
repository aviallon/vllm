"""gfx90a int8 W8A8 MoE dispatcher (v2-clean — staged kernels, raw ctypes).

Small-batch (<=32 tokens) MoE on gfx90a uses the staged int8 MFMA kernels:
  1. pertoken_quant (AITER): bf16 input -> int8 + scale
  2. moe_sorting (AITER): sort tokens by expert
  3. fused_fc1_act (custom): FC1 + SwiGLU/SiLU -> bf16 intermediate
  4. quant_per_token (custom): bf16 -> int8 + scale
  5. fc2 (custom): int8 GEMM + weighted scatter -> output
"""
import ctypes
import os
import torch

GFX90A_INT8_MOE_MAX_TOKENS = 32
_lib = None
BM = 64


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    so_paths = [
        os.path.join(os.environ.get("AITER_ROOT_DIR", "/root/aiter"), "fmoe_int8_gfx90a.so"),
        "/root/fmoe_int8_gfx90a.so",
        "/opt/aiter/fmoe_int8_gfx90a.so",
    ]
    for p in so_paths:
        if os.path.exists(p):
            _lib = ctypes.CDLL(p)
            break
    else:
        raise RuntimeError("fmoe_int8_gfx90a.so not found")
    _lib.fused_fc1_act_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    _lib.fused_fc1_act_launch.restype = None
    _lib.quant_per_token_launch.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    _lib.quant_per_token_launch.restype = None
    _lib.fc2_int8_launch.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    _lib.fc2_int8_launch.restype = None
    return _lib


def apply_gfx90a_int8_moe(layer, x, topk_weights, topk_ids, moe_quant_config):
    lib = _get_lib()
    M = x.shape[0]
    device = x.device
    hidden = layer.w13_weight.shape[2]
    N1 = layer.w13_weight.shape[1]
    inter_dim = layer.w2_weight.shape[2]
    is_g1u1 = (N1 == 2 * inter_dim)
    num_experts = layer.global_num_experts
    stream = torch.cuda.current_stream().cuda_stream

    from aiter import pertoken_quant
    from aiter.fused_moe import moe_sorting

    x_int8, x_scale = pertoken_quant(x, quant_dtype=torch.int8)
    x_int8 = x_int8.view(M, hidden)
    x_scale = x_scale.view(M)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        num_experts, hidden, x.dtype)

    M_sorted = sorted_ids.shape[0]
    M_padded = ((M_sorted + BM - 1) // BM) * BM
    if M_padded > M_sorted:
        sorted_ids_p = torch.full((M_padded,), M, dtype=torch.int32, device=device)
        sorted_ids_p[:M_sorted] = sorted_ids
        sorted_weights_p = torch.zeros(M_padded, dtype=torch.float32, device=device)
        sorted_weights_p[:M_sorted] = sorted_weights
    else:
        sorted_ids_p = sorted_ids
        sorted_weights_p = sorted_weights

    num_blocks = M_padded // BM
    if sorted_expert_ids.shape[0] >= num_blocks:
        expert_ids = sorted_expert_ids[:num_blocks].contiguous()
    else:
        expert_ids = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
        expert_ids[:sorted_expert_ids.shape[0]] = sorted_expert_ids

    safe_ids = sorted_ids_p.clamp(max=M - 1)
    x_gathered = x_int8[safe_ids]
    x_scale_gathered = x_scale[safe_ids]

    inter = torch.zeros(M_padded, inter_dim, dtype=torch.bfloat16, device=device)
    lib.fused_fc1_act_launch(
        ctypes.c_void_p(inter.data_ptr()),
        ctypes.c_void_p(x_gathered.data_ptr()),
        ctypes.c_void_p(layer.w13_weight.data_ptr()),
        ctypes.c_void_p(sorted_ids_p.data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(x_scale_gathered.data_ptr()),
        ctypes.c_void_p(layer.w13_weight_scale.data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(N1), ctypes.c_int(hidden),
        ctypes.c_int(inter_dim), ctypes.c_int(M), ctypes.c_int(int(is_g1u1)),
        ctypes.c_void_p(stream))

    inter_q = torch.empty(M_padded, inter_dim, dtype=torch.int8, device=device)
    inter_scale = torch.empty(M_padded, dtype=torch.float32, device=device)
    lib.quant_per_token_launch(
        ctypes.c_void_p(inter.data_ptr()),
        ctypes.c_void_p(inter_q.data_ptr()),
        ctypes.c_void_p(inter_scale.data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(inter_dim),
        ctypes.c_void_p(stream))

    out_f32 = torch.zeros(M, hidden, dtype=torch.float32, device=device)
    lib.fc2_int8_launch(
        ctypes.c_void_p(out_f32.data_ptr()),
        ctypes.c_void_p(inter_q.data_ptr()),
        ctypes.c_void_p(layer.w2_weight.data_ptr()),
        ctypes.c_void_p(sorted_ids_p.data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(inter_scale.data_ptr()),
        ctypes.c_void_p(layer.w2_weight_scale.data_ptr()),
        ctypes.c_void_p(sorted_weights_p.data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(hidden), ctypes.c_int(inter_dim),
        ctypes.c_int(M), ctypes.c_void_p(stream))

    result = out_f32.bfloat16()
    moe_buf.copy_(result)
    return moe_buf
