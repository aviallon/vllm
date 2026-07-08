"""gfx90a int8 W8A8 custom MoE kernel dispatcher.

Routes small-batch (<=32 tokens) MoE through custom int8 MFMA kernels on gfx90a.
Large batches fall back to vLLM's Triton fused_experts.

The custom kernel is a 3-stage pipeline:
  1. FC1: int8 X @ int8 W1 -> bf16 (raw dequant, no activation)
  2. Activation: SwiGLU (g1u1) or SiLU (g1u0)
  3. Re-quantize intermediate bf16 -> int8 (for FC2)
  4. FC2: int8 @ int8 W2 -> f32 (weighted atomicAdd)

Input to FC1 is already int8 (pre-quantized weights + per-token input quant).
The re-quantization in step 3 is ONLY for the FC1->FC2 intermediate.

Future: a hand-written ASM kernel can replace the .so without changing this interface.
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

    _lib.fc1_int8_launch.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*5 + [ctypes.c_void_p]
    _lib.fc1_int8_launch.restype = None
    _lib.swiglu_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _lib.swiglu_launch.restype = None
    _lib.silu_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _lib.silu_launch.restype = None
    _lib.quant_per_token_launch.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    _lib.quant_per_token_launch.restype = None
    _lib.fc2_int8_launch.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    _lib.fc2_int8_launch.restype = None
    return _lib

BM = 64  # kernel block size (4 waves x 16 rows)

def apply_gfx90a_int8_moe(layer, x, topk_weights, topk_ids, moe_quant_config):
    """Run custom int8 W8A8 MoE for gfx90a.

    Args:
        layer: FusedMoE layer with w13_weight, w2_weight, w13_weight_scale, w2_weight_scale
        x: [num_tokens, hidden] bf16 input (will be quantized to int8)
        topk_weights: [num_tokens, topk] float32
        topk_ids: [num_tokens, topk] int32
        moe_quant_config: FusedMoEQuantConfig (for scale info)
    Returns:
        output: [num_tokens, hidden] bf16
    """
    lib = _get_lib()
    M = x.shape[0]
    device = x.device
    hidden = layer.w13_weight.shape[2]
    N1 = layer.w13_weight.shape[1]
    inter_dim = layer.w2_weight.shape[2]
    is_g1u1 = (N1 == 2 * inter_dim)
    num_experts = layer.global_num_experts
    stream = torch.cuda.current_stream().cuda_stream

    # Step 1: Quantize input bf16 -> int8 (per-token, symmetric)
    from aiter import pertoken_quant
    x_int8, x_scale = pertoken_quant(x, quant_dtype=torch.int8)
    x_int8 = x_int8.view(M, hidden)
    x_scale = x_scale.view(M)

    # Step 2: moe_sorting
    from aiter.fused_moe import moe_sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        num_experts, hidden, x.dtype)

    num_tokens = M
    M_sorted = sorted_ids.shape[0]
    M_padded = ((M_sorted + BM - 1) // BM) * BM

    # Pad sorted arrays
    if M_padded > M_sorted:
        sorted_ids = torch.cat([sorted_ids, torch.full((M_padded - M_sorted,), num_tokens,
            dtype=torch.int32, device=device)])
        sorted_weights = torch.cat([sorted_weights, torch.zeros(M_padded - M_sorted,
            dtype=torch.float32, device=device)])

    # Create expert_ids (one per BM-block)
    num_blocks = M_padded // BM
    expert_ids = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
    for b in range(num_blocks):
        start = b * BM
        end = min(start + BM, M_sorted)
        if end > start:
            expert_ids[b] = sorted_expert_ids[start].item()

    # Gather input tokens in sorted order
    safe_ids = sorted_ids[:M_padded].clamp(max=num_tokens - 1)
    x_gathered = x_int8[safe_ids]
    x_scale_gathered = x_scale[safe_ids]

    # Step 3: FC1
    inter_raw = torch.zeros(M_padded, N1, dtype=torch.bfloat16, device=device)
    lib.fc1_int8_launch(
        ctypes.c_void_p(inter_raw.data_ptr()),
        ctypes.c_void_p(x_gathered.data_ptr()),
        ctypes.c_void_p(layer.w13_weight.data_ptr()),
        ctypes.c_void_p(sorted_ids[:M_padded].data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(x_scale_gathered.data_ptr()),
        ctypes.c_void_p(layer.w13_weight_scale.data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(N1), ctypes.c_int(hidden),
        ctypes.c_int(num_tokens), ctypes.c_int(is_g1u1),
        ctypes.c_void_p(stream))

    # Step 4: Activation
    inter_act = torch.empty(M_padded, inter_dim, dtype=torch.bfloat16, device=device)
    if is_g1u1:
        lib.swiglu_launch(
            ctypes.c_void_p(inter_act.data_ptr()),
            ctypes.c_void_p(inter_raw.data_ptr()),
            ctypes.c_int(M_padded), ctypes.c_int(inter_dim),
            ctypes.c_void_p(stream))
    else:
        lib.silu_launch(
            ctypes.c_void_p(inter_act.data_ptr()),
            ctypes.c_void_p(inter_raw.data_ptr()),
            ctypes.c_int(M_padded), ctypes.c_int(inter_dim),
            ctypes.c_void_p(stream))

    # Step 5: Re-quantize intermediate for FC2
    inter_q = torch.empty(M_padded, inter_dim, dtype=torch.int8, device=device)
    inter_scale = torch.empty(M_padded, dtype=torch.float32, device=device)
    lib.quant_per_token_launch(
        ctypes.c_void_p(inter_act.data_ptr()),
        ctypes.c_void_p(inter_q.data_ptr()),
        ctypes.c_void_p(inter_scale.data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(inter_dim),
        ctypes.c_void_p(stream))

    # Step 6: FC2
    out_f32 = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=device)
    lib.fc2_int8_launch(
        ctypes.c_void_p(out_f32.data_ptr()),
        ctypes.c_void_p(inter_q.data_ptr()),
        ctypes.c_void_p(layer.w2_weight.data_ptr()),
        ctypes.c_void_p(sorted_ids[:M_padded].data_ptr()),
        ctypes.c_void_p(expert_ids.data_ptr()),
        ctypes.c_void_p(inter_scale.data_ptr()),
        ctypes.c_void_p(layer.w2_weight_scale.data_ptr()),
        ctypes.c_void_p(sorted_weights[:M_padded].data_ptr()),
        ctypes.c_int(M_padded), ctypes.c_int(hidden), ctypes.c_int(inter_dim),
        ctypes.c_int(num_tokens), ctypes.c_void_p(stream))

    result = out_f32.bfloat16()
    moe_buf.copy_(result)
    return moe_buf
