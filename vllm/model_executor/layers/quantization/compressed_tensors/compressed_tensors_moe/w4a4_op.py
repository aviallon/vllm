# SPDX-License-Identifier: MIT
"""W4A4 MoE GEMM custom op for gfx90a.

Registers `vllm::w4a4_gfx90a_moe_gemm` — a single fused op that:
1. Quantizes BF16 activations to int4 (PyTorch GPU ops)
2. Sorts tokens by expert via moe_align_block_size (vLLM C++ op)
3. Launches the HIP W4A4 MoE kernel (opaque to compiler)

The op takes unquantized inputs and topk_ids, returns the per-expert
GEMM output. The experts class calls it twice (w1 + w2) with an
activation in between.

torch.compile: the op is a single opaque node (fake impl returns shape).
cudagraphs: grid size from tensor shapes (static), no CPU-GPU sync.
"""
import ctypes
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

__all__ = ["w4a4_gfx90a_moe_gemm", "is_gfx90a", "ensure_moe_loaded"]

# ─── Constants ─────────────────────────────────────────────────────
_NW = 4
_BM = _NW * 16  # 64
_BN = 16
_K_TILE_HALF = 64
_LDS_SIZE = _K_TILE_HALF * _BN  # 1024 bytes

# ─── HIP module loading ────────────────────────────────────────────
_hip = None
_module = None
_kernels = {}

def is_gfx90a() -> bool:
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_rocm():
            return False
        return "gfx90a" in str(current_platform.get_device_arch())
    except Exception:
        try:
            return torch.cuda.get_device_name(0).startswith("AMD Instinct MI210")
        except Exception:
            return False

def _get_hip():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        _hip.hipModuleLoad.restype = ctypes.c_int
        _hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleGetFunction.restype = ctypes.c_int
        _hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _hip.hipModuleLaunchKernel.restype = ctypes.c_int
        _hip.hipModuleGetGlobal.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_char_p]
        _hip.hipModuleGetGlobal.restype = ctypes.c_int
        _hip.hipMemcpyHtoD.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        _hip.hipMemcpyHtoD.restype = ctypes.c_int
        _hip.hipSetDevice.argtypes = [ctypes.c_int]
        _hip.hipSetDevice.restype = ctypes.c_int
    return _hip

def _chk(err, what):
    if err != 0:
        raise RuntimeError(f"HIP error {err} at {what}")

def _get_co_path() -> str:
    for p in [
        os.path.join(os.path.dirname(__file__), "w4a4_kernels.co"),
        "/root/w4a4-workspace/vllm_w4a4/w4a4_kernels.co",
        "/opt/vllm/lib/w4a4_kernels.co",
    ]:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("w4a4_kernels.co not found")

def ensure_moe_loaded():
    """Load HIP module + init LUT. Call before first kernel launch."""
    global _module, _kernels
    if _module is not None:
        return
    hip = _get_hip()
    co_path = _get_co_path()
    hip.hipSetDevice(torch.cuda.current_device())
    _module = ctypes.c_void_p()
    _chk(hip.hipModuleLoad(ctypes.byref(_module), co_path.encode()), "hipModuleLoad")
    for name in ["gemm_w4a4_moe_kernel", "gemm_w4a4_v3_kernel", "gemm_w8a8_kernel",
                 "w4a4_unpack_a_kernel", "w4a4_unpack_b_kernel"]:
        fn = ctypes.c_void_p()
        err = hip.hipModuleGetFunction(ctypes.byref(fn), _module, name.encode())
        if err == 0:
            _kernels[name] = fn
        else:
            logger.warning("W4A4 MoE: kernel %s not found in .co (error %d)", name, err)
    # Init LUT
    lut = []
    for i in range(256):
        lo = i & 0xF
        hi = (i >> 4) & 0xF
        if lo & 8: lo |= 0xF0
        if hi & 8: hi |= 0xF0
        lut.append((lo & 0xFF) | ((hi & 0xFF) << 8))
    lut_arr = (ctypes.c_uint32 * 256)(*lut)
    dev_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    _chk(hip.hipModuleGetGlobal(ctypes.byref(dev_ptr), ctypes.byref(size),
                                _module, b"w4a4_unpack_lut"),
         "hipModuleGetGlobal(w4a4_unpack_lut)")
    hip.hipMemcpyHtoD(dev_ptr, lut_arr, ctypes.c_size_t(256 * 4))
    logger.info("W4A4 MoE: HIP module loaded from %s", co_path)

def _launch(kernel_name, gx, gy, gz, bx, by, bz, shared, args_list):
    ensure_moe_loaded()
    hip = _get_hip()
    fn = _kernels[kernel_name]
    n = len(args_list)
    arg_ptrs = (ctypes.c_void_p * n)()
    storage = []
    for i, arg in enumerate(args_list):
        if isinstance(arg, int):
            v = ctypes.c_uint64(arg) if arg > 0xFFFFFFFF else ctypes.c_int(arg)
        elif isinstance(arg, float):
            v = ctypes.c_float(arg)
        else:
            v = ctypes.c_void_p(arg)
        storage.append(v)
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(storage[-1]), ctypes.c_void_p)
    _chk(hip.hipModuleLaunchKernel(
        fn, gx, gy, gz, bx, by, bz, shared,
        ctypes.c_void_p(0), arg_ptrs, ctypes.c_void_p(0)),
        f"hipModuleLaunchKernel({kernel_name})")

# ─── int4 activation quantization (PyTorch GPU ops) ────────────────

def _quantize_act_int4(x: torch.Tensor):
    """[M, K] BF16 → ([M, K/2] uint8, [M] float32)"""
    x_f = x.float()
    max_val = x_f.abs().amax(dim=-1, keepdim=True)
    scale = (max_val / 7.0).clamp(min=1e-8)
    q = (x_f / scale).round().clamp(-8, 7).to(torch.int8)
    flat = q.view(-1).to(torch.int16) & 0xF
    packed = (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8)
    packed = packed.view(x.shape[0], x.shape[1] // 2).contiguous()
    return packed, scale.squeeze(-1).float().contiguous()

# ─── Custom op: vllm::w4a4_gfx90a_moe_gemm ─────────────────────────

def w4a4_gfx90a_moe_gemm(
    x: torch.Tensor,              # [M, K] BF16
    w_packed: torch.Tensor,       # [E, N, K/2] bits4x2
    w_scale: torch.Tensor,        # [E, N] float32
    topk_ids: torch.Tensor,       # [M, topk] int32
) -> torch.Tensor:                # [M * topk, N] BF16
    """Fused W4A4 MoE GEMM with expert dispatch.

    Quantizes activations to int4, sorts tokens by expert, and runs
    the W4A4 MFMA kernel in a single launch.
    """
    M, K = x.shape
    E = w_packed.shape[0]
    N = w_packed.shape[1]
    topk = topk_ids.shape[1]

    # 1. Quantize activations (PyTorch GPU ops)
    x_packed, a_scale = _quantize_act_int4(x)

    # 2. Sort tokens by expert (vLLM C++ op)
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, _BM, E,
    )

    # 3. Allocate output [M * topk, N]
    output = torch.empty(M * topk, N, device=x.device, dtype=torch.bfloat16)

    # 4. Launch kernel — grid from tensor shapes (static for cudagraphs)
    EM = sorted_token_ids.shape[0]  # static: max_num_tokens_padded
    nN = N // 16
    em_blocks = (EM + _BM - 1) // _BM

    # View weights as uint8 for data_ptr (bits4x2 has same storage)
    w_uint8 = w_packed.view(torch.uint8)

    _launch("gemm_w4a4_moe_kernel", nN, em_blocks, 1,
            _NW * 64, 1, 1, _LDS_SIZE,
            [x_packed.data_ptr(), w_uint8.data_ptr(),
             a_scale.data_ptr(), w_scale.data_ptr(),
             0,  # no bias
             output.data_ptr(),
             sorted_token_ids.data_ptr(),
             expert_ids.data_ptr(),
             num_tokens_post_padded.data_ptr(),
             M, topk, N, K])
    return output

def w4a4_gfx90a_moe_gemm_fake(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Meta impl for torch.compile tracing."""
    M = x.shape[0]
    topk = topk_ids.shape[1]
    N = w_packed.shape[1]
    return x.new_empty(M * topk, N, dtype=torch.bfloat16)

# Register
direct_register_custom_op(
    op_name="w4a4_gfx90a_moe_gemm",
    op_func=w4a4_gfx90a_moe_gemm,
    fake_impl=w4a4_gfx90a_moe_gemm_fake,
)
