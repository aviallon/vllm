# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused residual-add + RMSNorm + per-token int8 quant (v2 "norm+quant"),
and its one-shot two-rank all-reduce variant (the "3-way": AR + norm + quant in
a single Triton kernel).

Numeric contract (research/ar-rms-quant-v2-spec-2026-10-22.md section 3.2) --
the target is the chain the engine *executes* at the w8a8 decode sites:

    x_f    = bf16(fp32(x_self) + fp32(x_peer))   # AR output, bf16 (no peer: x)
    s      = fp32(x_f) + fp32(residual)          # residual add IN FP32
    res_out = bf16(s)                            # residual rounded HERE
    var    = mean_over_hidden(s * s)             # fp32, non-rounded values
    y      = s * rsqrt(var + eps) * w_fp32       # weight is fp32 (Gemma 1+w)
    y_bf16 = bf16(y)                             # single norm output rounding
    amax   = max_over_hidden(|y_bf16|)           # order independent
    scale  = amax / 127                          # NO clamp
    inv_s  = 0 if amax == 0 else 127 / amax      # zero special-case, 1 division
    q      = clamp(rint(y_bf16 * inv_s), -128, 127)   # nearbyint, half-to-even

Only the reduction order of ``var`` (class-3, ~1e-7) and the ``rsqrt``
primitive may differ from the baseline: the baseline's ``torch.rsqrt`` is
inductor-compiled, and pinning its exact lowering against ours
(``tl.math.rsqrt`` -> ``v_rsq_f32``) is the still-open S0 item (REPORT
section 2). Everything else is bit-exact by construction.

This mirrors, expression for expression, the native
``vllm/ir/ops/layernorm.py::fused_add_rms_norm`` decomposition followed by
``_C.dynamic_scaled_int8_quant`` (``csrc/libtorch_stable/quantization/w8a8/
int8/scaled_quant.cu``: ``scale_out = absmax/127`` without clamp,
``inv_s = (absmax == 0) ? 0 : 127/absmax``, ``float_to_int8_rn`` =
``nearbyint`` + saturate). Only the reduction order of ``var`` can differ
(class-3 divergence, ~1e-7); the quant path is bit-exact given identical
``y_bf16`` (amax/scale/inv_s are order independent).

The one-shot all-reduce path uses the data shape proven on gfx90a by the
2026-10-22 micro-experiment (research/ar-rms-quant-v2-spec-2026-10-22.md
section 1.5): each rank *publishes* its row with a plain LOCAL store into its
own symmetric slot, releases a per-row int32 flag with
``atomic_xchg(sem="release", scope="sys")``, spins on the peer's flag with
``atomic_add(0, sem="acquire", scope="sys")`` (the non-hoistable spin form),
then *loads* the peer's row (the proven direction -- plain peer STORES were
anomalous in the probe, peer loads were exact). All flags/accumulators are
integers: MI200 Infinity Fabric has NO FP32 atomic add (silent NOP). The
sequence number is derived IN-KERNEL from the persistent flag slot, never
passed as an argument -- a baked scalar would be captured in the CUDA graph
and replay #2+ would sail through the barrier on a stale sequence.
"""

import threading
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# rounding primitive: half-to-even (FE_TONEAREST), the ROCm form of
# ``float_to_int8_rn``'s std::nearbyint. tl.math has no rint in Triton 3.8;
# libdevice is the documented lowering of llvm rint/nearbyint on both backends.
# ---------------------------------------------------------------------------
try:
    from triton.language.extra.hip.libdevice import nearbyint as _nearbyint
except ImportError:  # pragma: no cover - non-ROCm toolchains
    from triton.language.extra.cuda.libdevice import nearbyint as _nearbyint

# Fused-regime row bound. The spec borrowed the one-stage custom-AR bound
# (kMaxBlocks=80); the real decode universe of this deployment is bounded by
# max_num_seqs (128) and by the captured decode graphs (cudagraph_capture_sizes
# ends at 256), while prefill/mixed chunks run up to max_num_batched_tokens
# (4096). Sizing the comm buffers for 256 rows (1 MiB/rank) covers EVERY
# captured decode graph with the fused kernel and leaves only prefill-scale
# calls on the exact-chain fallback. At 80, the fallback would have taken
# c96..c256 decode -- and the eager fallback chain costs ~7 extra launches per
# site vs the compiled baseline (measured cost model, 2026-10-22).
MAX_ROWS = 256
"""Decode-only regime bound (matches the one-stage custom-AR limit)."""

FLAG_STRIDE_BYTES = 128
"""Padding between per-row flags (AITER Signal layout: flags must not share a
cache line -- the 2026-10-22 probe's data-then-flag anomaly tracked a
req/ack/data set sharing 16-byte allocation granularity)."""

_NUM_WARPS = 8


@triton.jit
def _norm_quant_int8_kernel(
    x_ptr,
    res_ptr,
    weight_ptr,
    out_ptr,
    scale_ptr,
    slot_ptr,
    peer_ptr,
    flag_ptr,
    peer_flag_ptr,
    eps,
    stride_x,
    stride_res,
    stride_out,
    stride_slot,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PEER: tl.constexpr,
    FLAG_STRIDE: tl.constexpr,
):
    """One program per row; the full row is reduced inside the program.

    PEER=1 adds the publish + barrier + peer-sum phases (3-way); PEER=0 takes
    ``x`` itself as ``x_f`` (norm+quant only). Both share one numeric chain.
    """
    row = tl.program_id(0)

    # int64 scalar args -> pointers (the peer buffers arrive as raw addresses
    # opened from hipIpc handles; the 2026-10-22 probe proved this exact cast
    # form on gfx90a). All zero when PEER=0, never dereferenced there.
    # The explicit promotion to i64 is load-bearing: the Triton JIT infers a
    # Python int argument as i32 when it fits (the PEER=0 dummies are 0), and
    # tt.int_to_ptr REJECTS an i32 operand (measured 2026-10-22 on aragorn:
    # "'tt.int_to_ptr' op operand #0 must be 64-bit signless integer").
    slot_p = tl.cast(slot_ptr.to(tl.int64), tl.pointer_type(tl.bfloat16))
    peer_p = tl.cast(peer_ptr.to(tl.int64), tl.pointer_type(tl.bfloat16))
    flag_p = tl.cast(flag_ptr.to(tl.int64), tl.pointer_type(tl.int32))
    pflag_p = tl.cast(peer_flag_ptr.to(tl.int64), tl.pointer_type(tl.int32))

    # ---- phase 0 (PEER): publish my row, then release my flag, then wait
    # for the peer's flag. Order is load-bearing: data stores strictly before
    # the release xchg (release/acquire pair gives the peer the data).
    if PEER:
        for base in range(0, H, BLOCK_H):
            offs = base + tl.arange(0, BLOCK_H)
            m = offs < H
            xv = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
            tl.store(slot_p + row * stride_slot + offs, xv, mask=m)
        f_off = row * FLAG_STRIDE
        # sequence derived from MY persistent flag slot, in-kernel (graph-safe)
        cur = tl.atomic_add(flag_p + f_off, 0, sem="acquire", scope="sys")
        seq = cur + 1
        tl.atomic_xchg(flag_p + f_off, seq, sem="release", scope="sys")
        # spin on the PEER's flag with an atomic RMW: effects-bearing, so
        # neither Triton's LICM nor LLVM may hoist it out of the loop.
        # `!= seq` (fail loud) rather than the spec's `< seq` (fail silent):
        # if the peer were ever a full call ahead on this row's monotone
        # counter (its next-call xchg lands between my xchg and my first
        # spin read -- a microsecond-scale window needing my wave to be
        # descheduled for the peer's whole next kernel), `!=` hangs both
        # ranks while `<` would silently read the NEXT call's slot data.
        # Hang beats silent corruption here (the spec's own doctrine); the
        # window is documented in REPORT section 3 note 9.
        v = tl.atomic_add(pflag_p + f_off, 0, sem="acquire", scope="sys")
        while v != seq:
            v = tl.atomic_add(
                pflag_p + f_off, 0, sem="acquire", scope="sys"
            )

    # ---- phase 1: residual add (fp32, unrounded) + variance accumulator
    var_acc = 0.0
    for base in range(0, H, BLOCK_H):
        offs = base + tl.arange(0, BLOCK_H)
        m = offs < H
        if PEER:
            a = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
            b = tl.load(peer_p + row * stride_slot + offs, mask=m, other=0.0)
            xf = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        else:
            xf = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
        rs = tl.load(res_ptr + row * stride_res + offs, mask=m, other=0.0)
        s = xf.to(tl.float32) + rs.to(tl.float32)
        var_acc += tl.sum(s * s)
    # H is a power of two for this model (2048 = 2^11): the division is exact,
    # only the summation order above can differ from the baseline tree.
    var = var_acc / H
    rstd = tl.math.rsqrt(var + eps)

    # ---- phase 2: weighted norm -> bf16 -> amax (order independent)
    amax = 0.0
    for base in range(0, H, BLOCK_H):
        offs = base + tl.arange(0, BLOCK_H)
        m = offs < H
        if PEER:
            a = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
            b = tl.load(peer_p + row * stride_slot + offs, mask=m, other=0.0)
            xf = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        else:
            xf = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
        rs = tl.load(res_ptr + row * stride_res + offs, mask=m, other=0.0)
        s = xf.to(tl.float32) + rs.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=m, other=0.0)
        y = (s * rstd) * w.to(tl.float32)
        yb = y.to(tl.bfloat16)
        amax = tl.maximum(amax, tl.max(tl.abs(yb.to(tl.float32))))

    # ---- scale + inverse scale (exact contract of dynamic_scaled_int8_quant)
    scale_val = amax / 127.0
    tl.store(scale_ptr + row, scale_val)
    inv_s = tl.where(amax == 0.0, 0.0, 127.0 / amax)

    # ---- phase 3: quantize + write the new residual last (each element is
    # read before it is written, so res_out == res is aliasing-safe).
    for base in range(0, H, BLOCK_H):
        offs = base + tl.arange(0, BLOCK_H)
        m = offs < H
        if PEER:
            a = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
            b = tl.load(peer_p + row * stride_slot + offs, mask=m, other=0.0)
            xf = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        else:
            xf = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
        rs = tl.load(res_ptr + row * stride_res + offs, mask=m, other=0.0)
        s = xf.to(tl.float32) + rs.to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=m, other=0.0)
        y = (s * rstd) * w.to(tl.float32)
        yb = y.to(tl.bfloat16)
        q = _nearbyint(yb.to(tl.float32) * inv_s)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(out_ptr + row * stride_out + offs, q.to(tl.int8), mask=m)
        tl.store(res_ptr + row * stride_res + offs, s.to(tl.bfloat16), mask=m)


@triton.jit
def _zero_kernel(ptr, n, BLOCK: tl.constexpr):
    p = tl.cast(ptr.to(tl.int64), tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    for base in range(0, n, BLOCK):
        o = base + offs
        tl.store(p + o, tl.zeros([BLOCK], dtype=tl.int32), mask=o < n)


# ---------------------------------------------------------------------------
# symmetric buffer plumbing (the one non-kernel piece). Reuses vLLM's own
# custom-all-reduce IPC mechanism (``_C`` hipIpc handles through
# ``CustomAllreduce.create_shared_buffer``) -- the same registration path the
# production all-reduce uses on this cluster -- and hands the raw peer
# addresses to the kernel as int64 scalars, exactly what the Triton side needs.
# ---------------------------------------------------------------------------
class ArNormQuantComm:
    """Own + peer symmetric buffers for the one-shot 2-rank AR kernel."""

    def __init__(self, group: Any, rows: int, hidden: int) -> None:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.rows = rows
        self.hidden = hidden
        data_bytes = rows * hidden * 2  # bf16
        flag_bytes = rows * FLAG_STRIDE_BYTES
        # registered BEFORE any graph capture; pointers are process-lifetime
        self.data_ptrs = CustomAllreduce.create_shared_buffer(
            data_bytes, group=group
        )
        self.flag_ptrs = CustomAllreduce.create_shared_buffer(
            flag_bytes, group=group
        )
        rank = torch.distributed.get_rank(group=group)
        peer = 1 - rank
        assert len(self.data_ptrs) == 2, "TP=2 only (v2 scope)"
        self.data_ptr = self.data_ptrs[rank]
        self.peer_data_ptr = self.data_ptrs[peer]
        self.flag_ptr = self.flag_ptrs[rank]
        self.peer_flag_ptr = self.flag_ptrs[peer]
        self.flag_stride = FLAG_STRIDE_BYTES // 4  # int32 elements
        # flags start at 0 -> first derived sequence is 1
        _zero_flags(self.flag_ptr, rows * self.flag_stride)

    @property
    def peer_args(self) -> tuple[int, int, int, int]:
        return (
            self.data_ptr,
            self.peer_data_ptr,
            self.flag_ptr,
            self.peer_flag_ptr,
        )


def _zero_flags(ptr: int, n: int) -> None:
    block = 1024
    _zero_kernel[((n + block - 1) // block,)](ptr, n, BLOCK=block)


# ---------------------------------------------------------------------------
# process-frozen runtime decision (collective, decided once)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_decision: dict[str, Any] | None = None
_logged: set[str] = set()


def _log_once(key: str, msg: str, level: str = "info") -> None:
    if key in _logged:
        return
    _logged.add(key)
    getattr(logger, level)(msg)


def _local_readiness() -> tuple[bool, str, Any]:
    if torch.distributed.get_world_size(group=None) != 2:
        return False, "world_size != 2", None
    try:
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        if tp.world_size != 2:
            return False, "tp world_size != 2", None
        cpu_group = tp.cpu_group
    except Exception as e:  # noqa: BLE001 - readiness probing must not crash
        return False, f"tp group unavailable: {e}", None
    try:
        import vllm._custom_ops as ops  # noqa: PLC0415

        if not hasattr(ops, "allocate_shared_buffer_and_handle"):
            return False, "_C shared-buffer IPC ops unavailable", None
    except Exception as e:  # noqa: BLE001
        return False, f"_C ops unavailable: {e}", None
    return True, "ready", cpu_group


def _collective_and(local_yes: bool, cpu_group: Any) -> bool:
    """Collective AND over the TP group. MUST be called on every first
    decision path, even for a local 'no' (send 0): a rank that skips the
    vote leaves the peer blocked in the collective. On ANY collective
    failure both sides freeze fallback (fail loud: one-shot WARN)."""
    if cpu_group is None:
        return False
    try:
        import torch.distributed as dist  # noqa: PLC0415

        flag = torch.tensor(
            [1 if local_yes else 0], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
        return bool(flag.item())
    except Exception as e:  # noqa: BLE001 - never crash on the critical path
        _log_once(
            "collective_failed",
            f"triton_ar_norm_quant collective readiness vote failed: {e} "
            f"(one-shot) -- falling back everywhere",
            "warning",
        )
        return False


def _norm_gate(rows: int) -> tuple[bool, str]:
    """Local gate for the rank-local norm+quant kernel. No collective needed:
    nothing here couples the ranks (a local fallback changes only which
    numerically-equivalent chain runs on this rank -- no spin, no hang)."""
    if rows > MAX_ROWS:
        return False, f"rows {rows} > {MAX_ROWS} (decode-only regime)"
    return True, "ready"


def get_decision(rows: int, hidden: int) -> tuple[bool, str, Any]:
    """The COLLECTIVE activation predicate, decided exactly once.

    Both TP ranks must take the same branch at every call site: a rank running
    the fused kernel against a rank running the fallback would spin forever on
    a flag nobody writes (GPU hang in a captured graph). So the decision is
    frozen at first (eager) use and voted on collectively TWICE:

    1. a readiness vote (all-reduce MIN over the TP group), ALWAYS run, even
       for a local 'no' -- a rank that skips the vote deadlocks the peer;
    2. a second vote after the shared-buffer init, because that init can
       fail on ONE rank only -- freezing 'fused' on one side and 'fallback'
       on the other is the exact hang class spec section 3.7 forbids.

    Any collective failure freezes FALLBACK on both sides (one-shot WARN).
    A first call during graph capture is a voted fallback too.

    Per-call shape/dtype conditions are checked separately and are
    rank-symmetric by construction (TP ranks execute the same graphs with
    the same shapes -- the all-reduce itself requires it); the rows/hidden
    bound against the comm buffer is re-checked on EVERY call (never launch
    out of bounds into local and peer memory).
    """
    global _decision
    with _lock:
        if _decision is not None:
            comm = _decision["comm"]
            if comm is not None:
                if comm.hidden != hidden:
                    return (
                        False,
                        f"hidden {hidden} != comm hidden {comm.hidden}",
                        None,
                    )
                if rows > comm.rows:
                    return (
                        False,
                        f"rows {rows} > comm rows {comm.rows}",
                        None,
                    )
            return _decision["enabled"], _decision["reason"], comm
        # FIRST CALL (warmup: eager, before any graph capture)
        ready, info, cpu_group = _local_readiness()
        try:
            if (
                torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            ):
                ready = False
                info = "first call during graph capture"
        except Exception:  # noqa: BLE001 - no CUDA context: not capturing
            pass
        # vote 1: ALWAYS collective (send 0 when not ready)
        voted = _collective_and(ready, cpu_group)
        if ready and not voted:
            info = "peer rank not ready (collective decision)"
        comm = None
        if voted:
            try:
                comm = ArNormQuantComm(cpu_group, MAX_ROWS, hidden)
            except Exception as e:  # noqa: BLE001
                info = f"shared-buffer init failed: {e}"
                comm = None
        # vote 2: the comm-init outcome is voted too (it can fail on ONE
        # rank only -- see the docstring)
        final = _collective_and(comm is not None, cpu_group)
        if comm is not None and not final:
            # one side failed -> NOBODY runs fused
            comm = None
            info = "peer comm init failed (collective decision)"
        enabled = bool(final)
        _decision = {
            "enabled": enabled,
            "reason": info if not enabled else "ready",
            "comm": comm,
        }
        # Per-call bounds apply to the FIRST call too: the engine's first call
        # is far out of regime (the 4096-row memory-profiling forward runs
        # through the same backed-dynamic graph) while the frozen decision must
        # stay enabled for the in-regime calls that follow. Launching the fused
        # kernel with rows > comm.rows writes out of bounds in local AND peer
        # memory (measured 2026-10-22: GPU fault "Write access to a read-only
        # page" at engine warmup). Rank-symmetric like every per-call
        # condition (TP ranks execute the same shapes).
        if comm is not None:
            if comm.hidden != hidden:
                return (
                    False,
                    f"hidden {hidden} != comm hidden {comm.hidden}",
                    None,
                )
            if rows > comm.rows:
                return (
                    False,
                    f"rows {rows} > comm rows {comm.rows}",
                    None,
                )
        return _decision["enabled"], _decision["reason"], comm


# ---------------------------------------------------------------------------
# launchers
# ---------------------------------------------------------------------------
def _launch(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
    comm: ArNormQuantComm | None,
) -> None:
    rows, hidden = input.shape
    block = min(triton.next_power_of_2(hidden), 2048)
    grid = (rows,)
    assert scale.stride(0) == 1, "scale must be [rows, 1] contiguous"
    if comm is None:
        slot_ptr = 0
        peer_ptr = 0
        flag_ptr = 0
        peer_flag_ptr = 0
        flag_stride = FLAG_STRIDE_BYTES // 4
    else:
        slot_ptr = comm.data_ptr
        peer_ptr = comm.peer_data_ptr
        flag_ptr = comm.flag_ptr
        peer_flag_ptr = comm.peer_flag_ptr
        flag_stride = comm.flag_stride
    _norm_quant_int8_kernel[grid](
        input,
        residual,
        weight,
        result,
        scale,
        slot_ptr,
        peer_ptr,
        flag_ptr,
        peer_flag_ptr,
        epsilon,
        input.stride(0),
        residual.stride(0),
        result.stride(0),
        hidden,  # stride_slot: the comm slot is contiguous [MAX_ROWS, hidden]
        H=hidden,
        BLOCK_H=block,
        PEER=comm is not None,
        FLAG_STRIDE=flag_stride,
        num_warps=_NUM_WARPS,
    )


def _fallback_norm_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
    reduced: torch.Tensor | None = None,
) -> None:
    """The exact chain this op replaces, executed inside the single node:
    (all-reduce +) the native fused_add_rms_norm decomposition
    (vllm/ir/ops/layernorm.py, expression for expression) +
    ``_C.dynamic_scaled_int8_quant`` into the same buffers. No extra kernels
    over the unfused graph beyond the chain itself.

    Note: the expressions are identical to the baseline chain, but this
    fallback runs EAGER torch while the baseline is inductor-compiled --
    the same class-3 reduction-order caveat (~1e-7) applies between them.
    """
    x_in = input if reduced is None else reduced
    orig_dtype = x_in.dtype
    xf = x_in.to(torch.float32)
    s = xf + residual.to(torch.float32)
    residual.copy_(s.to(residual.dtype))
    var = (s * s).mean(dim=-1, keepdim=True)
    y = s * torch.rsqrt(var + epsilon)
    y = y.to(weight.dtype) * weight
    normed = y.to(orig_dtype)
    torch.ops._C.dynamic_scaled_int8_quant(
        result=result, input=normed, scale=scale, azp=None
    )


# ---------------------------------------------------------------------------
# custom ops (the standard vLLM registration mechanism:
# ``direct_register_custom_op`` with a mutating schema mirroring
# ``_C.rms_norm_dynamic_per_token_quant`` so auto_functionalization treats the
# fused node exactly like the chain it replaces).
# ---------------------------------------------------------------------------
def _contract_ok(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    return (
        result.dtype == torch.int8
        and scale.dtype == torch.float32
        and residual.dtype == input.dtype
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.float32
    )


def _norm_quant_int8_impl(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> None:
    rows, hidden = input.shape
    use, reason = _norm_gate(rows)
    if not use or not _contract_ok(result, input, weight, scale, residual):
        if use:
            reason = "dtype contract (result/scale/residual/weight)"
        _log_once(
            "fallback_norm",
            f"triton_norm_quant_int8 falling back to the unfused chain "
            f"(one-shot): {reason}",
            "warning",
        )
        _fallback_norm_quant(result, input, weight, scale, residual, epsilon)
        return
    _log_once(
        "active_norm",
        f"triton_norm_quant_int8 path ACTIVE (one-shot; shape="
        f"{tuple(input.shape)})",
    )
    _launch(result, input, weight, scale, residual, epsilon, None)


def _norm_quant_int8_fake(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> None:
    # Mutating op: the buffers carry the metas, nothing to return.
    return None


def _ar_norm_quant_int8_impl(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> None:
    rows, hidden = input.shape
    use, reason, comm = get_decision(rows, hidden)
    if not use or comm is None or not _contract_ok(
        result, input, weight, scale, residual
    ):
        if use:
            reason = "dtype contract (result/scale/residual/weight)"
        _log_once(
            "fallback_ar",
            f"triton_ar_norm_quant_int8 falling back to the unfused chain "
            f"(one-shot): {reason}",
            "warning",
        )
        from vllm.distributed import tensor_model_parallel_all_reduce

        reduced = tensor_model_parallel_all_reduce(input)
        _fallback_norm_quant(
            result, input, weight, scale, residual, epsilon, reduced=reduced
        )
        return
    _log_once(
        "active_ar",
        f"triton_ar_norm_quant_int8 path ACTIVE (one-shot; shape="
        f"{tuple(input.shape)})",
    )
    _launch(result, input, weight, scale, residual, epsilon, comm)


def _ar_norm_quant_int8_fake(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="triton_norm_quant_int8",
    op_func=_norm_quant_int8_impl,
    mutates_args=["result", "scale", "residual"],
    fake_impl=_norm_quant_int8_fake,
)

direct_register_custom_op(
    op_name="triton_ar_norm_quant_int8",
    op_func=_ar_norm_quant_int8_impl,
    mutates_args=["result", "scale", "residual"],
    fake_impl=_ar_norm_quant_int8_fake,
)


# ---------------------------------------------------------------------------
# reference implementation (tests only): the contract above in plain torch.
# ---------------------------------------------------------------------------
def reference_norm_quant_int8(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    peer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (quant int8 [M,H], scale fp32 [M,1], res_out bf16 [M,H])."""
    if peer is None:
        xf = input
    else:
        xf = (input.to(torch.float32) + peer.to(torch.float32)).to(
            torch.bfloat16
        )
    s = xf.to(torch.float32) + residual.to(torch.float32)
    res_out = s.to(torch.bfloat16)
    var = (s * s).mean(dim=-1, keepdim=True)
    y = (s * torch.rsqrt(var + epsilon)).to(weight.dtype) * weight
    y_bf16 = y.to(torch.bfloat16)
    amax = y_bf16.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    inv_s = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(y_bf16.to(torch.float32) * inv_s)  # half-to-even
    q = q.clamp(-128, 127)
    return q.to(torch.int8), scale, res_out
