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

FUSE_MAX_ROWS = 32
"""Measured fused-regime row bound (research/ar-vol-count-2026-10-25, ETAPE-B3).

Above this bound the one-stage kernel is SLOWER than the chain it replaces
(serie M, 2026-10-25, 128 rows: 31.29 vs 27.74 us/call, +2.3% end-to-end at
c128), so the AR op takes the fused path at rows <= 32 only and answers with
the TWO-STAGE chain above (all-reduce + the rank-local norm+quant kernel of
``triton_norm_quant_int8``, same expressions). The decode universe stays
covered without the eager chain -- whose ~7 extra launches per site are
themselves a regression (measured cost model 2026-10-22).

``MAX_ROWS`` above is the COMM-BUFFER capacity; this is the PROFITABLE
regime. Two different facts sharing one kernel."""

FLAG_STRIDE_BYTES = 128
"""Padding between per-row flags (AITER Signal layout: flags must not share a
cache line -- the 2026-10-22 probe's data-then-flag anomaly tracked a
req/ack/data set sharing 16-byte allocation granularity)."""

_NUM_WARPS = 8


@triton.jit
def _ar_combined_s(
    x_ptr,
    x_scale_ptr,
    slot_ptr,
    peer_ptr,
    slot_scale_ptr,
    peer_scale_ptr,
    res_ptr,
    row,
    offs,
    m,
    stride_x,
    stride_slot,
    stride_x_scale,
    stride_res,
    SCALE_TILE: tl.constexpr,
    SCALE_TILES: tl.constexpr,
    PEER: tl.constexpr,
    IN8: tl.constexpr,
):
    """The AR combination + residual add, shared by the THREE reduction
    phases of ``_norm_quant_int8_kernel`` -- ONE definition so the phases
    cannot drift apart (the numeric contract is this expression).

    Returns ``s = fp32(x_f) + fp32(residual)`` per element of ``offs``:

    * no peer: x_f = x (bf16 loaded);
    * peer, bf16 transport: x_f = bf16(fp32(x_self) + fp32(x_peer)) -- the
      AR output is rounded HERE (the chain this kernel replaces);
    * peer, IN8 transport (increment 2b): the int8 contributions are
      requantized to the composed scale s_out = max(s_self, s_peer) with the
      two per-token factors, accumulated in INT32 (exact), and dequantized in
      REGISTERS only (zero HBM dequant) -- see the kernel docstring.
    """
    rs = tl.load(res_ptr + row * stride_res + offs, mask=m, other=0.0)
    if PEER:
        if IN8:
            a8 = tl.load(
                x_ptr + row * stride_x + offs, mask=m, other=0
            ).to(tl.int32)
            b8 = tl.load(
                tl.cast(peer_ptr.to(tl.int64), tl.pointer_type(tl.int8))
                + row * stride_slot
                + offs,
                mask=m,
                other=0,
            ).to(tl.int32)
            s_off = offs // SCALE_TILE
            s_self = tl.load(
                x_scale_ptr + row * stride_x_scale + s_off, mask=m, other=0.0
            )
            s_peer = tl.load(
                tl.cast(peer_scale_ptr.to(tl.int64), tl.pointer_type(tl.float32))
                + row * SCALE_TILES
                + s_off,
                mask=m,
                other=0.0,
            )
            s_out = tl.maximum(s_self, s_peer)
            f_self = tl.where(s_out == 0.0, 0.0, s_self / s_out)
            f_peer = tl.where(s_out == 0.0, 0.0, s_peer / s_out)
            acc = _nearbyint(a8.to(tl.float32) * f_self) + _nearbyint(
                b8.to(tl.float32) * f_peer
            )
            xf = acc.to(tl.float32) * s_out
            return xf + rs.to(tl.float32)
        a = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
        b = tl.load(
            tl.cast(peer_ptr.to(tl.int64), tl.pointer_type(tl.bfloat16))
            + row * stride_slot
            + offs,
            mask=m,
            other=0.0,
        )
        xf = (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
        return xf.to(tl.float32) + rs.to(tl.float32)
    xf = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0)
    return xf.to(tl.float32) + rs.to(tl.float32)


@triton.jit
def _norm_quant_int8_kernel(
    x_ptr,
    res_ptr,
    weight_ptr,
    out_ptr,
    scale_ptr,
    normed_ptr,
    slot_ptr,
    peer_ptr,
    flag_ptr,
    peer_flag_ptr,
    eps,
    stride_x,
    stride_res,
    stride_out,
    stride_normed,
    stride_slot,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PEER: tl.constexpr,
    FLAG_STRIDE: tl.constexpr,
    x_scale_ptr,
    slot_scale_ptr,
    peer_scale_ptr,
    stride_x_scale,
    SCALE_TILE: tl.constexpr = 0,
    SCALE_TILES: tl.constexpr = 0,
    SCALE_BLOCK: tl.constexpr = 0,
    IN8: tl.constexpr = False,
):
    """One program per row; the full row is reduced inside the program.

    PEER=1 adds the publish + barrier + peer-sum phases (3-way); PEER=0 takes
    ``x`` itself as ``x_f`` (norm+quant only). Both share one numeric chain.

    IN8=1 (increment 2b, AR int8 natif) is the PEER=1 int8 transport variant:
    ``x_ptr`` carries the INT8 contribution (the requant-epilogue output) and
    ``x_scale_ptr`` its per-(row, SCALE_TILE) fp32 scales; the symmetric slot
    carries the same pair (int8 row at offset 0, scales row in the scale
    area).  Contract (ETAPE-A 4.3 algebraic validity, named):

        s_out  = max(s_self, s_peer)        # composed scale, per scale element
        f_i    = s_i / s_out (0 if s_out == 0)   # the two factors per token
        q_i'   = nearbyint(q_i * f_i)       # requant to the common scale
        acc    = q_self' + q_peer'          # fp32-integral accumulation
                                          # (exact: |acc| <= 256 << 2^24 --
                                          # sums run in fp32 registers,
                                          # integral-valued, review M3)
        x_f    = acc * s_out                # dequantized IN REGISTERS only
        s      = fp32(x_f) + fp32(residual) # residual add in fp32 (unchanged)

    "Zero dequant": the dequantized value never touches HBM (the bf16 path
    publishes/reads bf16; here the transport is 1 byte/element and the scales
    travel with it).  The result is exact up to the final requant: each
    contribution carries at most s_out/2 of requant error (measured by the
    tests against an fp32 reference).  The norm+quant tail is the SAME
    expressions as the bf16 mode (lockstep within this kernel).
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
    # IN8 views of the same symmetric data buffer (int8 rows at offset 0) and
    # of the scale area (raw addresses, zero when IN8=0).
    slot_p8 = tl.cast(slot_ptr.to(tl.int64), tl.pointer_type(tl.int8))
    peer_p8 = tl.cast(peer_ptr.to(tl.int64), tl.pointer_type(tl.int8))
    slot_s_p = tl.cast(slot_scale_ptr.to(tl.int64), tl.pointer_type(tl.float32))
    peer_s_p = tl.cast(peer_scale_ptr.to(tl.int64), tl.pointer_type(tl.float32))

    # ---- phase 0 (PEER): publish my row, then release my flag, then wait
    # for the peer's flag. Order is load-bearing: data stores strictly before
    # the release xchg (release/acquire pair gives the peer the data).
    if PEER:
        if IN8:
            # int8 contribution + its per-(row, tile) scales, both published
            # (the scale row lives in the scale area of the symmetric buffer).
            for base in range(0, H, BLOCK_H):
                offs = base + tl.arange(0, BLOCK_H)
                m = offs < H
                xv = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0)
                tl.store(slot_p8 + row * stride_slot + offs, xv, mask=m)
            s_blk = tl.arange(0, SCALE_BLOCK)
            sv = tl.load(
                x_scale_ptr + row * stride_x_scale + s_blk,
                mask=s_blk < SCALE_TILES,
                other=0.0,
            )
            tl.store(
                slot_s_p + row * SCALE_TILES + s_blk,
                sv,
                mask=s_blk < SCALE_TILES,
            )
        else:
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
        s = _ar_combined_s(
            x_ptr, x_scale_ptr, slot_ptr, peer_ptr, slot_scale_ptr,
            peer_scale_ptr, res_ptr, row, offs, m, stride_x, stride_slot,
            stride_x_scale, stride_res, SCALE_TILE, SCALE_TILES, PEER, IN8,
        )
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
        s = _ar_combined_s(
            x_ptr, x_scale_ptr, slot_ptr, peer_ptr, slot_scale_ptr,
            peer_scale_ptr, res_ptr, row, offs, m, stride_x, stride_slot,
            stride_x_scale, stride_res, SCALE_TILE, SCALE_TILES, PEER, IN8,
        )
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
        s = _ar_combined_s(
            x_ptr, x_scale_ptr, slot_ptr, peer_ptr, slot_scale_ptr,
            peer_scale_ptr, res_ptr, row, offs, m, stride_x, stride_slot,
            stride_x_scale, stride_res, SCALE_TILE, SCALE_TILES, PEER, IN8,
        )
        w = tl.load(weight_ptr + offs, mask=m, other=0.0)
        y = (s * rstd) * w.to(tl.float32)
        yb = y.to(tl.bfloat16)
        q = _nearbyint(yb.to(tl.float32) * inv_s)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(out_ptr + row * stride_out + offs, q.to(tl.int8), mask=m)
        # emit the normed value too: the chain this kernel replaces produces
        # it for OTHER consumers (a second quant of the same y at the GDN
        # sites, the shared-expert alias) -- without this store the motif
        # cannot take any site whose norm output has a second user (measured
        # 2026-10-22: 10/68 sites reachable otherwise).
        tl.store(normed_ptr + row * stride_normed + offs, yb, mask=m)
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
        # IN8 layout (increment 2b): the symmetric DATA buffer holds int8
        # contribution rows at byte offset 0 ([rows, hidden] bytes) and the
        # per-(row, tile) scale rows in the SCALE AREA at byte offset
        # rows*hidden ([rows, tiles] fp32).  Capacity check: the bf16 sizing
        # (rows*hidden*2) covers int8 data + scales as long as tiles >= 4
        # (every GEMM tile is >= 16 wide).
        self.scale_area_offset = rows * hidden
        # scale area capacity: tiles <= hidden/16 (the narrowest GEMM tile is
        # 16 wide) -> rows*hidden/4 bytes, well inside the bf16 sizing.
        assert rows * hidden + rows * (hidden // 16) * 4 <= data_bytes

    @property
    def data_scale_ptr(self) -> int:
        """Byte address of MY scale area inside my symmetric data buffer."""
        return self.data_ptr + self.scale_area_offset

    @property
    def peer_data_scale_ptr(self) -> int:
        """Byte address of the PEER's scale area inside its data buffer."""
        return self.peer_data_ptr + self.scale_area_offset

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


def _log_capture_regime(rows: int, branch: str, why: str) -> None:
    """One-shot per capture bucket: which buckets baked the fused kernel and
    which baked the two-stage chain. Bounded (CUDA-graph capture only, never
    replay) and falsifiable -- the 128-row bucket must log ``two-stage`` and
    the 32-row bucket ``fused``. Lesson 2026-10-25: a log line that never
    executed proves nothing, so the battery asserts this one."""
    try:
        capturing = (
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
    except Exception:  # noqa: BLE001 - no CUDA context: not capturing
        return
    if not capturing:
        return
    _log_once(
        f"capture_regime_{branch}_{rows}",
        f"[ar3way] graph capture rows={rows} -> {branch} ({why})",
    )


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


def _per_call_bounds_ok(
    rows: int, hidden: int, comm: ArNormQuantComm
) -> tuple[bool, str]:
    """Per-call shape/regime conditions (rank-symmetric by construction: TP
    ranks execute the same graphs with the same shapes -- the all-reduce
    itself requires it). The fused kernel is only launched inside the comm
    buffer (never out of bounds into local AND peer memory) and inside the
    MEASURED regime (``FUSE_MAX_ROWS``); above either bound the caller takes
    the two-stage chain."""
    if comm.hidden != hidden:
        return False, f"hidden {hidden} != comm hidden {comm.hidden}"
    if rows > comm.rows:
        return False, f"rows {rows} > comm rows {comm.rows}"
    if rows > FUSE_MAX_ROWS:
        return False, f"rows {rows} > {FUSE_MAX_ROWS} (measured fused regime)"
    return True, ""


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
                ok, why = _per_call_bounds_ok(rows, hidden, comm)
                if not ok:
                    return (False, why, None)
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
            ok, why = _per_call_bounds_ok(rows, hidden, comm)
            if not ok:
                return (False, why, None)
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
    normed: torch.Tensor,
    epsilon: float,
    comm: ArNormQuantComm | None,
    x_scale: torch.Tensor | None = None,
    scale_tile: int = 0,
) -> None:
    """``x_scale`` (increment 2b) selects the IN8 transport: ``input`` is then
    the int8 contribution and ``x_scale`` its [rows, tiles] fp32 scales
    (``scale_tile`` = the producing GEMM's BLOCK_SIZE_N)."""
    rows, hidden = input.shape
    block = min(triton.next_power_of_2(hidden), 2048)
    grid = (rows,)
    assert scale.stride(0) == 1, "scale must be [rows, 1] contiguous"
    in8 = x_scale is not None
    if in8:
        assert comm is not None, "the IN8 transport is one-shot only"
        assert input.dtype == torch.int8
        assert x_scale.dtype == torch.float32 and x_scale.is_contiguous()
        assert x_scale.shape[0] == rows and scale_tile > 0
        assert (x_scale.shape[1] - 1) * scale_tile < hidden <= x_scale.shape[1] * scale_tile
        scale_tiles = x_scale.shape[1]
        scale_block = triton.next_power_of_2(scale_tiles)
        stride_x_scale = x_scale.stride(0)
    else:
        scale_tiles = 0
        scale_block = 0
        stride_x_scale = 1
    if comm is None:
        slot_ptr = 0
        peer_ptr = 0
        flag_ptr = 0
        peer_flag_ptr = 0
        slot_scale_ptr = 0
        peer_scale_ptr = 0
        flag_stride = FLAG_STRIDE_BYTES // 4
    else:
        slot_ptr = comm.data_ptr
        peer_ptr = comm.peer_data_ptr
        flag_ptr = comm.flag_ptr
        peer_flag_ptr = comm.peer_flag_ptr
        slot_scale_ptr = comm.data_scale_ptr if in8 else 0
        peer_scale_ptr = comm.peer_data_scale_ptr if in8 else 0
        flag_stride = comm.flag_stride
    _norm_quant_int8_kernel[grid](
        input,
        residual,
        weight,
        result,
        scale,
        normed,
        slot_ptr,
        peer_ptr,
        flag_ptr,
        peer_flag_ptr,
        epsilon,
        input.stride(0),
        residual.stride(0),
        result.stride(0),
        normed.stride(0),
        hidden,  # stride_slot: the comm slot is contiguous [MAX_ROWS, hidden]
        H=hidden,
        BLOCK_H=block,
        PEER=comm is not None,
        FLAG_STRIDE=flag_stride,
        x_scale_ptr=x_scale if in8 else 0,
        slot_scale_ptr=slot_scale_ptr,
        peer_scale_ptr=peer_scale_ptr,
        stride_x_scale=stride_x_scale,
        SCALE_TILE=scale_tile if in8 else 0,
        SCALE_TILES=scale_tiles,
        SCALE_BLOCK=scale_block,
        IN8=in8,
        num_warps=_NUM_WARPS,
    )


def _fallback_norm_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """The exact chain this op replaces, executed inside the single node: the
    native fused_add_rms_norm decomposition (vllm/ir/ops/layernorm.py,
    expression for expression) + ``_C.dynamic_scaled_int8_quant`` into the
    same buffers. The caller reduces first when the site carries an
    all-reduce; ``input`` is the already-reduced value. No extra kernels over
    the unfused graph beyond the chain itself.

    Note: the expressions are identical to the baseline chain, but this
    fallback runs EAGER torch while the baseline is inductor-compiled --
    the same class-3 reduction-order caveat (~1e-7) applies between them.
    This is the PREFILL-scale answer (rows > MAX_ROWS); decode sizes take the
    two-stage kernel chain (see ``_ar_norm_quant_int8_impl``) precisely because
    the eager launches above are not free.
    """
    x_in = input
    orig_dtype = x_in.dtype
    xf = x_in.to(torch.float32)
    s = xf + residual.to(torch.float32)
    residual.copy_(s.to(residual.dtype))
    var = (s * s).mean(dim=-1, keepdim=True)
    y = s * torch.rsqrt(var + epsilon)
    y = y.to(weight.dtype) * weight
    normed_val = y.to(orig_dtype)
    torch.ops._C.dynamic_scaled_int8_quant(
        result=result, input=normed_val, scale=scale, azp=None
    )
    return normed_val


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
) -> torch.Tensor:
    """Mutates result/scale/residual in place; RETURNS the normed value.

    The return is load-bearing: the chain this op replaces produces the
    normed value as a graph value with OTHER consumers (a second quant of the
    same y at the GDN sites, the shared-expert alias at the attn-out sites) --
    without emitting it the motif cannot take any site whose norm output has
    a second user (measured 2026-10-22: 10/68 sites reachable otherwise).
    """
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
        return _fallback_norm_quant(
            result, input, weight, scale, residual, epsilon
        )
    _log_once(
        "active_norm",
        f"triton_norm_quant_int8 path ACTIVE (one-shot; shape="
        f"{tuple(input.shape)})",
    )
    normed = torch.empty(
        (rows, hidden), dtype=input.dtype, device=input.device
    )
    _launch(result, input, weight, scale, residual, normed, epsilon, None)
    return normed


def _norm_quant_int8_fake(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    # Mutated buffers carry their metas; the normed value is returned.
    return torch.empty_like(input)


def _ar_norm_quant_int8_impl(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    rows, hidden = input.shape
    use, reason, comm = get_decision(rows, hidden)
    if use and comm is not None and _contract_ok(
        result, input, weight, scale, residual
    ):
        _log_once(
            "active_ar",
            f"triton_ar_norm_quant_int8 path ACTIVE (one-shot; shape="
            f"{tuple(input.shape)})",
        )
        _log_capture_regime(rows, "fused", "one-stage AR+norm+quant")
        normed = torch.empty(
            (rows, hidden), dtype=input.dtype, device=input.device
        )
        _launch(result, input, weight, scale, residual, normed, epsilon, comm)
        return normed
    # Out of the fused regime: the measured row gate (rows > FUSE_MAX_ROWS,
    # ETAPE-B3), a contract miss, or no comm. Take the TWO-STAGE chain -- the
    # all-reduce plus the rank-local norm+quant op, which owns its own bounds
    # and falls back to the exact eager chain only at prefill-scale rows. The
    # eager chain must NOT answer decode-sized calls: its ~7 extra launches
    # per site (measured cost model 2026-10-22) are themselves a regression,
    # bigger than the c128 kernel cost this gate exists to avoid.
    if use:
        reason = "dtype contract (result/scale/residual/weight)"
    if rows > FUSE_MAX_ROWS:
        # expected path for the out-of-regime buckets (c128): info, not a
        # warning -- and a distinct key so a real in-regime fallback can never
        # be masked by it.
        _log_once(
            "fallback_ar_regime",
            f"triton_ar_norm_quant_int8 two-stage chain above the fused "
            f"regime (one-shot; shape={tuple(input.shape)}): {reason}",
        )
    else:
        _log_once(
            "fallback_ar",
            f"triton_ar_norm_quant_int8 two-stage chain (one-shot; "
            f"shape={tuple(input.shape)}): {reason}",
            "warning",
        )
    _log_capture_regime(rows, "two-stage", reason)
    from vllm.distributed import tensor_model_parallel_all_reduce

    reduced = tensor_model_parallel_all_reduce(input)
    return _norm_quant_int8_impl(
        result, reduced, weight, scale, epsilon, residual
    )


def _ar_norm_quant_int8_fake(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(input)


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
# Increment 2b (research/ar-vol-count-2026-10-25, ETAPE-E section 4): the AR
# INT8 NATIVE path.  Same gate machinery as above (get_decision, collective,
# per-call bounds, row gate); the TRANSPORT carries int8 contributions + their
# per-(row, tile) scales (the requant-epilogue output of increment 2a) instead
# of bf16 values -- half the bytes on the wire and through HBM, int32
# accumulation of the composed contributions, zero dequantized HBM traffic.
# ---------------------------------------------------------------------------
def _contract_ok_in8(
    result: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
) -> bool:
    return (
        result.dtype == torch.int8
        and scale.dtype == torch.float32
        and residual.dtype == torch.bfloat16
        and x_q.dtype == torch.int8
        and x_scale.dtype == torch.float32
        and weight.dtype == torch.float32
    )


def _dequant_contributions(
    x_q: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """The IN8 fallback's dequant: per-(row, tile) scales -> bf16 contribution
    values (the pre-2b transport value).  The fallback chain is then the
    bf16 chain VERBATIM on the dequantized input -- the EXACT variant (the
    one-shot pays one requant per contribution instead).

    ``scale_tile`` is DERIVED from the scales' shape (never a baked constant:
    the fusion wiring passes this op around backed-dynamic shapes and a
    trace-time constant would be wrong for another capture bucket)."""
    rows, hidden = x_q.shape
    tiles = x_scale.shape[1]
    assert hidden % tiles == 0, (
        f"scale tiles {tiles} must divide hidden {hidden}"
    )
    scale_tile = hidden // tiles
    idx = torch.arange(hidden, device=x_q.device) // scale_tile
    v = x_q.to(torch.float32) * x_scale[:, idx]
    return v.to(torch.bfloat16)


def _ar_norm_quant_int8_in_impl(
    result: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    """AR int8 natif (increment 2b).  Mutates result/scale/residual in place,
    RETURNS the normed value (same return contract as the bf16 op: the normed
    value has OTHER consumers at some sites).

    ``scale_tile`` (the producing GEMM's BLOCK_SIZE_N) is DERIVED from
    ``x_scale.shape[1]`` at runtime -- never a baked argument: the fusion
    wiring sees backed-dynamic shapes and a trace-time constant would be
    wrong for another capture bucket."""
    rows, hidden = x_q.shape
    tiles = x_scale.shape[1]
    assert hidden % tiles == 0, (
        f"scale tiles {tiles} must divide hidden {hidden}"
    )
    scale_tile = hidden // tiles
    use, reason, comm = get_decision(rows, hidden)
    if use and comm is not None and _contract_ok_in8(
        result, x_q, x_scale, weight, scale, residual
    ):
        _log_once(
            "active_ar_in8",
            f"triton_ar_norm_quant_int8_in path ACTIVE (one-shot; shape="
            f"{tuple(x_q.shape)}, scale_tile={scale_tile})",
        )
        _log_capture_regime(
            rows, "fused", "one-stage AR+norm+quant (int8 transport)"
        )
        normed = torch.empty(
            (rows, hidden), dtype=residual.dtype, device=x_q.device
        )
        _launch(
            result,
            x_q,
            weight,
            scale,
            residual,
            normed,
            epsilon,
            comm,
            x_scale=x_scale,
            scale_tile=scale_tile,
        )
        return normed
    if use:
        reason = "dtype contract (result/scale/residual/weight)"
    if rows > FUSE_MAX_ROWS:
        _log_once(
            "fallback_ar_in8_regime",
            f"triton_ar_norm_quant_int8_in two-stage chain above the fused "
            f"regime (one-shot; shape={tuple(x_q.shape)}): {reason}",
        )
    else:
        _log_once(
            "fallback_ar_in8",
            f"triton_ar_norm_quant_int8_in two-stage chain (one-shot; "
            f"shape={tuple(x_q.shape)}): {reason}",
            "warning",
        )
    _log_capture_regime(rows, "two-stage", reason)
    from vllm.distributed import tensor_model_parallel_all_reduce

    # the pre-2b semantics, verbatim: dequantized contributions over the
    # standard all-reduce, then the rank-local norm+quant kernel.
    reduced = tensor_model_parallel_all_reduce(
        _dequant_contributions(x_q, x_scale)
    )
    return _norm_quant_int8_impl(
        result, reduced, weight, scale, epsilon, residual
    )


def _ar_norm_quant_int8_in_fake(
    result: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="triton_ar_norm_quant_int8_in",
    op_func=_ar_norm_quant_int8_in_impl,
    mutates_args=["result", "scale", "residual"],
    fake_impl=_ar_norm_quant_int8_in_fake,
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


def reference_ar_norm_quant_int8_in(
    x_q_self: torch.Tensor,
    s_self: torch.Tensor,
    x_q_peer: torch.Tensor,
    s_peer: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scale_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The IN8 contract in plain torch (tests only).  Returns
    ``(quant int8 [M,H], scale fp32 [M,1], res_out bf16 [M,H], x_f fp32)``.

    Mirrors the kernel expression for expression:
    s_out = max(s_self, s_peer) per (row, tile); q_i' = nearbyint(q_i * s_i /
    s_out) (half-to-even); acc = q_self' + q_peer' (fp32-integral, exact, review M3); x_f =
    acc * s_out (dequantized in registers); then the norm+quant chain of
    ``reference_norm_quant_int8``.
    """
    rows, hidden = x_q_self.shape
    if scale_tile is None:
        scale_tile = hidden // s_self.shape[1]
    idx = torch.arange(hidden, device=x_q_self.device) // scale_tile
    s_self_e = s_self[:, idx].to(torch.float32)
    s_peer_e = s_peer[:, idx].to(torch.float32)
    s_out = torch.maximum(s_self_e, s_peer_e)
    f_self = torch.where(s_out == 0, torch.zeros_like(s_out), s_self_e / s_out)
    f_peer = torch.where(s_out == 0, torch.zeros_like(s_out), s_peer_e / s_out)
    a_r = torch.round(x_q_self.to(torch.float32) * f_self)  # half-to-even
    b_r = torch.round(x_q_peer.to(torch.float32) * f_peer)
    acc = a_r + b_r
    x_f = acc * s_out
    s = x_f + residual.to(torch.float32)
    res_out = s.to(torch.bfloat16)
    var = (s * s).mean(dim=-1, keepdim=True)
    y = (s * torch.rsqrt(var + epsilon)).to(weight.dtype) * weight
    y_bf16 = y.to(torch.bfloat16)
    amax = y_bf16.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    inv_s = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(y_bf16.to(torch.float32) * inv_s)  # half-to-even
    q = q.clamp(-128, 127)
    return q.to(torch.int8), scale, res_out, x_f


def reference_ar_exact_sum(
    x_q_self: torch.Tensor,
    s_self: torch.Tensor,
    x_q_peer: torch.Tensor,
    s_peer: torch.Tensor,
    scale_tile: int | None = None,
) -> torch.Tensor:
    """The fp32-exact combined value (NO requant): x_f = q_self*s_self +
    q_peer*s_peer.  The IN8 kernel's x_f stays within one composed-scale LSB
    of this (each contribution pays at most s_out/2)."""
    rows, hidden = x_q_self.shape
    if scale_tile is None:
        scale_tile = hidden // s_self.shape[1]
    idx = torch.arange(hidden, device=x_q_self.device) // scale_tile
    return (
        x_q_self.to(torch.float32) * s_self[:, idx].to(torch.float32)
        + x_q_peer.to(torch.float32) * s_peer[:, idx].to(torch.float32)
    )
