# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-site introspection for the fusion patterns (the ``check_fn`` cure).

``pm.register_replacement``'s ``check_fn`` re-traces the pattern with the
matched values and re-matches it EXACTLY against the site; when it rejects, it
does so **silently** (measured 2026-10-19: structurally-matching sites
rejected one by one with no log anywhere). This module makes every candidate
site's fate observable, one line per site, at the hook stage, in the engine:

* the site's alloc provenance (which node/literal provides each alloc dim --
  the shared-placeholder trap of the exact re-match lives here),
* for the LIVE registered entries: ``entry.pattern.match`` result (with
  torch's own ``FailedMatch`` reason) and, when the initial match passes, the
  ``check_fn`` verdict plus the reason of the EXACT re-match (recovered by
  spying on ``fx_to_pattern`` and re-running the re-traced pattern's own
  match),
* for LEGACY entries -- a faithful replica of the pre-fix computed-alloc
  pattern (the form that matched 10/70 on the real model) -- the same
  verdicts, so "why did the old pattern reject THIS site" is measured per
  site on the same graph, side by side with the fixed entries.

The legacy entries are registered into a throwaway ``PatternMatcherPass``
that is never applied: they exist purely to reproduce the pre-fix match
behaviour. Set ``VLLM_FUSION_MATCH_DIAG=1`` to enable (it is one-shot per
process and off by default).
"""

import operator
import os
from typing import Any, ClassVar

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import (
    FailedMatch,
    Match,
    PatternMatcherPass,
)

import vllm.ir.ops  # noqa: F401  registers vllm_ir ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .matcher_utils import (
    MatcherQuantInt8,
    called_op,
    op_call_kwargs,
    walk_views,
)

logger = init_logger(__name__)

INT8_QUANT_OP_NAME = "dynamic_scaled_int8_quant"
_QUANT_OP: Any = getattr(
    getattr(torch.ops._C, INT8_QUANT_OP_NAME, None), "default", None
)


def enabled() -> bool:
    return os.getenv("VLLM_FUSION_MATCH_DIAG", "0") == "1"


def _dim_repr(arg: Any) -> str:
    """Compact provenance repr of one alloc dim: node name or literal."""
    if isinstance(arg, fx.Node):
        return arg.name
    return repr(arg)


def _alloc_dims(node: Any) -> str:
    if not isinstance(node, fx.Node):
        return repr(node)
    args = node.args
    if args and isinstance(args[0], (list, tuple)):
        return "[" + ", ".join(_dim_repr(a) for a in args[0]) + "]"
    return node.format_node()[:80]


def _entry_int8(entry: Any) -> bool:
    return INT8_QUANT_OP_NAME in repr(entry.pattern)


def _check_with_reason(entry: Any, match: Any, node: fx.Node) -> tuple[Any, str, str]:
    """Run an entry's ``check_fn`` (``extra_check``) and, when it rejects,
    recover the exact re-match's own failure reason.

    Returns ``(verdict, reason, retraced_form)``. ``check_fn`` re-traces the
    pattern via ``fx_to_pattern``; spying on that call captures the re-traced
    pattern, whose ``match(node)`` we re-run to get torch's reason instead of
    a silent ``False``.
    """
    import torch._inductor.pattern_matcher as _pm

    from ..vllm_inductor_pass import PatternMatchTelemetry

    captured: dict[str, Any] = {}
    orig = _pm.fx_to_pattern

    def spy(gm: Any, **kw: Any) -> Any:
        pat = orig(gm, **kw)
        captured["pat"] = pat
        captured["gm"] = gm
        return pat

    _pm.fx_to_pattern = spy
    prev_dry = PatternMatchTelemetry.dry_run
    PatternMatchTelemetry.dry_run = True
    try:
        try:
            verdict = entry.extra_check(match)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must not crash
            verdict = f"RAISED {type(exc).__name__}: {exc}"
    finally:
        PatternMatchTelemetry.dry_run = prev_dry
        _pm.fx_to_pattern = orig

    reason = ""
    retraced = ""
    if verdict is not True and "pat" in captured:
        try:
            res = captured["pat"].match(node)
            reason = str(res)[:160]
        except Exception as exc:  # noqa: BLE001
            reason = f"(re-match raised {type(exc).__name__})"
        try:
            retraced = str(captured["gm"].graph)[:400]
        except Exception:  # noqa: BLE001
            retraced = ""
    return verdict, reason, retraced


def _try_entry(entry: Any, anchors: list[fx.Node]) -> tuple[str, str, str]:
    """Match one pattern entry at a site.

    Returns ``(status, reason, retraced)`` where status is one of
    ``match=OK check_fn=OK`` / ``match=OK check_fn=FAIL`` /
    ``match=FAIL``.
    """
    last_fail = ""
    for anchor in anchors:
        try:
            res = entry.pattern.match(anchor)
        except Exception as exc:  # noqa: BLE001
            return "match=RAISED", f"{type(exc).__name__}: {exc}", ""
        if isinstance(res, Match):
            node = res.output_nodes()[0]
            verdict, reason, retraced = _check_with_reason(entry, res, node)
            if verdict is True:
                return "match=OK check_fn=OK", "", ""
            return (
                "match=OK check_fn=FAIL",
                f"check_fn={verdict!r} exact_rematch={reason}",
                retraced,
            )
        if isinstance(res, FailedMatch):
            last_fail = str(res)[:160]
    return "match=FAIL", last_fail, ""


class FusionSiteDiag:
    """One-shot per-site report over the real hook graph (see module doc)."""

    _done: ClassVar[bool] = False
    _result_done: ClassVar[bool] = False
    _retraced_logged: ClassVar[set[str]] = set()

    @classmethod
    def report(
        cls,
        graph: fx.Graph,
        live_pass: PatternMatcherPass,
        legacy_pass: PatternMatcherPass | None,
    ) -> None:
        if cls._done or not enabled() or _QUANT_OP is None:
            return
        cls._done = True

        live_entries = [
            e
            for entries in live_pass.patterns.values()
            for e in entries
            if _entry_int8(e)
        ]
        legacy_entries = [
            e
            for entries in (legacy_pass.patterns.values() if legacy_pass else [])
            for e in entries
            if _entry_int8(e)
        ]

        sites: list[dict[str, Any]] = []
        for node in graph.nodes:
            if called_op(node) is not _QUANT_OP:
                continue
            kwargs = op_call_kwargs(node, _QUANT_OP)
            qinput = kwargs.get("input")
            producer, chain = walk_views(qinput)
            pop = called_op(producer) if isinstance(producer, fx.Node) else None
            pop_name = (
                getattr(pop, "__name__", str(pop)).split("::")[-1]
                if pop is not None
                else "other"
            )
            result = kwargs.get("result")
            scale = kwargs.get("scale")
            anchors = [
                u
                for u in node.users
                if u.target is operator.getitem and isinstance(u.args, tuple)
            ]
            # outputs[0] of the registered patterns is the quant RESULT
            # getitem (index 1 in the auto_functionalized tuple); try it first.
            anchors.sort(
                key=lambda u: (u.args[1] if len(u.args) > 1 else 99) != 1
            )
            sites.append(
                {
                    "node": node,
                    "producer": producer,
                    "family": pop_name,
                    "chain": chain,
                    "result_dims": _alloc_dims(result),
                    "scale_dims": _alloc_dims(scale),
                    "input_dims": _alloc_dims(qinput),
                    "result_is_alloc": isinstance(result, fx.Node)
                    and result.op == "call_function"
                    and "empty" in str(result.target),
                    "anchors": anchors,
                }
            )

        logger.info(
            "fusion-site-diag: %d int8 quant site(s) on this graph "
            "(legacy entries=%d, live entries=%d)",
            len(sites),
            len(legacy_entries),
            len(live_entries),
        )

        for i, site in enumerate(sites):
            lines = []
            for label, entries in (
                ("legacy", legacy_entries),
                ("live", live_entries),
            ):
                for entry in entries:
                    status, reason, retraced = _try_entry(
                        entry, site["anchors"]
                    )
                    motif = _motif_of(entry)
                    lines.append(f"{label}:{motif}={status}({reason})")
                    if retraced and status == "match=OK check_fn=FAIL":
                        key = f"{label}:{motif}:{reason[:60]}"
                        if key not in cls._retraced_logged:
                            cls._retraced_logged.add(key)
                            logger.info(
                                "fusion-site-diag: re-traced specific pattern "
                                "form (%s, %s): %s",
                                label,
                                motif,
                                retraced,
                            )
            logger.info(
                "fusion-site-diag site#%02d quant=%s family=%s(view_chain=%d) "
                "result_dims=%s scale_dims=%s input_dims=%s result_is_alloc=%s "
                "-> %s",
                i,
                site["node"].name,
                site["family"],
                site["chain"],
                site["result_dims"],
                site["scale_dims"],
                site["input_dims"],
                site["result_is_alloc"],
                " | ".join(lines),
            )

        # Which sites the LIVE pass actually fused is reported after apply()
        # by :meth:`report_result` (the telemetry captures match node names).

    @classmethod
    def report_result(cls, graph: fx.Graph) -> None:
        """Post-apply: join the telemetry's confirmed-match node names with
        the site list to say which QUANT SITES got fused."""
        if not enabled() or not cls._done or _QUANT_OP is None:
            return
        if getattr(cls, "_result_done", False):
            return
        cls._result_done = True

        from ..vllm_inductor_pass import PatternMatchTelemetry

        site_names = [
            node.name for node in graph.nodes if called_op(node) is _QUANT_OP
        ]
        fused: set[str] = set()
        for names in PatternMatchTelemetry.matched_node_names.values():
            fused.update(names)
        matched = [n for n in site_names if n in fused]
        unmatched = [n for n in site_names if n not in fused]
        logger.info(
            "fusion-site-diag result: matched=%d/%d unmatched=%s",
            len(matched),
            len(site_names),
            unmatched,
        )


def _motif_of(entry: Any) -> str:
    """Short motif label for an entry (its pattern's norm form + epsilon)."""
    pat = repr(entry.pattern)
    if "fused_add_rms_norm" in pat:
        fam = "FusedAdd"
    elif "rms_norm" in pat:
        fam = "RMSNorm"
    else:
        fam = "Other"
    # The pattern repr hides constant args; walk the pattern tree for the
    # epsilon literal it compares against.
    eps = "eps?"
    seen: set[int] = set()

    def walk(obj: Any) -> None:
        nonlocal eps
        if id(obj) in seen or len(seen) > 400:
            return
        seen.add(id(obj))
        if isinstance(obj, float) and obj in (1e-5, 1e-6):
            eps = repr(obj)
        for attr in ("args", "kwargs", "outputs", "fns_set"):
            val = getattr(obj, attr, None)
            if isinstance(val, dict):
                val = list(val.items())
            if isinstance(val, (list, tuple, set)):
                for item in val:
                    walk(item)
            elif val is not None and val is not obj:
                walk(val)

    walk(entry.pattern)
    return f"{fam}[{eps}]"


# ---------------------------------------------------------------------------
# Legacy (pre-fix) computed-alloc entries -- DIAGNOSTIC ONLY, never applied.
# ---------------------------------------------------------------------------


def build_legacy_entries() -> PatternMatcherPass:
    """Register the PRE-FIX int8 norm->quant pattern forms for diagnosis.

    These mirror the patterns as they were when the real model matched 10/70
    sites (commit ``7ab440c937``): ``pm.fwd_only`` trace and COMPUTED allocs
    in ``MatcherQuantInt8`` (the buffer-less code path, kept above for the fp8
    family). They are registered into a throwaway pass that is never applied,
    so their ``check_fn`` verdicts can be compared site by site with the live
    buffer-as-input entries.

    Registered inside the pass's ``enable_fake_mode`` init context (pattern
    registration traces immediately).
    """
    pm_pass = PatternMatcherPass(pass_name="int8_legacy_alloc_diag")
    if _QUANT_OP is None:
        return pm_pass

    from .matcher_utils import kInt8DynamicTokenSym

    quant_matcher = MatcherQuantInt8(kInt8DynamicTokenSym)
    assert quant_matcher.QUANT_OP is _QUANT_OP

    def _example(*shape: int) -> torch.Tensor:
        return torch.empty(
            shape, dtype=torch.bfloat16, device=current_platform.device_type
        )

    def _make(
        eps: float, fused_add: bool
    ) -> tuple[Any, Any, list[torch.Tensor]]:
        if fused_add:

            def pattern(  # noqa: F811
                input: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                result_rms, res2 = vllm.ir.ops.fused_add_rms_norm(
                    input, residual, weight, eps
                )
                result, scale = quant_matcher(result_rms)
                return result, res2, scale

            def replacement(  # noqa: F811
                input: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return pattern(input, weight, residual)

            example_inputs = [
                _example(5, 16),  # input
                _example(16),  # weight
                _example(5, 16),  # residual
            ]
        else:

            def pattern(  # noqa: F811
                input: torch.Tensor, weight: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                result_rms = vllm.ir.ops.rms_norm(input, weight, eps)
                return quant_matcher(result_rms)  # type: ignore[return-value]

            def replacement(  # noqa: F811
                input: torch.Tensor, weight: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return pattern(input, weight)

            example_inputs = [
                _example(5, 16),  # input
                _example(16),  # weight
            ]

        return pattern, replacement, example_inputs

    for eps in (1e-5, 1e-6):
        for fused_add in (True, False):
            pattern, replacement, example_inputs = _make(eps, fused_add)
            pm.register_replacement(
                pattern,
                replacement,
                example_inputs,
                pm.fwd_only,
                pm_pass,
            )
    return pm_pass
