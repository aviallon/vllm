# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalise ``auto_functionalized_v2`` HOP calls to the ``auto_functionalized``
(v1) form.

Why this exists
---------------
On torch >= 2.13, AOTAutograd emits ``auto_functionalized_v2`` for custom ops
with mutable arguments: the mutated tensors travel as ``_all_bases`` with
``_<name>_base_index`` indirection instead of named kwargs. Every vLLM fusion
matcher traces the **v1** call convention
(``auto_functionalized(op, result=..., input=..., scale=..., azp=None)``), and
``FixFunctionalizationPass`` looks for v1 nodes too -- so on this torch **no
fusion pattern can match the real graph at all** (measured 2026-10-19: the int8
norm->quant patterns matched 0 of 70 candidate sites on the real w8a8 qwen3.6
graph, on production torch 2.13.0+git733fca1, purely because of this form
difference -- the site census showed every site present).

The pass rewrites each simple v2 node into the equivalent v1 node before the
fusion passes run. Complex v2 nodes (list-valued mutated args, ViewInfo
entries) are left untouched -- they are not forms any matcher targets.

Output-index mapping: v2 returns ``(op_returns..., *bases in _all_bases
order)``, v1 returns ``(op_returns..., *mutated args in schema order)``;
consumers are ``getitem`` calls, so their indices are remapped by base name.
"""

import operator
from typing import Any

import torch
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._ops import OpOverload

from vllm.logger import init_logger

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

_v2_hop = getattr(torch.ops.higher_order, "auto_functionalized_v2", None)


def _mutable_arg_names(op: OpOverload) -> list[str]:
    return [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]


class NormalizeAutoFunctionalizedPass(VllmInductorPass):
    """Rewrite simple auto_functionalized_v2 nodes into auto_functionalized.

    Runs first in the post-grad pass manager, before every fusion matcher.
    A no-op on torch versions that emit v1 (or none at all).
    """

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if _v2_hop is None:
            return
        wrapped = 0
        rewritten = 0
        for node in list(graph.nodes):
            if self._is_v2(node):
                rewritten += int(self._normalize_one(graph, node))
            elif self._is_raw_mutation_call(node):
                wrapped += int(self._wrap_raw_call(graph, node))
        if rewritten or wrapped:
            # The rewrites move values across the graph; lint so a
            # use-before-def can never pass silently.
            graph.lint()
            logger.info(
                "normalize_auto_functionalized: rewrote %d v2 node(s), "
                "wrapped %d raw mutation call(s) into the v1 call form",
                rewritten,
                wrapped,
            )

    @staticmethod
    def _is_raw_mutation_call(node: fx.Node) -> bool:
        """A plain ``op(result, input, ...)`` mutation call.

        This is the form inductor's decompose_auto_functionalized leaves in
        the FINAL graph, and the form some pipelines produce before the vLLM
        hook (observed 2026-10-19 in an in-pod capture that ran the full
        post_grad_passes). The w8a8 engine's hook graph currently carries v1
        nodes (fusion site census: quant_call_forms={'auto_functionalized':
        110}); this branch is defensive canonicalisation so the matchers see
        ONE form regardless of which stage/pipeline produced the call."""
        if node.op != "call_function" or not isinstance(node.target, OpOverload):
            return False
        schema = node.target._schema
        return bool(schema.is_mutable) and not schema.returns

    def _wrap_raw_call(self, graph: fx.Graph, node: fx.Node) -> bool:
        """Rewrite one raw mutation call into the equivalent auto_functionalized
        (v1) node, rewiring post-call users of the mutated buffers to the
        functionalised outputs."""
        op = node.target
        assert isinstance(op, OpOverload)
        schema = op._schema
        schema_args = list(schema.arguments)

        # Map the call's args/kwargs onto schema names (same normalisation
        # do_auto_functionalize performs).
        values: dict[str, Any] = {}
        for idx, arg in enumerate(schema_args):
            if arg.name in node.kwargs:
                values[arg.name] = node.kwargs[arg.name]
            elif idx < len(node.args):
                values[arg.name] = node.args[idx]
            else:
                try:
                    values[arg.name] = arg.default_value
                except Exception:
                    return self._refuse(node, "raw call misses required arg")
        if node.kwargs and set(node.kwargs) - {a.name for a in schema_args}:
            return self._refuse(node, "raw call has unknown kwargs")
        if len(node.args) > len(schema_args):
            return self._refuse(node, "raw call has too many positional args")
        if node.users:
            return self._refuse(node, "raw mutation call return value is used")

        mutable = _mutable_arg_names(op)
        # Pre-call users of a mutated buffer read the OLD value and must stay;
        # post-call users read the mutated value and move to the getitem.
        positions = {n: i for i, n in enumerate(graph.nodes)}
        call_pos = positions[node]
        moves: list[tuple[fx.Node, fx.Node]] = []
        for name in mutable:
            v = values[name]
            if v is None:
                continue
            if not isinstance(v, fx.Node):
                return self._refuse(node, "non-node mutated value")
            for user in list(v.users):
                if user is node:
                    continue
                if positions.get(user, call_pos) < call_pos:
                    # pre-call read of the same buffer: legal (reads the old
                    # value), keep it.
                    continue
                if isinstance(user.target, OpOverload) and (
                    user.target._schema.is_mutable
                ):
                    # a LATER mutation call targeting the same buffer: its
                    # semantics need the raw buffer, not a functional value.
                    return self._refuse(node, "mutated buffer re-mutated later")
                moves.append((user, v))
        non_mutated_values = {
            id(values[a.name]) for a in schema_args if a.name not in mutable
        }
        for name in mutable:
            if (
                isinstance(values[name], fx.Node)
                and id(values[name]) in non_mutated_values
            ):
                return self._refuse(node, "mutated buffer also used as non-mutated arg")

        v1_kwargs: dict[str, Any] = dict(values)
        # Insert at the replaced node's position (fx appends to the end by
        # default, which would leave the rewired consumers reading a node
        # defined after them -- a use-before-def that graph.lint() rejects).
        with graph.inserting_before(node):
            new_node = graph.call_function(
                auto_functionalized, args=(op,), kwargs=v1_kwargs
            )
        new_node.meta.update(
            {k: v for k, v in node.meta.items() if k != "eager_input_vals"}
        )

        k_returns = len(schema.returns) or 1
        getitems: dict[str, fx.Node] = {}
        with graph.inserting_after(new_node):
            for pos, name in enumerate(mutable):
                getitems[name] = graph.call_function(
                    operator.getitem, args=(new_node, k_returns + pos)
                )

        for user, base in moves:
            # find which mutated value this user read
            target = None
            for name in mutable:
                if values[name] is base:
                    target = getitems[name]
                    break
            if target is None:
                continue
            new_args = tuple(target if a is base else a for a in user.args)
            new_kwargs = {
                k: (target if a is base else a) for k, a in user.kwargs.items()
            }
            user.args = new_args
            user.kwargs = new_kwargs

        graph.erase_node(node)
        return True

    @staticmethod
    def _is_v2(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        if _v2_hop is not None and node.target == _v2_hop:
            return True
        name = getattr(node.target, "__name__", "")
        cls_name = type(node.target).__name__
        return name == "auto_functionalized_v2" or cls_name == "AutoFunctionalizedV2"

    _refusal_logged: set[str] = set()

    def _refuse(self, node: fx.Node, reason: str) -> bool:
        key = f"{getattr(node.target, '__name__', node.target)}:{reason}"
        if key not in NormalizeAutoFunctionalizedPass._refusal_logged:
            NormalizeAutoFunctionalizedPass._refusal_logged.add(key)
            logger.warning(
                "normalize_auto_functionalized: cannot rewrite %s node (%s); "
                "fusion patterns will not match it. node=%r",
                getattr(node.target, "__name__", node.target),
                reason,
                node,
            )
        return False

    def _normalize_one(self, graph: fx.Graph, node: fx.Node) -> bool:
        """Rewrite one v2 node. Returns False (graph untouched) if the node is
        a v2 shape this rewrite does not understand."""
        if not node.args or not isinstance(node.args[0], OpOverload):
            return self._refuse(node, "first arg is not an OpOverload")
        op = node.args[0]
        mutable = _mutable_arg_names(op)
        kwargs = dict(node.kwargs)

        all_bases = kwargs.pop("_all_bases", None)
        if all_bases is None or not hasattr(all_bases, "__getitem__"):
            return self._refuse(node, "missing/invalid _all_bases")

        # Collect base index indirection; refuse complex forms (list-valued
        # bases and ViewInfo extras have no v1 equivalent).
        base_index: dict[str, int | None] = {}
        for name in mutable:
            key = f"_{name}_base_index"
            if key not in kwargs:
                return self._refuse(node, f"missing {key}")
            idx = kwargs.pop(key)
            if idx is not None and not isinstance(idx, int):
                return self._refuse(node, f"non-int {key}")
            if idx is not None and not (0 <= idx < len(all_bases)):
                return self._refuse(node, f"out-of-range {key}")
            base_index[name] = idx

        # Any kwarg that is not a schema name and not already-handled
        # infrastructure keys means a v2 shape we do not understand.
        schema_names = [arg.name for arg in op._schema.arguments]
        infra = {"_only_clone_these_tensors", "_op_schema"}
        if set(kwargs) - set(schema_names) - infra:
            return self._refuse(node, "unrecognised extra kwargs")

        # Build the v1 call: every schema arg as a named kwarg, mutated args
        # taking their base tensor (or None).
        v1_kwargs: dict[str, Any] = {}
        for arg in op._schema.arguments:
            name = arg.name
            if name in mutable:
                idx = base_index[name]
                v1_kwargs[name] = None if idx is None else all_bases[idx]
            elif name in kwargs:
                v1_kwargs[name] = kwargs[name]
            else:
                try:
                    v1_kwargs[name] = arg.default_value
                except Exception:
                    # required non-mutated arg missing from the v2 call
                    return self._refuse(node, "missing required non-mutated arg")

        # v2 tail index (base position) -> v1 tail index (schema position)
        k_returns = len(op._schema.returns) or 1
        tail_map: dict[int, int] = {}
        for schema_pos, name in enumerate(mutable):
            idx = base_index[name]
            if idx is not None and idx not in tail_map:
                tail_map[idx] = schema_pos

        # Pre-validate consumers BEFORE mutating anything: every user must be
        # a getitem whose index we can remap.
        remap: list[tuple[fx.Node, int]] = []
        for user in node.users:
            if user.op != "call_function" or user.target is not operator.getitem:
                return self._refuse(node, "non-getitem consumer")
            old_idx = user.args[1]
            if not isinstance(old_idx, int) or old_idx < 0:
                return self._refuse(node, "non-int getitem index")
            if old_idx < k_returns:
                new_idx = old_idx
            else:
                base_pos = old_idx - k_returns
                if base_pos not in tail_map:
                    return self._refuse(node, "getitem index not mappable")
                new_idx = k_returns + tail_map[base_pos]
            remap.append((user, new_idx))

        # All checks passed: rewrite. Insert at the replaced node's position
        # (fx appends by default -> use-before-def for the rewired users).
        with graph.inserting_before(node):
            new_node = graph.call_function(
                auto_functionalized, args=(op,), kwargs=v1_kwargs
            )
        # Carry the meta over, EXCEPT eager_input_vals: that one describes the
        # ORIGINAL v2 call's (args, kwargs) and inductor's decompose handler
        # tree-maps it against the node's own args/kwargs -- keeping the stale
        # v2 shape there made replace_by_example raise
        # "Node arity mismatch; expected 4, but got 5" on every rewritten node
        # that survived to decompose_auto_functionalized.
        new_node.meta.update(
            {k: v for k, v in node.meta.items() if k != "eager_input_vals"}
        )
        for user, new_idx in remap:
            user.update_arg(0, new_node)
            user.update_arg(1, new_idx)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return True
