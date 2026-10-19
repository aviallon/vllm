# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for admission control (max_num_queued_reqs / max_num_queued_tokens).

These tests cover:
- OutputProcessor.get_num_queued_tokens() token counting
- AsyncLLM.check_admission() admission control logic
- Exception classes (GracefulHTTPError, QueueOverflowError, MaxQueuedTokensError)
- create_error_response() mapping GracefulHTTPError to HTTP 503 / 429
- RetryAfterEstimator() heuristic, bounds and staleness
- the retry-after-on-overload mode (HTTP 429 + Retry-After, both response forms)
- SchedulerConfig field defaults and validation
- human_readable_int CLI notation for max_num_queued_tokens
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from vllm.config.scheduler import SchedulerConfig
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
    error_response_to_json_response,
)
from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
    VLLMError,
)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import human_readable_int
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.admission_control import (
    RETRY_AFTER_FALLBACK_SECONDS,
    RETRY_AFTER_MAX_SECONDS,
    RETRY_AFTER_MIN_SECONDS,
    RETRY_AFTER_RATE_TAU_SECONDS,
    RetryAfterEstimator,
    SharedAdmissionStats,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_req_state(prompt_len: int, is_prefilling: bool = True):
    return SimpleNamespace(
        prompt_len=prompt_len,
        is_prefilling=is_prefilling,
    )


def _make_async_llm(
    max_num_queued_reqs: int | None = None,
    max_num_queued_tokens: int | None = None,
    num_unfinished: int = 0,
    num_queued_tokens: int = 0,
    retry_after_on_overload: bool = False,
) -> AsyncLLM:
    """Create a bare AsyncLLM with just the attributes needed for scheduling."""
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.scheduler_config = SimpleNamespace(
        max_num_queued_reqs=max_num_queued_reqs,
        max_num_queued_tokens=max_num_queued_tokens,
        retry_after_on_overload=retry_after_on_overload,
    )
    llm.output_processor = MagicMock()
    llm.output_processor.get_num_unfinished_requests.return_value = num_unfinished
    llm.output_processor.get_num_queued_tokens.return_value = num_queued_tokens
    llm.admission_stats = None
    llm.retry_estimator = RetryAfterEstimator() if retry_after_on_overload else None
    return llm


def _make_output_processor(**request_states) -> OutputProcessor:
    op = OutputProcessor.__new__(OutputProcessor)
    op.request_states = request_states
    return op


def _make_engine_request(request_id: str, n: int) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(n=n),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_request_test_llm(
    max_num_queued_reqs: int,
    add_request_async: Callable[[EngineCoreRequest], Awaitable[None]],
) -> AsyncLLM:
    llm = _make_async_llm(max_num_queued_reqs=max_num_queued_reqs)
    llm.output_processor = OutputProcessor(None, log_stats=False)
    llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        add_request_async=AsyncMock(side_effect=add_request_async),
        abort_requests_async=AsyncMock(),
        shutdown=MagicMock(),
    )
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    llm.output_handler = None
    llm.log_requests = False
    llm._run_output_handler = MagicMock()
    llm.input_processor = MagicMock()
    llm.input_processor.assign_request_id.side_effect = lambda request: setattr(
        request, "external_req_id", request.request_id
    )
    return llm


def _make_scheduler_config(**kwargs) -> SchedulerConfig:
    return SchedulerConfig(
        runner_type="generate",
        max_model_len=4096,
        is_encoder_decoder=False,
        **kwargs,
    )


def _make_shared_stats(client_count: int = 2) -> list[SharedAdmissionStats]:
    counters = multiprocessing.RawArray(
        "q", SharedAdmissionStats.num_counters(client_count)
    )
    return [
        SharedAdmissionStats({"mp_admission_counters": counters}, client_count, index)
        for index in range(client_count)
    ]


@pytest.mark.skip_global_cleanup
def test_shared_admission_stats_aggregate_api_servers():
    server_0, server_1 = _make_shared_stats()

    server_0.set_num_requests(2)
    assert server_0.get_num_requests() == 2
    server_1.set_num_requests(3)
    assert server_0.get_num_requests() == server_1.get_num_requests() == 5

    server_0.set_num_requests(1)
    assert server_0.get_num_requests() == 4


@pytest.mark.skip_global_cleanup
def test_shared_request_snapshot_enforces_global_limit():
    server_0, server_1 = _make_shared_stats()
    server_0.set_num_requests(2)
    llm = _make_async_llm(max_num_queued_reqs=2)
    llm.admission_stats = server_1

    with pytest.raises(QueueOverflowError):
        llm.check_admission()

    assert server_0.get_num_requests() == 2


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_graceful_http_error_carries_status_and_message():
    err = GracefulHTTPError("custom message", HTTPStatus.SERVICE_UNAVAILABLE)
    assert err.message == "custom message"
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert str(err) == "custom message"


def test_graceful_http_error_is_vllm_error():
    err = GracefulHTTPError("msg", HTTPStatus.TOO_MANY_REQUESTS)
    assert isinstance(err, VLLMError)


def test_queue_overflow_error():
    err = QueueOverflowError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "busy" in err.message.lower() or "try again" in err.message.lower()


def test_max_queued_tokens_error():
    err = MaxQueuedTokensError()
    assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
    assert isinstance(err, GracefulHTTPError)
    assert "backlog" in err.message.lower() or "try again" in err.message.lower()


@pytest.mark.parametrize("exc_cls", [QueueOverflowError, MaxQueuedTokensError])
def test_admission_exceptions_are_vllm_errors(exc_cls):
    """Admission rejections must reach the VLLMError HTTP handler."""
    assert issubclass(exc_cls, VLLMError)


# ---------------------------------------------------------------------------
# OutputProcessor.get_num_queued_tokens
# ---------------------------------------------------------------------------


def test_queued_tokens_empty():
    assert _make_output_processor().get_num_queued_tokens() == 0


def test_queued_tokens_sums_prefilling_requests():
    op = _make_output_processor(r1=_make_req_state(100), r2=_make_req_state(200))
    assert op.get_num_queued_tokens() == 300


def test_queued_tokens_excludes_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=True),
        r2=_make_req_state(200, is_prefilling=False),
        r3=_make_req_state(50, is_prefilling=True),
    )
    assert op.get_num_queued_tokens() == 150


def test_queued_tokens_all_non_prefilling():
    op = _make_output_processor(
        r1=_make_req_state(100, is_prefilling=False),
        r2=_make_req_state(200, is_prefilling=False),
    )
    assert op.get_num_queued_tokens() == 0


# ---------------------------------------------------------------------------
# AsyncLLM.check_admission
# ---------------------------------------------------------------------------


def test_admission_no_limits_allows_everything():
    llm = _make_async_llm(num_unfinished=999, num_queued_tokens=999)
    llm.check_admission()


# -- max_num_queued_reqs ----------------------------------------------------


def test_admission_reqs_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=5)
    llm.check_admission()


def test_admission_reqs_rejects_at_limit():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


def test_admission_reqs_rejects_with_n():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=8)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(3)


def test_admission_reqs_allows_n_at_boundary():
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=7)
    llm.check_admission(3)


def test_admission_reqs_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_reqs=0, num_unfinished=0)
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


# -- max_num_queued_tokens --------------------------------------------------


def test_admission_tokens_allows_when_under_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=500)
    llm.check_admission()


def test_admission_tokens_rejects_at_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1000)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_tokens_rejects_over_limit():
    llm = _make_async_llm(max_num_queued_tokens=1000, num_queued_tokens=1500)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_tokens_rejects_when_zero_limit():
    llm = _make_async_llm(max_num_queued_tokens=0, num_queued_tokens=0)
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


# -- interaction between both limits ----------------------------------------


def test_admission_both_limits_checked_independently():
    llm = _make_async_llm(
        max_num_queued_reqs=100,
        max_num_queued_tokens=1000,
        num_unfinished=5,
        num_queued_tokens=1000,
    )
    with pytest.raises(MaxQueuedTokensError):
        llm.check_admission()


def test_admission_req_limit_checked_before_token_limit():
    llm = _make_async_llm(
        max_num_queued_reqs=10,
        max_num_queued_tokens=1000,
        num_unfinished=10,
        num_queued_tokens=1000,
    )
    with pytest.raises(QueueOverflowError):
        llm.check_admission()


# -- n derived from params at the add_request call site ---------------------


@pytest.mark.parametrize("params", [SamplingParams(), PoolingParams()])
def test_admission_params_without_explicit_n_count_as_one_slot(params):
    """PoolingParams has no ``.n``; add_request must fall back to 1."""
    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=9)
    llm.check_admission(getattr(params, "n", 1) or 1)

    llm = _make_async_llm(max_num_queued_reqs=10, num_unfinished=10)
    with pytest.raises(QueueOverflowError):
        llm.check_admission(getattr(params, "n", 1) or 1)


@pytest.mark.asyncio
async def test_concurrent_single_request_admission_respects_limit():
    """Concurrent n=1 admissions cannot consume the same slot."""
    llm = _make_async_llm(max_num_queued_reqs=1)
    request_states: dict[str, SimpleNamespace] = {}
    llm.output_processor.get_num_unfinished_requests.side_effect = lambda: len(
        request_states
    )
    llm.output_processor.has_request.side_effect = request_states.__contains__

    def add_request(request, *_args):
        request_states[request.request_id] = _make_req_state(
            len(request.prompt_token_ids)
        )

    llm.output_processor.add_request.side_effect = add_request

    async def add_request_async(_request):
        await asyncio.sleep(0)

    llm.engine_core = SimpleNamespace(
        add_request_async=AsyncMock(side_effect=add_request_async),
        shutdown=MagicMock(),
    )
    llm.log_requests = False

    requests = [
        SimpleNamespace(request_id=f"request-{idx}", prompt_token_ids=[idx])
        for idx in range(2)
    ]
    results = await asyncio.gather(
        *(
            llm._add_request(request, None, None, 0, MagicMock())
            for request in requests
        ),
        return_exceptions=True,
    )

    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, QueueOverflowError) for result in results) == 1
    assert len(request_states) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("first_n", [1, 3])
async def test_parallel_admission_is_all_or_nothing(first_n: int):
    """An n>1 request reserves either all its capacity or none of it."""

    async def add_request_async(_request):
        await asyncio.sleep(0)

    llm = _make_request_test_llm(3, add_request_async)
    second_n = 3 if first_n == 1 else 1
    first = _make_engine_request("first", first_n)
    second = _make_engine_request("second", second_n)
    results = await asyncio.gather(
        llm.add_request("first", first, first.params),
        llm.add_request("second", second, second.params),
        return_exceptions=True,
    )

    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], QueueOverflowError)
    assert llm.output_processor.get_num_unfinished_requests() == first_n
    assert llm.engine_core.add_request_async.await_count == first_n


@pytest.mark.asyncio
async def test_parallel_admission_cancellation_cleans_up_all_children():
    """Cancellation during core submission must release every reserved slot."""
    first_submission_started = asyncio.Event()

    async def add_request_async(_request):
        first_submission_started.set()
        await asyncio.Event().wait()

    llm = _make_request_test_llm(3, add_request_async)
    request = _make_engine_request("parallel", 3)
    output = llm.generate(request, request.params, request.request_id)
    generate_task = asyncio.create_task(anext(output))

    await first_submission_started.wait()
    generate_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generate_task

    assert llm.engine_core.add_request_async.await_count == 1
    assert llm.output_processor.get_num_unfinished_requests() == 0
    aborted_ids = llm.engine_core.abort_requests_async.await_args.args[0]
    assert set(aborted_ids) == {"0_parallel", "1_parallel", "2_parallel"}


# ---------------------------------------------------------------------------
# create_error_response integration
# ---------------------------------------------------------------------------


def test_queue_overflow_maps_to_503():
    resp = create_error_response(QueueOverflowError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    assert resp.error.param is None
    msg = resp.error.message.lower()
    assert "busy" in msg or "try again" in msg


def test_max_queued_tokens_maps_to_503():
    resp = create_error_response(MaxQueuedTokensError())
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase
    msg = resp.error.message.lower()
    assert "backlog" in msg or "try again" in msg


def test_custom_graceful_error_maps_to_its_status():
    err = GracefulHTTPError("custom", HTTPStatus.SERVICE_UNAVAILABLE)
    resp = create_error_response(err)
    assert resp.error.code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert resp.error.type == HTTPStatus.SERVICE_UNAVAILABLE.phrase


# ---------------------------------------------------------------------------
# SchedulerConfig field defaults
# ---------------------------------------------------------------------------


def test_scheduler_config_defaults_are_none():
    config = _make_scheduler_config()
    assert config.max_num_queued_reqs is None
    assert config.max_num_queued_tokens is None


def test_scheduler_config_accepts_explicit_values():
    config = _make_scheduler_config(
        max_num_queued_reqs=100,
        max_num_queued_tokens=32000,
    )
    assert config.max_num_queued_reqs == 100
    assert config.max_num_queued_tokens == 32000


@pytest.mark.parametrize("field", ["max_num_queued_reqs", "max_num_queued_tokens"])
def test_scheduler_config_rejects_negative(field):
    with pytest.raises(ValidationError):
        _make_scheduler_config(**{field: -1})


# ---------------------------------------------------------------------------
# human_readable_int for CLI notation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("32k", 32_000),
        ("1k", 1_000),
        ("1K", 1_024),
        ("1m", 1_000_000),
        ("1M", 1_048_576),
        ("100", 100),
        ("2.5k", 2_500),
        ("0", 0),
    ],
)
def test_human_readable_int_parses_notation(input_str: str, expected: int):
    assert human_readable_int(input_str) == expected


@pytest.mark.parametrize("invalid", ["abc", "1x", "", "k", "1.5K"])
def test_human_readable_int_rejects_invalid(invalid: str):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        human_readable_int(invalid)


# ---------------------------------------------------------------------------
# Retry-After / HTTP 429 overload mode
# ---------------------------------------------------------------------------


def test_admission_errors_stay_503_without_retry_after():
    """Default behaviour is unchanged: 503, no retry hint."""
    for err in (QueueOverflowError(), MaxQueuedTokensError()):
        assert err.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        assert err.retry_after is None


def test_admission_errors_with_retry_after_are_429():
    for err in (
        QueueOverflowError(retry_after=7),
        MaxQueuedTokensError(retry_after=7),
    ):
        assert err.http_status == HTTPStatus.TOO_MANY_REQUESTS
        assert err.retry_after == 7


def test_graceful_http_error_carries_retry_after():
    err = GracefulHTTPError("slow down", HTTPStatus.TOO_MANY_REQUESTS, 15)
    assert err.retry_after == 15


def test_error_response_carries_retry_after_in_body():
    resp = create_error_response(QueueOverflowError(retry_after=7))
    assert resp.error.code == HTTPStatus.TOO_MANY_REQUESTS.value
    assert resp.error.type == HTTPStatus.TOO_MANY_REQUESTS.phrase
    assert resp.error.retry_after == 7


def test_error_response_without_hint_has_no_retry_after():
    resp = create_error_response(QueueOverflowError())
    assert resp.error.retry_after is None


def test_json_error_response_sets_retry_after_header():
    resp = error_response_to_json_response(
        create_error_response(QueueOverflowError(retry_after=7))
    )
    assert resp.status_code == HTTPStatus.TOO_MANY_REQUESTS.value
    assert resp.headers["Retry-After"] == "7"


def test_json_error_response_omits_header_without_hint():
    resp = error_response_to_json_response(create_error_response(QueueOverflowError()))
    assert resp.status_code == HTTPStatus.SERVICE_UNAVAILABLE.value
    assert "retry-after" not in resp.headers


def test_streaming_error_carries_retry_after_in_body():
    """A streaming rejection cannot carry headers: the 200 is already sent."""
    serving = GenerateBaseServing.__new__(GenerateBaseServing)
    payload = json.loads(
        serving.create_streaming_error_response(QueueOverflowError(retry_after=9))
    )
    assert payload["error"]["code"] == HTTPStatus.TOO_MANY_REQUESTS.value
    assert payload["error"]["retry_after"] == 9


def test_check_admission_disabled_has_no_retry_after():
    llm = _make_async_llm(max_num_queued_reqs=2, num_unfinished=2)
    with pytest.raises(QueueOverflowError) as exc_info:
        llm.check_admission()
    assert exc_info.value.retry_after is None
    assert exc_info.value.http_status == HTTPStatus.SERVICE_UNAVAILABLE


def test_check_admission_estimates_retry_after_on_queue_overflow():
    llm = _make_async_llm(
        max_num_queued_reqs=2,
        num_unfinished=2,
        num_queued_tokens=200,
        retry_after_on_overload=True,
    )
    assert llm.retry_estimator is not None
    llm.retry_estimator = RetryAfterEstimator(clock=lambda: 0.0)
    # One observation pins the rates: 100 prompt tokens and 1 completion
    # over tau => 10 tok/s and 0.1 req/s.
    llm.retry_estimator.observe(num_finished=1, prefill_tokens=100)
    with pytest.raises(QueueOverflowError) as exc_info:
        llm.check_admission()
    assert exc_info.value.http_status == HTTPStatus.TOO_MANY_REQUESTS
    # (200/10 + 2/0.1) * 1.1 = 44
    assert exc_info.value.retry_after == 44


def test_check_admission_estimates_retry_after_on_token_backlog():
    llm = _make_async_llm(
        max_num_queued_tokens=100,
        num_unfinished=3,
        num_queued_tokens=150,
        retry_after_on_overload=True,
    )
    assert llm.retry_estimator is not None
    llm.retry_estimator = RetryAfterEstimator(clock=lambda: 0.0)
    llm.retry_estimator.observe(num_finished=1, prefill_tokens=100)
    with pytest.raises(MaxQueuedTokensError) as exc_info:
        llm.check_admission()
    assert exc_info.value.http_status == HTTPStatus.TOO_MANY_REQUESTS
    # (150/10 + 3/0.1) * 1.1 = 49.5 -> 50
    assert exc_info.value.retry_after == 50


# ---------------------------------------------------------------------------
# RetryAfterEstimator heuristic
# ---------------------------------------------------------------------------


def test_estimator_falls_back_before_any_observation():
    est = RetryAfterEstimator()
    assert (
        est.estimate_seconds(num_in_flight=8, prefill_backlog_tokens=1000, now=0.0)
        == RETRY_AFTER_FALLBACK_SECONDS
    )


def test_estimator_uses_measured_rates():
    est = RetryAfterEstimator()
    est.observe(num_finished=1, prefill_tokens=100, now=0.0)
    # (200/10 + 5/0.1) * 1.1 = 77.0
    assert est.estimate_seconds(5, 200, now=0.0) == 77


def test_estimator_clamps_to_max():
    est = RetryAfterEstimator()
    est.observe(num_finished=1, prefill_tokens=100, now=0.0)
    assert (
        est.estimate_seconds(num_in_flight=10**6, prefill_backlog_tokens=10**9, now=0.0)
        == RETRY_AFTER_MAX_SECONDS
    )


def test_estimator_clamps_to_min():
    est = RetryAfterEstimator()
    est.observe(num_finished=1, prefill_tokens=100, now=0.0)
    assert (
        est.estimate_seconds(num_in_flight=0, prefill_backlog_tokens=0, now=0.0)
        == RETRY_AFTER_MIN_SECONDS
    )


def test_estimator_decays_to_fallback_after_idle():
    """A rate whose traffic stopped must not survive as a stale estimate."""
    est = RetryAfterEstimator()
    est.observe(num_finished=1, prefill_tokens=100, now=0.0)
    later = 60.0 * RETRY_AFTER_RATE_TAU_SECONDS
    assert (
        est.estimate_seconds(num_in_flight=8, prefill_backlog_tokens=1000, now=later)
        == RETRY_AFTER_FALLBACK_SECONDS
    )


def _make_feed_output_processor(est: RetryAfterEstimator) -> OutputProcessor:
    op = OutputProcessor.__new__(OutputProcessor)
    op.retry_estimator = est
    op.admission_stats = None
    return op


def test_output_processor_feeds_estimator_on_prefill_completion():
    """Rates that are never fed silently degrade the heuristic to its
    fallback; prompt work is observed when a request leaves prefill."""
    est = RetryAfterEstimator(clock=lambda: 0.0)
    op = _make_feed_output_processor(est)
    req_state = MagicMock()
    req_state.is_prefilling = True
    req_state.prompt_len = 100
    req_state.num_cached_tokens = 0
    req_state.remote_prefill_cached_tokens = None
    req_state.streaming_input = False
    req_state.queue = None
    req_state.detokenizer.update.return_value = None
    req_state.make_request_output.return_value = MagicMock()
    op.request_states = {"r1": req_state}

    eco = MagicMock()
    eco.request_id = "r1"
    eco.new_token_ids = [1]
    eco.finish_reason = None
    eco.prefill_stats = None
    eco.num_cached_tokens = 20
    eco.pooling_output = None
    op.process_outputs([eco], 0.0, None)

    # 80 computed prompt tokens over tau => 8 tok/s; 200/8 * 1.1 = 27.5 -> 28
    assert est.estimate_seconds(num_in_flight=0, prefill_backlog_tokens=200) == 28


def test_output_processor_feeds_estimator_on_request_finish():
    """Completions are observed when a request leaves the frontend."""
    est = RetryAfterEstimator(clock=lambda: 0.0)
    op = _make_feed_output_processor(est)
    req_state = MagicMock()
    req_state.request_id = "r1"
    req_state.external_req_id = "ext"
    req_state.parent_req = None
    op.request_states = {"r1": req_state}
    op.external_req_ids = {"ext": ["r1"]}
    op.lora_states = MagicMock()

    op._finish_request(req_state)

    # 1 completion over tau => 0.1 req/s; 4/0.1 * 1.1 = 44
    assert est.estimate_seconds(num_in_flight=4, prefill_backlog_tokens=0) == 44


# ---------------------------------------------------------------------------
# retry-after-on-overload config surface
# ---------------------------------------------------------------------------


def test_scheduler_config_retry_after_on_overload_defaults_off():
    assert _make_scheduler_config().retry_after_on_overload is False


def test_scheduler_config_retry_after_on_overload_accepts_true():
    assert _make_scheduler_config(retry_after_on_overload=True).retry_after_on_overload


def test_retry_after_on_overload_cli_flag():
    """The flag must parse into EngineArgs; an unparsed flag is a silent no-op."""
    code = textwrap.dedent(
        """
        import argparse
        from vllm.engine.arg_utils import EngineArgs

        parser = argparse.ArgumentParser()
        EngineArgs.add_cli_args(parser)
        default = parser.parse_args(["--model", "m"])
        enabled = parser.parse_args(["--model", "m", "--retry-after-on-overload"])
        assert default.retry_after_on_overload is False
        assert enabled.retry_after_on_overload is True
        """
    )
    env = os.environ | {"VLLM_TARGET_DEVICE": "cpu"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
