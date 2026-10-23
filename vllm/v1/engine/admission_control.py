# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frontend admission control: shared queue counters and retry-after estimation.

Enforced in the API server process (see ``AsyncLLM.check_admission``), so
these primitives stay free of engine-core state.
"""

import math
import time
from collections.abc import Callable
from typing import Any, Protocol, cast


class AdmissionCounterBuffer(Protocol):
    def __getitem__(self, index: int) -> int: ...

    def __setitem__(self, index: int, value: int) -> None: ...


class SharedAdmissionStats:
    """Lock-free admission stats shared by API server processes.

    Each process writes exclusively to its own cache-line-sized slot. Readers
    aggregate the slots, avoiding both inter-process locks and cache-line
    contention on the request path.
    """

    # Space per-process counters 64 bytes apart to avoid false sharing on CPUs
    # with 64-byte cache lines: one process's writes would otherwise invalidate
    # the cache line containing another process's counter.
    CACHE_LINE_BYTES = 64
    COUNTER_BYTES = 8
    COUNTERS_PER_SLOT = CACHE_LINE_BYTES // COUNTER_BYTES

    def __init__(
        self,
        client_addresses: dict[str, Any],
        client_count: int = 1,
        client_index: int = 0,
    ):
        self._counters = cast(
            AdmissionCounterBuffer,
            client_addresses.pop("mp_admission_counters"),
        )
        self._client_count = client_count
        self._slot_offset = client_index * self.COUNTERS_PER_SLOT

    @classmethod
    def num_counters(cls, client_count: int) -> int:
        return client_count * cls.COUNTERS_PER_SLOT

    def set_num_requests(self, count: int) -> None:
        self._counters[self._slot_offset] = count

    def get_num_requests(self) -> int:
        return sum(
            self._counters[client_index * self.COUNTERS_PER_SLOT]
            for client_index in range(self._client_count)
        )


RETRY_AFTER_MIN_SECONDS = 1
RETRY_AFTER_MAX_SECONDS = 300
RETRY_AFTER_FALLBACK_SECONDS = 30
RETRY_AFTER_MARGIN_FRACTION = 0.1
RETRY_AFTER_RATE_TAU_SECONDS = 10.0
# Rates with no observed traffic behind them are not estimates at all.
RETRY_AFTER_STALE_SECONDS = 3.0 * RETRY_AFTER_RATE_TAU_SECONDS


class RetryAfterEstimator:
    """Estimates how long a rejected client should wait before retrying.

    The estimate is the drain time of the work queued ahead of a rejected
    request::

        wait = (prefill_backlog / prefill_rate + num_in_flight / completion_rate) * (
            1 + margin
        )

    The first term is the pending prompt work divided by the observed
    prefill throughput (tokens/s). The second is the in-flight request
    count divided by the observed completion rate (req/s), which is the
    ``queue depth * mean request length / token rate`` form written out
    under the steady-state identity ``completion_rate = token_rate /
    mean_request_tokens``. ``margin``
    (``RETRY_AFTER_MARGIN_FRACTION``) covers estimator lag and synchronized
    client retries.

    It deliberately estimates the drain of the whole current backlog, not
    the time for one queue slot to free: a retry after this delay has a
    high probability of admission, whereas the one-slot wait is what turns
    a herd of rejected clients into a retry storm.

    Rates are exponentially decayed moving averages with time constant
    ``RETRY_AFTER_RATE_TAU_SECONDS``, so recent steps weigh more and a rate
    whose traffic stopped decays back to zero (an idle period cannot leave
    a stale rate behind). Once no traffic has been observed for
    ``RETRY_AFTER_STALE_SECONDS`` — e.g. a burst arriving after an idle
    period, before its first outputs — the rates are treated as unknown.
    The result is clamped to
    [``RETRY_AFTER_MIN_SECONDS``, ``RETRY_AFTER_MAX_SECONDS``]; a term whose
    rate is unknown makes the estimate fall back to
    ``RETRY_AFTER_FALLBACK_SECONDS``.

    Observation is cheap — one decayed addition per finished request and
    per processed output batch — and is fed by the frontend
    ``OutputProcessor``. Each API server process estimates from the
    traffic it observes.
    """

    def __init__(
        self,
        tau: float = RETRY_AFTER_RATE_TAU_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tau = tau
        self._clock = clock
        self._prefill_rate = 0.0
        self._completion_rate = 0.0
        self._last_decay: float | None = None
        self._last_observe: float | None = None

    def observe(
        self,
        num_finished: int = 0,
        prefill_tokens: int = 0,
        now: float | None = None,
    ) -> None:
        """Record finished requests and completed prompt work.

        Args:
            num_finished: Requests that left the frontend.
            prefill_tokens: Prompt tokens whose prefill completed (cache
                hits excluded, matching the work the engine really did).
            now: Monotonic timestamp; defaults to the estimator's clock.

        """
        now = now if now is not None else self._clock()
        self._decay(now)
        if num_finished <= 0 and prefill_tokens <= 0:
            return
        self._completion_rate += num_finished / self._tau
        self._prefill_rate += prefill_tokens / self._tau
        self._last_observe = now

    def estimate_seconds(
        self,
        num_in_flight: int,
        prefill_backlog_tokens: int,
        now: float | None = None,
    ) -> int:
        """Estimate the retry-after delay for a rejected request.

        Args:
            num_in_flight: Requests currently waiting or running.
            prefill_backlog_tokens: Prompt tokens of requests still in
                prefill.
            now: Monotonic timestamp; defaults to the estimator's clock.

        Returns:
            Whole seconds in [``RETRY_AFTER_MIN_SECONDS``,
            ``RETRY_AFTER_MAX_SECONDS``], or
            ``RETRY_AFTER_FALLBACK_SECONDS`` when the rates are unknown.

        """
        now = now if now is not None else self._clock()
        self._decay(now)
        if (
            self._last_observe is None
            or now - self._last_observe > RETRY_AFTER_STALE_SECONDS
        ):
            return RETRY_AFTER_FALLBACK_SECONDS
        wait = 0.0
        if prefill_backlog_tokens > 0:
            if self._prefill_rate <= 0.0:
                return RETRY_AFTER_FALLBACK_SECONDS
            wait += prefill_backlog_tokens / self._prefill_rate
        if num_in_flight > 0:
            if self._completion_rate <= 0.0:
                return RETRY_AFTER_FALLBACK_SECONDS
            wait += num_in_flight / self._completion_rate
        wait *= 1.0 + RETRY_AFTER_MARGIN_FRACTION
        return max(
            RETRY_AFTER_MIN_SECONDS, min(RETRY_AFTER_MAX_SECONDS, math.ceil(wait))
        )

    def _decay(self, now: float) -> None:
        if self._last_decay is None:
            self._last_decay = now
            return
        dt = now - self._last_decay
        if dt <= 0.0:
            return
        decay = math.exp(-dt / self._tau)
        self._prefill_rate *= decay
        self._completion_rate *= decay
        self._last_decay = now
