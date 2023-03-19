from abc import ABC, abstractmethod
from collections import deque
import copy
from dataclasses import dataclass
import random
from typing import Any, Deque, Tuple, List
import math

from shared import TIME_DURATION_MS, TimeDuration, TimePoint


@dataclass
class ServerCallContext:
    attempted_calls: int


@dataclass
class RequestEvent:
    timepoint: TimePoint
    is_ok: bool
    is_retry: bool


class RetryStrategy(ABC):
    def add_call_to_stats(self, timepoint: TimePoint, is_ok: bool, is_retry: bool):
        pass

    @abstractmethod
    # need_retry returns do we need to retry and after what time duration.
    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, TimeDuration]:
        pass

    def copy(self) -> Any:
        return self


class FixedRetryCountStrategy(RetryStrategy):
    def __init__(self, retry_count: int, fixed_sleep_time: TimeDuration = 0):
        self.retry_count = retry_count
        self.fixed_sleep_time = fixed_sleep_time

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, TimeDuration]:
        return (ctx.attempted_calls <= self.retry_count, self.fixed_sleep_time)


class FixedRetryCountWithExpBackoffStrategy(RetryStrategy):
    def __init__(self, retry_count: int, exp_base: float, sleep_base: TimeDuration, max_sleep_time: TimeDuration, use_jitter: bool = False):
        self.retry_count = retry_count
        self.exp_base = exp_base
        self.sleep_base = sleep_base
        self.max_sleep_time = max_sleep_time
        self.use_jitter = use_jitter

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, TimeDuration]:
        if ctx.attempted_calls > self.retry_count:
            return (False, 0)

        coef = math.pow(self.exp_base, ctx.attempted_calls - 1)  # 1, base, base^2, ...
        sleep_time = min(TimeDuration(self.sleep_base * coef), self.max_sleep_time)
        if self.use_jitter:
            # Using Full Jitter from https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
            sleep_time = TimeDuration(random.uniform(0, sleep_time))

            # Not using Equal Jitter from https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
            # sleep_time = sleep_time//2 + TimeDuration(random.uniform(0, sleep_time//2))
        return (True, sleep_time)


class RetryCircuitBreakerStrategy(RetryStrategy):
    def __init__(self, underlying_strategy: RetryStrategy, max_failures_rate: float, window_duration: float):
        self.max_failures_rate = max_failures_rate
        self.window_events: Deque[RequestEvent] = deque()
        self.window_failed_req_count = 0
        self.window_duration = window_duration
        self.underlying_strategy = underlying_strategy

    def add_call_to_stats(self, timepoint: TimePoint, is_ok: bool, is_retry: bool):
        while len(self.window_events) != 0 and timepoint - self.window_events[0].timepoint >= self.window_duration:
            ev = self.window_events.popleft()
            self.window_failed_req_count -= 0 if ev.is_ok else 1
        self.window_events.append(RequestEvent(timepoint, is_ok, is_retry))
        self.window_failed_req_count += 0 if is_ok else 1

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, TimeDuration]:
        if len(self.window_events) == 0:
            # Similarly to token bucket, we don't allow to retry if we don't have any successfully finished requests.
            # It's in essence a concurrency control: when all requests are inflight, we don't allow to retry.
            return (False, 0)

        if self.window_failed_req_count / len(self.window_events) >= self.max_failures_rate:
            return (False, 0)

        return self.underlying_strategy.need_retry(ctx)

    def copy(self) -> RetryStrategy:
        return copy.deepcopy(self)


class RetryBudgetStrategy(RetryStrategy):
    def __init__(self, underlying_strategy: RetryStrategy, retry_budget_ratio: float):
        assert retry_budget_ratio >= 0.01 and retry_budget_ratio <= 1
        self.tokens_to_sub_for_retry = 100  # TODO: differentiate between kind of failures: timed out, server throttled, other.
        self.tokens_to_add_for_success = int(retry_budget_ratio * self.tokens_to_sub_for_retry)

        MAX_CONSECUTIVE_RETRIES_BEFORE_THROTTLING = 30
        self.max_tokens = self.tokens_to_sub_for_retry * MAX_CONSECUTIVE_RETRIES_BEFORE_THROTTLING
        self.tokens = self.max_tokens

        self.underlying_strategy = underlying_strategy

    def add_call_to_stats(self, timepoint: TimePoint, is_ok: bool, is_retry: bool):
        if not is_ok:
            return

        # Return 0 tokens for successfull retry otherwise retry budget will allow 2 x budget retries in partial failure scenarios.
        self.tokens = min(self.max_tokens, self.tokens + (0 if is_retry else self.tokens_to_add_for_success))

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, TimeDuration]:
        if self.tokens < self.tokens_to_sub_for_retry:
            return (False, 0)

        need_retry, delay = self.underlying_strategy.need_retry(ctx)
        if not need_retry:
            return (False, 0)

        self.tokens -= self.tokens_to_sub_for_retry
        return (True, delay)

    def copy(self) -> RetryStrategy:
        return copy.deepcopy(self)