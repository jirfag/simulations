from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class ServerCallContext:
    attempted_calls: int


class RetryStrategy(ABC):
    failed_count: int = 0
    total_count: int = 0

    def add_call_to_stats(self, is_success: bool):
        self.total_count += 1
        if not is_success:
            self.failed_count += 1

    @abstractmethod
    # need_retry returns do we need to retry and after what time duration.
    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, float]:
        pass

class FixedRetryCountStrategy(RetryStrategy):
    def __init__(self, retry_count: int):
        self.retry_count = retry_count

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, float]:
        return (ctx.attempted_calls <= self.retry_count, 0)


class FixedRetryCountWithExpBackoffStrategy(RetryStrategy):
    def __init__(self, retry_count: int, exp_base: float, max_sleep_time: float):
        self.retry_count = retry_count
        self.exp_base = exp_base
        self.max_sleep_time = max_sleep_time

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, float]:
        return (
            ctx.attempted_calls <= self.retry_count,
            min(math.pow(self.exp_base, ctx.attempted_calls - 1), self.max_sleep_time)
        )

class RetryCircuitBreakerStrategy(RetryStrategy):
    def __init__(self, max_failures_rate: float):
        self.max_failures_rate = max_failures_rate

    def need_retry(self, ctx: ServerCallContext) -> Tuple[bool, float]:
        return (self.failed_count / self.total_count < self.max_failures_rate, 0)