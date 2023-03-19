from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque

import simpy

from shared import TIME_DURATION_MS, TimePoint


class CircuitBreakerState(Enum):
    CLOSED = 1
    OPEN = 2
    HALF_OPEN = 3


@dataclass
class RequestEvent:
    timepoint: TimePoint
    is_ok: bool


class CircuitBreaker:
    def __init__(self, failure_rate_threshold: float):
        self.state = CircuitBreakerState.CLOSED
        self.failure_rate_threshold = failure_rate_threshold
        self.window_duration = 200 * TIME_DURATION_MS  # such small values only for simulation purposes
        self.wait_duration_in_open_state = 100 * TIME_DURATION_MS  # such small values only for simulation purposes

        self.min_window_size = 100
        self.permitted_num_of_calls_in_half_open_state = 100
        assert self.min_window_size <= self.permitted_num_of_calls_in_half_open_state

        self.last_open_time = TimePoint(0)
        self.requests_window: Deque[RequestEvent] = deque()
        self.window_failed_req_count = 0

    def record_request(self, timepoint: TimePoint, is_ok: bool):
        self.requests_window.append(RequestEvent(timepoint, is_ok))
        self.window_failed_req_count += 0 if is_ok else 1
        while len(self.requests_window) and timepoint - self.requests_window[0].timepoint > self.window_duration:
            self.window_failed_req_count -= 0 if self.requests_window[0].is_ok else 1
            self.requests_window.popleft()

    def _is_failure_threshold_reached(self) -> bool:
        return (
            len(self.requests_window) != 0
            and len(self.requests_window) >= self.min_window_size
            and self.window_failed_req_count / len(self.requests_window) >= self.failure_rate_threshold
        )

    def _reset_window(self):
        self.requests_window.clear()
        self.window_failed_req_count = 0

    def _set_state(self, state: CircuitBreakerState, env: simpy.Environment):
        self.state = state
        if state == CircuitBreakerState.OPEN:
            self.last_open_time = TimePoint(env.now)
        self._reset_window()

    def is_request_allowed(self, env: simpy.Environment) -> bool:
        if self.state == CircuitBreakerState.CLOSED:
            if not self._is_failure_threshold_reached():
                return True
            self._set_state(CircuitBreakerState.OPEN, env)
            return False

        if self.state == CircuitBreakerState.OPEN:
            if env.now - self.last_open_time < self.wait_duration_in_open_state:
                return False
            self._set_state(CircuitBreakerState.HALF_OPEN, env)
            assert self.permitted_num_of_calls_in_half_open_state > 0
            return True

        if self.state == CircuitBreakerState.HALF_OPEN:
            if len(self.requests_window) < self.permitted_num_of_calls_in_half_open_state:
                return True
            if self._is_failure_threshold_reached():
                self._set_state(CircuitBreakerState.OPEN, env)
                return False
            self._set_state(CircuitBreakerState.CLOSED, env)
            return True

        raise Exception(f"Unknown state: {self.state}")
