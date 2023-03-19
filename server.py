from abc import ABC, abstractmethod
import random
from typing import Callable, List, Optional

import simpy
from events import ServerResult
from shared import TimeDuration, TimeInterval, TimePoint
from dataclasses import dataclass


@dataclass
class InjectedFailure:
    interval: TimeInterval
    failure_prob: Optional[float] = None
    use_failure_prob_per_user: bool = False
    is_system_freeze: bool = False


@dataclass
class Request:
    deadline: TimePoint
    user_id: int


class Server(ABC):
    injected_failures: List[InjectedFailure]
    target_uptime: float
    events: List[ServerResult]
    handle_duration_builder: Callable[[], TimeDuration]
    enable_deadline_propagation: bool

    def __init__(
        self,
        injected_failures: List[InjectedFailure],
        target_uptime: float,
        handle_duration_builder: Callable[[], TimeDuration],
        enable_deadline_propagation: bool,
    ):
        self.injected_failures = injected_failures
        assert all(f.interval.begin < f.interval.end for f in injected_failures)

        self.target_uptime = target_uptime
        self.events = []
        self.handle_duration_builder = handle_duration_builder
        self.enable_deadline_propagation = enable_deadline_propagation

    @abstractmethod
    def handle(
        self,
        env: simpy.Environment,
        connection: simpy.Store,
        req: Request,
    ):
        pass

    def _base_handle(
        self,
        env: simpy.Environment,
        connection: simpy.Store,
        pending_req_count: int,
        req: Request,
    ):
        # Drop all failures that are in the past.
        while self.injected_failures and self.injected_failures[0].interval.end < env.now:
            self.injected_failures.pop(0)

        is_ok = True
        started_at = TimePoint(env.now)

        normal_handle_duration = self.handle_duration_builder()
        handle_duration = normal_handle_duration
        if self.enable_deadline_propagation:
            deadline_rest = req.deadline - started_at
            if deadline_rest < 0:
                # Let's set that fast request reject takes 5% of normal request time.
                handle_duration = normal_handle_duration // 20
                is_ok = False
            elif deadline_rest < normal_handle_duration:
                # Use normal_handle_duration // 5 because typically we check deadline in some checkpoints.
                handle_duration = min(deadline_rest + normal_handle_duration // 5, normal_handle_duration)
                is_ok = False

        if handle_duration != 0:
            yield env.timeout(handle_duration)

        # Inject failure if needed.
        failure_prob = 1 - self.target_uptime
        use_failure_prob_per_user = False
        if self.injected_failures and env.now >= self.injected_failures[0].interval.begin:
            f = self.injected_failures[0]
            if f.is_system_freeze:
                freeze_duration = f.interval.end - max(started_at, f.interval.begin)
                handle_duration += freeze_duration
                yield env.timeout(freeze_duration)
            else:
                assert f.failure_prob is not None
                failure_prob = f.failure_prob
                use_failure_prob_per_user = f.use_failure_prob_per_user
                if use_failure_prob_per_user:
                    assert req.user_id
        if is_ok:
            is_ok = ((req.user_id % 1000) / 1000.0 >= failure_prob) if use_failure_prob_per_user else (random.random() >= failure_prob)

        self.events.append(ServerResult(TimePoint(env.now), handle_duration, pending_req_count, is_ok))
        yield connection.put(is_ok)


class SimpleServer(Server):
    def handle(self, env: simpy.Environment, connection: simpy.Store, req: Request):
        yield from self._base_handle(env, connection, 0, req)


class CpuBoundServer(Server):
    cpus: Optional[simpy.Resource] = None
    cpu_count: int

    def __init__(
        self,
        cpu_count: int,
        handle_duration_builder: Callable[[], TimeDuration],
        injected_failures: List[InjectedFailure],
        target_uptime: float,
        enable_deadline_propagation: bool = False,
    ):
        super(CpuBoundServer, self).__init__(injected_failures, target_uptime, handle_duration_builder, enable_deadline_propagation)
        self.cpu_count = cpu_count

    def handle(self, env: simpy.Environment, connection: simpy.Store, req: Request):
        if self.cpus is None:
            self.cpus = simpy.Resource(env, capacity=self.cpu_count)
        with self.cpus.request() as cpu_req:
            yield cpu_req
            yield from self._base_handle(env, connection, len(self.cpus.queue), req)
