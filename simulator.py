from __future__ import annotations
from dataclasses import dataclass
import random
from circuit_breaker import CircuitBreaker
from client import Client
from retry_strategies import RetryStrategy
import simpy
import time
from typing import Callable, Tuple, Optional, List
from server import Server
from shared import TIME_DURATION_SECOND, TimeInterval, TimePoint, TimeDuration
from events import ClientResultList, ServerResultList


@dataclass
class LoadSpike:
    interval: TimeInterval
    load_multiplier: float


@dataclass
class RunParams:
    id: str
    retry_strategy: RetryStrategy
    server: Server
    target_duration: TimeDuration
    target_rps: int

    cpu_count: int
    client_timeout_builder: Callable[[], TimeDuration]

    static_clients_count: int = 0

    max_inflight_requests: Optional[int] = None
    load_spike: Optional[LoadSpike] = None
    use_global_client_state: bool = True
    circuit_breaker: Optional[CircuitBreaker] = None


class MinimalParams:
    id: str
    target_duration: TimeDuration
    target_rps: int
    cpu_count: int

    def __init__(self, p: RunParams):
        self.id = p.id
        self.target_duration = p.target_duration
        self.target_rps = p.target_rps
        self.cpu_count = p.cpu_count


@dataclass
class SimulationResult:
    client: ClientResultList
    server: ServerResultList
    params: MinimalParams  # use minimal params for fast pickling


@dataclass
class MultiSimulationResult:
    results: List[SimulationResult]


class Simulator:
    env = simpy.Environment()

    def _run_parametrized(self, params: RunParams, result_collector: List[SimulationResult]):
        inflight_semaphore: Optional[simpy.Resource] = None
        if params.max_inflight_requests is not None:
            assert params.static_clients_count == 0  # not supported yet
            inflight_semaphore = simpy.Resource(self.env, capacity=params.max_inflight_requests)

        clients_count: int = 0
        client_target_rps: float = 0

        if params.static_clients_count == 0:
            new_clients_per_second = params.target_rps
            run_duration_seconds = params.target_duration / TIME_DURATION_SECOND
            clients_count = int(new_clients_per_second * run_duration_seconds)
            if params.load_spike is not None:
                spike_duration_seconds = params.load_spike.interval.length() / TIME_DURATION_SECOND
                clients_count += int(new_clients_per_second * spike_duration_seconds * (params.load_spike.load_multiplier - 1))
        else:
            assert params.load_spike is None  # not supported
            clients_count = params.static_clients_count
            client_target_rps = params.target_rps / clients_count

        print(f"Running simulation id={params.id} with {clients_count} clients and inflight limit {params.max_inflight_requests}...")

        # It's important to use generator here for lazy loading and performance when inflight limit is set.
        clients = [
            Client(
                env=self.env,
                id=f"{params.id}/{i}",
                retry_strategy=(params.retry_strategy if params.use_global_client_state else params.retry_strategy.copy()),
                timeout=params.client_timeout_builder(),
                target_rps=client_target_rps,
                run_until=params.target_duration,
                circuit_breaker=params.circuit_breaker,
            )
            for i in range(clients_count)
        ]

        for c in clients:
            self.env.process(self._call_client(c, params.server, inflight_semaphore, params.target_duration))  # no yield to be async

            if params.static_clients_count != 0:
                continue

            if params.load_spike is not None and params.load_spike.interval.is_in(TimePoint(self.env.now)):
                sleep_time = random.expovariate(new_clients_per_second * params.load_spike.load_multiplier)
                yield self.env.timeout(TimeDuration(TIME_DURATION_SECOND * sleep_time))
            else:
                # Use poisson distribution to simulate clients arrival.
                sleep_time = random.expovariate(new_clients_per_second)
                yield self.env.timeout(TimeDuration(TIME_DURATION_SECOND * sleep_time))

        client_results: ClientResultList = []
        for c in clients:
            results = yield c.results.get()
            client_results += results
        res = SimulationResult(client=client_results, server=params.server.events, params=MinimalParams(params))
        result_collector.append(res)

    def _call_client(self, client: Client, server: Server, inflight_semaphore: Optional[simpy.Resource], client_deadline: TimePoint):
        if inflight_semaphore is not None:
            with inflight_semaphore.request() as req:
                yield req
                yield from client.run(self.env, server)
        else:
            yield from client.run(self.env, server)

    def run_multiple_simulations(self, params: List[RunParams]) -> MultiSimulationResult:
        started_at = time.time()
        results: List[SimulationResult] = []
        for p in params:
            self.env.process(self._run_parametrized(p, results))
        self.env.run()
        assert len(results) == len(params), [len(results), len(params)]

        simulation_params = f"id={','.join([p.id for p in params])}/rps={params[0].target_rps}/cpu={params[0].cpu_count}/duration={params[0].target_duration/TIME_DURATION_SECOND}s"
        print(f"Ran simulation {simulation_params} in {time.time() - started_at}s")
        return MultiSimulationResult(results)
