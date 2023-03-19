import random
from typing import List, Optional

import simpy
from circuit_breaker import CircuitBreaker
from events import ClientResult
from retry_strategies import RetryStrategy, ServerCallContext
from server import Request, Server
from shared import MEAN_RTT_MS, TIME_DURATION_MS, TIME_DURATION_SECOND, TimeDuration, TimePoint


def net_rtt() -> TimeDuration:
    return TimeDuration(TIME_DURATION_MS * random.expovariate(1 / MEAN_RTT_MS))


class Client:
    def __init__(
        self,
        env: simpy.Environment,
        id: str,
        retry_strategy: RetryStrategy,
        timeout: TimeDuration,
        target_rps: float = 0,
        run_until: TimeDuration = 0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.retry_strategy = retry_strategy
        self.id = id
        self.results = simpy.Store(env)
        self.timeout = timeout
        self.target_rps = target_rps
        self.run_until = run_until
        self.circuit_breaker = circuit_breaker

    def run(self, env: simpy.Environment, server: Server):
        if self.target_rps == 0:  # one call per client
            res_store = simpy.Store(env)
            yield from self._call_once(env, server, res_store)
            res = yield res_store.get()
            yield self.results.put([res])
            return

        # Multiple calls per client.
        result_futures: List[simpy.Store] = []
        assert self.run_until > 0
        while env.now < self.run_until:
            result_futures.append(simpy.Store(env))

            # Use env.process to simulate open loop. If closed loop needed then need to use semaphore.
            env.process(self._call_once(env, server, result_futures[-1]))

            if env.now >= self.run_until:
                break

            # Use poisson distribution to simulate new requests arrival.
            sleep_time = TIME_DURATION_SECOND * random.expovariate(self.target_rps)
            yield env.timeout(TimeDuration(sleep_time))

        results: List[ClientResult] = []
        for fut in result_futures:
            res = yield fut.get()
            results.append(res)
        yield self.results.put(results)

    def _call_once(self, env: simpy.Environment, server: Server, call_result: simpy.Store):
        started_at = TimePoint(env.now)
        call_ctx = ServerCallContext(0)

        if self.circuit_breaker and not self.circuit_breaker.is_request_allowed(env):
            yield call_result.put(ClientResult(timepoint=started_at, is_ok=False, duration=TimeDuration(0)))
            return

        user_id = random.randint(0, 10**9)

        def call_server(connection: simpy.Store):
            rtt = net_rtt()

            # Wait full RTT before handling (not rtt/2 before and rtt/2 after) to simplify code.
            yield env.timeout(rtt)
            yield from server.handle(env, connection, Request(user_id=user_id, deadline=started_at + self.timeout))

        while True:
            connection = simpy.Store(env)
            env.process(call_server(connection))
            call_ctx.attempted_calls += 1
            response_future = connection.get()
            res = yield (response_future | env.timeout(self.timeout))
            is_ok = response_future in res and response_future.value == True
            self.retry_strategy.add_call_to_stats(TimePoint(env.now), is_ok, call_ctx.attempted_calls != 1)
            if self.circuit_breaker:
                self.circuit_breaker.record_request(TimePoint(env.now), is_ok)
            duration = TimeDuration(env.now - started_at)

            # It may be env.now, but it makes charts more flaky.
            client_res_tp = TimePoint(env.now)
            if is_ok:
                yield call_result.put(ClientResult(timepoint=client_res_tp, is_ok=True, duration=duration))
                return

            need_retry, delay = self.retry_strategy.need_retry(call_ctx)
            if not need_retry:
                yield call_result.put(ClientResult(timepoint=client_res_tp, is_ok=False, duration=duration))
                return
            yield env.timeout(delay)  # used e.g. for exponential backoff retry strategy
