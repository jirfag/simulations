from retry_strategies import FixedRetryCountStrategy, FixedRetryCountWithExpBackoffStrategy, RetryCircuitBreakerStrategy, RetryStrategy, ServerCallContext
import simpy
import random
from typing import Tuple, Optional, List, Dict

from visualize import CallResult, CallResultList, visualize_stats


def net_rtt():
    # mean RTT is 0.01s = 10ms
    return random.expovariate(1 / 0.01)

class Server:
    def __init__(self, failure_prob: float):
        self.failure_prob = failure_prob

    def handle(self, env: simpy.Environment, connection: simpy.Store):
        yield env.timeout(random.uniform(0, 0.04))
        connection.put(random.random() >= self.failure_prob)  # send response (OK/FAIL) to connection

class Client:
    def __init__(self, env: simpy.Environment, id: str, retry_strategy: RetryStrategy, server: Server):
        self.retry_strategy = retry_strategy
        self.server = server
        self.id = id
        self.result = simpy.Store(env)
    
    def call(self, env: simpy.Environment):
        started_at = env.now
        yield env.timeout(net_rtt())
        call_ctx = ServerCallContext(0)
        connection = simpy.Store(env)
        while True:
            yield from self.server.handle(env, connection)
            call_ctx.attempted_calls += 1
            is_ok = yield connection.get()  # get response from server
            self.retry_strategy.add_call_to_stats(is_ok)
            if is_ok:
                print(f"{env.now}: call from client {self.id} to server succeeded from #{call_ctx.attempted_calls} attempt and in {env.now - started_at}s")
                self.result.put(CallResult(env.now, True, call_ctx.attempted_calls))
                break

            need_retry, delay = self.retry_strategy.need_retry(call_ctx)
            if not need_retry:
                print(f"{env.now}: call from client {self.id} to server finally failed after all attempts")
                self.result.put(CallResult(env.now, False, call_ctx.attempted_calls))
                return
            print(f"{env.now}: call from client {self.id} to server failed, sleep {delay} and retry")
            yield env.timeout(delay)


class Simulator:
    env = simpy.Environment()
    call_stats: Dict[str, CallResultList] = {}

    def run_parametrized(self, id: str, clients_count: int, retry_strategy: RetryStrategy, server_failure_prob: float, target_rps: int):
        server = Server(server_failure_prob)
        clients = [Client(self.env, f"{id}/{i}", retry_strategy, server) for i in range(clients_count)]

        for c in clients:
            self.env.process(c.call(self.env))  # no yield to be async
            yield self.env.timeout(1 / target_rps)

        call_results: CallResultList = []
        for c in clients:
            res = yield c.result.get()
            call_results.append(res)
        self.call_stats[id] = call_results

    def run(self):
        target_rps = 100
        clients_count = target_rps * 15
        server_failure_prob = 0.3
        params = {
            'clients_count': clients_count,
            'server_failure_prob': server_failure_prob,
            'target_rps': target_rps,
        }
        self.env.process(self.run_parametrized('fixed-1x', retry_strategy=FixedRetryCountStrategy(1), **params))
        self.env.process(self.run_parametrized('fixed-2x', retry_strategy=FixedRetryCountStrategy(2), **params))
        # self.env.process(self.run_parametrized('fixed-3x', retry_strategy=FixedRetryCountStrategy(3), **params))
        self.env.process(self.run_parametrized('exp-backoff-1x', retry_strategy=FixedRetryCountWithExpBackoffStrategy(retry_count=1, exp_base=2, max_sleep_time=3), **params))
        self.env.process(self.run_parametrized('exp-backoff-2x', retry_strategy=FixedRetryCountWithExpBackoffStrategy(retry_count=2, exp_base=2, max_sleep_time=3), **params))
        # self.env.process(self.run_parametrized('exp-backoff-3x', retry_strategy=FixedRetryCountWithExpBackoffStrategy(retry_count=3, exp_base=2, max_sleep_time=3), **params))
        self.env.process(self.run_parametrized('circuit-breaker-0.1', retry_strategy=RetryCircuitBreakerStrategy(0.1), **params))
        # self.env.process(self.run_parametrized('circuit-breaker-0.2', retry_strategy=RetryCircuitBreakerStrategy(0.2), **params))
        self.env.run()

        visualize_stats(self.call_stats, int(clients_count / target_rps), target_rps)


if __name__ == '__main__':
    random.seed(42)
    sim = Simulator()
    sim.run()