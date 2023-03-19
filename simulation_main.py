from __future__ import annotations
import copy
from circuit_breaker import CircuitBreaker
from retry_strategies import (
    FixedRetryCountStrategy,
    FixedRetryCountWithExpBackoffStrategy,
    RetryBudgetStrategy,
    RetryCircuitBreakerStrategy,
    RetryStrategy,
    ServerCallContext,
)
import random
import time
import pickle
from typing import Callable, Optional, Tuple, List, Union, cast
from server import CpuBoundServer, InjectedFailure
from shared import MEAN_RTT_MS, TIME_DURATION_MS, TIME_DURATION_SECOND, TimeInterval, TimeDuration, TimePoint
import argparse
from multiprocessing import Pool
import numpy as np

from simulator import MultiSimulationResult, RunParams, SimulationResult, Simulator
from visualize_params import *


def run_one_process(params: RunParams) -> MultiSimulationResult:
    sim = Simulator()
    return sim.run_multiple_simulations([params])


def run_simulations(args: argparse.Namespace, params: List[RunParams]) -> MultiSimulationResult:
    if args.load_from:
        started_at = time.time()
        print("Loading simulation result...")
        with open(args.load_from, "rb") as f:
            res: MultiSimulationResult = pickle.load(f)
            print(f"Loaded simulation result in {time.time() - started_at}s")
            return res

    mp_pool_size = min(6, len(params))
    print(f"Running simulation in {mp_pool_size} processes")
    started_at = time.time()
    total_res: MultiSimulationResult
    with Pool() as mp_pool:
        subresults = mp_pool.map(run_one_process, params)
        flatten_results: List[SimulationResult] = sum([sr.results for sr in subresults], [])
        print(f"got {len(subresults)} subresults: {[res.params.id for res in flatten_results]}")
        total_res = MultiSimulationResult(flatten_results)
    print(f"Ran all simulations {[res.params.id for res in total_res.results]} in {time.time() - started_at}s")
    if args.save_to:
        ran_at = time.time()
        with open(args.save_to, "wb") as f:
            pickle.dump(total_res, f)
        print(f"Saved results of simulations in {time.time() - ran_at}s")
    return total_res


def visualize(res: MultiSimulationResult, vis_params: VisualizeParams):
    if not vis_params.out_image_path:
        print("Dont need to visualize")
        return

    started_at = time.time()
    print("Simulation ended, visualizing it ...")
    from visualize import visualize_stats_by_time

    visualize_stats_by_time(res, vis_params)
    simulation_perf_params = f"rps={res.results[0].params.target_rps}/cpu={res.results[0].params.cpu_count}/duration={res.results[0].params.target_duration/TIME_DURATION_SECOND}s"
    print(f"Visualized simulation {simulation_perf_params} in {time.time() - started_at}s")


SIMPLE_HANDLER_BASE_DURATION = 10 * TIME_DURATION_MS
SIMPLE_HANDLER_MU, SIMPLE_HANDLER_SIGMA = 0, 0.25

# https://en.wikipedia.org/wiki/Log-normal_distribution#Arithmetic_moments
SIMPLE_HANLDER_MEAN_DURATION_FLOAT = SIMPLE_HANDLER_BASE_DURATION * np.exp(
    SIMPLE_HANDLER_MU + SIMPLE_HANDLER_SIGMA * SIMPLE_HANDLER_SIGMA / 2
)


def simple_handler_duration_builder() -> TimeDuration:
    return TimeDuration(SIMPLE_HANDLER_BASE_DURATION * np.random.lognormal(SIMPLE_HANDLER_MU, SIMPLE_HANDLER_SIGMA))


CLIENT_TIMEOUT = 100 * TIME_DURATION_MS


def default_client_timeout_builder() -> TimeDuration:
    return CLIENT_TIMEOUT


MAX_SLEEP_TIME = CLIENT_TIMEOUT * 10
SLEEP_BASE = 100 * TIME_DURATION_MS
INFLIGHT_LIMIT_AS_RPS_RATIO = 0.1

ParamsTransformerRet = Union[List[RunParams], RunParams, None]
ParamsTransformer = Callable[[RunParams], ParamsTransformerRet]

exp_backoff_with_jitter_2x = FixedRetryCountWithExpBackoffStrategy(
    retry_count=2, exp_base=2, sleep_base=SLEEP_BASE, max_sleep_time=MAX_SLEEP_TIME, use_jitter=True
)
fixed_retry_2x = FixedRetryCountStrategy(2)

all_retry_strategies = [
    ("retry-3x", FixedRetryCountStrategy(3), 0),
    ("retry-3x-with-delay", FixedRetryCountStrategy(3, fixed_sleep_time=50 * TIME_DURATION_MS), 1),
    (
        "retry-3x-with-exp-backoff",
        FixedRetryCountWithExpBackoffStrategy(retry_count=3, exp_base=2, sleep_base=SLEEP_BASE, max_sleep_time=MAX_SLEEP_TIME),
        2,
    ),
    (
        "retry-3x-with-exp-backoff+jitter",
        FixedRetryCountWithExpBackoffStrategy(
            retry_count=3, exp_base=2, sleep_base=SLEEP_BASE, max_sleep_time=MAX_SLEEP_TIME, use_jitter=True
        ),
        3,
    ),
    ("retry-2x", fixed_retry_2x, 0),
    ("retry-2x-with-exp-backoff+jitter", exp_backoff_with_jitter_2x, 3),
    ("no-retries", FixedRetryCountStrategy(0), 4),
    (
        "retry-cb-10%+retry-2x",
        RetryCircuitBreakerStrategy(max_failures_rate=0.1, underlying_strategy=fixed_retry_2x, window_duration=200 * TIME_DURATION_MS),
        5,
    ),
    ("retry-budget-10%+retry-2x", RetryBudgetStrategy(retry_budget_ratio=0.1, underlying_strategy=fixed_retry_2x), 6),
    (
        "retry-cb-10%+exp-backoff-2x",
        RetryCircuitBreakerStrategy(
            max_failures_rate=0.1, underlying_strategy=exp_backoff_with_jitter_2x, window_duration=200 * TIME_DURATION_MS
        ),
        7,
    ),
    ("retry-budget-10%+exp-backoff-2x", RetryBudgetStrategy(retry_budget_ratio=0.1, underlying_strategy=exp_backoff_with_jitter_2x), 8),
]


def inject_full_downtime(from_sec: float, to_sec: float) -> List[InjectedFailure]:
    return [
        InjectedFailure(
            interval=TimeInterval(begin=TimePoint(from_sec * TIME_DURATION_SECOND), end=TimePoint(to_sec * TIME_DURATION_SECOND)),
            failure_prob=1.0,
        )
    ]


simple_injected_failures = inject_full_downtime(0.5, 1.0)


def build_input(
    args: argparse.Namespace,
    cpu_count_coef: float = 1,
    target_uptime: float = 0.995,
    params_transformer: Optional[ParamsTransformer] = None,
    injected_failures: List[InjectedFailure] = simple_injected_failures,
    client_timeout_builder: Callable[[], TimeDuration] = default_client_timeout_builder,
    duration: TimeDuration = TimeDuration(1.5 * TIME_DURATION_SECOND),
    vis_params_override: Dict = {},
    params_override: Dict = {},
    server_params_override: Dict = {},
) -> Tuple[List[RunParams], VisualizeParams]:
    strategies = all_retry_strategies
    cpu_count = int(4 * cpu_count_coef * args.target_rps * SIMPLE_HANLDER_MEAN_DURATION_FLOAT / TIME_DURATION_SECOND)

    id_to_color_num = {id: color_num for id, _, color_num in strategies}
    all_params: List[RunParams] = []
    for id, strategy, color_num in strategies:
        server = CpuBoundServer(
            cpu_count=cpu_count,
            handle_duration_builder=simple_handler_duration_builder,
            injected_failures=injected_failures,
            target_uptime=target_uptime,
            **server_params_override,
        )
        params_list = [
            RunParams(
                id=id,
                retry_strategy=copy.deepcopy(strategy),  # strategies can be stateful, so don't share state between runs
                server=server,
                load_spike=None,
                target_duration=duration,
                target_rps=args.target_rps,
                cpu_count=cpu_count,
                client_timeout_builder=client_timeout_builder,
                **params_override,
            )
        ]
        if params_transformer is not None:
            transform_res = params_transformer(params_list[0])
            if not transform_res:
                continue
            new_params_list = [transform_res] if isinstance(transform_res, RunParams) else transform_res
            id_to_color_num[new_params_list[0].id] = color_num
            params_list = new_params_list
        all_params += params_list

    simple_vis_params_kwargs = dict(
        needed_panels=[
            PANEL_SERVER_UPTIME,
            PANEL_SERVER_RPS_AMPLIFICATION,
            PANEL_CLIENT_LATENCY_P50,
            PANEL_SERVER_AVG_CPU_USAGE,
        ],
        id_to_color_num=id_to_color_num,
        row_n=0,
        col_n=0,
        shareX=False,
        out_image_path=args.render_to,
        highlight_interval=simple_injected_failures[0].interval,
    )
    for k, v in vis_params_override.items():
        simple_vis_params_kwargs[k] = v

    return all_params, VisualizeParams(**simple_vis_params_kwargs)


def different_client_timeouts_builder() -> TimeDuration:
    return random.choice([100, 200, 300]) * TIME_DURATION_MS


def remove_suffix(s: str, suffix: str) -> str:
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", help="load simulation result from the file")
    parser.add_argument("--save-to", help="save simulation result to the file")
    parser.add_argument("--target-rps", help="target rps affects data flakiness and simulation speed", default=1000, type=int)
    parser.add_argument("--render-to", help="need visualization or not", type=str)
    parser.add_argument("--simulation-name", help="name of specific simulation", type=str)
    args = parser.parse_args()

    random.seed(1)

    if args.simulation_name == "simple_retry_only":
        params, vis_params = build_input(
            args,
            params_transformer=lambda p: p if p.id == "retry-3x" else None,
            vis_params_override=dict(needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION]),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "simple_retry_vs_backoff":
        params, vis_params = build_input(
            args,
            params_transformer=lambda p: p if p.id in ["retry-3x", "retry-3x-with-exp-backoff"] else None,
            vis_params_override=dict(needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION]),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "simple_retry_vs_backoff_closed_loop":
        params, vis_params = build_input(
            args,
            duration=TimeDuration(1.75 * TIME_DURATION_SECOND),
            params_transformer=lambda p: p if p.id in ["retry-3x", "retry-3x-with-exp-backoff"] else None,
            vis_params_override=dict(needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION]),
            params_override=dict(max_inflight_requests=int(args.target_rps * INFLIGHT_LIMIT_AS_RPS_RATIO)),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "jitter_vs_no_jitter":
        params, vis_params = build_input(
            args,
            duration=TimeDuration(1.75 * TIME_DURATION_SECOND),
            params_transformer=lambda p: p if p.id.startswith("retry-3x-with-exp-backoff") else None,
            vis_params_override=dict(needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION]),
            params_override=dict(max_inflight_requests=int(args.target_rps * INFLIGHT_LIMIT_AS_RPS_RATIO)),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "jitter_vs_no_jitter_cpu_half":
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.5,
            duration=TimeDuration(2.2 * TIME_DURATION_SECOND),
            params_transformer=lambda p: p if p.id in ["retry-3x-with-exp-backoff", "retry-3x-with-exp-backoff+jitter"] else None,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_RPS_AMPLIFICATION, PANEL_CLIENT_LATENCY_P50],
            ),
            params_override=dict(max_inflight_requests=int(args.target_rps * INFLIGHT_LIMIT_AS_RPS_RATIO)),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "mfs":
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.5,
            params_transformer=lambda p: p if p.id == "retry-3x" else None,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION, PANEL_SERVER_AVG_REQ_QUEUE, PANEL_SERVER_AVG_CPU_USAGE]
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "exponential_backoff_just_delays_problem_open_loop":
        failures = inject_full_downtime(0.5, 1.5)
        params, vis_params = build_input(
            args,
            duration=TimeDuration(2.0 * TIME_DURATION_SECOND),
            injected_failures=failures,
            params_transformer=lambda p: p if p.id in ["retry-3x", "retry-3x-with-exp-backoff+jitter"] else None,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                row_n=1,
                highlight_interval=failures[0].interval,
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "exponential_backoff_just_delays_problem_good_closed_loop":
        failures = inject_full_downtime(0.5, 1.5)
        params, vis_params = build_input(
            args,
            duration=TimeDuration(2.0 * TIME_DURATION_SECOND),
            injected_failures=failures,
            params_transformer=lambda p: p if p.id in ["retry-3x", "retry-3x-with-exp-backoff+jitter"] else None,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                row_n=1,
                highlight_interval=failures[0].interval,
            ),
            params_override=dict(max_inflight_requests=int(args.target_rps * 0.3)),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "exponential_backoff_just_delays_problem_bad_closed_loop":
        failures = inject_full_downtime(0.5, 1.5)
        params, vis_params = build_input(
            args,
            duration=TimeDuration(2.0 * TIME_DURATION_SECOND),
            injected_failures=failures,
            params_transformer=lambda p: p if p.id in ["retry-3x", "retry-3x-with-exp-backoff+jitter"] else None,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                row_n=1,
                highlight_interval=failures[0].interval,
            ),
            params_override=dict(max_inflight_requests=int(args.target_rps * 0.4)),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "any_retries_are_slowing_down_recovery":
        failures = inject_full_downtime(0.5, 1.5)
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.54,
            client_timeout_builder=different_client_timeouts_builder,
            injected_failures=failures,
            duration=TimeDuration(2.5 * TIME_DURATION_SECOND),
            params_transformer=lambda p: p if p.id in ["no-retries", "retry-2x", "retry-2x-with-exp-backoff+jitter"] else None,
            vis_params_override=dict(
                needed_panels=[
                    PANEL_SERVER_UPTIME,
                    PANEL_SERVER_RPS_AMPLIFICATION,
                    PANEL_CLIENT_UPTIME,
                    PANEL_SERVER_AVG_REQ_QUEUE,
                    # PANEL_SERVER_AVG_CPU_USAGE,
                    # PANEL_CLIENT_LATENCY_P50,
                ],
                row_n=2,
                highlight_interval=failures[0].interval,
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "retry_cb_or_budget_vs_exp_backoff":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id in [
                "no-retries",
                "retry-2x",
                "retry-2x-with-exp-backoff+jitter",
                "retry-cb-10%+retry-2x",
                "retry-budget-10%+retry-2x",
            ]:
                p.id = remove_suffix(p.id, "+retry-2x")
                return p
            else:
                return None

        params, vis_params = build_input(
            args,
            cpu_count_coef=0.54,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[PANEL_SERVER_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "retry_cb_or_budget_vs_exp_backoff_partial_failure":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id in [
                "no-retries",
                "retry-2x",
                "retry-2x-with-exp-backoff+jitter",
                "retry-cb-10%+retry-2x",
                "retry-budget-10%+retry-2x",
            ]:
                p.id = remove_suffix(p.id, "+retry-2x")
                return p
            else:
                return None

        failures = [
            InjectedFailure(
                interval=TimeInterval(begin=TimePoint(0.5 * TIME_DURATION_SECOND), end=TimePoint(1.0 * TIME_DURATION_SECOND)),
                failure_prob=0.3,
            )
        ]
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.54,
            injected_failures=failures,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[PANEL_CLIENT_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                highlight_interval=failures[0].interval,
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "retry_cb_vs_budget":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id in [
                "retry-cb-10%+retry-2x",
                "retry-budget-10%+retry-2x",
            ]:
                p.id = remove_suffix(p.id, "+retry-2x")
                return p
            else:
                return None

        failures = [
            InjectedFailure(
                interval=TimeInterval(begin=TimePoint(0.5 * TIME_DURATION_SECOND), end=TimePoint(1.0 * TIME_DURATION_SECOND)),
                failure_prob=0.3,
            )
        ]
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.54,
            injected_failures=failures,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[PANEL_CLIENT_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                highlight_interval=failures[0].interval,
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "retry_budget_over_what":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id in [
                "retry-budget-10%+retry-2x",
                "retry-budget-10%+exp-backoff-2x",
            ]:
                return p
            else:
                return None

        failures = [
            InjectedFailure(
                interval=TimeInterval(begin=TimePoint(0.5 * TIME_DURATION_SECOND), end=TimePoint(1.0 * TIME_DURATION_SECOND)),
                failure_prob=1.0,
            )
        ]
        params, vis_params = build_input(
            args,
            cpu_count_coef=0.54,
            injected_failures=failures,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[PANEL_CLIENT_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                highlight_interval=failures[0].interval,
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "retry_cb_budget_global_vs_local":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id in [
                "retry-cb-10%+retry-2x",
                "retry-budget-10%+retry-2x",
            ]:
                p.id = remove_suffix(p.id, "+retry-2x")
                p.static_clients_count = p.target_rps // 200

                p_copy = copy.copy(p)
                p_copy.id += "-local"
                p_copy.use_global_client_state = False
                p.id += "-global"
                return [p, p_copy]
            else:
                return None

        failures = [
            InjectedFailure(
                interval=TimeInterval(begin=TimePoint(0.5 * TIME_DURATION_SECOND), end=TimePoint(1.0 * TIME_DURATION_SECOND)),
                failure_prob=1.0,
            )
        ]
        params, vis_params = build_input(
            args,
            # cpu_count_coef=0.54,
            injected_failures=failures,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[PANEL_CLIENT_UPTIME, PANEL_SERVER_RPS_AMPLIFICATION],
                highlight_interval=failures[0].interval,
            ),
        )

        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name.startswith("circuit_breaker_vs_retry_budget/"):
        # circuit_breaker_vs_retry_budget/cb_threshold=xx%/fail_rate=yy%

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id == "retry-budget-10%+exp-backoff-2x":
                p.id = remove_suffix(p.id, "+exp-backoff-2x")
                return p
            if p.id == "retry-2x":
                p_cb = copy.deepcopy(p)
                cb_threshold_pct = int(args.simulation_name[len("circuit_breaker_vs_retry_budget/cb_threshold=") :][:2])
                p_cb.id += "-circuit-breaker-" + str(cb_threshold_pct) + "%"
                p_cb.circuit_breaker = CircuitBreaker(failure_rate_threshold=cb_threshold_pct / 100)
                return [p, p_cb]
            else:
                return None

        failure_pct = int(args.simulation_name[len("circuit_breaker_vs_retry_budget/cb_threshold=00%/fail_rate=") :][:2])
        failures = [
            InjectedFailure(
                interval=TimeInterval(begin=TimePoint(0.5 * TIME_DURATION_SECOND), end=TimePoint(1.0 * TIME_DURATION_SECOND)),
                failure_prob=failure_pct / 100,
                use_failure_prob_per_user=True,
            )
        ]

        params, vis_params = build_input(
            args,
            cpu_count_coef=0.5,
            params_transformer=transform,
            injected_failures=failures,
            vis_params_override=dict(
                needed_panels=[
                    PANEL_SERVER_UPTIME,
                    PANEL_SERVER_RPS_AMPLIFICATION,
                    PANEL_CLIENT_UPTIME,
                    PANEL_SERVER_AVG_CPU_USAGE,
                ]
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    elif args.simulation_name == "dp_vs_retry_budget":

        def transform(p: RunParams) -> ParamsTransformerRet:
            if p.id == "retry-budget-10%+exp-backoff-2x":
                p.id = remove_suffix(p.id, "+exp-backoff-2x")
                return p
            if p.id == "retry-2x":
                p_dp = copy.deepcopy(p)
                p_dp.id += "-deadline-propagation"
                p_dp.server.enable_deadline_propagation = True
                return [p, p_dp]
            else:
                return None

        params, vis_params = build_input(
            args,
            cpu_count_coef=0.58,
            params_transformer=transform,
            vis_params_override=dict(
                needed_panels=[
                    PANEL_SERVER_UPTIME,
                    PANEL_SERVER_RPS_AMPLIFICATION,
                    PANEL_SERVER_AVG_REQ_QUEUE,
                    PANEL_SERVER_AVG_CPU_USAGE,
                ]
            ),
        )
        res = run_simulations(args, params)
        visualize(res, vis_params)
    else:
        raise Exception(f"Unknown simulation name: {args.simulation_name}")
