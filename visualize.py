from dataclasses import dataclass
from math import ceil
import pandas as pd
from typing import List, Tuple
import time
from events import ClientResultList, ServerResultList

from shared import (
    TIME_DURATION_MS,
    TimeDuration,
    TimeInterval,
    TimePoint,
    TIME_DURATION_SECOND,
)
from simulator import MultiSimulationResult, SimulationResult
from visualize_params import *


def p50_latency(arr: List[float]) -> float:
    assert len(arr) != 0
    sorted_arr = sorted(arr)
    return sorted_arr[int(len(sorted_arr) / 2)]


def p99_latency(arr: List[float]) -> float:
    assert len(arr) != 0
    sorted_arr = sorted(arr)
    return sorted_arr[int(len(sorted_arr) * 0.99)]


def avg_latency(arr: List[float]) -> float:
    assert len(arr) != 0
    return sum(arr) / len(arr)


CHART_POINTS_PER_SECOND = pow(10, 1) * 5


def get_chart_point_count(duration: TimeDuration) -> int:
    return ceil(duration * CHART_POINTS_PER_SECOND / TIME_DURATION_SECOND)


def timepoint_to_chart_point(relative_tp: TimePoint) -> int:
    return (relative_tp * CHART_POINTS_PER_SECOND) // TIME_DURATION_SECOND


def get_chart_point_time_interval(
    chart_point_num: int, from_time: TimePoint
) -> TimeInterval:
    return TimeInterval(
        begin=from_time
        + TimePoint(chart_point_num * TIME_DURATION_SECOND / CHART_POINTS_PER_SECOND),
        end=from_time
        + TimePoint(
            (chart_point_num + 1) * TIME_DURATION_SECOND / CHART_POINTS_PER_SECOND - 1
        ),
    )


def get_chart_point_time_length() -> TimeDuration:
    return TimeDuration(TIME_DURATION_SECOND / CHART_POINTS_PER_SECOND)


def fill_missing_values(values: List[float]):
    last_value = None
    for i in range(len(values)):
        if values[i] is None:
            values[i] = last_value
        else:
            last_value = values[i]


# calc_client_metrics calcs client stats for every point in interval [from_time; to_time).
def calc_client_metrics(
    client_results: ClientResultList, from_time: TimePoint, to_time: TimePoint
):
    point_count = get_chart_point_count(to_time - from_time)
    total_counts, succeeded_counts = [0] * point_count, [0] * point_count
    all_latencies: List[List[float]] = [[] for _ in range(point_count)]
    for cr in client_results:
        if cr.timepoint >= to_time or cr.timepoint < from_time:
            continue
        chart_point = timepoint_to_chart_point(cr.timepoint - from_time)
        total_counts[chart_point] += 1
        if cr.is_ok:
            succeeded_counts[chart_point] += 1
        all_latencies[chart_point].append(cr.duration)
    uptimes = [
        100 * succeeded_counts[i] / total_counts[i] if total_counts[i] != 0 else None
        for i in range(point_count)
    ]
    p50_latencies = [
        (
            p50_latency(all_latencies[i]) / TIME_DURATION_MS
            if len(all_latencies[i]) != 0
            else None
        )
        for i in range(point_count)
    ]
    p99_latencies = [
        (
            p99_latency(all_latencies[i]) / TIME_DURATION_MS
            if len(all_latencies[i]) != 0
            else None
        )
        for i in range(point_count)
    ]
    avg_latencies = [
        (
            avg_latency(all_latencies[i]) / TIME_DURATION_MS
            if len(all_latencies[i]) != 0
            else None
        )
        for i in range(point_count)
    ]
    return uptimes, p50_latencies, p99_latencies, avg_latencies


def calc_server_cpu_usages(
    server_results: ServerResultList,
    from_time: TimePoint,
    to_time: TimePoint,
    cpu_count: int,
):
    point_count = get_chart_point_count(to_time - from_time)
    cpu_usages_sum = [0] * point_count

    for sr in server_results:
        if (
            sr.timepoint < from_time
        ):  # don't check to_time because sr.timepoint is request end time and it can overlap
            continue

        req_started_at, req_finished_at = (
            sr.timepoint - sr.handle_duration,
            sr.timepoint,
        )
        begin_cp = timepoint_to_chart_point(req_started_at - from_time)
        end_cp = timepoint_to_chart_point(req_finished_at - from_time)
        sr_interval = TimeInterval(req_started_at, req_finished_at)
        for cp in range(begin_cp, min(end_cp + 1, point_count)):
            cp_interval = get_chart_point_time_interval(cp, from_time)
            intersect_interval = sr_interval.intersect(cp_interval)
            assert intersect_interval is not None
            cpu_usages_sum[cp] += intersect_interval.length()

    chart_point_time_len = get_chart_point_time_length()
    return [
        100 * cpu_usages_sum[i] / chart_point_time_len / cpu_count
        for i in range(point_count)
    ]


# calc_server_metrics calcs server stats for every point in interval [from_time; to_time).
def calc_server_metrics(
    server_results: ServerResultList,
    from_time: TimePoint,
    to_time: TimePoint,
    target_rps: int,
    cpu_count: int,
):
    point_count = get_chart_point_count(to_time - from_time)
    total_counts, succeeded_counts, pending_reqs, max_pending_reqs, latency_sums = (
        [0] * point_count,
        [0] * point_count,
        [0] * point_count,
        [0] * point_count,
        [0] * point_count,
    )
    for sr in server_results:
        if sr.timepoint >= to_time or sr.timepoint < from_time:
            continue
        p = timepoint_to_chart_point(sr.timepoint - from_time)
        total_counts[p] += 1
        if sr.is_ok:
            succeeded_counts[p] += 1
        pending_reqs[p] += sr.pending_req_count
        max_pending_reqs[p] = max(max_pending_reqs[p], sr.pending_req_count)
        latency_sums[p] += sr.handle_duration
    rps_amplifications = [
        100 * tc * CHART_POINTS_PER_SECOND / target_rps for tc in total_counts
    ]
    uptimes = [
        100 * succeeded_counts[i] / total_counts[i] if total_counts[i] != 0 else None
        for i in range(point_count)
    ]
    avg_queue_sizes = [
        pending_reqs[i] / total_counts[i] if total_counts[i] != 0 else None
        for i in range(point_count)
    ]
    avg_cpu_usages = calc_server_cpu_usages(
        server_results, from_time, to_time, cpu_count
    )
    avg_latencies = [
        (
            latency_sums[i] / total_counts[i] / TIME_DURATION_MS
            if total_counts[i] != 0
            else None
        )
        for i in range(point_count)
    ]
    return (
        rps_amplifications,
        uptimes,
        avg_queue_sizes,
        max_pending_reqs,
        avg_cpu_usages,
        avg_latencies,
    )


X_AXIS_TIME = localize("Time (s)", "время, с")
LINE_VARIABLE = localize("Retry Strategy", "стратегия ретраев")
Y_AXIS_CLIENT_UPTIME = localize("Client Uptime (%)", "аптайм клиентов, %")
Y_AXIS_SERVER_RPS_AMPLIFICATION = localize(
    "Relative Server RPS (%)", "относительный серверный rps, %"
)
Y_AXIS_CLIENT_LATENCY_P50 = localize(
    "Client Latency p50 (ms)", "p50 клиентские тайминги, мс"
)
Y_AXIS_CLIENT_LATENCY_P99 = localize(
    "Client Latency p99 (ms)", "p99 клиентские тайминги, мс"
)
Y_AXIS_CLIENT_LATENCY_AVG = localize(
    "Client Latency Avg (ms)", "средние клиентские тайминги, мс"
)
Y_AXIS_SERVER_FAILURE_RATE = localize(
    "Server Failure Rate (%)", "доля ошибок сервера, %"
)
Y_AXIS_AVG_PENDING_REQ_COUNT = localize(
    "Avg Server Queue Size", "средний размер очереди запросов сервера"
)
Y_AXIS_MAX_PENDING_REQ_COUNT = localize(
    "Max Server Queue Size", "макс размер очереди запросов сервера"
)
Y_AXIS_SERVER_UPTIME = localize("Server Uptime (%)", "аптайм сервера, %")
Y_AXIS_SERVER_AVG_CPU_USAGE = localize(
    "Avg Server CPU Usage (%)", "средний cpu usage сервера, %"
)
Y_AXIS_SERVER_AVG_LATENCY = localize(
    "Avg Server Latency (ms)", "средние серверные тайминги, мс"
)

ALL_TIME_FRAME_COLUMNS = [
    X_AXIS_TIME,
    LINE_VARIABLE,
    PANEL_CLIENT_UPTIME,
    PANEL_SERVER_RPS_AMPLIFICATION,
    PANEL_CLIENT_LATENCY_P50,
    PANEL_CLIENT_LATENCY_P99,
    PANEL_CLIENT_LATENCY_AVG,
    PANEL_SERVER_AVG_REQ_QUEUE,
    PANEL_SERVER_MAX_REQ_QUEUE,
    PANEL_SERVER_UPTIME,
    PANEL_SERVER_AVG_CPU_USAGE,
    PANEL_SERVER_AVG_LATENCY,
]

PANEL_TO_Y_AXIS_NAME = {
    PANEL_CLIENT_UPTIME: Y_AXIS_CLIENT_UPTIME,
    PANEL_SERVER_RPS_AMPLIFICATION: Y_AXIS_SERVER_RPS_AMPLIFICATION,
    PANEL_CLIENT_LATENCY_P50: Y_AXIS_CLIENT_LATENCY_P50,
    PANEL_CLIENT_LATENCY_P99: Y_AXIS_CLIENT_LATENCY_P99,
    PANEL_CLIENT_LATENCY_AVG: Y_AXIS_CLIENT_LATENCY_AVG,
    PANEL_SERVER_FAILURE_RATE: Y_AXIS_SERVER_FAILURE_RATE,
    PANEL_SERVER_AVG_REQ_QUEUE: Y_AXIS_AVG_PENDING_REQ_COUNT,
    PANEL_SERVER_MAX_REQ_QUEUE: Y_AXIS_MAX_PENDING_REQ_COUNT,
    PANEL_SERVER_UPTIME: Y_AXIS_SERVER_UPTIME,
    PANEL_SERVER_AVG_CPU_USAGE: Y_AXIS_SERVER_AVG_CPU_USAGE,
    PANEL_SERVER_AVG_LATENCY: Y_AXIS_SERVER_AVG_LATENCY,
}


def fill_dataframe_by_time(df: pd.DataFrame, res: SimulationResult):
    max_time = res.params.target_duration
    client_uptimes, p50_client_latencies, p99_client_latencies, avg_client_latencies = (
        calc_client_metrics(res.client, 0, max_time)
    )
    (
        server_loads,
        uptimes,
        avg_pending_counts,
        max_pending_counts,
        avg_cpu_usages,
        avg_server_latencies,
    ) = calc_server_metrics(
        res.server, 0, max_time, res.params.target_rps, res.params.cpu_count
    )

    fill_missing_values(client_uptimes)
    fill_missing_values(p50_client_latencies)
    fill_missing_values(p99_client_latencies)
    fill_missing_values(avg_client_latencies)
    fill_missing_values(server_loads)
    fill_missing_values(uptimes)
    fill_missing_values(avg_pending_counts)
    fill_missing_values(max_pending_counts)
    fill_missing_values(avg_cpu_usages)
    fill_missing_values(avg_server_latencies)

    for i in range(get_chart_point_count(max_time)):
        row_num = len(df)
        df.at[row_num, X_AXIS_TIME] = float(i) / CHART_POINTS_PER_SECOND
        df.at[row_num, LINE_VARIABLE] = res.params.id
        df.at[row_num, PANEL_CLIENT_UPTIME] = client_uptimes[i]
        df.at[row_num, PANEL_SERVER_RPS_AMPLIFICATION] = server_loads[i]
        df.at[row_num, PANEL_CLIENT_LATENCY_P50] = p50_client_latencies[i]
        df.at[row_num, PANEL_CLIENT_LATENCY_P99] = p99_client_latencies[i]
        df.at[row_num, PANEL_CLIENT_LATENCY_AVG] = avg_client_latencies[i]
        df.at[row_num, PANEL_SERVER_AVG_REQ_QUEUE] = avg_pending_counts[i]
        df.at[row_num, PANEL_SERVER_MAX_REQ_QUEUE] = max_pending_counts[i]
        df.at[row_num, PANEL_SERVER_UPTIME] = uptimes[i]
        df.at[row_num, PANEL_SERVER_AVG_CPU_USAGE] = avg_cpu_usages[i]
        df.at[row_num, PANEL_SERVER_AVG_LATENCY] = avg_server_latencies[i]


# visualize_stats_by_time visualizes only events in [0; sim_res.target_duration).
def visualize_stats_by_time(sim_res: MultiSimulationResult, params: VisualizeParams):
    started_at = time.time()
    df = pd.DataFrame(columns=ALL_TIME_FRAME_COLUMNS)
    for res in sim_res.results:
        if params.needed_lines and res.params.id not in params.needed_lines:
            continue
        fill_dataframe_by_time(df, res)
    print(
        f"Built dataframe for {len(sim_res.results)} results in {time.time() - started_at}s"
    )
    visualize_dataframe(df, X_AXIS_TIME, params)


def visualize_dataframe(df: pd.DataFrame, x_axis_column: str, params: VisualizeParams):
    from matplotlib import pyplot as plt
    import seaborn as sns

    # Apply the default theme
    sns.set_theme()
    sns.set(font_scale=0.7)
    panels = params.needed_panels

    # Create a visualization
    row_n = params.row_n if params.row_n else len(panels) // 2
    col_n = params.col_n if params.col_n else len(panels) // row_n
    assert row_n * col_n == len(panels)
    args = {
        "nrows": row_n,
        "ncols": col_n,
        "figsize": (4 * col_n, 4 * row_n),
        "sharex": params.shareX,
    }

    var_to_color = {}
    all_colors = sns.color_palette().as_hex()
    max_color_num = max(params.id_to_color_num.values())
    for i, var in enumerate(df[LINE_VARIABLE].unique()):
        color_num = params.id_to_color_num.get(var)
        if color_num is None:
            color_num = max_color_num + 1
            max_color_num += 1
        var_to_color[var] = all_colors[color_num % len(all_colors)]

    f, axes_matrix = plt.subplots(**args)
    axes = axes_matrix.flatten()
    for i, panel in enumerate(panels):
        kwargs = {}
        if i != 0:
            kwargs["legend"] = 0
        sns.lineplot(
            data=df,
            x=x_axis_column,
            y=panel,
            hue=LINE_VARIABLE,
            ax=axes[i],
            palette=var_to_color,
            **kwargs,
        )
        axes[i].set_title(panel)
        axes[i].set(ylabel=PANEL_TO_Y_AXIS_NAME[panel])
        axes[i].set(xlim=params.xlim)
        if params.highlight_interval:
            axes[i].vlines(
                x=[
                    params.highlight_interval.begin / TIME_DURATION_SECOND,
                    params.highlight_interval.end / TIME_DURATION_SECOND,
                ],
                ymin=df[panel].min(),
                ymax=df[panel].max(),
                linestyles="dashed",
                colors="gray",
                linewidths=0.75,
            )
    f.tight_layout(h_pad=2.4, w_pad=1.6)

    # Make place for legend.
    f.subplots_adjust(top=1 - (0.6 / args["figsize"][1]))
    handles, labels = axes[0].get_legend_handles_labels()
    f.legend(handles, labels, ncol=7, loc="upper center", frameon=False, fontsize=9)
    sns.move_legend(
        axes[0],
        loc=(-100, -100),  # hide legend
    )

    # render image
    started_render_at = time.time()
    f.savefig(params.out_image_path, dpi=params.dpi)
    print(f"Rendered image in {time.time() - started_render_at}s")
