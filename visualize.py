from dataclasses import dataclass
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Dict, Tuple, List

@dataclass
class CallResult:
    timepoint: float
    is_ok: bool
    attempt_count: int

CallResultList = List[CallResult]

# visualize_stats accepts max_time to visualize only events in [0; max_time).
def visualize_stats(stats: Dict[str, CallResultList], max_time: int, target_rps: int):
    x: List[int] = list(range(max_time))
    df = pd.DataFrame(columns=['time', 'retry strategy', 'client uptime', 'rps', 'rps ratio'])

    for id, call_results in stats.items():
        total_counts, succeeded_counts, attempt_counts = [0] * len(x), [0] * len(x), [0] * len(x)
        for cr in call_results:
            timepoint_rounded = int(cr.timepoint)
            if timepoint_rounded >= max_time:
                continue
            total_counts[timepoint_rounded] += 1
            if cr.is_ok:
                succeeded_counts[timepoint_rounded] += 1
            attempt_counts[timepoint_rounded] += cr.attempt_count
        for i in range(len(x)):
            client_uptime = succeeded_counts[i] / total_counts[i] if total_counts[i] != 0 else 1
            df.loc[len(df)] = [x[i], id, client_uptime, attempt_counts[i], attempt_counts[i] / target_rps]

    # Apply the default theme
    sns.set_theme()

    # Create a visualization
    print(df)
    f, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
    sns.lineplot(data=df, x='time', y='client uptime', hue='retry strategy', ax=axs[0])
    sns.lineplot(data=df, x='time', y='rps ratio', hue='retry strategy', ax=axs[1], legend=0)
    f.tight_layout()

    # show legend
    f.subplots_adjust(top=0.9)  # make place for legend
    handles, labels = axs[0].get_legend_handles_labels()
    f.legend(handles, labels, ncol=5, loc='upper center', frameon=False)
    sns.move_legend(
        axs[0], loc=(-100, -100),  # hide legend
    )

    # render image
    f.savefig("out.png", dpi=200)