#!/usr/bin/env python3
"""Visualize one-candidate RT-V2-prior-only trajectories for selected seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONTROL_DT = 0.1
ACTION_ARROW_SCALE = 2.5


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_seed(axis: plt.Axes, trace: list[dict[str, str]], outcome: dict[str, str]) -> None:
    positions = np.asarray([[float(row["position_x"]), float(row["position_y"])] for row in trace])
    actions = np.asarray([[float(row["action_x"]), float(row["action_y"])] for row in trace])
    target = np.asarray([float(trace[0]["target_x"]), float(trace[0]["target_y"])])
    axis.plot(np.r_[0.0, positions[:, 0]], np.r_[0.0, positions[:, 1]], color="#2563eb", linewidth=1.5)
    sample = slice(None, None, 3)
    axis.quiver(
        positions[sample, 0],
        positions[sample, 1],
        ACTION_ARROW_SCALE * CONTROL_DT * actions[sample, 0],
        ACTION_ARROW_SCALE * CONTROL_DT * actions[sample, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#334155",
        alpha=0.65,
        width=0.006,
        headwidth=4.5,
        zorder=3,
    )
    axis.scatter(0, 0, color="#1f2937", s=55, zorder=4)
    axis.scatter(target[0], target[1], color="#059669", s=130, marker="*", zorder=4)
    final = positions[-1]
    axis.scatter(final[0], final[1], color="#dc2626" if outcome["outcome"] == "timeout" else "#059669", s=36, zorder=4)
    axis.set(
        aspect="equal",
        xlim=(-0.8, 10.5),
        ylim=(-9.8, 9.8),
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"seed {outcome['seed']}: {outcome['outcome']} ({outcome['steps']} steps)",
    )
    axis.grid(alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv", type=Path)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--seeds", default="0,1,9,16")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "prior_only_diagnostic.png")
    args = parser.parse_args()
    requested_seeds = [int(value) for value in args.seeds.split(",")]
    trace_rows = read_rows(args.trace_csv)
    result_rows = read_rows(args.results_csv)
    outcomes = {int(row["seed"]): row for row in result_rows}
    traces = {seed: [row for row in trace_rows if int(row["seed"]) == seed] for seed in requested_seeds}
    missing = [seed for seed, trace in traces.items() if not trace or seed not in outcomes]
    if missing:
        parser.error(f"Missing trace or outcome for seeds: {missing}")
    columns = min(2, len(requested_seeds))
    rows = int(np.ceil(len(requested_seeds) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 5.2 * rows), constrained_layout=True)
    for axis, seed in zip(np.atleast_1d(axes).flat, requested_seeds):
        plot_seed(axis, traces[seed], outcomes[seed])
    for axis in np.atleast_1d(axes).flat[len(requested_seeds) :]:
        axis.set_visible(False)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#2563eb", label="RT-V2 prior-only trajectory"),
            Line2D([0], [0], color="#334155", marker=r"$\rightarrow$", markersize=12, linewidth=0, label="Prior action every 3 steps (2.5x display scale)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#059669", markersize=12, label="Only candidate target"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=7, label="Timeout final position"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle("One-candidate RT-V2 prior-only diagnostic", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote visualization to {args.output}")


if __name__ == "__main__":
    main()