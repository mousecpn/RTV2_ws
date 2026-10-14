#!/usr/bin/env python3
"""Plot RT-V2 trajectories and highlight safety-takeover segments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.bci_noise_robustness.environment import FIXED_GOAL_POSITIONS


CONTROL_DT = 0.1
COMMAND_ARROW_SCALE = 2.5


def read_trace(trace_csv: Path, seed: int, noise_levels: list[float]) -> dict[float, list[dict[str, float]]]:
    with trace_csv.open(newline="", encoding="utf-8") as trace_file:
        rows = list(csv.DictReader(trace_file))
    traces: dict[float, list[dict[str, float]]] = {noise: [] for noise in noise_levels}
    for row in rows:
        noise = float(row["noise_std"])
        if row["condition"] == "rtv2_full" and int(row["system_seed"]) == seed and noise in traces:
            traces[noise].append({key: float(value) for key, value in row.items() if key not in {"condition", "safety_takeover"}} | {"safety_takeover": row["safety_takeover"] == "True"})
    return traces


def scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    goals = FIXED_GOAL_POSITIONS[rng.permutation(len(FIXED_GOAL_POSITIONS))]
    return goals, goals[0, :2] / rng.uniform(1.6, 2.4)


def plot_trace(axis: plt.Axes, trace: list[dict[str, float]], goals: np.ndarray, obstacle: np.ndarray, noise: float) -> None:
    axis.scatter(0, 0, color="#1f2937", s=50, zorder=4)
    axis.scatter(goals[1:, 0], goals[1:, 1], color="#cbd5e1", s=50, zorder=2)
    axis.scatter(goals[0, 0], goals[0, 1], color="#059669", s=115, marker="*", zorder=4)
    axis.scatter(obstacle[0], obstacle[1], color="#dc2626", s=65, marker="s", zorder=4)
    axis.add_patch(plt.Circle(obstacle, 1.3, fill=False, color="#dc2626", linestyle=":", linewidth=1.2))
    axis.add_patch(plt.Circle(obstacle, 1.5, fill=False, color="#f59e0b", linestyle=":", linewidth=1.2))
    positions = np.asarray([[row["position_x"], row["position_y"]] for row in trace])
    positions = np.vstack(([0.0, 0.0], positions))
    for index, row in enumerate(trace):
        color = "#f59e0b" if row["safety_takeover"] else "#2563eb"
        width = 3.2 if row["safety_takeover"] else 1.2
        axis.plot(positions[index : index + 2, 0], positions[index : index + 2, 1], color=color, linewidth=width, solid_capstyle="round")
    command_rows = trace[::3]
    axis.quiver(
        [row["position_x"] for row in command_rows],
        [row["position_y"] for row in command_rows],
        [COMMAND_ARROW_SCALE * CONTROL_DT * row["command_x"] for row in command_rows],
        [COMMAND_ARROW_SCALE * CONTROL_DT * row["command_y"] for row in command_rows],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#334155",
        alpha=0.6,
        width=0.006,
        headwidth=4.5,
        zorder=5,
    )
    takeover_steps = sum(row["safety_takeover"] for row in trace)
    axis.set(
        aspect="equal",
        xlim=(-0.8, 10.5),
        ylim=(-9.8, 9.8),
        title=f"noise={noise:g}; {len(trace)} steps; takeover={takeover_steps}",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axis.grid(alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noise-levels", default="0,0.25,0.5")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "trajectory_takeover_comparison.png")
    args = parser.parse_args()
    noise_levels = [float(value) for value in args.noise_levels.split(",")]
    traces = read_trace(args.trace_csv, args.seed, noise_levels)
    missing = [noise for noise, trace in traces.items() if not trace]
    if missing:
        parser.error(f"No rtv2_full trace for seed {args.seed}, noise levels {missing}")
    goals, obstacle = scene(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(noise_levels), figsize=(5.1 * len(noise_levels), 5.3), constrained_layout=True)
    for axis, noise in zip(np.atleast_1d(axes), noise_levels):
        plot_trace(axis, traces[noise], goals, obstacle, noise)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#2563eb", label="RT-V2 action without takeover"),
            Line2D([0], [0], color="#f59e0b", linewidth=3.2, label="Safety takeover: RT-V2 prior"),
            Line2D([0], [0], color="#334155", marker=r"$\rightarrow$", markersize=12, linewidth=0, label="User command every 3 steps (2.5x display scale)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#dc2626", label="Obstacle"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#059669", markersize=11, label="True target"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle(f"RT-V2 full trajectories, fixed-five-target scene, seed {args.seed}", fontsize=14)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote trajectory visualization to {args.output}")


if __name__ == "__main__":
    main()