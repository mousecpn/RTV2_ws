#!/usr/bin/env python3
"""Visualize fixed-five-target geometry and RT-V2 noise-sweep outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIXED_GOAL_POSITIONS = np.asarray(
    [[6.0, -7.0], [8.5, -3.5], [9.2, 0.0], [8.5, 3.5], [6.0, 7.0]],
    dtype=np.float32,
)


def read_rtv2_rows(summary_csv: Path) -> list[dict[str, float]]:
    with summary_csv.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    return [
        {
            "noise_std": float(row["noise_std"]),
            "success_rate": float(row["success_rate"]),
            "collision_rate": float(row["collision_rate"]),
            "timeout_rate": float(row["timeout_rate"]),
            "mean_steps": float(row["mean_steps"]),
        }
        for row in rows
        if row["condition"] == "rtv2_full"
    ]


def fixed_scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    goals = FIXED_GOAL_POSITIONS[rng.permutation(len(FIXED_GOAL_POSITIONS))]
    return goals, goals[0] / rng.uniform(1.6, 2.4)


def plot_geometry(axis: plt.Axes, seed: int) -> None:
    goals, obstacle = fixed_scene(seed)
    axis.scatter(0, 0, c="#1f2937", s=90, marker="o", label="Start")
    axis.scatter(goals[1:, 0], goals[1:, 1], c="#94a3b8", s=90, marker="o", label="Distractor targets")
    axis.scatter(goals[0, 0], goals[0, 1], c="#059669", s=130, marker="*", label="True target")
    axis.scatter(obstacle[0], obstacle[1], c="#dc2626", s=105, marker="s", label="Obstacle")
    axis.plot([0, goals[0, 0]], [0, goals[0, 1]], "--", color="#475569", label="Zero-noise user-command ray")
    axis.add_patch(plt.Circle(obstacle, 1.3, fill=False, color="#dc2626", linestyle=":", label="Collision radius (1.3 m)"))
    axis.add_patch(plt.Circle(obstacle, 1.5, fill=False, color="#f59e0b", linestyle=":", label="Takeover threshold (1.5 m)"))
    axis.set(xlim=(-0.8, 10.5), ylim=(-9.8, 9.8), aspect="equal", xlabel="x (m)", ylabel="y (m)", title=f"Fixed scene geometry (seed {seed})")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)


def plot_outcomes(axis: plt.Axes, rows: list[dict[str, float]]) -> None:
    noise = np.asarray([row["noise_std"] for row in rows])
    success = 100 * np.asarray([row["success_rate"] for row in rows])
    timeout = 100 * np.asarray([row["timeout_rate"] for row in rows])
    collision = 100 * np.asarray([row["collision_rate"] for row in rows])
    steps = np.asarray([row["mean_steps"] for row in rows])
    axis.plot(noise, success, "o-", color="#059669", label="Success")
    axis.plot(noise, timeout, "o-", color="#d97706", label="Timeout")
    axis.plot(noise, collision, "o-", color="#dc2626", label="Collision")
    axis.set(xlabel="BCI command noise std (m/s)", ylabel="Trial rate (%)", ylim=(-3, 103), title="RT-V2 full outcomes")
    axis.grid(alpha=0.25)
    step_axis = axis.twinx()
    step_axis.plot(noise, steps, "s--", color="#2563eb", label="Mean steps")
    step_axis.set_ylabel("Mean steps")
    axis.legend(axis.lines + step_axis.lines, [line.get_label() for line in axis.lines + step_axis.lines], loc="center right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="Fixed-scene seed to illustrate.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "fixed_scene_noise_explanation.png")
    args = parser.parse_args()
    rows = sorted(read_rtv2_rows(args.summary_csv), key=lambda row: row["noise_std"])
    if not rows:
        parser.error("summary CSV has no rtv2_full rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    plot_geometry(axes[0], args.seed)
    plot_outcomes(axes[1], rows)
    figure.suptitle("Fixed-five-target BCI noise sweep diagnostic", fontsize=15)
    figure.savefig(args.output, dpi=200)
    print(f"wrote visualization to {args.output}")


if __name__ == "__main__":
    main()