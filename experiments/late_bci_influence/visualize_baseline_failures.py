#!/usr/bin/env python3
"""Reproduce and plot selected Experiment 4 baseline trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.late_bci_influence.environment import LateInfluenceEnv
from experiments.late_bci_influence.run_experiment import ActionProbe, simulated_bci_command, target_progress


ACTION_DISPLAY_SCALE = 2.5


def replay_baseline(
    probe: ActionProbe,
    environment: LateInfluenceEnv,
    trial: int,
    seed: int,
    max_steps: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, str]:
    goals, obstacles = environment.reset(seed)
    probe.reset(goals, obstacles, seed)
    start = environment.position().copy()
    trace: list[dict[str, object]] = []
    outcome = "timeout"
    for step in range(1, max_steps + 1):
        position = environment.position().copy()
        command = simulated_bci_command(position, goals[0])
        action = probe._predict(position, command)
        executed_action = action if action is not None else command
        environment.move(executed_action)
        next_position = environment.position().copy()
        progress = target_progress(position, start, goals[0])
        trace.append(
            {
                "step": step,
                "position": position,
                "next_position": next_position,
                "action": executed_action,
                "progress": progress,
            }
        )
        outcome = environment.terminal_outcome() or "timeout"
        if outcome != "timeout":
            break
    return trace, goals, obstacles, outcome


def plot_replay(axis: plt.Axes, trace: list[dict[str, object]], goals: np.ndarray, obstacles: np.ndarray, seed: int, outcome: str) -> None:
    positions = np.asarray([row["position"] for row in trace], dtype=float)
    next_positions = np.asarray([row["next_position"] for row in trace], dtype=float)
    actions = np.asarray([row["action"] for row in trace], dtype=float)
    progress = np.asarray([row["progress"] for row in trace], dtype=float)
    path = np.vstack((positions[0], next_positions))
    late_indices = np.flatnonzero(progress >= 2.0 / 3.0)

    axis.scatter(0.0, 0.0, color="#1f2937", s=48, zorder=5)
    axis.scatter(goals[1:, 0], goals[1:, 1], color="#cbd5e1", s=48, zorder=2)
    axis.scatter(goals[0, 0], goals[0, 1], color="#059669", s=135, marker="*", zorder=5)
    axis.scatter(obstacles[:, 0], obstacles[:, 1], color="#dc2626", s=65, marker="s", zorder=5)
    for obstacle in obstacles:
        axis.add_patch(plt.Circle(obstacle, 1.3, fill=False, color="#dc2626", linestyle=":", linewidth=1.2))
    axis.plot(path[:, 0], path[:, 1], color="#2563eb", linewidth=1.35, zorder=3)
    if len(late_indices):
        late_start = positions[late_indices[0]]
        axis.scatter(late_start[0], late_start[1], color="#f59e0b", s=42, zorder=6)
        axis.plot(path[late_indices[0] :, 0], path[late_indices[0] :, 1], color="#f59e0b", linewidth=1.8, zorder=4)
    sample = slice(None, None, 5)
    axis.quiver(
        positions[sample, 0],
        positions[sample, 1],
        ACTION_DISPLAY_SCALE * 0.1 * actions[sample, 0],
        ACTION_DISPLAY_SCALE * 0.1 * actions[sample, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#334155",
        alpha=0.6,
        width=0.006,
        headwidth=4.5,
        zorder=5,
    )
    final_position = next_positions[-1]
    final_color = "#dc2626" if outcome == "timeout" else "#059669"
    axis.scatter(final_position[0], final_position[1], color=final_color, s=38, zorder=6)
    final_goal_distance = np.linalg.norm(final_position - goals[0])
    axis.set(
        aspect="equal",
        xlim=(-5.0, 13.0),
        ylim=(-14.0, 14.0),
        xlabel="x (m)",
        ylabel="y (m)",
        title=(f"seed {seed}: {outcome}, {len(trace)} steps\n"
               f"final target distance={final_goal_distance:.1f} m"),
    )
    axis.grid(alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="12,18,30,36", help="Comma-separated system seeds to replay.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "baseline_failure_diagnostic.png")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or args.max_steps < 1:
        parser.error("provide at least one seed and a positive max-steps value")

    probe = ActionProbe(args.checkpoint, args.config, user_covariance=0.3)
    environment = LateInfluenceEnv(gui=False)
    try:
        replays = [(seed, *replay_baseline(probe, environment, seed, seed, args.max_steps)) for seed in seeds]
    finally:
        environment.close()

    columns = min(2, len(replays))
    rows = int(np.ceil(len(replays) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 5.6 * rows), constrained_layout=True)
    for axis, (seed, trace, goals, obstacles, outcome) in zip(np.atleast_1d(axes).flat, replays):
        plot_replay(axis, trace, goals, obstacles, seed, outcome)
    for axis in np.atleast_1d(axes).flat[len(replays) :]:
        axis.set_visible(False)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#2563eb", label="Baseline posterior trajectory"),
            Line2D([0], [0], color="#f59e0b", linewidth=2, label="Trajectory from 67% progress"),
            Line2D([0], [0], color="#334155", marker=r"$\rightarrow$", markersize=12, linewidth=0, label="Executed action every 5 steps (2.5x display scale)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#dc2626", label="Obstacle / 1.3 m collision radius"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#059669", markersize=12, label="True target"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", label="Timeout final position"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle("Experiment 4 baseline: selected timeout trajectory diagnostics", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    for seed, trace, goals, _, outcome in replays:
        final_position = np.asarray(trace[-1]["next_position"])
        print(f"seed={seed} outcome={outcome} steps={len(trace)} final_goal_distance={np.linalg.norm(final_position - goals[0]):.3f}")
    print(f"wrote visualization to {args.output}")


if __name__ == "__main__":
    main()