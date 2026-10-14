#!/usr/bin/env python3
"""Compare full RT-V2 and AI-only prior as candidate-target count increases."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributions as distributions


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.candidate_target_ai_only.environment import CandidateTargetEnv  # noqa: E402


MAX_SPEED = 3.0
EMPTY_OBSTACLES = np.empty((0, 2), dtype=np.float32)


@dataclass(frozen=True)
class Condition:
    name: str
    uses_user_command: bool


CONDITIONS = (Condition("rtv2_full", True), Condition("rtv2_prior_only", False))


def parse_candidate_counts(raw_counts: str) -> list[int]:
    counts = [int(value) for value in raw_counts.split(",")]
    if not counts or any(count < 1 or count > 20 for count in counts):
        raise argparse.ArgumentTypeError("candidate counts must be integers in [1, 20]")
    return counts


def parse_seed_list(raw_seeds: str) -> list[int]:
    seeds = [int(value) for value in raw_seeds.split(",")]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def clip_speed(velocity: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(velocity)
    return velocity if norm <= MAX_SPEED else velocity * (MAX_SPEED / norm)


def simulated_user_command(position: np.ndarray, true_target: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    direction = true_target - position
    distance = np.linalg.norm(direction)
    nominal = np.zeros(2, dtype=np.float32) if distance == 0 else MAX_SPEED * direction / distance
    return clip_speed(nominal + rng.normal(0.0, noise_std, size=2)).astype(np.float32)


class RTV2Controller:
    def __init__(self, checkpoint: Path, config: Path, user_covariance: float):
        if not torch.cuda.is_available():
            raise RuntimeError("RT-V2 inference requires CUDA because TrajectronNode's batch adapter uses .cuda().")
        from TrajectronNode import trajectron_service
        from model.trajectron import Trajectron

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        with config.open(encoding="utf-8") as config_file:
            hyperparams = json.load(config_file)
        hyperparams.update({"edge_encoding": False, "batch_size": 128, "k_eval": 10, "map_encoding": True, "frequency": 10})
        trajectron = Trajectron(hyperparams, log_writer=None, device=self.device)
        trajectron.model.node_modules = torch.load(checkpoint, map_location=self.device)
        trajectron.set_annealing_params()
        trajectron.model.to(self.device)
        trajectron.model.eval()

        logger = logging.getLogger("candidate_target_ai_only")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_covariance = user_covariance

    def reset_trial(self, targets: np.ndarray, trial_seed: int) -> None:
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        self.service.goal_position = targets.copy()
        self.service.obs_position = EMPTY_OBSTACLES.copy()
        self.service.traj_log.queue.clear()
        self.service.dist = None
        self.service.v_dist = None
        self.service.op_count = 0
        self.service.mpc.nominal_u.zero_()

    def action(self, position: np.ndarray, user_command: np.ndarray, condition: Condition) -> np.ndarray:
        from msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = position.tolist()
        if condition.uses_user_command:
            mean = torch.tensor(user_command.reshape(1, 2), dtype=torch.float32, device=self.device)
            covariance = torch.eye(2, device=self.device) * self.user_covariance
            self.service.user_model = distributions.MultivariateNormal(mean, covariance)
        else:
            self.service.user_model = None

        self.service.trajectory_prediction(pose)
        if self.service.dist is None:
            return user_command.copy() if condition.uses_user_command else np.zeros(2, dtype=np.float32)
        action, _ = self.service.mpc_control(self.service.goal_position, self.service.obs_position, position)
        return clip_speed(np.asarray(action, dtype=np.float32))


def execute_trial(
    controller: RTV2Controller,
    environment: CandidateTargetEnv,
    condition: Condition,
    candidate_count: int,
    trial_seed: int,
    noise_std: float,
    max_steps: int,
    trace_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    targets = environment.reset(trial_seed, candidate_count)
    controller.reset_trial(targets, trial_seed)
    rng = np.random.default_rng(trial_seed)
    true_target = targets[0]

    for step in range(1, max_steps + 1):
        position = environment.position()
        user_command = simulated_user_command(position, true_target, rng, noise_std)
        action = controller.action(position, user_command, condition)
        environment.move(action)
        if trace_rows is not None:
            trace_rows.append(
                {
                    "condition": condition.name,
                    "candidate_count": candidate_count,
                    "seed": trial_seed,
                    "step": step,
                    "position_x": position[0],
                    "position_y": position[1],
                    "target_x": true_target[0],
                    "target_y": true_target[1],
                    "action_x": action[0],
                    "action_y": action[1],
                }
            )
        reached_index = environment.reached_target_index()
        if reached_index is not None:
            outcome = "correct_target" if reached_index == 0 else "wrong_target"
            return {"outcome": outcome, "steps": step, "seed": trial_seed}
    return {"outcome": "timeout", "steps": max_steps, "seed": trial_seed}


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_fields = ["condition", "candidate_count", "trial", "seed", "outcome", "steps"]
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(results)

    summary = []
    for condition in sorted({str(row["condition"]) for row in results}):
        for candidate_count in sorted({int(row["candidate_count"]) for row in results}):
            rows = [row for row in results if row["condition"] == condition and row["candidate_count"] == candidate_count]
            total = len(rows)
            summary.append(
                {
                    "condition": condition,
                    "candidate_count": candidate_count,
                    "trials": total,
                    "correct_target_success": sum(row["outcome"] == "correct_target" for row in rows) / total,
                    "any_target_success": sum(row["outcome"] in {"correct_target", "wrong_target"} for row in rows) / total,
                    "wrong_target_rate": sum(row["outcome"] == "wrong_target" for row in rows) / total,
                    "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / total,
                    "mean_steps": float(np.mean([row["steps"] for row in rows])),
                }
            )
    fields = list(summary[0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)


def write_trace(rows: list[dict[str, object]], trace_dir: Path) -> None:
    if not rows:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "trajectory_trace.csv").open("w", newline="", encoding="utf-8") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the candidate-target / AI-only RT-V2 experiment.")
    parser.add_argument("--trials", type=int, default=100, help="Trials per candidate count and condition.")
    parser.add_argument("--candidate-counts", default="1,2,3,4,5,10,15,20", help="Candidate target counts, comma-separated.")
    parser.add_argument("--noise-std", type=float, default=0.25, help="Simulated-user Gaussian command-noise standard deviation in m/s.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum 10 Hz control steps per trial.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; conditions share trial scenes and user noise.")
    parser.add_argument("--seeds", type=parse_seed_list, help="Explicit comma-separated trial seeds; overrides --trials and --seed.")
    parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], help="Run one condition only.")
    parser.add_argument("--trace-dir", type=Path, help="Write per-step trajectory traces to this directory.")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet scene and run at 10 Hz.")
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.noise_std < 0 or args.max_steps <= 0:
        parser.error("trials and max-steps must be positive; noise-std must be non-negative")

    candidate_counts = parse_candidate_counts(args.candidate_counts)
    trial_seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.trials))
    conditions = [condition for condition in CONDITIONS if args.condition is None or condition.name == args.condition]
    controller = RTV2Controller(args.checkpoint, args.config, args.noise_std**2)
    environment = CandidateTargetEnv(gui=args.gui)
    results: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] | None = [] if args.trace_dir else None
    try:
        for candidate_count in candidate_counts:
            for condition in conditions:
                for trial, trial_seed in enumerate(trial_seeds):
                    row = execute_trial(controller, environment, condition, candidate_count, trial_seed, args.noise_std, args.max_steps, trace_rows)
                    row.update({"condition": condition.name, "candidate_count": candidate_count, "trial": trial})
                    results.append(row)
                    if trace_rows is not None:
                        write_trace(trace_rows, args.trace_dir)
                print(f"completed candidates={candidate_count}, condition={condition.name}")
    finally:
        environment.close()
    write_results(results, args.output_dir)
    if trace_rows is not None:
        write_trace(trace_rows, args.trace_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()