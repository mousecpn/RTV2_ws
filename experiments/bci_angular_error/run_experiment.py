#!/usr/bin/env python3
"""Experiment 3 angular-error sweep with fixed-magnitude BCI commands."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributions as distributions


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.bci_noise_robustness.environment import NoiseRobustnessEnv  # noqa: E402


COMMAND_SPEED = 3.0


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    blend_weight: float | None = None


def parse_nonnegative_floats(raw_values: str) -> list[float]:
    values = [float(value) for value in raw_values.split(",")]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be non-negative")
    return values


def build_conditions(blend_weights: list[float]) -> list[Condition]:
    return [
        Condition("user_only", "user_only"),
        Condition("rtv2_full", "rtv2_full"),
        *(Condition(f"fixed_blend_{weight:g}", "fixed_blend", weight) for weight in blend_weights),
    ]


def angular_user_command(position: np.ndarray, true_goal: np.ndarray, standard_normal: float, angular_sd_degrees: float) -> tuple[np.ndarray, float]:
    """Rotate the goal-directed unit vector while preserving the command speed."""
    direction = true_goal - position
    nominal_angle = math.atan2(direction[1], direction[0])
    error_radians = math.radians(angular_sd_degrees) * standard_normal
    rotated_angle = nominal_angle + error_radians
    command = COMMAND_SPEED * np.asarray([math.cos(rotated_angle), math.sin(rotated_angle)], dtype=np.float32)
    return command, math.degrees(error_radians)


class RTV2Controller:
    def __init__(self, checkpoint: Path, config: Path, user_covariance: float):
        if not torch.cuda.is_available():
            raise RuntimeError("RT-V2 inference requires CUDA because TrajectronNode's adapter uses .cuda().")
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
        logger = logging.getLogger("bci_angular_error")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_covariance = user_covariance
        self.safety_count = 0
        self.take_control_threshold = 1.5

    def reset_trial(self, goals: np.ndarray, obstacles: np.ndarray, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.service.goal_position = goals.copy()
        self.service.obs_position = obstacles.copy()
        self.service.traj_log.queue.clear()
        self.service.dist = self.service.v_dist = None
        self.service.op_count = self.safety_count = 0
        self.service.mpc.nominal_u.zero_()

    def action(self, position: np.ndarray, command: np.ndarray, condition: Condition) -> tuple[np.ndarray, bool]:
        from msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = position.tolist()
        use_prior = condition.kind == "fixed_blend"
        if condition.kind == "rtv2_full":
            self.safety_count -= 1
            predicted_position = position + self.service.dt * command
            if np.linalg.norm(predicted_position[None] - self.service.obs_position, axis=1).min() < self.take_control_threshold:
                self.safety_count = 3
            use_prior = self.safety_count > 0
        if use_prior:
            self.service.user_model = None
        else:
            mean = torch.tensor(command.reshape(1, 2), dtype=torch.float32, device=self.device)
            self.service.user_model = distributions.MultivariateNormal(mean, torch.eye(2, device=self.device) * self.user_covariance)
        self.service.trajectory_prediction(pose)
        if self.service.dist is None:
            return command.copy(), use_prior
        action, _ = self.service.mpc_control(self.service.goal_position, self.service.obs_position, position)
        return np.asarray(action, dtype=np.float32), use_prior


def execute_trial(
    controller: RTV2Controller,
    environment: NoiseRobustnessEnv,
    condition: Condition,
    angular_sd_degrees: float,
    system_seed: int,
    max_steps: int,
    trace_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    goals, obstacles = environment.reset(system_seed)
    controller.reset_trial(goals, obstacles, system_seed)
    error_rng = np.random.default_rng(system_seed)
    for step in range(1, max_steps + 1):
        position = environment.position()
        command, angular_error = angular_user_command(position, goals[0], error_rng.normal(), angular_sd_degrees)
        safety_takeover = False
        if condition.kind == "user_only":
            action = command
        else:
            rtv2_action, safety_takeover = controller.action(position, command, condition)
            action = (1.0 - condition.blend_weight) * command + condition.blend_weight * rtv2_action if condition.kind == "fixed_blend" else rtv2_action
        environment.move(action)
        if trace_rows is not None:
            trace_rows.append({"condition": condition.name, "angular_sd_degrees": angular_sd_degrees, "system_seed": system_seed, "step": step, "command_speed": float(np.linalg.norm(command)), "angular_error_degrees": angular_error, "action_x": action[0], "action_y": action[1], "safety_takeover": safety_takeover})
        if environment.is_collision():
            return {"outcome": "collision", "steps": step}
        if environment.is_true_goal_achieved():
            return {"outcome": "success", "steps": step}
    return {"outcome": "timeout", "steps": max_steps}


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "angular_sd_degrees", "trial", "system_seed", "outcome", "steps"]
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for error_sd in sorted({float(row["angular_sd_degrees"]) for row in results}):
        for condition in sorted({str(row["condition"]) for row in results}):
            rows = [row for row in results if row["angular_sd_degrees"] == error_sd and row["condition"] == condition]
            summary.append({"angular_sd_degrees": error_sd, "condition": condition, "trials": len(rows), "success_rate": sum(row["outcome"] == "success" for row in rows) / len(rows), "collision_rate": sum(row["outcome"] == "collision" for row in rows) / len(rows), "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / len(rows), "mean_steps": float(np.mean([row["steps"] for row in rows]))})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def write_trace(trace_rows: list[dict[str, object]], trace_dir: Path) -> None:
    if not trace_rows:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "angular_error_trace.csv").open("w", newline="", encoding="utf-8") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-magnitude angular-error BCI robustness experiments.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--angular-sds", default="0,10,20,40,60", help="Angular-error standard deviations in degrees.")
    parser.add_argument("--blend-weights", default="0.5")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", help="Run one condition only.")
    parser.add_argument("--trace-dir", type=Path, help="Write per-step command and angular-error traces.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scene-layout", choices=["fixed_five_target", "random_occluded"], default="fixed_five_target")
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.max_steps <= 0:
        parser.error("trials and max-steps must be positive")
    angular_sds = parse_nonnegative_floats(args.angular_sds)
    selected_conditions = build_conditions(parse_nonnegative_floats(args.blend_weights))
    if any(condition.blend_weight is not None and condition.blend_weight > 1 for condition in selected_conditions):
        parser.error("blend weights must be in [0, 1]")
    if args.condition:
        selected_conditions = [condition for condition in selected_conditions if condition.name == args.condition]
        if not selected_conditions:
            parser.error("unknown condition")
    controller = RTV2Controller(args.checkpoint, args.config, 0.3)
    environment = NoiseRobustnessEnv(gui=args.gui, scene_layout=args.scene_layout)
    results: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] | None = [] if args.trace_dir else None
    try:
        for angular_sd in angular_sds:
            for condition in selected_conditions:
                for trial in range(args.trials):
                    system_seed = args.seed + trial
                    row = execute_trial(controller, environment, condition, angular_sd, system_seed, args.max_steps, trace_rows)
                    row.update({"condition": condition.name, "angular_sd_degrees": angular_sd, "trial": trial, "system_seed": system_seed})
                    results.append(row)
                print(f"completed angular_sd={angular_sd} deg, condition={condition.name}")
    finally:
        environment.close()
    write_results(results, args.output_dir)
    if trace_rows is not None:
        write_trace(trace_rows, args.trace_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()