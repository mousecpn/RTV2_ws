#!/usr/bin/env python3
"""Evaluate shared control as BCI command noise increases."""

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

from experiments.bci_noise_robustness.environment import NoiseRobustnessEnv  # noqa: E402


MAX_SPEED = 3.0


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    blend_weight: float | None = None


def parse_float_list(raw_values: str) -> list[float]:
    values = [float(value) for value in raw_values.split(",")]
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def conditions(weights: list[float]) -> list[Condition]:
    return [Condition("user_only", "user_only"), Condition("rtv2_full", "rtv2_full"), *(Condition(f"fixed_blend_{weight:g}", "fixed_blend", weight) for weight in weights)]


def clip_speed(velocity: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(velocity)
    return velocity if norm <= MAX_SPEED else velocity * MAX_SPEED / norm


def user_command(position: np.ndarray, true_goal: np.ndarray, unit_noise: np.ndarray, noise_std: float) -> np.ndarray:
    direction = true_goal - position
    distance = np.linalg.norm(direction)
    nominal = np.zeros(2, dtype=np.float32) if distance == 0 else MAX_SPEED * direction / distance
    return clip_speed(nominal + noise_std * unit_noise).astype(np.float32)


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
        logger = logging.getLogger("bci_noise_robustness")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_covariance = user_covariance
        self.safety_count = 0
        self.take_control_threshold = 1.5

    def reset_trial(self, goals: np.ndarray, obstacles: np.ndarray, system_seed: int) -> None:
        torch.manual_seed(system_seed)
        torch.cuda.manual_seed_all(system_seed)
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
        return clip_speed(np.asarray(action, dtype=np.float32)), use_prior


def execute_trial(
    controller: RTV2Controller,
    environment: NoiseRobustnessEnv,
    condition: Condition,
    noise_std: float,
    system_seed: int,
    max_steps: int,
    trace_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    goals, obstacles = environment.reset(system_seed)
    controller.reset_trial(goals, obstacles, system_seed)
    noise_rng = np.random.default_rng(system_seed)
    for step in range(1, max_steps + 1):
        position = environment.position()
        command = user_command(position, goals[0], noise_rng.normal(size=2), noise_std)
        safety_takeover = False
        if condition.kind == "user_only":
            action = command
        else:
            rtv2_action, safety_takeover = controller.action(position, command, condition)
            action = clip_speed((1.0 - condition.blend_weight) * command + condition.blend_weight * rtv2_action) if condition.kind == "fixed_blend" else rtv2_action
        environment.move(action)
        if trace_rows is not None:
            trace_rows.append(
                {
                    "condition": condition.name,
                    "noise_std": noise_std,
                    "system_seed": system_seed,
                    "step": step,
                    "position_x": position[0],
                    "position_y": position[1],
                    "command_x": command[0],
                    "command_y": command[1],
                    "action_x": action[0],
                    "action_y": action[1],
                    "safety_takeover": safety_takeover,
                }
            )
        if environment.is_collision():
            return {"outcome": "collision", "steps": step}
        if environment.is_true_goal_achieved():
            return {"outcome": "success", "steps": step}
    return {"outcome": "timeout", "steps": max_steps}


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_fields = ["condition", "noise_std", "trial", "system_seed", "outcome", "steps"]
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for noise_std in sorted({float(row["noise_std"]) for row in results}):
        for condition in sorted({str(row["condition"]) for row in results}):
            rows = [row for row in results if row["noise_std"] == noise_std and row["condition"] == condition]
            summary.append({"noise_std": noise_std, "condition": condition, "trials": len(rows), "success_rate": sum(row["outcome"] == "success" for row in rows) / len(rows), "collision_rate": sum(row["outcome"] == "collision" for row in rows) / len(rows), "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / len(rows), "mean_steps": float(np.mean([row["steps"] for row in rows]))})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def write_trace(rows: list[dict[str, object]], trace_dir: Path) -> None:
    if not rows:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / "trajectory_trace.csv").open("w", newline="", encoding="utf-8") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RT-V2 robustness experiments across BCI noise levels.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise-levels", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--blend-weights", default="0.5")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", help="Run one condition only.")
    parser.add_argument("--trace-dir", type=Path, help="Write per-step trajectory traces to this directory.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--scene-layout",
        choices=["fixed_five_target", "random_occluded"],
        default="random_occluded",
        help="Fixed five-target scene or random occluded scene.",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.max_steps <= 0:
        parser.error("trials and max-steps must be positive")
    noise_levels, blend_weights = parse_float_list(args.noise_levels), parse_float_list(args.blend_weights)
    selected_conditions = conditions(blend_weights)
    if args.condition:
        selected_conditions = [condition for condition in selected_conditions if condition.name == args.condition]
        if not selected_conditions:
            parser.error("unknown condition")
    controller = RTV2Controller(args.checkpoint, args.config, 0.3)
    environment = NoiseRobustnessEnv(gui=args.gui, scene_layout=args.scene_layout)
    results: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] | None = [] if args.trace_dir else None
    try:
        for noise_std in noise_levels:
            for condition in selected_conditions:
                for trial in range(args.trials):
                    system_seed = args.seed + trial
                    row = execute_trial(controller, environment, condition, noise_std, system_seed, args.max_steps, trace_rows)
                    row.update({"condition": condition.name, "noise_std": noise_std, "trial": trial, "system_seed": system_seed})
                    results.append(row)
                print(f"completed noise={noise_std}, condition={condition.name}")
    finally:
        environment.close()
    write_results(results, args.output_dir)
    if trace_rows is not None:
        write_trace(trace_rows, args.trace_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()