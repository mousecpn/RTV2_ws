#!/usr/bin/env python3
"""Compare hard-coded obstacle-distance assistance with shared-control baselines."""

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

from experiments.shared_control_sim.environment import SharedControlBatchEnv  # noqa: E402


MAX_SPEED = 3.0


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    blend_weight: float | None = None
    switch_radius: float | None = None


def parse_float_list(raw_values: str, lower: float, upper: float | None = None) -> list[float]:
    values = [float(value) for value in raw_values.split(",")]
    if not values or any(value < lower or (upper is not None and value > upper) for value in values):
        raise argparse.ArgumentTypeError("invalid numeric value")
    return values


def build_conditions(blend_weights: list[float], switch_radii: list[float]) -> list[Condition]:
    return [
        Condition("user_only", "user_only"),
        Condition("rtv2_full", "rtv2_full"),
        *(Condition(f"fixed_blend_{weight:g}", "fixed_blend", blend_weight=weight) for weight in blend_weights),
        Condition("rtv2_prior_only", "rtv2_prior_only"),
        *(Condition(f"phase_prior_R{radius:g}", "phase_prior", switch_radius=radius) for radius in switch_radii),
    ]


def clip_speed(velocity: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(velocity)
    return velocity if norm <= MAX_SPEED else velocity * MAX_SPEED / norm


def simulated_user_command(position: np.ndarray, true_goal: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    direction = true_goal - position
    distance = np.linalg.norm(direction)
    nominal = np.zeros(2, dtype=np.float32) if distance == 0 else MAX_SPEED * direction / distance
    return clip_speed(nominal + rng.normal(0.0, noise_std, size=2)).astype(np.float32)


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
        logger = logging.getLogger("phase_specific_assistance")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_covariance = user_covariance
        self.safety_count = 0
        self.take_control_threshold = 1.5

    def reset_trial(self, goals: np.ndarray, obstacles: np.ndarray, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.service.goal_position, self.service.obs_position = goals.copy(), obstacles.copy()
        self.service.traj_log.queue.clear()
        self.service.dist = self.service.v_dist = None
        self.service.op_count = self.safety_count = 0
        self.service.mpc.nominal_u.zero_()

    def model_action(self, position: np.ndarray, command: np.ndarray | None) -> np.ndarray | None:
        from msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = position.tolist()
        if command is None:
            self.service.user_model = None
        else:
            mean = torch.tensor(command.reshape(1, 2), dtype=torch.float32, device=self.device)
            covariance = torch.eye(2, device=self.device) * self.user_covariance
            self.service.user_model = distributions.MultivariateNormal(mean, covariance)
        self.service.trajectory_prediction(pose)
        if self.service.dist is None:
            return None
        action, _ = self.service.mpc_control(self.service.goal_position, self.service.obs_position, position)
        return clip_speed(np.asarray(action, dtype=np.float32))

    def full_action(self, position: np.ndarray, command: np.ndarray) -> np.ndarray:
        self.safety_count -= 1
        predicted_position = position + self.service.dt * command
        obstacle_distances = np.linalg.norm(predicted_position[None] - self.service.obs_position, axis=1)
        if obstacle_distances.size and obstacle_distances.min() < self.take_control_threshold:
            self.safety_count = 3
        action = self.model_action(position, None if self.safety_count > 0 else command)
        return action if action is not None else command.copy()


def execute_trial(controller: RTV2Controller, environment: SharedControlBatchEnv, condition: Condition, seed: int, noise_std: float, max_steps: int) -> dict[str, object]:
    goals, obstacles = environment.reset(seed)
    controller.reset_trial(goals, obstacles, seed)
    rng = np.random.default_rng(seed)
    phase_prior_steps = 0
    for step in range(1, max_steps + 1):
        position = environment.position()
        command = simulated_user_command(position, goals[0], rng, noise_std)
        nearest_distance = float(np.linalg.norm(position[None] - obstacles, axis=1).min()) if len(obstacles) else float("inf")
        if condition.kind == "user_only":
            action = command
        elif condition.kind == "rtv2_full":
            action = controller.full_action(position, command)
        elif condition.kind == "rtv2_prior_only":
            action = controller.model_action(position, None)
            action = action if action is not None else np.zeros(2, dtype=np.float32)
        elif condition.kind == "fixed_blend":
            prior = controller.model_action(position, None)
            prior = prior if prior is not None else np.zeros(2, dtype=np.float32)
            action = clip_speed((1.0 - condition.blend_weight) * command + condition.blend_weight * prior)
        else:
            # Sole arbitration rule for phase-specific assistance: current nearest-obstacle distance.
            phase_prior_active = nearest_distance < condition.switch_radius
            model_action = controller.model_action(position, None if phase_prior_active else command)
            action = model_action if phase_prior_active and model_action is not None else command
            phase_prior_steps += int(phase_prior_active)
        environment.move(action)
        if environment.is_collision():
            return {"outcome": "collision", "steps": step, "phase_prior_steps": phase_prior_steps}
        if environment.is_true_goal_achieved():
            return {"outcome": "success", "steps": step, "phase_prior_steps": phase_prior_steps}
    return {"outcome": "timeout", "steps": max_steps, "phase_prior_steps": phase_prior_steps}


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "switch_radius", "trial", "system_seed", "outcome", "steps", "phase_prior_steps"]
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for name in sorted({str(row["condition"]) for row in results}):
        rows = [row for row in results if row["condition"] == name]
        total = len(rows)
        summary.append({"condition": name, "switch_radius": rows[0]["switch_radius"], "trials": total, "success_rate": sum(row["outcome"] == "success" for row in rows) / total, "collision_rate": sum(row["outcome"] == "collision" for row in rows) / total, "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / total, "mean_steps": float(np.mean([row["steps"] for row in rows])), "mean_phase_prior_steps": float(np.mean([row["phase_prior_steps"] for row in rows]))})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-radius phase-prior assistance.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--switch-radii", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--blend-weights", default="0.5")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", help="Run one condition only.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scene-layout", choices=["fixed_five_target", "random_occluded"], default="fixed_five_target")
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.noise_std < 0 or args.max_steps <= 0:
        parser.error("trials and max-steps must be positive; noise-std must be non-negative")
    conditions = build_conditions(parse_float_list(args.blend_weights, 0.0, 1.0), parse_float_list(args.switch_radii, 0.0))
    if args.condition:
        conditions = [condition for condition in conditions if condition.name == args.condition]
        if not conditions:
            parser.error("unknown condition")
    controller = RTV2Controller(args.checkpoint, args.config, user_covariance=0.3)
    environment = SharedControlBatchEnv(gui=args.gui, scene_layout=args.scene_layout)
    results: list[dict[str, object]] = []
    try:
        for condition in conditions:
            for trial in range(args.trials):
                system_seed = args.seed + trial
                row = execute_trial(controller, environment, condition, system_seed, args.noise_std, args.max_steps)
                row.update({"condition": condition.name, "switch_radius": condition.switch_radius, "trial": trial, "system_seed": system_seed})
                results.append(row)
            print(f"completed {condition.name}")
    finally:
        environment.close()
    write_results(results, args.output_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()