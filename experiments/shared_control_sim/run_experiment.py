#!/usr/bin/env python3
"""Batch shared-control comparison in the five-target simulated scene.

The scene geometry and termination thresholds match ``sim_env.py``.  RT-V2 is
loaded in-process so the benchmark can evaluate all controller conditions with
the same simulated-user commands and no GUI or ZMQ dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


def parse_weights(raw_weights: str) -> list[float]:
    weights = [float(value) for value in raw_weights.split(",")]
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise argparse.ArgumentTypeError("blend weights must be in [0, 1]")
    return weights


def build_conditions(weights: Iterable[float]) -> list[Condition]:
    conditions = [
        Condition("rtv2_full", "rtv2_full"),
        Condition("user_only", "user_only"),
    ]
    conditions.extend(
        Condition(f"fixed_blend_{weight:g}", "fixed_blend", weight)
        for weight in weights
    )
    conditions.append(Condition("rtv2_prior_only", "rtv2_prior_only"))
    return conditions


def clip_speed(velocity: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(velocity)
    return velocity if norm <= MAX_SPEED else velocity * (MAX_SPEED / norm)


def simulated_user_command(
    position: np.ndarray, true_goal: np.ndarray, rng: np.random.Generator, noise_std: float
) -> np.ndarray:
    """Return a fixed-speed command toward the true goal with Gaussian noise."""
    direction = true_goal - position
    distance = np.linalg.norm(direction)
    nominal = np.zeros(2, dtype=np.float32) if distance == 0 else MAX_SPEED * direction / distance
    return clip_speed(nominal + rng.normal(0.0, noise_std, size=2)).astype(np.float32)


class RTV2Controller:
    def __init__(self, checkpoint: Path, config: Path, user_sigma: float):
        if not torch.cuda.is_available():
            raise RuntimeError("RT-V2 experiment inference requires CUDA because the existing batch adapter uses .cuda().")

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

        logger = logging.getLogger("shared_control_sim")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_sigma = user_sigma
        self.take_control_threshold = 1.5
        # self.take_control_threshold = 3

        self.safety_count = 0

    def reset_trial(self, goals: np.ndarray, obstacles: np.ndarray) -> None:
        self.service.goal_position = goals.copy()
        self.service.obs_position = obstacles.copy()
        self.service.traj_log.queue.clear()
        self.service.dist = None
        self.service.v_dist = None
        self.service.op_count = 0
        self.service.mpc.nominal_u.zero_()
        self.safety_count = 0

    def use_prior(self, position: np.ndarray, user_command: np.ndarray, condition: Condition) -> bool:
        """Match TrajectronNode's three-step obstacle-triggered safety takeover."""
        if condition.kind != "rtv2_full":
            return True

        self.safety_count -= 1
        next_position = position + self.service.dt * user_command
        obstacle_distances = np.linalg.norm(next_position[None, :] - self.service.obs_position, axis=1)
        if obstacle_distances.size and obstacle_distances.min() < self.take_control_threshold:
            self.safety_count = 3
        return self.safety_count > 0

    def action(self, position: np.ndarray, user_command: np.ndarray, condition: Condition) -> np.ndarray:
        from msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = position.tolist()
        if self.use_prior(position, user_command, condition):
            self.service.user_model = None
        else:
            mean = torch.tensor(user_command.reshape(1, 2), dtype=torch.float32, device=self.device)
            covariance = torch.eye(2, device=self.device) * self.user_sigma
            self.service.user_model = distributions.MultivariateNormal(mean, covariance)

        self.service.trajectory_prediction(pose)
        if self.service.dist is None:
            return user_command.copy() if condition.kind != "rtv2_prior_only" else np.zeros(2, dtype=np.float32)

        action, _ = self.service.mpc_control(self.service.goal_position, self.service.obs_position, position)
        return clip_speed(np.asarray(action, dtype=np.float32))


def execute_trial(
    controller: RTV2Controller | None,
    environment: SharedControlBatchEnv,
    condition: Condition,
    trial_seed: int,
    noise_std: float,
    max_steps: int,
) -> dict[str, object]:
    rng = np.random.default_rng(trial_seed)
    goals, obstacles = environment.reset(trial_seed)
    true_goal = goals[0]
    if controller is not None:
        controller.reset_trial(goals, obstacles)

    for step in range(1, max_steps + 1):
        position = environment.position()
        user_command = simulated_user_command(position, true_goal, rng, noise_std)
        if condition.kind == "user_only":
            applied = user_command
        else:
            assert controller is not None
            rtv2_action = controller.action(position, user_command, condition)
            if condition.kind == "fixed_blend":
                applied = clip_speed((1.0 - condition.blend_weight) * user_command + condition.blend_weight * rtv2_action)
            else:
                applied = rtv2_action

        environment.move(applied)
        if environment.is_collision():
            return {"outcome": "collision", "steps": step, "seed": trial_seed}
        if environment.is_true_goal_achieved():
            return {"outcome": "success", "steps": step, "seed": trial_seed}

    return {"outcome": "timeout", "steps": max_steps, "seed": trial_seed}


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["condition", "trial", "seed", "outcome", "steps"])
        writer.writeheader()
        writer.writerows(results)

    summary = []
    for condition in sorted({str(result["condition"]) for result in results}):
        rows = [result for result in results if result["condition"] == condition]
        total = len(rows)
        summary.append(
            {
                "condition": condition,
                "trials": total,
                "success_rate": sum(row["outcome"] == "success" for row in rows) / total,
                "collision_rate": sum(row["outcome"] == "collision" for row in rows) / total,
                "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / total,
                "mean_steps": float(np.mean([row["steps"] for row in rows])),
            }
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simulated-user shared-control comparison.")
    parser.add_argument("--trials", type=int, default=100, help="Trials per condition.")
    parser.add_argument("--noise-std", type=float, default=0.25, help="Gaussian command-noise standard deviation in m/s.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum 10 Hz control steps per trial.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; every condition receives matching trial seeds.")
    parser.add_argument("--blend-weights", default="0.25,0.5,0.75", help="RT-V2 weights for fixed blends, comma-separated.")
    parser.add_argument("--condition", help="Run one condition only, for example rtv2_full or fixed_blend_0.5.")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet scene and run at the 10 Hz control rate.")
    parser.add_argument(
        "--scene-layout",
        choices=["fixed_five_target", "random_occluded"],
        default="fixed_five_target",
        help="Scene layout: fixed five targets or the random occluded layout from sim_env.py.",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.noise_std < 0 or args.max_steps <= 0:
        parser.error("trials and max-steps must be positive; noise-std must be non-negative")

    conditions = build_conditions(parse_weights(args.blend_weights))
    if args.condition:
        conditions = [condition for condition in conditions if condition.name == args.condition]
        if not conditions:
            available = ", ".join(condition.name for condition in build_conditions(parse_weights(args.blend_weights)))
            parser.error(f"unknown condition '{args.condition}'; choose one of: {available}")

    needs_rtv2 = any(condition.kind != "user_only" for condition in conditions)
    learned_controller = RTV2Controller(args.checkpoint, args.config, 0.3) if needs_rtv2 else None
    environment = SharedControlBatchEnv(gui=args.gui, scene_layout=args.scene_layout)
    results: list[dict[str, object]] = []
    try:
        for condition in conditions:
            controller = None if condition.kind == "user_only" else learned_controller
            for trial in range(args.trials):
                row = execute_trial(controller, environment, condition, args.seed + trial, args.noise_std, args.max_steps)
                row.update({"condition": condition.name, "trial": trial})
                results.append(row)
            print(f"completed {condition.name}")
    finally:
        environment.close()

    write_results(results, args.output_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()