#!/usr/bin/env python3
"""Experiment 4: quantify BCI influence early, middle, and late in a trial."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributions as distributions
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.late_bci_influence.environment import LateInfluenceEnv  # noqa: E402


MAX_SPEED = 3.0


def clip_speed(action: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(action)
    return action if norm <= MAX_SPEED else action * MAX_SPEED / norm


def target_progress(position: np.ndarray, start: np.ndarray, true_goal: np.ndarray) -> float:
    """Return online projected progress along the start-to-true-target axis."""
    start_to_goal = true_goal - start
    distance = np.linalg.norm(start_to_goal)
    if distance == 0:
        return 1.0
    return float(np.dot(position - start, start_to_goal) / distance**2)


def phase_from_progress(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def load_recorded_bci_commands(path: Path, trial: int, max_steps: int) -> np.ndarray:
    """Load a trial's recorded `trial,step,vx,vy` BCI command sequence."""
    commands = np.zeros((max_steps, 2), dtype=np.float32)
    with path.open(newline="", encoding="utf-8") as command_file:
        for row in csv.DictReader(command_file):
            if int(row["trial"]) == trial and 1 <= int(row["step"]) <= max_steps:
                commands[int(row["step"]) - 1] = [float(row["vx"]), float(row["vy"])]
    return commands


def simulated_bci_command(position: np.ndarray, true_goal: np.ndarray) -> np.ndarray:
    """Return a goal-directed simulated BCI command from the current position."""
    direction = true_goal - position
    distance = np.linalg.norm(direction)
    nominal = np.zeros(2, dtype=np.float32) if distance == 0 else MAX_SPEED * direction / distance
    return nominal.astype(np.float32)


class ActionProbe:
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
        logger = logging.getLogger("late_bci_influence")
        logger.addHandler(logging.NullHandler())
        self.service = trajectron_service(trajectron, ph=12, method_idx=2, logger=logger, timelimit=0.0)
        self.user_covariance = user_covariance

    def reset(self, goals: np.ndarray, obstacles: np.ndarray, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.service.goal_position, self.service.obs_position = goals.copy(), obstacles.copy()
        self.service.traj_log.queue.clear()
        self.service.dist = self.service.v_dist = None
        self.service.op_count = 0
        self.service.mpc.nominal_u.zero_()

    def _predict(self, position: np.ndarray, command: np.ndarray | None) -> np.ndarray | None:
        from msg import Pose

        pose = Pose()
        pose.position.x, pose.position.y = position.tolist()
        if command is None:
            self.service.user_model = None
        else:
            mean = torch.tensor(command.reshape(1, 2), dtype=torch.float32, device=self.device)
            self.service.user_model = distributions.MultivariateNormal(mean, torch.eye(2, device=self.device) * self.user_covariance)
        self.service.trajectory_prediction(pose)
        if self.service.dist is None:
            return None
        action, _ = self.service.mpc_control(self.service.goal_position, self.service.obs_position, position)
        return clip_speed(np.asarray(action, dtype=np.float32))

    def paired_actions(self, position: np.ndarray, command: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Evaluate prior and posterior from cloned history and identical RNG state."""
        history = list(self.service.traj_log.queue)
        op_count = self.service.op_count
        nominal = self.service.mpc.nominal_u.clone()
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        prior = self._predict(position, None)
        prior_history = list(self.service.traj_log.queue)
        prior_op_count = self.service.op_count
        prior_nominal = self.service.mpc.nominal_u.clone()
        self.service.traj_log.queue.clear()
        self.service.traj_log.queue.extend(history)
        self.service.op_count, self.service.mpc.nominal_u = op_count, nominal.clone()
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, self.device)
        posterior = self._predict(position, command)
        self.service.traj_log.queue.clear()
        self.service.traj_log.queue.extend(prior_history)
        self.service.op_count = prior_op_count
        self.service.mpc.nominal_u = prior_nominal
        return prior, posterior


def dry_run(probe: ActionProbe, environment: LateInfluenceEnv, recorded_commands: np.ndarray | None, seed: int, max_steps: int) -> list[dict[str, object]]:
    goals, obstacles = environment.reset(seed)
    probe.reset(goals, obstacles, seed)
    start = environment.position().copy()
    true_goal = goals[0]
    rows = []
    for step in range(1, max_steps + 1):
        position = environment.position()
        command = simulated_bci_command(position, true_goal) if recorded_commands is None else recorded_commands[step - 1]
        progress = target_progress(position, start, true_goal)
        prior, posterior = probe.paired_actions(position, command)
        action = prior if prior is not None else command
        environment.move(action)
        if prior is not None and posterior is not None:
            delta = posterior - prior
            rows.append({"step": step, "progress": progress, "phase": phase_from_progress(progress), "prior_x": prior[0], "prior_y": prior[1], "posterior_x": posterior[0], "posterior_y": posterior[1], "action_difference": float(np.linalg.norm(delta)), "command_alignment": float(np.dot(delta, command) / (np.linalg.norm(delta) * np.linalg.norm(command) + 1e-8))})
        if environment.terminal_outcome() is not None:
            break
    return rows


def execute_perturbation(probe: ActionProbe, environment: LateInfluenceEnv, recorded_commands: np.ndarray | None, seed: int, max_steps: int, direction: np.ndarray, magnitude: float, perturb: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    goals, obstacles = environment.reset(seed)
    probe.reset(goals, obstacles, seed)
    start = environment.position().copy()
    true_goal = goals[0]
    late_position = None
    late_activated = False
    late_trace: list[dict[str, object]] = []
    for step in range(1, max_steps + 1):
        position = environment.position()
        command = simulated_bci_command(position, true_goal) if recorded_commands is None else recorded_commands[step - 1]
        progress = target_progress(position, start, true_goal)
        if progress >= 2.0 / 3.0 and not late_activated:
            late_activated = True
            late_position = position.copy()
        if perturb and late_activated:
            command = magnitude * direction
        action = probe._predict(position, command)
        environment.move(action if action is not None else command)
        if late_activated:
            late_trace.append({"late_step": len(late_trace) + 1, "position_x": float(environment.position()[0]), "position_y": float(environment.position()[1])})
        outcome = environment.terminal_outcome()
        if outcome is not None:
            break
    final_position = environment.position()
    return ({"outcome": outcome or "timeout", "steps": step, "late_activated": late_activated, "late_start_progress": target_progress(late_position, start, true_goal) if late_position is not None else None, "final_x": float(final_position[0]), "final_y": float(final_position[1])}, late_trace)


def trial_mean(rows: list[dict[str, object]], value_key: str) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for row in rows:
        values.setdefault(int(row["trial"]), []).append(float(row[value_key]))
    return {trial: float(np.mean(samples)) for trial, samples in values.items()}


def one_sample_statistics(values: np.ndarray, null_mean: float, label: str) -> dict[str, object]:
    """Return a two-sided one-sample t test and 95% t confidence interval."""
    sample_count = len(values)
    mean = float(np.mean(values)) if sample_count else math.nan
    if sample_count < 2:
        return {"comparison": label, "n_trials": sample_count, "mean": mean, "ci95_low": math.nan, "ci95_high": math.nan, "t_statistic": math.nan, "p_value_two_sided": math.nan, "p_value_greater_than_zero": math.nan}
    standard_error = float(stats.sem(values))
    critical_value = float(stats.t.ppf(0.975, df=sample_count - 1))
    t_statistic, p_value = stats.ttest_1samp(values, popmean=null_mean)
    return {"comparison": label, "n_trials": sample_count, "mean": mean, "ci95_low": mean - critical_value * standard_error, "ci95_high": mean + critical_value * standard_error, "t_statistic": float(t_statistic), "p_value_two_sided": float(p_value), "p_value_greater_than_zero": float(stats.t.sf(t_statistic, df=sample_count - 1))}


def paired_trajectory_differences(late_trajectory_rows: list[dict[str, object]]) -> np.ndarray:
    """Return one post-activation paired average displacement error per trial."""
    trajectories: dict[int, dict[str, list[dict[str, object]]]] = {}
    for row in late_trajectory_rows:
        trajectories.setdefault(int(row["trial"]), {}).setdefault(str(row["condition"]), []).append(row)
    trial_differences = []
    for conditions in trajectories.values():
        if "baseline" not in conditions or "perturbed" not in conditions:
            continue
        baseline = sorted(conditions["baseline"], key=lambda row: int(row["late_step"]))
        perturbed = sorted(conditions["perturbed"], key=lambda row: int(row["late_step"]))
        common_steps = min(len(baseline), len(perturbed))
        if common_steps:
            baseline_positions = np.asarray([[row["position_x"], row["position_y"]] for row in baseline[:common_steps]], dtype=float)
            perturbed_positions = np.asarray([[row["position_x"], row["position_y"]] for row in perturbed[:common_steps]], dtype=float)
            trial_differences.append(float(np.mean(np.linalg.norm(perturbed_positions - baseline_positions, axis=1))))
    return np.asarray(trial_differences, dtype=float)


def calculate_trial_statistics(action_rows: list[dict[str, object]], late_trajectory_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    statistics_rows = []
    paired_difference = paired_trajectory_differences(late_trajectory_rows)
    statistics_rows.append(one_sample_statistics(paired_difference, 0.0, "perturbed_vs_baseline_post_activation_ade"))
    for segment in ("early", "middle", "late"):
        rows = [row for row in action_rows if row["phase"] == segment]
        per_trial_alignment = np.asarray(list(trial_mean(rows, "command_alignment").values()), dtype=float)
        statistics_rows.append(one_sample_statistics(per_trial_alignment, 0.0, f"{segment}_command_alignment_above_zero"))
    return statistics_rows


def write_results(action_rows: list[dict[str, object]], perturbation_rows: list[dict[str, object]], late_trajectory_rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "action_trace.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(action_rows[0]))
        writer.writeheader()
        writer.writerows(action_rows)
    with (output_dir / "late_perturbation.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(perturbation_rows[0]))
        writer.writeheader()
        writer.writerows(perturbation_rows)
    with (output_dir / "late_trajectory.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["trial", "system_seed", "condition", "late_step", "position_x", "position_y"])
        writer.writeheader()
        writer.writerows(late_trajectory_rows)
    summary = []
    for segment in ("early", "middle", "late"):
        rows = [row for row in action_rows if row["phase"] == segment]
        summary.append({"analysis": "action_difference", "phase": segment, "trials_or_steps": len(rows), "mean_action_difference": float(np.mean([row["action_difference"] for row in rows])), "mean_command_alignment": float(np.mean([row["command_alignment"] for row in rows]))})
    paired_difference = paired_trajectory_differences(late_trajectory_rows)
    baseline_rows = [row for row in perturbation_rows if row["condition"] == "baseline"]
    perturbed_rows = [row for row in perturbation_rows if row["condition"] == "perturbed"]
    summary.append({"analysis": "late_perturbation", "phase": "perturbed_vs_baseline", "trials_or_steps": len(paired_difference), "mean_paired_post_activation_ade": float(np.mean(paired_difference)), "baseline_success_rate": sum(row["outcome"] == "success" for row in baseline_rows) / len(baseline_rows), "perturbed_success_rate": sum(row["outcome"] == "success" for row in perturbed_rows) / len(perturbed_rows)})
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        fields = sorted({key for row in summary for key in row})
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    trial_statistics = calculate_trial_statistics(action_rows, late_trajectory_rows)
    with (output_dir / "trial_statistics.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(trial_statistics[0]))
        writer.writeheader()
        writer.writerows(trial_statistics)
    with (output_dir / "trial_statistics.json").open("w", encoding="utf-8") as output_file:
        json.dump(trial_statistics, output_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure late-trial BCI influence on RT-V2 actions.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bci-csv", type=Path, help="Optional trial,step,vx,vy BCI command CSV.")
    parser.add_argument("--late-direction", default="0,-1", help="Perturbation direction as dx,dy.")
    parser.add_argument("--late-magnitude", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=REPOSITORY_ROOT / "model_checkpoint.pth")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.trials <= 0 or args.max_steps < 6 or not 0 <= args.late_magnitude <= MAX_SPEED:
        parser.error("trials must be positive, max-steps must be at least 6, and late-magnitude must be between 0 and 3 m/s")
    direction = np.asarray([float(value) for value in args.late_direction.split(",")], dtype=np.float32)
    if direction.shape != (2,) or np.linalg.norm(direction) == 0:
        parser.error("late-direction must be a non-zero dx,dy pair")
    direction /= np.linalg.norm(direction)
    probe, environment = ActionProbe(args.checkpoint, args.config, 0.3), LateInfluenceEnv(gui=args.gui)
    action_rows, perturbation_rows, late_trajectory_rows = [], [], []
    try:
        for trial in range(args.trials):
            seed = args.seed + trial
            goals, _ = environment.reset(seed)
            recorded_commands = load_recorded_bci_commands(args.bci_csv, trial, args.max_steps) if args.bci_csv else None
            for row in dry_run(probe, environment, recorded_commands, seed, args.max_steps):
                row.update({"trial": trial, "system_seed": seed})
                action_rows.append(row)
            for label, perturb in (("baseline", False), ("perturbed", True)):
                row, late_trace = execute_perturbation(probe, environment, recorded_commands, seed, args.max_steps, direction, args.late_magnitude, perturb)
                row.update({"trial": trial, "system_seed": seed, "condition": label})
                perturbation_rows.append(row)
                for trace_row in late_trace:
                    trace_row.update({"trial": trial, "system_seed": seed, "condition": label})
                    late_trajectory_rows.append(trace_row)
            print(f"completed trial {trial}")
    finally:
        environment.close()
    write_results(action_rows, perturbation_rows, late_trajectory_rows, args.output_dir)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()