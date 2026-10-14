#!/usr/bin/env python3
"""Render Experiment 4's action-difference and perturbation summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--trial-statistics", type=Path, help="Optional trial_statistics.csv from run_experiment.py.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    action_rows = [row for row in rows if row["analysis"] == "action_difference"]
    perturb_rows = [row for row in rows if row["analysis"] == "late_perturbation"]
    lines = ["# Late BCI Influence Summary", "", "## Prior vs Posterior", "", "Phases are assigned online from projected geometric progress toward the true target: early is below 33%, middle is 33-67%, and late is at least 67%.", "", "| Phase | Samples | Mean action difference | Mean command alignment |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['phase']} | {row['trials_or_steps']} | {float(row['mean_action_difference']):.4f} | {float(row['mean_command_alignment']):.4f} |" for row in action_rows)
    lines.extend(["", "## Late Command Perturbation", "", "The perturbation activates at 67% projected progress. For each paired trial, both post-activation trajectories are trimmed to their shared length; the effect is their average displacement error (ADE), the mean per-step Euclidean separation.", "", "| Comparison | Trials | Mean post-activation ADE | Baseline success | Perturbed success |", "|---|---:|---:|---:|---:|"])
    lines.extend(f"| {row['phase']} | {row['trials_or_steps']} | {float(row['mean_paired_post_activation_ade']):.4f} | {float(row['baseline_success_rate']) * 100:.1f}% | {float(row['perturbed_success_rate']) * 100:.1f}% |" for row in perturb_rows)
    trial_statistics_path = args.trial_statistics or args.summary_csv.with_name("trial_statistics.csv")
    if trial_statistics_path.is_file():
        with trial_statistics_path.open(newline="", encoding="utf-8") as statistics_file:
            trial_statistics = list(csv.DictReader(statistics_file))
        lines.extend(["", "## Trial-Level Statistical Tests", "", "Each trial contributes one observation: its post-activation paired ADE or its within-phase mean command alignment. The ADE comparison uses a two-sided one-sample t test against zero. Alignment rows also report the one-sided p-value for H1: mean alignment > 0. All 95% CIs use the t distribution.", "", "| Comparison | Trials | Mean | 95% CI | t | p (two-sided) | p (greater than 0) |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in trial_statistics:
            ci = f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]"
            lines.append(f"| {row['comparison']} | {row['n_trials']} | {float(row['mean']):.4f} | {ci} | {float(row['t_statistic']):.4f} | {float(row['p_value_two_sided']):.4g} | {float(row['p_value_greater_than_zero']):.4g} |")
    output = args.output or args.summary_csv.with_name("summary_table.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote table to {output}")


if __name__ == "__main__":
    main()