#!/usr/bin/env python3
"""Render a summary table for the angular-error BCI experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    headers = ["Angular SD (deg)", "Condition", "Trials", "Success", "Collision", "Timeout", "Mean steps"]
    values = [[row["angular_sd_degrees"], row["condition"], row["trials"], *(f"{float(row[key]) * 100:.1f}%" for key in ("success_rate", "collision_rate", "timeout_rate")), f"{float(row['mean_steps']):.1f}"] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    lines = ["# Fixed-Magnitude Angular-Error BCI Summary", "", "Every simulated BCI command has magnitude 3 m/s. Its goal-directed angle is perturbed by a zero-mean Gaussian error with the listed standard deviation. The same scene seeds, unit-normal angle samples, and RT-V2/MPPI random seeds are reused at every angular-error level.", "", "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |", "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines.extend("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in values)
    output = args.output or args.summary_csv.with_name("summary_table.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote table to {output}")


if __name__ == "__main__":
    main()