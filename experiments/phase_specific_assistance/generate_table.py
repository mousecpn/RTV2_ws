#!/usr/bin/env python3
"""Render fixed-radius phase-specific assistance results as Markdown."""

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
    headers = ["Condition", "Switch R (m)", "Trials", "Success", "Collision", "Timeout", "Mean steps", "Mean prior steps"]
    values = [[row["condition"], "-" if not row["switch_radius"] else f"{float(row['switch_radius']):g}", row["trials"], *(f"{float(row[key]) * 100:.1f}%" for key in ("success_rate", "collision_rate", "timeout_rate")), f"{float(row['mean_steps']):.1f}", f"{float(row['mean_phase_prior_steps']):.1f}"] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    lines = ["# Fixed-Radius Phase-Specific Assistance Summary", "", "For `phase_prior_R`, the controller executes the user command when the current nearest-obstacle distance is at least R, and executes the 100% RT-V2 prior action when the distance is below R. The switch uses only current distance; it does not use confidence, entropy, a safety counter, or imminent-collision prediction.", "", "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |", "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines.extend("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in values)
    output = args.output or args.summary_csv.with_name("summary_table.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote table to {output}")


if __name__ == "__main__":
    main()