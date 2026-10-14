#!/usr/bin/env python3
"""Generate a Markdown table for BCI-noise robustness results."""

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
    headers = ["Noise std", "Condition", "Trials", "Success", "Collision", "Timeout", "Mean steps"]
    values = [[row["noise_std"], row["condition"], row["trials"], *(f"{float(row[key]) * 100:.1f}%" for key in ("success_rate", "collision_rate", "timeout_rate")), f"{float(row['mean_steps']):.1f}"] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    lines = ["| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |", "|" + "|".join("-" * (width + 2) for width in widths) + "|"]
    lines.extend("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in values)
    output = args.output or args.summary_csv.with_name("summary_table.md")
    output.write_text("# BCI Noise Robustness Summary\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote table to {output}")


if __name__ == "__main__":
    main()