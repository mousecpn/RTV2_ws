#!/usr/bin/env python3
"""Generate a Markdown table for candidate-target experiment results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a candidate-target result table.")
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    headers = ["Condition", "Candidates", "Trials", "Correct target", "Any target", "Wrong target", "Timeout", "Mean steps"]
    values = [
        [row["condition"], row["candidate_count"], row["trials"], *(f"{float(row[key]) * 100:.1f}%" for key in ("correct_target_success", "any_target_success", "wrong_target_rate", "timeout_rate")), f"{float(row['mean_steps']):.1f}"]
        for row in rows
    ]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    table = ["| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"]
    table.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    table.extend("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in values)
    text = "# Candidate-Target / AI-Only Summary\n\n" + "\n".join(table) + "\n"
    output = args.output or args.summary_csv.with_name("summary_table.md")
    output.write_text(text, encoding="utf-8")
    print("\n".join(table))
    print(f"wrote table to {output}")


if __name__ == "__main__":
    main()