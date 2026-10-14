#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/late_bci_influence"
TRIALS="${TRIALS:-100}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED="${SEED:-0}"
LATE_DIRECTION="${LATE_DIRECTION:-0,-1}"
LATE_MAGNITUDE="${LATE_MAGNITUDE:-1.0}"
BCI_CSV="${BCI_CSV:-}"
RUN_LABEL="late_bci_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

cd "$REPOSITORY_ROOT"
COMMAND=(python -B "$EXPERIMENT_DIR/run_experiment.py" --trials "$TRIALS" --max-steps "$MAX_STEPS" --seed "$SEED" --late-direction "$LATE_DIRECTION" --late-magnitude "$LATE_MAGNITUDE" --output-dir "$OUTPUT_DIR")
if [[ -n "$BCI_CSV" ]]; then
  COMMAND+=(--bci-csv "$BCI_CSV")
fi
"${COMMAND[@]}"
python -B "$EXPERIMENT_DIR/generate_table.py" \
  "$OUTPUT_DIR/summary.csv" \
  --trial-statistics "$OUTPUT_DIR/trial_statistics.csv"

echo "Complete experiment results: $OUTPUT_DIR"
echo "Trial-level statistics: $OUTPUT_DIR/trial_statistics.csv"