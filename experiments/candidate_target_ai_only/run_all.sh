#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/candidate_target_ai_only"
TRIALS="${TRIALS:-100}"
NOISE_STD="${NOISE_STD:-0.25}"
CANDIDATE_COUNTS="${CANDIDATE_COUNTS:-1,2,3,4,5,10,15,20}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
RUN_LABEL="candidates_${CANDIDATE_COUNTS//,/--}_noise_${NOISE_STD//./_}_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

cd "$REPOSITORY_ROOT"
python -B "$EXPERIMENT_DIR/run_experiment.py" \
  --trials "$TRIALS" \
  --candidate-counts "$CANDIDATE_COUNTS" \
  --noise-std "$NOISE_STD" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR"
python -B "$EXPERIMENT_DIR/generate_table.py" "$OUTPUT_DIR/summary.csv"

echo "Complete experiment results: $OUTPUT_DIR"