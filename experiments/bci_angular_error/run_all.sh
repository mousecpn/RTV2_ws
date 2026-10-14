#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/bci_angular_error"
TRIALS="${TRIALS:-100}"
ANGULAR_SDS="${ANGULAR_SDS:-0,10,20,40,60}"
BLEND_WEIGHTS="${BLEND_WEIGHTS:-0.5}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
SCENE_LAYOUT="${SCENE_LAYOUT:-random_occluded}" # random_occluded fixed_five_target
RUN_LABEL="${SCENE_LAYOUT}_angular_error_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

cd "$REPOSITORY_ROOT"
python -B "$EXPERIMENT_DIR/run_experiment.py" --trials "$TRIALS" --angular-sds "$ANGULAR_SDS" --blend-weights "$BLEND_WEIGHTS" --seed "$SEED" --max-steps "$MAX_STEPS" --scene-layout "$SCENE_LAYOUT" --output-dir "$OUTPUT_DIR"
python -B "$EXPERIMENT_DIR/generate_table.py" "$OUTPUT_DIR/summary.csv"

echo "Complete experiment results: $OUTPUT_DIR"