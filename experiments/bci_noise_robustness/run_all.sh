#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/bci_noise_robustness"
TRIALS="${TRIALS:-100}"
NOISE_LEVELS="${NOISE_LEVELS:-0,0.1,0.25,0.5,0.75,1.0}"
BLEND_WEIGHTS="${BLEND_WEIGHTS:-0.5}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
SCENE_LAYOUT="${SCENE_LAYOUT:-fixed_five_target}"
RUN_LABEL="${SCENE_LAYOUT}_noise_sweep_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

cd "$REPOSITORY_ROOT"
python -B "$EXPERIMENT_DIR/run_experiment.py" --trials "$TRIALS" --noise-levels "$NOISE_LEVELS" --blend-weights "$BLEND_WEIGHTS" --seed "$SEED" --max-steps "$MAX_STEPS" --scene-layout "$SCENE_LAYOUT" --output-dir "$OUTPUT_DIR"
python -B "$EXPERIMENT_DIR/generate_table.py" "$OUTPUT_DIR/summary.csv"

echo "Complete experiment results: $OUTPUT_DIR"