#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/shared_control_sim"
TRIALS="${TRIALS:-100}"
NOISE_STD="${NOISE_STD:-0.25}"
# BLEND_WEIGHTS="${BLEND_WEIGHTS:-}"
BLEND_WEIGHTS="${BLEND_WEIGHTS:-0.25,0.5,0.75,0.80,0.85,0.90,0.95}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-200}"
SCENE_LAYOUT="${SCENE_LAYOUT:-fixed_five_target}" # random_occluded fixed_five_target
RUN_LABEL="${SCENE_LAYOUT}_noise_${NOISE_STD//./_}_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

# cd "$REPOSITORY_ROOT"
# python -B "$EXPERIMENT_DIR/run_experiment.py" \
#   --trials "$TRIALS" \
#   --noise-std "$NOISE_STD" \
#   --blend-weights "$BLEND_WEIGHTS" \
#   --seed "$SEED" \
#   --max-steps "$MAX_STEPS" \
#   --scene-layout "$SCENE_LAYOUT" \
#   --output-dir "$OUTPUT_DIR"
python -B "$EXPERIMENT_DIR/generate_table.py" "$OUTPUT_DIR/summary.csv"

echo "Complete experiment results: $OUTPUT_DIR"