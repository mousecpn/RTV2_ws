#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPOSITORY_ROOT/experiments/phase_specific_assistance"
TRIALS="${TRIALS:-100}"
NOISE_STD="${NOISE_STD:-0.25}"
SWITCH_RADII="${SWITCH_RADII:-1.5,2.0,2.5,3.0}"
BLEND_WEIGHTS="${BLEND_WEIGHTS:-0.5}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-200}" 
SCENE_LAYOUT="${SCENE_LAYOUT:-random_occluded}" # random_occluded fixed_five_target
RUN_LABEL="${SCENE_LAYOUT}_phase_prior_trials_${TRIALS}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/$RUN_LABEL}"

cd "$REPOSITORY_ROOT"
IFS=',' read -r -a RADII <<< "$SWITCH_RADII"
for RADIUS in "${RADII[@]}"; do
	printf -v RADIUS_LABEL '%g' "$RADIUS"
	python -B "$EXPERIMENT_DIR/run_experiment.py" \
		--condition "phase_prior_R${RADIUS_LABEL}" \
		--trials "$TRIALS" \
		--noise-std "$NOISE_STD" \
		--switch-radii "$RADIUS" \
		--blend-weights "$BLEND_WEIGHTS" \
		--seed "$SEED" \
		--max-steps "$MAX_STEPS" \
		--scene-layout "$SCENE_LAYOUT" \
		--output-dir "$OUTPUT_DIR/radius_${RADIUS}"
done

{
	head -n 1 "$OUTPUT_DIR/radius_${RADII[0]}/summary.csv"
	for RADIUS in "${RADII[@]}"; do
		tail -n +2 "$OUTPUT_DIR/radius_${RADIUS}/summary.csv"
	done
} > "$OUTPUT_DIR/summary.csv"
python -B "$EXPERIMENT_DIR/generate_table.py" "$OUTPUT_DIR/summary.csv"

echo "Complete experiment results: $OUTPUT_DIR"