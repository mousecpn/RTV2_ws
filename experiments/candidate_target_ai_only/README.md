# Candidate-Target / AI-Only Experiment

This standalone experiment varies the number of visible candidate targets
while retaining exactly one true target in every trial. Candidate counts are
`1, 2, 3, 4, 5, 10, 15, 20` by default. Targets are sampled at least `6 m`
from the start and at least `1 m` apart.

`rtv2_full` conditions RT-V2 on the same simulated-user command used in the
shared-control experiment: a `3 m/s` command toward the true target with
Gaussian noise. `rtv2_prior_only` sets the user measurement to `None` and
therefore acts using the RT-V2 prior only.

The experiment ends at the first target reached. A correct-target success is a
reach of the true target. An any-target success is a reach of either the true
target or a distractor. The summary also reports wrong-target and timeout rates.

## Run All Counts

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/candidate_target_ai_only/run_all.sh
experiments/candidate_target_ai_only/run_all.sh
```

Override the default sweep or trial count:

```bash
TRIALS=200 CANDIDATE_COUNTS=1,5,10,15,20 \
  experiments/candidate_target_ai_only/run_all.sh
```

## Run One Condition With GUI

```bash
python experiments/candidate_target_ai_only/run_experiment.py \
  --condition rtv2_prior_only \
  --candidate-counts 20 \
  --gui \
  --trials 1 \
  --seed 0
```

The output directory contains `trial_results.csv`, `summary.csv`,
`summary.json`, and `summary_table.md`.