# Simulated-User Shared-Control Experiment

This standalone experiment runs a headless batch adapter of the shared-control
PyBullet scene in `sim_env.py`: five candidate targets, one true target
selected by a per-trial permutation, and a single obstacle on the line from
the start to the true target. It does not modify or start `sim_env.py` or
`TrajectronNode.py`.

Each 10 Hz simulated user command points toward the true target at `3 m/s` and
has independent Gaussian noise added to its two planar components. All
conditions reuse the same seed for a given trial, so target order, obstacle
placement, and user-command noise are matched.

## Conditions

`rtv2_full` conditions RT-V2 on the simulated-user command and applies its
MPPI action. It preserves the original `TrajectronNode.py` safety takeover: if
the user's next predicted position is within `1.5 m` of an obstacle, RT-V2
uses its prior for that step and the next two control steps. `user_only`
applies the simulated-user command. `fixed_blend_W` applies
$(1-W)u_{user}+W u_{RT-V2-prior}$ for each configured weight.
`rtv2_prior_only` always sets RT-V2's user measure to `None`.

## Scene Layouts

`fixed_five_target` is the default: five fixed candidate targets and one
obstacle on the true-target ray. `random_occluded` reproduces the commented
`setting_objects` configuration in `sim_env.py`: two or three targets beyond
the radius-6 boundary, three to five obstacles in the radius-3-to-6 annulus,
at least `1 m` between all objects, and the original axis-aligned occlusion
test between the start and every target.

## Run

The runner reuses the model adapter in `TrajectronNode.py`, which currently
allocates tensors with `.cuda()`; run it in the same CUDA-enabled environment
used for the RT-V2 service.

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/shared_control_sim/run_all.sh
experiments/shared_control_sim/run_all.sh
```

The script uses 100 trials, `0.25 m/s` command noise, and fixed RT-V2 blend
weights `0.25, 0.5, 0.75` by default. Override any parameter using environment
variables:

```bash
TRIALS=200 NOISE_STD=0.5 BLEND_WEIGHTS=0.1,0.3,0.5,0.7,0.9 \
  experiments/shared_control_sim/run_all.sh
```

## Run One Condition With GUI

Use `--condition` to run a single controller and `--gui` to display the
PyBullet scene. GUI mode advances at the experiment's 10 Hz control rate.

```bash
cd /home/u0161364/Desktop/RTV2_ws
python experiments/shared_control_sim/run_experiment.py \
  --condition rtv2_full \
  --scene-layout random_occluded \
  --gui \
  --trials 1 \
  --seed 0
```

For a specific fixed blend, include its weight in `--blend-weights` and use
the generated condition name:

```bash
python experiments/shared_control_sim/run_experiment.py \
  --blend-weights 0.5 \
  --condition fixed_blend_0.5 \
  --gui \
  --trials 1
```

`user_only` can be previewed without loading RT-V2 or requiring CUDA:

```bash
python experiments/shared_control_sim/run_experiment.py --condition user_only --gui --trials 1
```

The output directory contains `trial_results.csv` with one row per trial,
`summary.csv` / `summary.json`, and `summary_table.md`, a readable table with
success, collision, timeout, and mean-step metrics. Success uses the original
`0.5 m` distance threshold; collision uses the original `1.3 m` threshold. The
avatar is kinematically advanced at the shared-control environment's 10 Hz
control rate, because the source environment creates its avatar with
`useFixedBase=True`.