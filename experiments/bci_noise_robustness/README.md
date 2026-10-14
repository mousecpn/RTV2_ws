# Robustness to Poor BCI Decoding

This standalone Experiment 3 measures how shared control changes as the
simulated user's command becomes less reliable. Every command points toward
the true target with additive Gaussian noise.

For a given trial, the system seed is held fixed across all noise levels: it
fixes the occluded target/obstacle scene, the true target, the unit Gaussian
noise sequence, and PyTorch's RT-V2/MPPI sampling sequence. Noise level only
scales the same unit Gaussian command samples. This preserves $\Sigma_{sys}$
across the noise sweep.

## Scene Layouts

`random_occluded` remains the default. Set `fixed_five_target` to use the
five target positions from `sim_env.py`, select the first seed-shuffled target
as the true goal, and place one obstacle on its ray from the start. For a
given system seed, this layout is identical at every noise level.

At every noise level the suite compares `user_only`, `rtv2_full` (including
the original three-step obstacle safety takeover), and each requested
`fixed_blend_W`. Fixed blending uses
$(1-W)u_{user} + W u_{RT-V2-prior}$.

## Run

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/bci_noise_robustness/run_all.sh
experiments/bci_noise_robustness/run_all.sh
```

Default noise standard deviations are `0,0.1,0.25,0.5,0.75,1.0 m/s`; the
default fixed blend weight is `0.5`. Override them as needed:

```bash
TRIALS=200 NOISE_LEVELS=0,0.25,0.5,1.0,1.5 BLEND_WEIGHTS=0.25,0.5,0.75 \
  experiments/bci_noise_robustness/run_all.sh
```

Run the fixed five-target noise sweep:

```bash
SCENE_LAYOUT=fixed_five_target \
  experiments/bci_noise_robustness/run_all.sh
```

To view one condition at one noise level:

```bash
python experiments/bci_noise_robustness/run_experiment.py \
  --condition rtv2_full --noise-levels 1.0 --scene-layout fixed_five_target \
  --gui --trials 1 --seed 0
```

Results include `trial_results.csv`, `summary.csv`, `summary.json`, and
`summary_table.md`, with success, collision, timeout, and mean-step metrics.