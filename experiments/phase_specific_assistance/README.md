# Fixed-Radius Phase-Specific Assistance

This reviewer-requested experiment compares a hard-coded assistance switch with
the existing shared-control baselines on matched trial seeds.

For a fixed radius $R$, the phase-specific controller calculates the current
two-dimensional distance $d_t$ to the nearest obstacle and executes

$$
a_t =
\begin{cases}
u_{\mathrm{RT\text{-}V2\ prior}}, & d_t < R, \\
u_{\mathrm{user}}, & d_t \ge R.
\end{cases}
$$

This decision depends only on current nearest-obstacle distance. It does not
use confidence, entropy, a safety counter, predicted collision distance, or
hysteresis. Outside the radius, RT-V2 still observes the current position and
user command to retain a comparable history, but its action is not executed.

Default radii are `1.5, 2.0, 2.5, 3.0 m`. The same run reports `user_only`,
`rtv2_full` with its existing safety behavior, `fixed_blend_0.5`, and
`rtv2_prior_only` for direct comparison.

## Run

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/phase_specific_assistance/run_all.sh
experiments/phase_specific_assistance/run_all.sh
```

Use a custom set of radii:

```bash
SWITCH_RADII=1.3,1.5,2.0,2.5,3.0 TRIALS=200 \
  experiments/phase_specific_assistance/run_all.sh
```

Inspect one radius in the GUI:

```bash
python experiments/phase_specific_assistance/run_experiment.py \
  --condition phase_prior_R2.5 \
  --switch-radii 2.5 \
  --gui --trials 1
```

The result directory contains `trial_results.csv`, `summary.csv`,
`summary.json`, and `summary_table.md`. The table reports success, collision,
timeout, mean episode length, and mean number of prior-controlled steps.