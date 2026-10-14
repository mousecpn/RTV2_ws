# Fixed-Magnitude Angular-Error BCI Experiment

This Experiment 3 variant evaluates RT-V2 under direction-only BCI errors. At
each control step, the nominal command points toward the true target with a
fixed speed of `3 m/s`. Its angle is rotated by a zero-mean Gaussian error:

$$
u_{BCI} = 3
\begin{bmatrix}
\cos(\theta_{goal} + \epsilon) \\
\sin(\theta_{goal} + \epsilon)
\end{bmatrix}, \qquad
\epsilon \sim \mathcal{N}(0, \sigma_\theta^2).
$$

Default angular standard deviations are `0, 10, 20, 40, 60` degrees. The
command magnitude is exactly `3 m/s` before it is passed to all conditions.
For each trial seed, the scene, true target, standard-normal angular-error
sequence, and RT-V2/MPPI random stream are held fixed across all error levels.

Every angular-error level compares `user_only`, `rtv2_full` with the original
three-step obstacle safety takeover, and `fixed_blend_0.5`, which blends the
angular-error command with the RT-V2 prior action.

## Run

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/bci_angular_error/run_all.sh
experiments/bci_angular_error/run_all.sh
```

Use a different sweep or write a per-step audit trace:

```bash
python experiments/bci_angular_error/run_experiment.py \
  --angular-sds 0,5,10,20,40,60 \
  --trace-dir experiments/bci_angular_error/traces \
  --trials 100
```

The output includes `trial_results.csv`, `summary.csv`, `summary.json`, and
`summary_table.md`. Per-step traces include the sampled angular error and the
actual command speed.