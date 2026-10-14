# Experiment 4: Late BCI Influence Protocol

## Purpose

Experiment 4 tests two related questions about RT-V2 shared control in a
goal-directed navigation task:

1. Does conditioning RT-V2 on a BCI command change its action at early,
   middle, and late stages of a trial?
2. If a BCI command is changed only after the avatar has made substantial
   progress toward the target, do the resulting trajectories diverge from an
   otherwise identical baseline trajectory?

The experiment is implemented by `run_experiment.py` in this directory.

## Scene

Each trial uses a planar PyBullet scene with a kinematic avatar starting at the
origin, five candidate targets, and one obstacle. The candidate target
coordinates are:

| Target | World position (m) |
|---|---:|
| 1 | `(6.0, -7.0)` |
| 2 | `(8.5, -3.5)` |
| 3 | `(9.2, 0.0)` |
| 4 | `(8.5, 3.5)` |
| 5 | `(6.0, 7.0)` |

At reset, the five targets are deterministically shuffled using the trial
seed. The first target in the shuffled order is the true target; the remaining
four are distractor targets. The obstacle is placed on the ray from the origin
to the true target:

$$
o = \frac{g}{s}, \qquad s \sim U(1.6, 2.4),
$$

where $g$ is the true target position. Therefore, the obstacle is placed
between the start and the true target, rather than at an unrelated random
location.

The environment runs at 10 Hz ($\Delta t=0.1$ seconds). A trial ends when one
of the following occurs:

- **Success:** the avatar is within $1.0$ m of the true target.
- **Collision:** the avatar is within $1.3$ m of the obstacle.
- **Timeout:** neither condition occurs within `max_steps` control steps
  (200 by default).

## Simulated User Input

Unless `--bci-csv` is supplied, the simulated user/BCI input is deterministic
and re-aims toward the true target at every control step. Given avatar position
$p_t$ and true target $g$, the command is

$$
u_t =
\begin{cases}
3\dfrac{g-p_t}{\lVert g-p_t\rVert_2}, & p_t \ne g, \\
0, & p_t = g.
\end{cases}
$$

The command has a maximum magnitude of $3$ m/s and contains no simulated
noise. Consequently, a baseline trial represents continuous, goal-directed
user intent rather than a command fixed at the initial heading.

## RT-V2 Action Computation


The experiment explicitly compares two RT-V2 modes:

- **Prior action:** RT-V2 is evaluated with no user measurement.
- **Posterior action:** RT-V2 is evaluated after conditioning on the current
  BCI command.

The implementation preserves and restores the controller history, MPPI nominal
control sequence, and CPU/CUDA random-number states between the two evaluations.
Thus, a prior/posterior pair is evaluated from the same state and controller
history; the BCI measurement is the intended difference.

## Progress Phases

Progress is measured by projecting the current avatar position onto the axis
from the start position $p_0$ to the true target:

$$
r_t = \frac{(p_t-p_0)^\top(g-p_0)}{\lVert g-p_0\rVert_2^2}.
$$

The phases are defined from this online projected progress:

| Phase | Progress interval |
|---|---:|
| Early | $r_t < 1/3$ |
| Middle | $1/3 \le r_t < 2/3$ |
| Late | $r_t \ge 2/3$ |

Projected progress is used rather than elapsed time, so the phase definition
remains tied to physical advancement toward the true target.

## Analysis A: Prior-versus-Posterior Action Probe

This analysis follows a **prior-only rollout**. At each visited state, it
evaluates both the prior action $a_t^{\mathrm{prior}}$ and the BCI-conditioned
posterior action $a_t^{\mathrm{posterior}}$, but advances the environment with
the prior action. This keeps the sampled states independent of the posterior
action being measured.

For every valid pair of actions, the experiment records two step-level
quantities.

### Action Difference

$$
\delta_t =
\left\|a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}}\right\|_2.
$$

This is measured in m/s. A value near zero means the BCI measurement had little
effect on the RT-V2 action at that state. A larger value means the BCI
measurement changed the controller output more strongly.

### Command Alignment

$$
c_t =
\frac{
\left(a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}}\right)^\top u_t
}{
\left\|a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}}\right\|_2
\left\|u_t\right\|_2 + 10^{-8}
}.
$$

This is a cosine similarity in approximately $[-1,1]$:

- $c_t > 0$: the BCI shifts the RT-V2 action in the requested command
  direction.
- $c_t \approx 0$: the shift is orthogonal to the command or negligible.
- $c_t < 0$: the BCI shifts the action opposite to the command direction.

For descriptive output, the experiment reports the mean step-level action
difference and command alignment within early, middle, and late phases.

For inferential statistics, it first averages command alignment within each
trial and phase. Each trial therefore contributes at most one observation per
phase. This avoids treating correlated control steps from a single trajectory
as independent observations.

## Analysis B: Late Command Perturbation

This analysis compares two matched posterior-controlled rollouts for every
trial:

- **Baseline:** the user command remains goal-directed throughout the trial.
- **Perturbed:** the user command is identical to the baseline command before
  late activation. Once $r_t \ge 2/3$, it receives a sustained fixed-direction
  replacement command.

The perturbed user command is

$$
\tilde{u}_t =
\begin{cases}
u_t, & r_t < 2/3, \\
m d, & r_t \ge 2/3,
\end{cases}
$$

where $d$ is the normalized `--late-direction` vector and $m$ is
`--late-magnitude`. The defaults are $d=(0,-1)$ and $m=1.0$ m/s; $m$ is
restricted to $[0,3]$ m/s, so the replacement command requires no clipping.
The direction selects the input manipulation; it is **not** used to project the
trajectory effect metric.

Both members of a pair use the same trial seed, scene, true target, obstacle,
controller seed, and unperturbed user input. The intended intervention is only
the sustained replacement of the perturbed user command after activation.

## Primary Outcome: Post-Activation Paired ADE

The primary late-perturbation outcome is the post-activation paired average
displacement error (ADE). It measures the average physical separation between
the perturbed and baseline trajectories after late activation.

For trial $i$, let $p^{B}_{i,k}$ and $p^{P}_{i,k}$ be the baseline and
perturbed positions at matched relative post-activation step $k$. If one
trajectory terminates earlier, both trajectories are trimmed to the shorter
post-activation length:

$$
K_i = \min(T_i^B, T_i^P).
$$

The trial-level ADE is

$$
\mathrm{ADE}_i = \frac{1}{K_i}\sum_{k=1}^{K_i}
\left\|p^{P}_{i,k}-p^{B}_{i,k}\right\|_2.
$$

ADE is measured in metres:

- **ADE = 0 m:** the matched trajectories are identical after activation.
- **Larger ADE:** the sustained BCI perturbation causes a larger trajectory
  divergence from the baseline.
- **No sign:** ADE is an Euclidean distance, not a directional displacement.
  It measures magnitude of divergence regardless of whether the perturbed
  trajectory moves up, down, left, or right relative to baseline.

The experiment stores the post-activation positions in `late_trajectory.csv`.
Its rows contain `trial`, `system_seed`, `condition`, `late_step`, `position_x`,
and `position_y`, so the ADE calculation can be reproduced independently.

## Statistical Analysis

The independent unit for all reported tests is the trial.

For the late perturbation analysis, one $\mathrm{ADE}_i$ value is computed for
each matched pair. The experiment tests
whether the mean ADE differs from zero with a two-sided one-sample $t$ test:

$$
t = \frac{\overline{\mathrm{ADE}}}
{s_{\mathrm{ADE}}/\sqrt{n}},
$$

with $n-1$ degrees of freedom. It reports the mean ADE, a two-sided 95% t-based
confidence interval, the $t$ statistic, and the two-sided p-value. This is
reported as `perturbed_vs_baseline_post_activation_ade` in
`trial_statistics.csv`.

For each action-probe phase, one within-trial mean command alignment is tested
against zero using a one-sample t test. The outputs include a two-sided p-value
and a one-sided p-value for the directional alternative
$H_1: \mu_{\mathrm{alignment}}>0$.
