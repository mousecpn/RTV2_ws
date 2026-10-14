# Experiment 4: Methods and Statistical Analysis

This document answers the methodological questions for Experiment 4, *Does the
BCI Still Influence Movement Later in the Trial?* The method descriptions refer
to the current implementation in `run_experiment.py`.

## 1. What Is the Primary Late-Perturbation Outcome?

**Answer:** The primary outcome is the paired post-activation average
displacement error (ADE), not a signed or directional displacement.

Each trial has one baseline rollout and one perturbed rollout with the same
trial index and system seed. They start from the same state and use the same
target permutation, true target, obstacle placement, controller seed, and user
input until late activation. Their post-activation positions are recorded at
every control step in `late_trajectory.csv`.

Let $p^B_{i,k}$ and $p^P_{i,k}$ be the baseline and perturbed positions for
trial $i$ at relative post-activation step $k$. The trajectories are trimmed to
their shared length,

$$
K_i = \min(T_i^B, T_i^P),
$$

and the trial-level paired ADE is

$$
\mathrm{ADE}_i = \frac{1}{K_i}\sum_{k=1}^{K_i}
\left\|p^P_{i,k}-p^B_{i,k}\right\|_2.
$$

ADE is measured in metres. An ADE of zero means that the matched trajectories
coincide after activation. A larger ADE means that the late BCI manipulation
causes greater average physical divergence from baseline. ADE has no sign and
does not determine whether the divergence improves or degrades task success.

## 2. How Are the ADE Statistics Calculated?

**Answer:** The independent observations are trial-level ADE values, not
individual time steps. One $\mathrm{ADE}_i$ is calculated for each matched
pair, and a two-sided one-sample $t$ test against zero is performed:

$$
t = \frac{\overline{\mathrm{ADE}}}
{s_{\mathrm{ADE}}/\sqrt{n}}.
$$

The reported 95% confidence interval is

$$
\overline{\mathrm{ADE}} \pm
 t_{0.975,n-1}\frac{s_{\mathrm{ADE}}}{\sqrt{n}}.
$$

The statistic is stored under
`perturbed_vs_baseline_post_activation_ade` in `trial_statistics.csv`.

### Stored ADE Result

The current `results/late_bci_trials_100_seed_0/summary_table.md` contains the
following paired-ADE result:

| Comparison | Trials | Mean ADE | 95% CI | t | Two-sided p-value |
|---|---:|---:|---:|---:|---:|
| Perturbed vs baseline | 100 | 1.2286 m | [1.1295, 1.3277] m | 24.5996 | $5.755\times10^{-44}$ |

This says that the stored perturbed trajectories differ from their matched
baselines by an average of about $1.23$ m over their matched post-activation
steps. The stored result is significantly different from zero under the stated
trial-level test.

The result directory does not record the full perturbation-command
configuration. Because the current implementation uses pure $md$ replacement
after late activation, rerun the experiment before treating these stored values
as the final result for that exact configuration.

## 3. How Are Prior-versus-Posterior Actions Measured?

**Answer:** A separate prior-only dry run evaluates two RT-V2 actions at each
visited state:

- the **prior** action, with no user measurement; and
- the **posterior** action, conditioned on the current BCI command.

The controller history, MPPI nominal control sequence, and CPU/CUDA random
states are restored between the evaluations. The environment then advances
with the prior action only, so both actions are measured from the same visited
state and controller history.

The step-level action difference is

$$
\delta_t = \left\|a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}}\right\|_2,
$$

and command alignment is

$$
c_t =
\frac{(a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}})^\top u_t}
{\left\|a_t^{\mathrm{posterior}}-a_t^{\mathrm{prior}}\right\|_2
 \left\|u_t\right\|_2 + 10^{-8}}.
$$

Positive alignment means BCI conditioning shifts the RT-V2 action in the same
direction as the BCI command. Before statistical testing, alignments are
averaged within each trial and phase, so each trial contributes at most one
independent observation per phase.

### Stored Alignment Results

| Phase | Trial-level mean alignment | 95% CI | t | Two-sided p-value | One-sided p-value for $\mu>0$ |
|---|---:|---:|---:|---:|---:|
| Early | 0.7747 | [0.7405, 0.8089] | 44.9470 | $1.125\times10^{-67}$ | $5.627\times10^{-68}$ |
| Middle | 0.8400 | [0.8144, 0.8656] | 65.0906 | $4.540\times10^{-83}$ | $2.270\times10^{-83}$ |
| Late | 0.9147 | [0.9080, 0.9215] | 268.8080 | $1.390\times10^{-143}$ | $6.951\times10^{-144}$ |

These stored values show strong positive command alignment in every phase,
including the late phase.

## 4. How Is Trial Progress Defined?

**Answer:** Progress projects the current position $p_t$ onto the axis from the
start position $p_0$ to the true target $g$:

$$
r_t = \frac{(p_t-p_0)^\top(g-p_0)}{\lVert g-p_0\rVert_2^2}.
$$

| Phase | Projected progress |
|---|---:|
| Early | $r_t < 1/3$ |
| Middle | $1/3 \le r_t < 2/3$ |
| Late | $r_t \ge 2/3$ |

Late replacement begins at the first control step satisfying $r_t\ge2/3$.

## 5. How Is the Simulated BCI Command Generated?

**Answer:** The default simulated BCI command is deterministic and is re-aimed
toward the true target at every control step. It contains no Gaussian noise:

$$
u_t =
\begin{cases}
3\dfrac{g-p_t}{\lVert g-p_t\rVert_2}, & p_t \ne g, \\
0, & p_t = g.
\end{cases}
$$

Thus, the baseline represents continuous, closed-loop goal-directed user
intent. This differs from an implementation that holds a fixed initial heading
or adds Gaussian noise. When `--bci-csv` is supplied, the experiment instead
replays the recorded `trial,step,vx,vy` commands.

## 6. What Is the Late Command Manipulation?

**Answer:** Baseline uses the goal-directed command $u_t$ throughout the trial.
Before late activation, perturbed uses exactly the same input. From activation
onward, the perturbed input is replaced, rather than offset, by a pure fixed
velocity:

$$
\tilde{u}_t =
\begin{cases}
u_t, & r_t < 2/3, \\
m d, & r_t \ge 2/3,
\end{cases}
$$

where $d$ is the normalized `--late-direction` vector and $m$ is
`--late-magnitude`. There is no clipping in this replacement branch. Command
line validation restricts $m$ to $0\le m\le3$ m/s, so the exact command $md$
remains within the system speed limit.

The current defaults are:

- `LATE_DIRECTION=0,-1`, so $d=(0,-1)$ in the world frame;
- `LATE_MAGNITUDE=1.0`, so the late perturbed BCI input is exactly
  $(0,-1)$ m/s.

The executed avatar action is not $md$ directly. RT-V2/MPPI produces a
posterior action conditioned on that replacement BCI input.

## 7. Does Experiment 4 Use the Same Scene Geometry as Experiment 1?

**Answer:** Experiment 4 uses a separate environment implementation but the
same fixed five-target geometry as Experiment 1's `fixed_five_target`
configuration. The candidate targets are

$$
(6,-7),\ (8.5,-3.5),\ (9.2,0),\ (8.5,3.5),\ (6,7).
$$

The target order is deterministically shuffled from the trial seed; the first
shuffled target is the true target. One obstacle is placed on the ray from the
origin to that target:

$$
o = \frac{g_{\mathrm{true}}}{s}, \qquad s\sim U(1.6,2.4).
$$

Both environments use a 10 Hz kinematic avatar update.

## 8. How Many Trials Were Used?

**Answer:** The current stored summary reports 100 trials with seeds 0 through
99. A dry run stops at success, collision, or timeout; consequently, the
number of valid action pairs differs by phase. Phase-level alignment statistics
nevertheless use at most one within-phase mean per trial.

## 9. Was a Safety Override Active?

**Answer:** No. Experiment 4 does not use the `safety_count` override. It
obtains a prior action by setting the user measurement to `None` and a posterior
action by conditioning on the BCI command.

For Experiment 1 fixed blending, the safety override is also not active. The
RT-V2 component uses the prior action and blends it with the simulated user
command. The `safety_count` mechanism is used only by Experiment 1
`rtv2_full`.

## 10. What Are the Success and Collision Criteria?

**Answer:** The environment uses two-dimensional Euclidean distances:

$$
\lVert p-g_{\mathrm{true}}\rVert_2 < 1.0\ \mathrm{m}
$$

for correct-target success, and

$$
\min_o\lVert p-o\rVert_2 < 1.3\ \mathrm{m}
$$

for collision. A trial times out if neither event occurs within 200 control
steps by default.

## Reproducibility Note

The stored values above are read from
`results/late_bci_trials_100_seed_0/summary_table.md`. Re-run the experiment
after changing `LATE_DIRECTION`, `LATE_MAGNITUDE`, the trial count, or any
controller setting, and retain the run configuration with the results.