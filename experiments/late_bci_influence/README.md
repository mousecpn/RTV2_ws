# Does BCI Influence Movement Later in the Trial?

Experiment 4 contains two matched analyses in the fixed five-target scene.

1. **Prior dry run / action probe:** the trial is executed with RT-V2 prior
   actions only. At every state, the code evaluates the prior action and the
   posterior action conditioned on the BCI command from the same history and
   random-number state. It reports their Euclidean difference and its alignment
   with the BCI command in early, middle, and late geometric-progress phases.
   At each control step, progress is the current position projected onto the
   start-to-true-target direction, divided by the start-to-target distance.
   The phases are `0-33%`, `33-67%`, and `67-100%` projected progress.
2. **Late-command perturbation:** the same BCI command sequence is replayed
   twice. Once projected progress first reaches `67%`, the perturbed replay
   replaces the **BCI command input** with a sustained fixed-direction velocity
   and passes it to RT-V2 to obtain the executed posterior action.
   The report aligns paired baseline and perturbed trajectories from activation,
   trims them to their shared duration, and reports their post-activation
   average displacement error (ADE), as well as their success rates.

By default the BCI dry run is deterministic. At every control step, it outputs
a 3 m/s command from the current avatar position toward the true target. The
baseline uses this command throughout; the perturbed replay uses the identical
command until 67% projected progress, then continuously replaces it with the
configured fixed-direction velocity. The magnitude must be between 0 and 3 m/s.
To replay recorded BCI files,
provide a CSV with `trial,step,vx,vy` columns through `--bci-csv`.

## Run

```bash
cd /home/u0161364/Desktop/RTV2_ws
chmod +x experiments/late_bci_influence/run_all.sh
experiments/late_bci_influence/run_all.sh
```

Use recorded commands and a downward late perturbation:

```bash
BCI_CSV=/absolute/path/to/bci_commands.csv LATE_DIRECTION=0,-1 LATE_MAGNITUDE=1.0 \
  experiments/late_bci_influence/run_all.sh
```

For a single GUI trial:

```bash
python experiments/late_bci_influence/run_experiment.py --trials 1 --gui
```

Output files are `action_trace.csv`, `late_perturbation.csv`, `summary.csv`,
`summary.json`, `trial_statistics.csv`, `trial_statistics.json`, and
`summary_table.md`. The statistical output uses trial-level independent
observations: paired perturbed-minus-baseline directional displacement, and
one within-phase mean command-alignment value per trial. It reports two-sided
tests and t-based 95% confidence intervals; command-alignment rows also report
the one-sided p-value for whether mean alignment is greater than zero.