# Experiment 09 — Warm-start σ-max sweep

**Status:** _scaffolded._ Driver shell script + downstream taxonomy
post-processing are wired up; runs through exp 05's existing
`run_can.py`.

Tests the "preserve BC structure vs. give the planner room to
explore" trade-off narrated informally in the Exp 05 README.
Currently only `σ_max ∈ {0.25, 0.5}` are documented; this exp
fills in the curve.

## Running

Smoke (one JSON per σ, 2×1 protocol, ~5 min total):

```bash
N_DEMOS=2 N_ROLLOUTS=1 ./experiments/09_sigma_sweep/run.sh
```

Headline 20×3 protocol (long; budget ~7 h on the MacBook):

```bash
./experiments/09_sigma_sweep/run.sh
```

Outputs land in `results/sigma_<value>.json` plus
`results/taxonomy.{json,md}` so the per-σ behaviour distribution
is one Markdown table.

## Reading the result

The hypothesis is a U: small σ "freezes" the BC trajectory and
inherits its 6.7 % BC-alone ceiling, large σ "erases" the BC and
descends back toward Exp 04's 0/60. The ablation already documents
0.25 (1/60) and 0.5 (presumably worse but never run); this fills
the curve in between.
