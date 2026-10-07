# Experiment 10 — K × n_replans grid

> **DEPRECATED for new work**: This experiment delegates to
> Exp 05's driver, which is now superseded by Exp 15
> (`experiments/15_unified_hybrid/`). The `pareto_*.json` files
> here remain valid for archival reproducibility (Exp 15
> reproduces them within ±1/60 at `--lambda-prior=0`). New K × R
> grids should target Exp 15 directly. See
> [`experiments/15_unified_hybrid/README.md`](../15_unified_hybrid/README.md)
> § Migration.

**Status:** _scaffolded._ Driver shell script reuses exp 05's
runner; downstream taxonomy already wired.

Answers a question the Exp 05b README explicitly raises but does
not resolve: at fixed wall-clock budget, is denoising deeper
(`K`) or replanning more often (`n_replans`) the better spend?
A 3 × 3 grid is enough for a single appendix figure.

## Running

Smoke (a few minutes):

```bash
KS="5" REPLANS="0 1" N_DEMOS=2 N_ROLLOUTS=1 \
    ./experiments/10_kxreplans_grid/run.sh
```

Full grid (long; ~ ∑ K(R+1) ≈ 8× single-shot wall = ~10 h):

```bash
./experiments/10_kxreplans_grid/run.sh
```

Outputs: one `K{K}_R{R}.json` per cell + `taxonomy.{json,md}`.

## Suggested headline plot

For each cell, plot `success_rate` vs.
`wall_s ∝ K * (replans + 1)`. Look for a clean Pareto frontier; if
points cluster on a single line, "K is just compute" and the grid
collapses; if `R > 0` consistently dominates `R = 0` at fixed
compute, that's the cleanest evidence for the MPC argument.
