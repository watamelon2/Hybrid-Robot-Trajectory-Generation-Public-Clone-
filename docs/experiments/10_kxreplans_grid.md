# Experiment 10 — K × n_replans Pareto grid

**Lifecycle:** **superseded → [`15_unified_hybrid`](15_unified_hybrid.md)**.
The `pareto_*.json` files in this directory remain valid for
archival reproducibility (exp 15 reproduces them within ±1/60 at
`--lambda-prior=0`). New K × R grids should target exp 15
directly.

**In-tree deep doc:**
[`experiments/10_kxreplans_grid/README.md`](../../experiments/10_kxreplans_grid/README.md).

## Purpose

Answer a question that exp 05b's MPC ablation raises but does not
resolve: **at fixed wall-clock budget, is denoising deeper (`K`)
or replanning more often (`n_replans`) the better spend?**

A 3 × 3 grid is enough for a single appendix figure:

| K \ R   | 0 | 1 | 2 |
|---------|---|---|---|
| 5       | ⌐ | ⌐ | ⌐ |
| 10      | ⌐ | ⌐ | ⌐ |
| 20      | ⌐ | ⌐ | ⌐ |

Per-cell wall ∝ `K × (R+1)`; full 9-cell sweep is ~10 h on
M-series.

## Headline reading

Pareto frontier of `success` vs `K × (R+1)`:

- If points cluster on a single line → "K is just compute", the
  grid collapses to a 1-D sweep on total compute and you should
  pick whichever axis is cheapest.
- If `R > 0` consistently dominates `R = 0` at fixed compute →
  cleanest evidence for the MPC argument.
- If `K → ∞` saturates and `R → ∞` keeps improving → the
  bottleneck is information access (env state), not search depth.

The published K=10 R=2 cell gave **18.3 % single-seed** at λ=0
(`docs/results.md`), which informed Job C's choice to pool 4 seeds
at K=10 R=1 (Job C = 37.5 %).

## Reproduce

```bash
# Smoke.
KS="5" REPLANS="0 1" N_DEMOS=2 N_ROLLOUTS=1 \
    ./experiments/10_kxreplans_grid/run.sh

# Full grid (~10 h).
./experiments/10_kxreplans_grid/run.sh
```

Or use the canonical successor (exp 15):

```bash
# Equivalent to exp 10 K=10 R=2 at λ=0.
python experiments/15_unified_hybrid/run_can.py \
    --prior bc --lambda-prior 0 --mpc-replans 2 \
    --K 10 --M 16 --sigma-max 0.25 \
    --n-demos 20 --n-rollouts 3 --seed 0
```

Or simply use exp 15's `regression_test.py` which evaluates this
exact cell against the published 11/60 reference.

## Files

| Path | Role |
|------|------|
| [`experiments/10_kxreplans_grid/run.sh`](../../experiments/10_kxreplans_grid/run.sh) | Driver shell. Delegates to exp 05's runner. |
| [`experiments/10_kxreplans_grid/analyze.py`](../../experiments/10_kxreplans_grid/analyze.py) | Pareto plotter (`success` vs `wall_s`). |
| `experiments/10_kxreplans_grid/results/_archive/K*_R*.json` | Per-cell result JSONs. |

## Cross-references

- **Predecessors**:
  [`05_hybrid_method.md`](05_hybrid_method.md).
- **Canonical successor**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) — Phase 2 K × R
  grid (7 / 9 cells) executed at the canonical driver.
- **Re-run analysis**:
  The discrepancy between this experiment's headline K=10 R=2
  number and exp 07's λ=5 number drove
  [`docs/bc_swap_rerun_analysis.md`](../bc_swap_rerun_analysis.md).
- **Branch context**:
  Branch C roadmap in [`docs/branch_plan.md`](../branch_plan.md).
