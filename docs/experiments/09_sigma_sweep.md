# Experiment 09 — σ-max sweep on the BC warm-start hybrid

**Lifecycle:** **superseded → [`15_unified_hybrid`](15_unified_hybrid.md)**.
The σ-sweep cells executed under this experiment are now reproduced
in exp 15 (Slate E `--sigma-tail` cells). The `sigma_*.json` files
in this directory remain valid for archival reproducibility.

**In-tree deep doc:**
[`experiments/09_sigma_sweep/README.md`](../../experiments/09_sigma_sweep/README.md).

## Purpose

Test the "preserve BC structure vs. give the planner room to
explore" trade-off introduced informally in exp 05's README.
Under the BC warm-start, σ_max controls how much the planner can
perturb the BC trajectory at iteration 1. If σ_max is too small,
the planner inherits BC's 6.7 % ceiling unchanged; if too large,
the BC is overwritten and the planner descends back to exp 04's
0 / 60.

This experiment sweeps σ_max ∈ {0.02, 0.03, 0.05, 0.10, 0.15,
0.20, 0.25, 0.30, 0.50}, 9 cells, on the exp-05 driver.

This is **Branch A — A1.4** in
[`docs/branch_plan.md`](../branch_plan.md).

## Headline result

| Cell                                                | Success      |
|-----------------------------------------------------|--------------|
| σ=0.05 M=64 K=10 R=1, seed 0 (BC + score-bridge λ=5) | **30 / 60 = 50 %** (cross-branch peak) |
| σ=0.03 M=16, 4-seed pool (Job B)                    | **91 / 240 = 37.9 %** (CI [31.6, 43.8]) |
| σ=0.25 M=16, single-seed (exp-05 default)           | 1 / 60 = 1.7 % |
| σ=0.50 M=16, single-seed (exp-04 baseline)          | 0 / 60 = 0 % |

The **50 % cross-branch peak** at σ=0.05 M=64 is recorded in
[`docs/results.md`](../results.md) as one of the better
single-seed cells before the May-17 retrieval-prior pivot.

The hypothesis was a U-curve: small σ freezes the BC trajectory
and inherits its ceiling; large σ erases BC and descends back
toward 0 / 60. The data is consistent with the U-curve at this
single seed; the 4-seed pool moderates the peak.

## Reproduce

```bash
# Smoke (one JSON per σ, 2×1 protocol, ~5 min total).
N_DEMOS=2 N_ROLLOUTS=1 ./experiments/09_sigma_sweep/run.sh

# Headline 20×3 protocol (long; budget ~7 h on M-series).
./experiments/09_sigma_sweep/run.sh
```

Outputs land in `experiments/09_sigma_sweep/results/sigma_<value>.json`
plus `results/taxonomy.{json,md}` so the per-σ behaviour
distribution is one markdown table.

Or use the canonical successor:

```bash
# Exp 15 Slate-E sigma-tail equivalent.
for s in 0.05 0.10 0.15 0.20 0.25 0.30 0.50; do
  python experiments/15_unified_hybrid/run_can.py \
      --prior bc --lambda-prior 5 --mpc-replans 0 \
      --K 10 --M 64 --sigma-max $s \
      --n-demos 20 --n-rollouts 3 --seed 0 \
      --output experiments/15_unified_hybrid/results/sigma_${s}.json
done
```

## Files

| Path | Role |
|------|------|
| [`experiments/09_sigma_sweep/run.sh`](../../experiments/09_sigma_sweep/run.sh) | Driver shell. Delegates to exp 05's `run_can.py`. |
| `experiments/09_sigma_sweep/results/_archive/sigma_*.json` | Per-σ result JSONs. |
| `experiments/09_sigma_sweep/results/taxonomy.{json,md}` | Per-σ behaviour distribution. |

## Cross-references

- **Predecessors**:
  [`04_mbd_baseline.md`](04_mbd_baseline.md),
  [`05_hybrid_method.md`](05_hybrid_method.md).
- **Canonical successor**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md). The Slate E
  `--sigma-tail` cells reproduce these results at the unified
  driver.
- **Branch context**:
  Branch A roadmap in [`docs/branch_plan.md`](../branch_plan.md)
  (A1.4 = "σ-sweep on the unified driver at K=10 R=1 λ=5", the
  promotion of this experiment to exp 15).
