# Experiment 14 — 3-seed pool for Exp 04 / 05 / 05b

**Status:** _scaffolded._ Driver shell + Wilson-CI pooler ship in
this branch.

This is **Tier 2c** in `docs/plans.md`. Pools `n=180` per method
across seeds {0, 1, 2}. Runs Exp 05 / 05b with the new
`--deterministic-bc-reset` flag so the BC-alone-arm bug is
already retired in this sweep.

## Running

```bash
SEEDS="0 1 2" ./experiments/14_seed_sweep/run.sh
```

Approximate wall (sequential sim, MacBook):

| method            | per-seed | x3      |
| ----------------- | -------: | ------: |
| Exp 04            | 78 min   | 3.9 h   |
| Exp 05 single     | 81 min   | 4.1 h   |
| Exp 05b MPC=1     | 222 min  | 11.1 h  |
| **total**         |          | **~19 h** |

Run overnight; the script is idempotent (existing JSONs are
overwritten so a partial re-run finishes cleanly).

## Outputs

```
results/
  exp04_seed0.json    exp04_seed1.json    exp04_seed2.json
  exp05_seed0.json    exp05_seed1.json    exp05_seed2.json
  exp05b_seed0.json   exp05b_seed1.json   exp05b_seed2.json
  summary.json        # per-method pooled k/n + Wilson 95% CI
  taxonomy.json       # per-rollout bucketisation across all 9 files
```

The Wilson CI sanity check on the published 12/180 hypothetical
rate gives [3.9 %, 11.3 %], strictly disjoint from 0/180's
[0, 2.1 %]. That separation is the goal of this experiment.

## Pool standalone

```bash
python experiments/14_seed_sweep/pool.py \
    --in experiments/14_seed_sweep/results \
    --out experiments/14_seed_sweep/results/summary.json
```
