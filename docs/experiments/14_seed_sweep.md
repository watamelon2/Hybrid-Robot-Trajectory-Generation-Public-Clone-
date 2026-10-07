# Experiment 14 — 3-seed pool for Exp 04 / 05 / 05b

**Lifecycle:** **scaffolded**. Driver + Wilson-CI pooler ship in
this branch. Has been **superseded for the canonical hybrid** by
the seed pools embedded in exp 15 (Job C, Slate E).

**In-tree deep doc:**
[`experiments/14_seed_sweep/README.md`](../../experiments/14_seed_sweep/README.md).

## Purpose

Pools `n = 180` per method across seeds `{0, 1, 2}` for the three
exp-04 / exp-05 / exp-05b cells. The motivation is the
n=60-rollout protocol is **underpowered**: it can't statistically
separate 4 / 60 from 1 / 60 (Fisher `p ≈ 0.36`). 180-rollout
pools give Wilson 95 % CIs tight enough to make small effects
detectable.

This is **Tier 2c** in [`docs/plans.md`](../plans.md). Runs exp 05
/ 05b with the `--deterministic-bc-reset` flag so the BC-alone-arm
RNG anomaly (see [`05_hybrid_method.md`](05_hybrid_method.md)) is
already retired in this sweep.

## Expected output

The Wilson CI on a hypothetical 12 / 180 (6.7 %) rate gives
**[3.9 %, 11.3 %]**, strictly disjoint from 0 / 180's
**[0, 2.1 %]**. That separation is the goal — being able to
report "MPC R=1 is statistically distinguishable from
single-shot at α=0.05" or, if the data doesn't support it,
documenting that the n=60 14.8 % strong-reach → success rate
was likely variance.

## Reproduce

```bash
# Override SEEDS to taste.
SEEDS="0 1 2" ./experiments/14_seed_sweep/run.sh
```

Approximate wall (sequential sim, MacBook):

| method            | per-seed | × 3        |
|-------------------|---------:|-----------:|
| Exp 04            | 78 min   | 3.9 h      |
| Exp 05 single     | 81 min   | 4.1 h      |
| Exp 05b MPC=1     | 222 min  | 11.1 h     |
| **total**         |          | **~19 h**  |

Run overnight; the script is idempotent (existing JSONs are
overwritten so a partial re-run finishes cleanly).

Outputs:

```
experiments/14_seed_sweep/results/
  exp04_seed0.json    exp04_seed1.json    exp04_seed2.json
  exp05_seed0.json    exp05_seed1.json    exp05_seed2.json
  exp05b_seed0.json   exp05b_seed1.json   exp05b_seed2.json
  summary.json        # per-method pooled k/n + Wilson 95 % CI
  taxonomy.json       # per-rollout bucketisation across all 9 files
```

### Standalone pool re-aggregation

```bash
python experiments/14_seed_sweep/pool.py \
    --in experiments/14_seed_sweep/results \
    --out experiments/14_seed_sweep/results/summary.json
```

## Files

| Path | Role |
|------|------|
| [`experiments/14_seed_sweep/run.sh`](../../experiments/14_seed_sweep/run.sh) | Driver shell. Loops over `SEEDS` × `{04, 05, 05b}`. |
| [`experiments/14_seed_sweep/pool.py`](../../experiments/14_seed_sweep/pool.py) | Wilson-CI pooler (consumes per-seed JSONs). |

## Cross-references

- **Pooled in canonical driver**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) Slate E ran a
  4-seed pool on the canonical exp-15 cell (177 / 240 = 73.8 %).
  Job B (May 14) was an earlier 4-seed pool on the exp 09 σ=0.03
  M=16 cell (91 / 240 = 37.9 %).
- **Motivation**: [`05_hybrid_method.md`](05_hybrid_method.md)
  Ablation 05b's Fisher-exact `p = 0.36`.
- **Companion analysis**:
  [`06_failure_taxonomy.md`](06_failure_taxonomy.md) over the 9
  output JSONs yields a per-seed × per-method behaviour
  distribution.

## Status / open questions

- [x] Scripts ship.
- [ ] Run the full 19 h overnight pool and commit the JSONs +
      summary.
- [ ] If MPC R=1 separates from single-shot at α=0.05, update
      [`docs/results.md`](../results.md). If not, the n=60
      14.8 % strong-reach→success was likely variance and the
      MPC paragraph should be softened.
