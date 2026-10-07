# Experiment 13 — Cross-task generalisation onto `Square`

**Lifecycle:** **partial**. Random and pure-MBD baselines run out of
the box; BC + hybrid runs need an `OBS_KEYS_SQUARE` shim in
[`experiments/03_bc_baseline/dataset.py`](../../experiments/03_bc_baseline/dataset.py).
Tracked under "Open items" in [`docs/results.md`](../results.md) and
Tier 3(e) in [`docs/plans.md`](../plans.md).

The actual Square cross-task evaluations have since moved to exp 15
Slates F / G / G-aux / I, which use task-aware `OBS_KEYS`.

**In-tree deep doc:**
[`experiments/13_square_task/README.md`](../../experiments/13_square_task/README.md).

## Goal

Add one `Square (ph)` row to the headline table for every method
that lands on main. Answers **"is the hybrid advantage
Can-specific?"**

## What runs as-is

The exp 04 baseline reads `--task square`:

```bash
python -m robomimic.scripts.download_datasets \
    --tasks square --hdf5_types low_dim

python experiments/04_mbd_baseline/run_can.py \
    --task square --n-demos 20 --n-rollouts 3 \
    --seed 0 --no-live-view \
    --output experiments/13_square_task/results/exp04_square.json
```

The same flag works for exp 02 (random baseline).

## What needs adapting

[`experiments/03_bc_baseline/dataset.py:OBS_KEYS`](../../experiments/03_bc_baseline/dataset.py)
hardcodes `Can_pos` / `Can_quat` and the `_DATASET_OBJECT_SLICES`
for the Can dataset's `obs/object` layout. To run BC + hybrid on
Square:

1. Add `SquareNut_pos`, `SquareNut_quat` to `OBS_KEYS` (or a
   sibling tuple `OBS_KEYS_SQUARE`).
2. Verify the dataset's `obs/object` layout for `square/ph`
   (typically `[nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat]`).
3. Add a runtime-key remap if robosuite 1.5 names the live obs
   differently (analogous to the existing `_RUNTIME_KEY_REMAP`
   in `experiments/05_hybrid_method/run_can.py`).

Intentionally not done on this branch because the obs shim
duplicated across exps 03 / 05 / 07 / 08 wants a small refactor
first (single source of truth for `OBS_KEYS`).

## Headline results (cross-task)

The actual cross-task numbers now live in exp 15 (see
[`15_unified_hybrid.md`](15_unified_hybrid.md) and the May-22
cross-task headline in [`docs/results.md`](../results.md)):

| Task   | Best Square success                | Source |
|--------|------------------------------------|--------|
| Lift   | **100.0 %** (Slate G G1 λ=1)       | exp 15 Slate G |
| Can    | **73.8 %** pooled                  | exp 15 Slate E |
| Square | **8.3 %** σ=0.30 (Slate G-aux AUX1) | exp 15 Slate G-aux |
| Tool hang | 0/40 sparse-reward (Slate H H1) | exp 15 Slate H (sparse-reward bug; dense-reward shim landed in commit `18af830`) |

Square's mechanism is consistently weaker than Can's; the
"mechanism-limited" verdict is documented in Slate G-aux and
Slate F final readings ([`docs/slate_f_final_reading.md`](../slate_f_final_reading.md)).

## Reproduce

```bash
# The exp-13 driver runs what works today (02, 04 on Square).
./experiments/13_square_task/run.sh

# For BC + hybrid on Square, use exp 15 with the task-aware
# OBS_KEYS path (Slate F+ machinery):
python experiments/15_unified_hybrid/run_can.py \
    --task square --prior demo_1nn --lambda-prior 1 \
    --K 10 --M 16 --sigma-max 0.30 \
    --n-demos 20 --n-rollouts 3 --seed 0
```

## Files

| Path | Role |
|------|------|
| [`experiments/13_square_task/run.sh`](../../experiments/13_square_task/run.sh) | Runs the methods that already work (02, 04) on Square at the 20×3 protocol. |

## Cross-references

- **Refactor blockers**:
  [`03_bc_baseline.md`](03_bc_baseline.md) (the OBS_KEYS
  hardcoding).
- **Canonical successor**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) Slates F / G /
  G-aux / I cover Square cross-task; [`priors_zerotrain.py`](../../experiments/15_unified_hybrid/priors_zerotrain.py)
  uses task-aware OBS_KEYS via `get_obs_keys(task)`.
- **Reading**:
  [`docs/slate_f_final_reading.md`](../slate_f_final_reading.md)
  documents the "Square mech-limit" verdict.
