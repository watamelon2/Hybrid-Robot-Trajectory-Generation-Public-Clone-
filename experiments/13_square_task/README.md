# Experiment 13 — `Square` second task (Tier 3e)

**Status:** _partial scaffold._ Reproduces Exp 02 (random) and Exp 04
(MBD without prior) on Square out of the box; Exps 03 / 05 / 07
need the obs-key shim adjusted because `experiments/03_bc_baseline/dataset.py`
hardcodes `Can_pos` / `Can_quat`.

## Goal

Add one `Square (ph)` row to the headline table for every method
that lands on `main`. Answers "is the hybrid advantage
can-specific?".

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

The same flag works for exp 02 (random baseline). These two
columns can land *today*.

## What needs adapting

`experiments/03_bc_baseline/dataset.py:OBS_KEYS` hardcodes
`Can_pos` / `Can_quat` and the `_DATASET_OBJECT_SLICES` for the
Can dataset's `obs/object` layout. To run BC + the hybrid on
Square:

1. Add `SquareNut_pos`, `SquareNut_quat` to `OBS_KEYS` (or a
   sibling tuple `OBS_KEYS_SQUARE`).
2. Verify the dataset's `obs/object` layout for `square/ph`
   (it's typically `[nut_pos, nut_quat, nut_to_eef_pos,
   nut_to_eef_quat]`).
3. Add a runtime-key remap if robosuite 1.5 names the live obs
   differently (analogous to the existing `_RUNTIME_KEY_REMAP` in
   `experiments/05_hybrid_method/run_can.py`).

This is intentionally not done on this branch because the obs
shim duplicated across exp 03 / 05 / 07 / 08 wants a small refactor
first (single source of truth for the OBS_KEYS list). Tracked under
"Open items" in [`docs/results.md`](../../docs/results.md) and as
Tier 3 (e) in [`docs/plans.md`](../../docs/plans.md).

## Driver

```bash
./experiments/13_square_task/run.sh
```

runs the methods that already work (02, 04) on Square at the
20×3 protocol and writes the JSONs into `results/`.
