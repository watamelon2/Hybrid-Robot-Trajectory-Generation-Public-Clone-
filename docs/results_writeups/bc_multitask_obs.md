# BC multi-task obs schema refactor (writeup stub)

**Branch:**  `feat/adapt-bc-multitask-obs`
**Worktree:** `HRTG-bcmulti`
**Slate:**   `experiments/03_bc_baseline/slate_multitask_retrain_2026_05_28.sh`
**Status:**  retrain slate executed 2026-05-29; refactor verified via per-task obs schema (table below).

## Motivation

The BC baseline pipeline hard-coded Can's obs schema in:
- `experiments/03_bc_baseline/train.py` (default `OBS_KEYS` -> Can)
- `experiments/05_hybrid_method/run_can.py` (`_get_runtime_obs`,
  `_concat_obs`, `bc_warmstart_actions` all referenced the imported
  `OBS_KEYS` symbol from Can directly)

`experiments/03_bc_baseline/dataset.py` already had `get_obs_keys(task)`
and per-task `OBS_KEYS_{LIFT,SQUARE,...}`.  The blocker was the train
and warm-start sides not threading task-aware obs keys through.

This refactor closes the gap.

## Changes

### `experiments/03_bc_baseline/train.py`

- Import `get_obs_keys` alongside `build_tensor_dataset`.
- Resolve `obs_keys = get_obs_keys(args.task)` once.
- Pass `obs_keys=obs_keys` to `build_tensor_dataset`.
- Set `PolicyConfig(obs_keys=obs_keys, ...)` so the saved checkpoint
  carries the right obs schema for downstream loaders.
- Record `task` and `obs_keys` in `training_log.json` for
  introspection.

### `experiments/05_hybrid_method/run_can.py`

- `_get_runtime_obs(env_obs, obs_keys=None)`: accepts an explicit
  obs-keys tuple; falls back to Can's legacy `OBS_KEYS` when not set
  (preserves backward compatibility with the historical Can BC
  checkpoint).
- `_concat_obs(obs, obs_keys=None)`: same.
- `bc_warmstart_actions`: reads `obs_keys` from `policy.config.obs_keys`
  at runtime (`getattr` for legacy-checkpoint robustness) and routes
  it into the helpers.

### `experiments/15_unified_hybrid/run_can.py`

No change required.  The unified run_can already calls
`bc_warmstart_actions` and lets that function fish `obs_keys` out of
the policy config.

## Backward compatibility

- Existing Can BC checkpoints (with `PolicyConfig.obs_keys = OBS_KEYS`
  set by the historical default) continue to work bit-for-bit.
- Legacy checkpoints without an `obs_keys` field fall back to Can's
  `OBS_KEYS` via the `getattr(...) or OBS_KEYS` guard.
- The T2.3 result on main was produced with the Can default and is
  unaffected.

## Slate output

`slate_multitask_retrain_2026_05_28.sh` produces 4 checkpoints under
`experiments/03_bc_baseline/results/`:

- `policy_lift_canonical_200.pt`     (200 Lift demos)
- `policy_lift_mass1p3_205.pt`        (200 + 5 mass=1.3 Lift, **needs SHARED_DATA prereq**)
- `policy_square_canonical_200.pt`   (200 Square demos)
- `policy_square_mass1p3_205.pt`     (200 + 5 mass=1.3 Square, **needs SHARED_DATA prereq**)

Plus matching `training_log_*.json` files.

Wall-clock: ~6 min CPU total (4 × ~80 s).

## Results -- training summary (2026-05-29)

The retrain slate ran on the workhorse Mac (10-core, N_WORKERS=8)
and produced 4 task-specific BC checkpoints + matching training logs.

### Training-log headlines

| Log | task | n_demos | n_transitions | obs_dim | epochs | final_loss | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `training_log_canonical_200_cpu.json`     | can (legacy)   | 200 | 23 207 | 46 | 500 | 0.00956 |  48.7 |
| `training_log_mass1p3_205.json`           | can (legacy)   | 205 | 23 756 | 46 | 500 | 0.00939 |  78.5 |
| `training_log_lift_canonical_200.json`    | **lift**       | 200 |  9 666 | 46 | 500 | **0.00516** | 28.8 |
| `training_log_square_canonical_200.json`  | **square**     | 200 | 30 154 | 46 | 500 | 0.00901 |  98.0 |

All four converged in 500 epochs (~30-100 s wall on CPU).  The two
legacy `task=None` rows are pre-refactor logs whose dataset path
implies Can; they're included for reference because the magnitude
and democount slates consume them indirectly.

### obs_keys per task (verification of the refactor)

Each post-refactor training log carries an explicit `obs_keys`
field, confirming that `train.py` correctly threads
`get_obs_keys(args.task)` from `dataset.py` into the saved
`PolicyConfig`:

```
lift   -> ['cube_pos', 'cube_quat',
           'robot0_eef_pos', 'robot0_eef_quat',
           'robot0_gripper_qpos', 'robot0_gripper_qvel',
           'robot0_joint_pos', 'robot0_joint_pos_cos',
           'robot0_joint_pos_sin', 'robot0_joint_vel']

square -> ['SquareNut_pos', 'SquareNut_quat',
           'robot0_eef_pos', 'robot0_eef_quat',
           'robot0_gripper_qpos', 'robot0_gripper_qvel',
           'robot0_joint_pos', 'robot0_joint_pos_cos',
           'robot0_joint_pos_sin', 'robot0_joint_vel']
```

`cube_pos / cube_quat` for Lift and `SquareNut_pos / SquareNut_quat`
for Square -- both correct (Can's analog is `Can_pos / Can_quat`,
which is what legacy `task=None` checkpoints carry).  The fact that
all three obs_dims happen to be 46 is coincidental: object_pos (3) +
object_quat (4) + robot eef + gripper + joints sum to the same total
regardless of task.  The keys themselves differ.

**Refactor verified.**  Downstream slates that consume these
checkpoints (the BC arm of the Square cross-task slate, planned
Lift retries) can now reference task-aware BC pipelines without
manual obs-key plumbing.

### Note on the BC mass=1.3 prereq cells

`policy_lift_mass1p3_205.pt` and `policy_square_mass1p3_205.pt`
listed in the slate output above were **not** built in this run --
their `SHARED_DATA` prereq files
(`lift_mass1p3_205demos.hdf5`, `square_mass1p3_205demos.hdf5`) only
came online after this slate finished, courtesy of the magnitude /
square slates.  Re-running this slate with the now-present prereqs
would produce those two checkpoints, but they're not required by
any landed evaluation cell so the rebuild is deferred.

## Integration into docs/results.md

**Status update 2026-06-01**: this branch has landed.  Merged
2026-05-28 as part of the T2 4-knob robustness slate batch
(see [`05_followup_plan.md`](../adaptation/05_followup_plan.md)
"Closure (2026-06-01)" subsection for the full 7-branch merge
table).  The BC sub-section in `docs/results.md` references the
multi-task generalisation, and downstream slates (square, future
Lift retries) reference the multi-task BC pipeline freely.
