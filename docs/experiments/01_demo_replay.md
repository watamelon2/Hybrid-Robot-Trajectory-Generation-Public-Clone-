# Experiment 01 — Demo replay (sim/dataset sanity check)

**Lifecycle:** **done**. No new evaluation planned; the recorded
90 % replay rate is the env's open-loop ceiling and is treated as
a permanent reference point.

**In-tree deep doc:**
[`experiments/01_demo_replay/README.md`](../../experiments/01_demo_replay/README.md).

## Purpose

This is the project's most basic experiment. It is **not** a
learning experiment; it is a sanity check on the
`robosuite` + `robomimic` + `mujoco` pipeline that every later
experiment depends on.

For each demonstration in a Robomimic proficient-human (`ph`)
dataset we:

1. Re-instantiate the matching `robosuite` env from
   `env_args` stored in the HDF5 metadata.
2. Set the env to the demo's recorded initial simulator state
   (`states[0]`).
3. Replay the recorded action sequence open-loop.
4. Ask the env whether the task succeeded at the final step.

If the dataset and environment are paired correctly and the
simulator is deterministic, the replay success rate should be
near 100 %. Anything below ~80 % would invalidate every learned
baseline that follows.

## Headline result

| Dataset  | n demos | success      | s / demo |
|----------|--------:|-------------:|---------:|
| `can/ph` |      20 | **18 / 20 = 90 %** | ~0.37 |

The 10 % gap is **expected drift**, not a bug. The dataset was
recorded with robosuite 1.4.1 + the legacy `mujoco_py` bindings
+ MuJoCo 2.x; we replay it with robosuite 1.5.2 + the new
`mujoco` 3.x bindings. Robosuite 1.5 redesigned its controller
schema; we map the legacy flat `OSC_POSE` config onto 1.5's
`BASIC` composite controller with the same `OSC_POSE`
parameters by default, but the underlying physics integration
and OSC implementation are not bit-identical.

**This 90 % is the *upper bound* any policy consuming raw demo
actions can achieve in our env.** Closing the gap between this
and a learned policy's number is the only thing later
experiments should be read for.

## Reproduce

```bash
conda activate hybrid-traj

# Default: 5 demos (smoke check; <10 s).
python experiments/01_demo_replay/replay.py

# Reproduce the headline 90 % number (~7 s).
python experiments/01_demo_replay/replay.py --task can --n-demos 20

# Cross-task variant (Square has its own success criterion).
python experiments/01_demo_replay/replay.py --task square --n-demos 20
```

Output lands at `experiments/01_demo_replay/results/<task>_ph.json`.

## Files

| Path | Role |
|------|------|
| [`experiments/01_demo_replay/replay.py`](../../experiments/01_demo_replay/replay.py) | Main entry point; CLI + replay loop. |
| [`experiments/01_demo_replay/visualize.ipynb`](../../experiments/01_demo_replay/visualize.ipynb) | Inline-render visualisation of replayed demos. |
| [`experiments/common.py`](../../experiments/common.py) | Shared env-construction + sim-state restore + success probe (used by exps 02, 03 as well). |
| `experiments/01_demo_replay/results/can_ph.json` | Headline replay result (committed). |

## Cross-references

- **Companion**: [`02_random_baseline.md`](02_random_baseline.md)
  is the lower-bound counterpart on the same protocol.
- **Inherited by**: [`03_bc_baseline.md`](03_bc_baseline.md),
  [`04_mbd_baseline.md`](04_mbd_baseline.md), and everything
  downstream uses the same `experiments/common.py` env-construction
  pipeline that this experiment validates.
- **Cross-task**: see [`13_square_task.md`](13_square_task.md)
  for the `square/ph` replay row.

## Caveats

- The first call into `robosuite` is slow (~3–5 s) because
  robosuite builds an MJCF on the fly. Subsequent demos reuse
  the same env.
- Robosuite's `mink` IK warning at import is benign for `Panda`
  + `PickPlaceCan`; we are not using GR1 or whole-body IK here.
- On a headless Linux machine, set `MUJOCO_GL=egl` before
  invoking the script. No effect on macOS.
- The replay protocol does not use replanning or any policy —
  it is strictly a re-execution of recorded action sequences.
  The 90 % ≠ 100 % gap is therefore a property of the sim
  itself, not of any planner downstream.
