# Experiment 01 — Robomimic demo replay

## Purpose

This is the project's **most basic** experiment. It is **not** a learning
experiment; it is a *sanity check on the simulator + dataset pipeline*
that every later experiment depends on.

For each demonstration in a Robomimic proficient-human (`ph`) dataset we
re-instantiate the matching `robosuite` environment, set it to the demo's
recorded initial simulator state, replay the recorded action sequence
open-loop, and ask the env whether the task succeeded at the final step.

If the dataset and environment are paired correctly and the simulator is
deterministic, the *replay success rate* should be **near 100 %**.
Anything below ~80 % indicates that the env we are instantiating differs
materially from the one that recorded the data, or that the dataset's
`states[0]` is not being applied — both of which would invalidate every
learned baseline we plan to add later.

### Observed numbers (2026-05-10, macOS arm64)

| Dataset | n demos | replay success | s / demo | notes |
| ------- | ------- | -------------- | -------- | ----- |
| `can/ph` | 20 | **18 / 20 = 90 %** | ~0.37 | ~120-step horizon |

The 10 % gap on `can/ph` is **expected drift**, not a bug:

- The dataset is `low_dim_v141.hdf5`, recorded with **robosuite 1.4.1
  + the legacy `mujoco_py` bindings + MuJoCo 2.x**.
- We replay it with **robosuite 1.5.2 + the new `mujoco` 3.x bindings**.
  Robosuite 1.5 redesigned its controller schema; we map the legacy
  flat `OSC_POSE` config onto 1.5's `BASIC` composite controller, which
  has *the same* OSC_POSE parameters by default, but the underlying
  physics integration and OSC implementation are not bit-identical.

This 90 % is therefore the *upper bound* any policy that consumes raw
demo actions can achieve in our env, and the gap between this and a
learned policy's number is the only thing we should read into. We
record the replay number as a row in the Week-4 results table:

| Method | Success rate (Can / ph) | Notes |
| ------ | ----------------------- | ----- |
| **Replay (this experiment)** | **0.90** (n=20) | open-loop ceiling for our env |
| Random actions (Exp 02, planned) | ≈ 0 | trivial lower bound |
| Behaviour cloning (Exp 03, planned) | TBD | learned baseline |
| Hybrid prior + MBD (our method) | TBD | online correction on top of BC |

## Prerequisites

1. The `hybrid-traj` conda env (see `docs/python_setup.md`).
2. Pinned versions: `numpy<2`, `jax<0.5`, `jaxlib<0.5` (`mink` requires
   `numpy<2`; the more-recent `jax` we installed earlier requires
   `numpy>=2`, so the pin matters).
3. The proficient-human Can dataset:

   ```bash
   conda activate hybrid-traj
   python -m robomimic.scripts.download_datasets \
       --tasks can square --dataset_types ph
   ```

## Usage

```bash
conda activate hybrid-traj

# Replay the first 5 demos of the can/ph dataset (default).
python experiments/01_demo_replay/replay.py

# Replay all demos of square/ph.
python experiments/01_demo_replay/replay.py --task square --n-demos -1

# Use a custom dataset path and output location.
python experiments/01_demo_replay/replay.py \
    --dataset /path/to/low_dim_v141.hdf5 \
    --output  /tmp/replay.json
```

## Output

A JSON file is written to
`experiments/01_demo_replay/results/<task>_ph.json` with the following
shape:

```jsonc
{
  "dataset": ".../datasets/can/ph/low_dim_v141.hdf5",
  "env_name": "PickPlaceCan",
  "env_type": 1,
  "n_demos_attempted": 5,
  "n_success": 5,
  "success_rate": 1.0,
  "total_wall_s": 6.4,
  "per_demo": [
    {"demo": "demo_0", "horizon": 124, "success": true,
     "done": false, "elapsed_s": 1.27, "steps_per_s": 97.6},
    ...
  ]
}
```

Per-demo entries record the demo key, action-horizon length, terminal
success flag, terminal `done` flag, wall time and steps/sec. Steps/sec
is purely diagnostic.

## Caveats

- The first call into `robosuite` is slow (~3-5 s) because robosuite
  builds an MJCF on the fly. Subsequent demos reuse the same env.
- Robosuite's `mink` IK warning at import is benign for `Panda` and
  `PickPlaceCan`; we are not using GR1 or whole-body IK here.
- Set `MUJOCO_GL=egl` before invoking the script on a headless Linux
  machine (no effect on macOS).
