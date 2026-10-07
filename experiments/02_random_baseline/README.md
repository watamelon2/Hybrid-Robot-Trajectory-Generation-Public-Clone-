# Experiment 02 — Random-action baseline

## Purpose

Companion to [`01_demo_replay`](../01_demo_replay/README.md). Where
exp 01 measures the **upper bound** any open-loop method can achieve in
our env (replay the expert action stream), this one measures the
**lower bound**: replay a *uniformly-random* action stream of the same
horizon from the same recorded initial state.

For `PickPlaceCan` this should be near 0 %. If it isn't, the env's
success criterion is too lax and our headline result table is
uninformative.

## What it does

For each of `n_demos` demos in a Robomimic `ph` HDF5 file:

1. Reset the matching `robosuite` env to the demo's recorded initial
   state.
2. Sample a fresh action sequence
   $a_t \sim \mathcal{U}(\texttt{action\_low}, \texttt{action\_high})$
   of length equal to the demo's horizon (so exp 01 and exp 02 are
   step-for-step comparable).
3. Step the env open-loop and record the terminal task-success flag.
4. Repeat `n_rollouts` times per init-state with independent RNG
   streams (`numpy.random.Generator.spawn`) for a lower-variance
   estimate.

## Observed numbers (2026-05-10, macOS arm64)

| Dataset | n init-states × n rollouts | success | s / rollout |
| ------- | -------------------------- | ------- | ----------- |
| `can/ph` | 20 × 3 = 60 | **0 / 60 = 0 %** | ~0.34 |

Action space: `Box(low=-1, high=1, shape=(7,))` — 6 OSC delta-pose
components plus 1 gripper command.

This is the trivial lower bound row in the Week-4 results table:

| Method | Success rate (Can / ph) |
| ------ | ----------------------- |
| Replay (Exp 01)              | 0.90 (n=20) |
| **Random actions (this experiment)** | **0.00 (n=60)** |
| Behaviour cloning (Exp 03, planned) | TBD |
| Hybrid prior + MBD (our method) | TBD |

## Prerequisites

Same as exp 01: the `hybrid-traj` conda env, the pinned
`numpy<2 / jax<0.5 / jaxlib<0.5` triple, and the `can/ph`
dataset downloaded via:

```bash
conda activate hybrid-traj
python -m robomimic.scripts.download_datasets \
    --tasks can square --dataset_types ph
```

## Usage

```bash
conda activate hybrid-traj

# Default: 20 demos × 1 rollout
python experiments/02_random_baseline/random_actions.py

# Same protocol used in our results table (20 init states, 3 rollouts each).
python experiments/02_random_baseline/random_actions.py \
    --task can --n-demos 20 --n-rollouts 3 --seed 0

# All demos, single rollout
python experiments/02_random_baseline/random_actions.py --n-demos -1
```

## Output

Writes `experiments/02_random_baseline/results/<task>_ph.json`:

```jsonc
{
  "experiment": "02_random_baseline",
  "env_name": "PickPlaceCan",
  "seed": 0,
  "n_demos_attempted": 20,
  "n_rollouts_per_demo": 3,
  "n_total": 60,
  "n_success": 0,
  "success_rate": 0.0,
  "action_low":  [-1, -1, -1, -1, -1, -1, -1],
  "action_high": [ 1,  1,  1,  1,  1,  1,  1],
  "per_rollout": [
    {"demo": "demo_0", "rollout_idx": 0, "horizon": 118,
     "success": false, "done": false, "elapsed_s": 0.36, "steps_per_s": 328},
    ...
  ]
}
```

## Implementation notes

- All env construction, controller-config translation, sim-state
  restore and success probing live in
  [`experiments/common.py`](../common.py) and are shared with exp 01.
- We use `numpy.random.Generator.spawn` to give each `(demo_i,
  rollout_j)` pair an independent RNG stream so the per-rollout
  results don't shift if we change the iteration order.
- Rollouts are sequential CPU `mujoco` steps; for the Week-4 numbers
  the wall-clock for 60 rollouts on this Mac was ~21 s, which is fine
  to commit to git as a single JSON.
