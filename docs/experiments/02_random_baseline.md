# Experiment 02 — Random-action baseline (lower bound)

**Lifecycle:** **done**. The recorded 0 / 60 result is the env's
trivial floor; no new evaluation planned.

**In-tree deep doc:**
[`experiments/02_random_baseline/README.md`](../../experiments/02_random_baseline/README.md).

## Purpose

Companion to [`01_demo_replay.md`](01_demo_replay.md). Where exp 01
measures the **upper bound** any open-loop method can achieve in our
env (replay the expert action stream), this one measures the
**lower bound**: replay a uniformly-random action stream of the same
horizon from the same recorded initial state.

For `PickPlaceCan` this should be near 0 %. If it isn't, the env's
success criterion is too lax and the headline-result table is
uninformative.

## Method

For each of `n_demos` demos in a Robomimic `ph` HDF5 file:

1. Reset the matching `robosuite` env to the demo's recorded initial
   state via `experiments/common.py`.
2. Sample a fresh action sequence
   $a_t \sim \mathcal{U}(\texttt{action\_low}, \texttt{action\_high})$
   of length equal to the demo's horizon (so exp 01 / 02 are
   step-for-step comparable).
3. Step the env open-loop and record the terminal task-success flag.
4. Repeat `n_rollouts` times per init-state with independent RNG
   streams (`numpy.random.Generator.spawn`) for a lower-variance
   estimate.

The action space is `Box(low=-1, high=1, shape=(7,))` — 6 OSC
delta-pose components plus 1 gripper command.

## Headline result

| Dataset  | n init-states × n rollouts | success | s / rollout |
|----------|----------------------------|---------|-------------|
| `can/ph` | 20 × 3 = 60 | **0 / 60 = 0 %** | ~0.34 |

This is the trivial lower bound row in the headline table.

## Reproduce

```bash
conda activate hybrid-traj

# Default: 20 demos × 1 rollout (~7 s).
python experiments/02_random_baseline/random_actions.py

# Same protocol used in the headline table (20 init-states × 3 rollouts).
python experiments/02_random_baseline/random_actions.py \
    --task can --n-demos 20 --n-rollouts 3 --seed 0
```

Output: `experiments/02_random_baseline/results/<task>_ph.json`.

## Files

| Path | Role |
|------|------|
| [`experiments/02_random_baseline/random_actions.py`](../../experiments/02_random_baseline/random_actions.py) | Main entry point. |
| [`experiments/02_random_baseline/visualize.ipynb`](../../experiments/02_random_baseline/visualize.ipynb) | Optional inline-render of random rollouts. |
| [`experiments/common.py`](../../experiments/common.py) | Shared env-construction + sim-state restore. |
| `experiments/02_random_baseline/results/can_ph.json` | Headline result (committed). |

## Cross-references

- **Companion**: [`01_demo_replay.md`](01_demo_replay.md) — upper bound.
- **Successor learned baselines**:
  [`03_bc_baseline.md`](03_bc_baseline.md),
  [`04_mbd_baseline.md`](04_mbd_baseline.md).
- **Square variant**: see [`13_square_task.md`](13_square_task.md)
  — random baseline runs unchanged with `--task square`.

## Implementation notes

- All env construction, controller-config translation, sim-state
  restore and success probing live in
  [`experiments/common.py`](../../experiments/common.py) and are
  shared with exp 01 and exp 03.
- We use `numpy.random.Generator.spawn` to give each
  `(demo_i, rollout_j)` pair an independent RNG stream so the
  per-rollout results don't shift if iteration order changes.
- Rollouts are sequential CPU `mujoco` steps; 60 rollouts complete in
  ~21 s on M-series hardware.
