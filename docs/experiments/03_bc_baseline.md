# Experiment 03 — Behaviour cloning baseline (first learned policy)

**Lifecycle:** **done**. The 6.7 %/21.7 % numbers below are reference
points used by every subsequent hybrid experiment as the "BC alone"
yardstick.

**In-tree deep doc:**
[`experiments/03_bc_baseline/README.md`](../../experiments/03_bc_baseline/README.md).

## Purpose

The first **learned** row in the results table. Trains a small MLP
policy on the proficient-human (`ph`) Can demonstrations via plain
MSE behaviour cloning, then evaluates it closed-loop using the same
20 init-states × 3 rollouts protocol as exp 02.

This sits between the two open-loop bookends:

| Method | Can / ph success | Bound |
|--------|------------------|-------|
| Demo replay (Exp 01) | **90 %** | open-loop ceiling |
| Random actions (Exp 02) | **0 %** | trivial floor |
| **BC, headline (this experiment)** | **21.7 %** (13 / 60) | first learned policy |
| MBD with no prior (Exp 04) | **0 %** | training-free planner |
| Unified hybrid (Exp 15) | **73.8 %** (pooled CI [67.8, 78.9]) | canonical method |

The 21.7 % number is the **May-15** post-canonicalisation BC
headline at the **canonical 256-256 MLP** configuration. The
earlier 6.7 % number (May 10) was at `--epochs 500 --hidden 512`
with a higher-loss config; both are committed for reproducibility
but only the canonical 21.7 % is used downstream.

## Method

### Training

1. Read all 200 demos from `can/ph/low_dim_v141.hdf5` via
   [`dataset.py`](../../experiments/03_bc_baseline/dataset.py).
2. Concatenate a fixed list of 10 low-dim observation keys
   (`OBS_KEYS`) into a 46-D per-step observation, paired with the
   recorded 7-D action.
3. Compute per-dimension `(mean, std)` over the ~23 k transitions.
4. Train a 2-layer MLP (256 → 256, ReLU, tanh-squashed output)
   with MSE loss against the actions, Adam, batch 256.
5. Save weights, obs stats, and config in a single `policy.pt`
   checkpoint.

### Robosuite 1.4 → 1.5 obs-key drift

Two observation channels in the dataset are not available in our
runtime env at the same exact keys:

- `robot0_eef_vel_lin` / `robot0_eef_vel_ang` — recorded in
  the dataset but rolled into `robot0_proprio-state` in 1.5.
  Excluded (same exclusion as the canonical
  `robomimic.algo.bc` `low_dim` config).
- `obs/object` (concatenated can pose) — sub-field order and the
  `Can_to_eef_*` reference frame differ between 1.4 and 1.5. We
  pull only `Can_pos` and `Can_quat` (matched exactly) by
  slicing `obs/object[:, :7]` at training time and reading the
  dedicated keys at runtime; the MLP can recover relative
  geometry from these and `robot0_eef_pos`.

### Evaluation

1. Load the checkpoint.
2. For each of 20 init-states (same as exp 02), do `n_rollouts`
   closed-loop rollouts: reset env, restore demo's sim state,
   normalise the live obs, query the policy, clip, step.
3. Aggregate to `results/can_ph.json`.

## Headline result

| n total | n success | success rate | training | eval wall |
|---------|-----------|---------------|----------|-----------|
| 60 | 13 | **21.7 %** (256-256 MLP, canonical config) | ~90 s | ~24 s |
| 60 | 4 | 6.7 % (May-10 `--hidden 512 --epochs 500`) | ~88 s | ~24 s |

Cross-task: see [`docs/results.md`](../results.md) for `lift/ph`,
`square/ph`, and `tool_hang/ph` rows where BC is the baseline
that the hybrid is compared against.

## Reproduce

```bash
conda activate hybrid-traj

# Canonical 256-256 MLP (21.7 % headline). ~90 s training + 24 s eval.
python experiments/03_bc_baseline/train.py --seed 0
python experiments/03_bc_baseline/eval.py \
    --checkpoint experiments/03_bc_baseline/results/policy.pt \
    --task can --n-demos 20 --n-rollouts 3 --seed 0

# Smoke (5 epochs, 20 demos; ~15 s).
python experiments/03_bc_baseline/train.py --n-demos 20 --epochs 5
```

## Files

| Path | Role |
|------|------|
| [`experiments/03_bc_baseline/dataset.py`](../../experiments/03_bc_baseline/dataset.py) | HDF5 → 46-D obs vector + action pairs; defines `OBS_KEYS` (used by exps 05, 07, 08, 15, 16). |
| [`experiments/03_bc_baseline/train.py`](../../experiments/03_bc_baseline/train.py) | Training loop. |
| [`experiments/03_bc_baseline/policy.py`](../../experiments/03_bc_baseline/policy.py) | `Policy` class + `from_checkpoint` loader (re-used by exps 05, 07, 15). |
| [`experiments/03_bc_baseline/eval.py`](../../experiments/03_bc_baseline/eval.py) | Closed-loop evaluation harness. |
| `experiments/03_bc_baseline/results/policy.pt` | Trained checkpoint (gitignored; ~3 MB). |
| `experiments/03_bc_baseline/results/can_ph.json` | Headline eval result. |

## Cross-references

- **Predecessors**: [`01_demo_replay.md`](01_demo_replay.md),
  [`02_random_baseline.md`](02_random_baseline.md) (define the
  open-loop bookends this BC row sits between).
- **Direct consumer**: [`05_hybrid_method.md`](05_hybrid_method.md)
  warm-starts MBD with this BC policy's rollout. The canonical
  hybrid in [`15_unified_hybrid.md`](15_unified_hybrid.md) does
  the same plus the per-step score-bridge.
- **Cross-task wiring**: [`13_square_task.md`](13_square_task.md)
  needs `OBS_KEYS_SQUARE` added to `dataset.py` (the Can obs
  layout is hardcoded today).
- **Related**: BC-swap re-run analysis in
  [`docs/bc_swap_rerun_analysis.md`](../bc_swap_rerun_analysis.md)
  resolved a long-standing 18.3 % vs 33.3 % discrepancy between
  exps 05/10 and exp 07.

## Caveats / known gaps

- **MSE on multimodal actions** is the primary BC failure mode:
  predicted xyz deltas converge to a single direction while the
  expert's recorded ones oscillate. A GMM head (standard
  `robomimic.algo.bc` upgrade) would likely move the number up
  by 10+ pp.
- **Dropped obs dims**: the 1.4 → 1.5 drift forced exclusion of
  ~13 dims (`Can_to_eef_*` + `eef_vel_*`). Re-deriving these
  consistently across robosuite versions is an open task.
- The dataset itself replays at only 90 % under the new bindings
  (exp 01 ceiling), so any closed-loop number inherits that drift.
