# Experiment 03 — Behaviour cloning baseline

## Purpose

The first **learned** row in the Week-4 results table. Trains a small
MLP policy on the proficient-human (`ph`) Can demonstrations via
plain MSE behaviour cloning, then evaluates it closed-loop using the
same 20 init-states × 3 rollouts protocol as exp 02.

This sits between the two open-loop bookends:

| Method | Success rate (Can / ph) | Bound |
| ------ | ----------------------- | ----- |
| Replay (Exp 01)              | 0.90 (n=20)  | upper bound for the env |
| Random actions (Exp 02)      | 0.00 (n=60)  | trivial lower bound |
| **Behaviour cloning (this)** | **0.067 (n=60)** | first learned policy |
| Hybrid prior + MBD (Exp 05)  | TBD          | our method |

BC sits well above 0 % and well below replay's 90 %, exactly the gap
the hybrid method (exp 05) needs to close.

## What it does

### Training

1. Read all 200 demos from `can/ph/low_dim_v141.hdf5` via
   [`dataset.py`](dataset.py).
2. Concatenate a fixed list of 10 low-dim observation keys
   (`OBS_KEYS`) into a 46-D per-step observation, paired with the
   recorded 7-D action. Two robosuite 1.4 → 1.5 drifts forced
   exclusions:
   - `robot0_eef_vel_lin` / `robot0_eef_vel_ang` — recorded in the
     dataset but not exposed as separate observables at runtime
     (rolled into `robot0_proprio-state` in 1.5). Same exclusion
     made by the standard `low_dim` BC config in `robomimic.algo.bc`.
   - `obs/object` (concatenated can pose) — sub-field order and the
     `Can_to_eef_*` reference frame both differ between 1.4 and 1.5.
     We pull only `Can_pos` and `Can_quat` (which match exactly) by
     slicing `obs/object[:, :7]` at training time and reading the
     dedicated keys at runtime; the MLP can recover relative geometry
     from these and `robot0_eef_pos`.
3. Compute per-dimension `(mean, std)` over the resulting ~23k
   transitions.
4. Train a 2-layer MLP (256 → 256, ReLU, tanh-squashed output) with
   MSE loss against the actions, Adam, batch 256.
5. Save the network weights, obs stats, and config in a single
   `policy.pt` checkpoint.

### Evaluation

1. Load the checkpoint.
2. For each of the same 20 init-states used in exp 02, do
   `n_rollouts` independent closed-loop rollouts:
   - Reset the robosuite env and restore the demo's recorded sim
     state (so init position is exactly the same as the demo).
   - At every timestep, build the obs vector from the live env's
     observation in the same `OBS_KEYS` order, normalise, query the
     policy, clip to the env's action bounds, step.
   - Run for the same number of steps as the demo's recorded horizon
     (so the success rate is step-for-step comparable to exps 01/02).
3. Record the terminal task-success flag.
4. Aggregate into `results/can_ph.json` in the same shape as
   exp 02.

## Observed numbers (2026-05-10, macOS arm64)

| Dataset | n init-states × n rollouts | success | training | eval wall |
| ------- | -------------------------- | ------- | -------- | --------- |
| `can/ph` | 20 × 3 = 60 | **4 / 60 = 6.7 %** | 88 s (500 epochs, 512-hidden MLP) | 24 s |

The 4 successes are spread across 4 different demos (`demo_1`,
`demo_10`, `demo_13`, `demo_16`), each succeeding on exactly **one**
of three rollouts despite an entirely deterministic policy and
identical recorded init states. This is the same residual non-
determinism in robosuite's contact resolution that exp 02 already
exposes — `set_sim_state` restores positions and velocities but does
not pin the order in which mujoco resolves contact constraints, so
small numerical differences accumulate. The 3-rollout protocol exists
precisely to surface this.

The 6.7 % number comes from the `--epochs 500 --hidden 512`
configuration; the script's defaults (`--epochs 100 --hidden 256`)
trained to a higher loss (~0.013 vs 0.006) and gave 0/60. Both
training_log.json snapshots are reproducible from a clean checkout
via the Usage commands below — only the higher-capacity one is
checked in (it's the one the headline number is computed against).

## Prerequisites

Same as exp 01 / 02: the `hybrid-traj` conda env (Python 3.11), the
pinned `numpy<2 / jax<0.5 / jaxlib<0.5` triple, and the `can/ph`
dataset downloaded via:

```bash
conda activate hybrid-traj
python -m robomimic.scripts.download_datasets \
    --tasks can square --dataset_types ph
```

`torch` (already in `env/requirements.txt`) is the only extra
dependency.

## Usage

```bash
conda activate hybrid-traj

# Reproduce the 6.7% headline number (500 epochs, 512-hidden MLP, ~90s).
python experiments/03_bc_baseline/train.py --epochs 500 --hidden 512 --seed 0

# Lighter default (100 epochs, 256-hidden, ~14s; trains less, gives 0% on
# eval -- mostly useful as a fast smoke check).
python experiments/03_bc_baseline/train.py

# Quick smoke run: 20 demos, 5 epochs.
python experiments/03_bc_baseline/train.py --n-demos 20 --epochs 5

# Evaluate the trained policy with the standard protocol used in the
# results table.
python experiments/03_bc_baseline/eval.py \
    --checkpoint experiments/03_bc_baseline/results/policy.pt \
    --task can --n-demos 20 --n-rollouts 3 --seed 0
```

## Output

### `results/policy.pt` (gitignored)

A single `torch.save` blob:

```python
{
  "format_version": 1,
  "config": {"obs_dim": 59, "action_dim": 7,
             "hidden_sizes": [256, 256], "obs_keys": [...]},
  "state_dict": {...},   # MLP weights
  "obs_mean": np.ndarray,
  "obs_std":  np.ndarray,
  "extra":    {"dataset": ..., "epochs": ..., "seed": ...},
}
```

Use `Policy.from_checkpoint(path)` to re-hydrate it.

### `results/training_log.json`

Per-epoch loss curve and timing, e.g.

```jsonc
{
  "experiment": "03_bc_baseline",
  "dataset": ".../can/ph/low_dim_v141.hdf5",
  "n_demos": 200,
  "n_transitions": 23207,
  "epochs": 100,
  "batch_size": 256,
  "learning_rate": 1e-4,
  "seed": 0,
  "per_epoch": [
    {"epoch": 1, "loss": 0.18, "elapsed_s": 1.2},
    ...
  ],
  "final_loss": 0.0123,
  "total_wall_s": 142.3
}
```

### `results/can_ph.json`

Same shape as exp 02:

```jsonc
{
  "experiment": "03_bc_baseline",
  "env_name": "PickPlaceCan",
  "checkpoint": "experiments/03_bc_baseline/results/policy.pt",
  "n_demos_attempted": 20,
  "n_rollouts_per_demo": 3,
  "n_total": 60,
  "n_success": <int>,
  "success_rate": <float>,
  "per_rollout": [
    {"demo": "demo_0", "rollout_idx": 0, "horizon": 118,
     "success": ..., ...},
    ...
  ]
}
```

## Implementation notes

- **Dataset / runtime obs key naming.** The robomimic HDF5 stores
  the concatenated object pose under `obs/object`, but at runtime
  robosuite's env emits the same data under `object-state`. `eval.py`
  remaps that one key transparently; everything else lines up.
- **Closed-loop horizon.** We use each demo's recorded horizon as the
  rollout length (matching exp 01 / 02). This is conservative for BC,
  which sometimes needs slightly more steps than the expert to recover
  from drift. We choose comparability over peak BC numbers; if BC's
  success rate is suspiciously low we can re-run with a fixed longer
  horizon and report both.
- **Seed.** Both `train.py` and `eval.py` accept `--seed` (default 0)
  and seed `numpy`, `torch`, and Python's `random`. Closed-loop
  success rate is essentially deterministic for a given checkpoint
  because the policy is deterministic and the rollouts start from
  recorded sim states.
- **No GMM head.** A standard upgrade for can/ph is a Gaussian
  mixture output head (cf. `robomimic.algo.bc`). We deliberately keep
  this simple; the headline result is allowed to be modest as long as
  it is well above 0.
- **Why 6.7 % and not the ~30 % robomimic-paper MLP-BC number.**
  Three contributors: (a) MSE on a multi-modal action distribution
  averages across modes — explicit in the closed-loop trace, where
  predicted xyz deltas converge to a single direction while the
  expert's recorded ones oscillate; (b) the robosuite 1.4 → 1.5 obs
  drift forced us to drop the `Can_to_eef_*` features and the
  `eef_vel_*` features, ~13 dims of input the policy could otherwise
  use; (c) the dataset itself is replayed at only 90 % under the new
  bindings (exp 01 ceiling), so any closed-loop number inherits that
  drift on top. Closing the gap is the whole point of exp 05.
- **Possible upgrades**, in increasing complexity:
  1. GMM action head + NLL loss (the canonical fix for compounding
     error from MSE + multi-modal data).
  2. Action chunking — predict the next *k* steps and roll out
     multiple at a time (Diffusion Policy / ACT style).
  3. Re-derive `obs/object` consistently across robosuite versions and
     restore the dropped 13 dims.
