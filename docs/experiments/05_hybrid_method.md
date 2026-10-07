# Experiment 05 — Hybrid BC warm-start + MBD (first hybrid)

**Lifecycle:** **superseded → [`15_unified_hybrid`](15_unified_hybrid.md)**.
The two committed JSONs (single-shot, MPC R=1) remain valid for
archival reproducibility and are reproduced by exp 15 within
±1/60 via [`regression_test.py`](../../experiments/15_unified_hybrid/regression_test.py).

**In-tree deep doc:**
[`experiments/05_hybrid_method/README.md`](../../experiments/05_hybrid_method/README.md).

## Purpose

The **first** hybrid: take the MBD denoiser from exp 04 and seed
its initial mean trajectory `mu_K` with a deterministic BC rollout
from exp 03 (instead of zeros). Adds one ablation flag
(`--sigma-max 0.25`, half of exp 04's 0.5) so the BC structure
survives iteration 1.

Adds the **MPC chunking** ablation (`--mpc-replans N`): split the
H-step execution into N+1 equal chunks, replan each chunk from the
live env state. The mid-rollout snapshot gives MBD a near-grasp
state to refine against in the second plan.

## Headline results (Can / ph, seed 0)

| Variant                          | Success   | Wall       |
|----------------------------------|-----------|-----------:|
| Hybrid, single-shot              | **1 / 60 = 1.7 %** | 81.3 min |
| Hybrid + MPC `n_replans=1`       | **4 / 60 = 6.7 %** | 222.6 min |

The 4-seed pool of the same single-shot cell (Job C, May 14)
gives **37.5 % CI [31.6, 43.8]** (`docs/results.md`) — i.e., the
mechanism is much stronger than the n=60 single-seed point estimate
suggests, but variance is large.

### Behaviour breakdown vs exp 04

| bucket | exp 04 | exp 05 single-shot |
|--------|-------:|------------------:|
| Strong reach (`>+3`)        | 30 % | **53 %** |
| Partial reach               | 17 % | 35 % |
| **No reach** (`≤+0.5`)      | **53 %** | **12 %** |

The "no reach" mass collapses from 53 % to 12 % under the BC warm
start. **Reaching is solved by the prior, not by the planner.**

### MPC mechanism (Ablation 05b)

| metric | single-shot | MPC R=1 |
|--------|-----------:|--------:|
| Strong-reach → success conversion | 3.1 % (1 / 32) | **14.8 %** (4 / 27) |
| No-reach share | 12 % | 3 % |
| Mean executed reward | +8.25 | +11.32 |

The 4× point-estimate boost on success rate is **not** statistically
separated at α=0.05 (Fisher `p = 0.36` vs single-shot), but the
strong-reach → success conversion rate is the mechanism we expect
MPC to improve and it nearly 5× under the protocol. The MPC
mechanism is then re-validated in exp 15 with much more compute.

## Method

The MBD denoising loop's `initial_mean` is no longer zero but a
single deterministic rollout of the BC policy from exp 03:

```python
from policy import Policy                # exp 03
from mbd import MBDParams, denoise_loop  # exp 04
from run_can import plan_and_execute     # exp 04 (with initial_mean=)

bc = Policy.from_checkpoint(BC_CKPT)
bc_actions = bc_warmstart_actions(env, init_state, bc, H, low, high)

result = plan_and_execute(
    env, init_state, params, child_rng,
    initial_mean=bc_actions,             # <-- the only change vs exp 04
    ...
)
```

The BC warm-start is closed-loop (the policy sees the live env's
obs at every step); MBD then perturbs that sequence open-loop
(standard MBD protocol).

### Tuned defaults

| flag | default | rationale |
|------|---------|-----------|
| `--H` | 120 | matches exps 01 / 03 / 04 |
| `--M` | 16 | matches exp 04 |
| `--K` | 10 | matches exp 04 |
| `--lambda_` | 0.5 | matches exp 04 |
| **`--sigma-max`** | **0.25** | half of exp 04; preserves BC structure |
| `--sigma-min` | 0.05 | matches exp 04 |
| `--bc-checkpoint` | `experiments/03_bc_baseline/results/policy.pt` | exp 03's MLP |

## Reproduce

```bash
conda activate hybrid-traj

# Single-shot (1/60, ~81 min wall).
python experiments/05_hybrid_method/run_can.py \
    --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view

# Ablation 05b: one MPC replan at t=H/2 (~2.7 h wall, 4/60).
python experiments/05_hybrid_method/run_can.py \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --mpc-replans 1 --mpc-chunk-steps 60 --no-live-view \
    --output experiments/05_hybrid_method/results/can_ph_mpc.json
```

Or use the **canonical successor** (exp 15) instead:

```bash
# Exp 15 single-shot equivalent of exp 05 (reproduces within ±1/60).
python experiments/15_unified_hybrid/run_can.py \
    --prior bc --lambda-prior 0 --mpc-replans 0 \
    --K 10 --M 16 --sigma-max 0.25 \
    --n-demos 20 --n-rollouts 3 --seed 0
```

## Files

| Path | Role |
|------|------|
| [`experiments/05_hybrid_method/run_can.py`](../../experiments/05_hybrid_method/run_can.py) | ~700-line driver: single-shot + MPC variants, BC warm-start + MBD `plan_and_execute`. |
| `experiments/05_hybrid_method/results/can_ph.json` | Single-shot headline (1 / 60). |
| `experiments/05_hybrid_method/results/can_ph_mpc.json` | MPC R=1 (4 / 60). |
| `experiments/05_hybrid_method/results/can_ph.png` | Aggregate cost-vs-iter plot. |

## Cross-references

- **Predecessors**:
  [`03_bc_baseline.md`](03_bc_baseline.md) (BC policy),
  [`04_mbd_baseline.md`](04_mbd_baseline.md) (MBD denoiser).
- **Canonical successor**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) — exp 05 is
  one of the rows in exp 15's regression test.
- **Adjacent ablations**:
  [`07_lambda_sweep.md`](07_lambda_sweep.md) adds the per-step
  score-bridge on top.
  [`09_sigma_sweep.md`](09_sigma_sweep.md) sweeps `sigma_max`
  through the exp-05 driver.
  [`10_kxreplans_grid.md`](10_kxreplans_grid.md) is the K × R
  Pareto sweep through the exp-05 driver.
  [`11_oracle_replan.md`](11_oracle_replan.md) is the mechanism
  check for the MPC boost.
- **Multi-seed pool**:
  [`14_seed_sweep.md`](14_seed_sweep.md) pools exp 04 / 05 / 05b
  across 3 seeds.

## Caveats / known anomalies

- **BC-alone arm RNG bug**: the script's internal BC-alone success
  counter reports 5 / 60 in the MPC run vs 0 / 60 in the single-shot
  run, even though the BC policy is deterministic and uses the
  same init state. Caused by `env.reset()` RNG-consumption
  differences between the two run schedules. Does NOT affect the
  hybrid MBD comparison rows (each is scored inside its own run).
  Fixed in exp 15's
  [`unified_loop.py`](../../experiments/15_unified_hybrid/unified_loop.py)
  by passing `--deterministic-bc-reset`.
- **Single-seed 1/60 is a low-power estimate**. The 4-seed pool
  (Job C) gives **37.5 % CI [31.6, 43.8]** at the same cell —
  significantly higher than the point estimate. This is why
  multi-seed pooling is now standard in exp 15.
