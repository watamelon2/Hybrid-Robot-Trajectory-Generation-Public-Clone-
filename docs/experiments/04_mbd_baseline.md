# Experiment 04 — Online MBD baseline (no prior)

**Lifecycle:** **done**. The 0 / 60 result is a permanent reference
that motivates every hybrid experiment downstream.

**In-tree deep doc:**
[`experiments/04_mbd_baseline/README.md`](../../experiments/04_mbd_baseline/README.md).

## Purpose

Validate that **online Model-Based Diffusion** (MBD,
[Pan et al. 2024](https://arxiv.org/abs/2407.01573)) can solve our
chosen task when given **no expert prior**. This isolates "what does
MBD on its own buy us" from "what does the BC prior buy us"; exp 05
then layers BC on top.

```
| Method                          | Can / ph success | Bound |
| Replay (Exp 01)                 |  90 % (n=20)     | open-loop ceiling |
| Random actions (Exp 02)         |   0 % (n=60)     | trivial floor |
| Behaviour cloning (Exp 03)      |  21.7 %          | first learned policy |
| **Online MBD, no prior (this)** |  **0 % (0/60)**  | training-free planner |
| Hybrid prior + MBD (Exp 15)     |  73.8 % (pooled) | canonical method |
```

## Method outline

For a horizon-`H` trajectory `tau = (a_0, ..., a_{H-1})` with task
cost `J(tau)`, MBD treats `tau` as a sample from a target density
`p*(tau) ∝ exp(-J(tau) / λ)` and runs a denoising-style sampler:

1. Initialise `tau_K` from a base distribution (Gaussian noise, or
   in exp 05+ a BC-warm-started distribution).
2. For `k = K-1, ..., 0`:
   a. Draw `M` perturbed copies
      `tau_k^{(m)} = mu_k + sigma_k * eps^{(m)}`,
      `eps^{(m)} ~ N(0, I)`.
   b. Roll out each `tau_k^{(m)}` through the simulator from the
      same initial state; get cost `J^{(m)}` per copy.
   c. Compute weights `w^{(m)} = softmax(-J^{(m)} / λ)`.
   d. Update mean `mu_{k-1} = Σ_m w^{(m)} · tau_k^{(m)}` (the
      score-matching estimator of `∇ log p*`).
   e. Schedule `sigma_{k-1} < sigma_k` (cosine / DDPM-style).
3. Execute `tau_0` open-loop (or its first `chunk_steps` actions
   and replan; introduced in exp 05).

The cost `J` is the negative dense reward from the env (distance to
target + grasp / lift bonuses for `PickPlaceCan`), summed over the
rollout.

## Two-stage validation

### Stage 1 — LQ sanity check ([`lq_sanity.py`](../../experiments/04_mbd_baseline/lq_sanity.py))

A 2-D point mass with linear dynamics and quadratic cost.
Closed-form optimal feedback gain `K*` available from
`scipy.linalg.solve_discrete_are`. Pure numpy / jax-numpy, no sim;
runs in <1 s.

**Observed (5 seeds 0..4):** ratios MBD / optimum are
`[1.059, 1.073, 1.075, 1.073, 1.068]` — consistently 5–8 % above
optimum, comfortably under the 10 % threshold. This validates that
the algorithm in `mbd.py` is wired correctly (schedule,
normalisation, weighted mean) before stepping into sim.

### Stage 2 — Full MBD on `PickPlaceCan` ([`run_can.py`](../../experiments/04_mbd_baseline/run_can.py))

Same algorithm, rollouts through the same robosuite env used by
exps 01–03 (no MJX). Sequential rollouts; ~78 min wall for 60.

**Tuned defaults (`run_can.py`):**

| flag | default | rationale |
|------|---------|-----------|
| `--H` | 120 | matches exps 01 / 03 demo horizon |
| `--M` | 16 | smallest reliable; M=8 too noisy |
| `--K` | 10 | descent saturates around k=8–10 |
| `--lambda_` | 0.5 | dense-reward differences are O(2–4); λ=5 averages too softly |
| `--sigma-max` | 0.5 | OSC_POSE actions clip to [-1, 1]; 0.5 explores meaningfully |
| `--sigma-min` | 0.05 | cosine schedule end |

## Headline result

| quantity                          |  mean |  median |
|-----------------------------------|------:|--------:|
| init best cost (k = 0)            | −0.82 |   −0.49 |
| final best cost (k = K−1)         | −2.36 |   −0.40 |
| cost descent                      | +1.54 |   +0.10 |
| executed `mu_final` reward        | +1.78 |   +0.31 |
| **success**                       | **0 / 60 = 0 %** |

### Behaviour breakdown (bucketed by `exec_reward`)

| bucket | criterion | count | share |
|--------|-----------|------:|------:|
| Strong reach | `exec_reward > +3.0` | 18 | 30 % |
| Partial reach | `0.5 < exec_reward ≤ +3.0` | 10 | 17 % |
| **No reach** | `exec_reward ≤ +0.5` | **32** | **53 %** |

The dominant failure mode is "**planner can't even reach**" (53 %),
not "reaches but can't grasp". This drops to 12 % under the BC
warm-start in exp 05 and to <5 % under retrieval priors in exp 15.

## Reproduce

```bash
conda activate hybrid-traj

# Stage 1: LQ math validation (<1 s).
python experiments/04_mbd_baseline/lq_sanity.py --seed 0

# Stage 2 smoke: 5 demos × 1 rollout, ~6 min.
python experiments/04_mbd_baseline/run_can.py --n-demos 5 --seed 0

# Stage 2 full headline: 20 × 3 = 60 rollouts, ~78 min headless.
python experiments/04_mbd_baseline/run_can.py \
    --n-demos 20 --n-rollouts 3 --no-live-view --seed 0
```

## Files

| Path | Role |
|------|------|
| [`experiments/04_mbd_baseline/mbd.py`](../../experiments/04_mbd_baseline/mbd.py) | `MBDParams` + `denoise_loop(...)`. Pure algorithm with optional `iter_callback` hook for live preview. Imported by exps 05, 07, 12. |
| [`experiments/04_mbd_baseline/lq_sanity.py`](../../experiments/04_mbd_baseline/lq_sanity.py) | Stage 1 — LQ regulator + DARE solution. |
| [`experiments/04_mbd_baseline/run_can.py`](../../experiments/04_mbd_baseline/run_can.py) | Stage 2 — live robosuite `PickPlaceCan` evaluation with optional matplotlib live preview. |
| [`experiments/04_mbd_baseline/worker.py`](../../experiments/04_mbd_baseline/worker.py) | Multiprocess worker for parallel rollouts (used by exp 15). |
| [`experiments/04_mbd_baseline/jax_osc.py`](../../experiments/04_mbd_baseline/jax_osc.py) | JAX port of robosuite's OSC controller, used by the MJX migration. |
| [`experiments/04_mbd_baseline/mjx_rollout.py`](../../experiments/04_mbd_baseline/mjx_rollout.py) | JAX/MJX batched-rollout context (see [`docs/mjx_migration.md`](../mjx_migration.md)). |
| [`experiments/04_mbd_baseline/scenes/`](../../experiments/04_mbd_baseline/scenes/) | Simplified MJX scene + Panda torque-controlled XML. |
| [`experiments/04_mbd_baseline/test_*.py`](../../experiments/04_mbd_baseline/) | OSC parity + MJX smoke gates for the MJX migration. |

## MJX migration

There is a parallel batched-rollout pathway in
[`mjx_rollout.py`](../../experiments/04_mbd_baseline/mjx_rollout.py)
that runs the same MBD inner loop on the MJX simplified scene with
JAX-batched copies (see [`docs/mjx_migration.md`](../mjx_migration.md)
for the full runbook). The CPU robosuite path remains the
canonical / reproducibility path; MJX is for fast iteration only.
The Colab L4 Gate-4 measurement completed on 2026-06-07 at
`13.80x` over the nominal 5 h Mac/robosuite CPU baseline, below
the prior promotion threshold.

## Cross-references

- **Predecessor pipeline**:
  [`01_demo_replay.md`](01_demo_replay.md),
  [`02_random_baseline.md`](02_random_baseline.md),
  [`03_bc_baseline.md`](03_bc_baseline.md).
- **Successors**:
  [`05_hybrid_method.md`](05_hybrid_method.md) is exp 04 plus a
  BC warm-start; [`15_unified_hybrid.md`](15_unified_hybrid.md)
  is the canonical successor.
- **Companion (failure modes)**:
  [`06_failure_taxonomy.md`](06_failure_taxonomy.md) formalises
  the "no-reach / partial / strong" buckets used in this README.
- **MJX**: [`docs/mjx_migration.md`](../mjx_migration.md).

## Open questions / WIP

- MJX follow-up is now about improving throughput or scaling `M`,
  not proving basic GPU execution; Gate-4 Colab L4 measurement is
  recorded in `docs/mjx_migration.md`.
- MPC replanning ablation (single-shot vs `--mpc-replans=1`) is
  exp 05 / exp 15 territory, not exp 04.
- Multi-seed variance estimate on the 0 / 60 headline (exp 14
  pools 3 seeds for exps 04 / 05 / 05b).
