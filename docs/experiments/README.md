# Experiments index

This directory contains per-experiment documentation that complements
the in-tree `experiments/<NN>_<name>/README.md` files. Each
`docs/experiments/<NN>_<name>.md` is a navigational document that:

1. States the **current lifecycle status** of the experiment
   (active, superseded, scaffolded, or archived).
2. Quotes the **most recent headline number** from
   [`docs/results.md`](../results.md).
3. Provides the **canonical "how to reproduce"** command for the
   headline number.
4. Maps **cross-experiment relationships** (predecessors, successors,
   companion experiments, ablations).
5. Surfaces **bugs, caveats, and open questions** so a future
   reader knows what is and isn't trustworthy.

For deep technical content — exact algorithm derivations, file-by-file
breakdowns, full output schemas — follow the link to the in-tree
README at the top of each doc.

## Quick map

| #  | Name                              | Doc | Status | Role in the story |
|---:|-----------------------------------|-----|--------|-------------------|
|  1 | `demo_replay`                     | [01](01_demo_replay.md) | **done** | Open-loop replay; upper bound for the env (Can replay = 90 %) |
|  2 | `random_baseline`                 | [02](02_random_baseline.md) | **done** | Random actions; trivial lower bound (Can = 0 %) |
|  3 | `bc_baseline`                     | [03](03_bc_baseline.md) | **done** | MLP behaviour cloning policy; first learned baseline (Can = 6.7 %) |
|  4 | `mbd_baseline`                    | [04](04_mbd_baseline.md) | **done** | Online Model-Based Diffusion, no prior (Can = 0 %) |
|  5 | `hybrid_method`                   | [05](05_hybrid_method.md) | **superseded → 15** | First BC + MBD hybrid (Can = 1.7 % single-shot, 6.7 % MPC R=1) |
|  6 | `failure_taxonomy`                | [06](06_failure_taxonomy.md) | **done** | Post-hoc bucket counts (no-reach / partial / strong / success) |
|  7 | `lambda_sweep`                    | [07](07_lambda_sweep.md) | **superseded → 15** | First implementation of the per-step λ score-bridge (Can λ=5 = 33.3 %) |
|  8 | `flow_matching_prior`             | [08](08_flow_matching_prior.md) | **WIP** | Train `v_θ` flow-matching score prior (v1 = 0/60, v2 = 16.7 %) |
|  9 | `sigma_sweep`                     | [09](09_sigma_sweep.md) | **superseded → 15** | σ-max curve on exp 05 driver (Can σ=0.05 M=64 = 50 %) |
| 10 | `kxreplans_grid`                  | [10](10_kxreplans_grid.md) | **superseded → 15** | K × n_replans grid on exp 05 driver |
| 11 | `oracle_replan`                   | [11](11_oracle_replan.md) | **scaffolded** | Mechanism check for the MPC `strong-reach → success` boost |
| 12 | `action_basis_mbd`                | [12](12_action_basis_mbd.md) | **scaffolded** | DCT-II low-frequency action-basis perturbations |
| 13 | `square_task`                     | [13](13_square_task.md) | **partial** | Cross-task generalisation onto `square/ph` |
| 14 | `seed_sweep`                      | [14](14_seed_sweep.md) | **scaffolded** | 3-seed pool for Exp 04 / 05 / 05b (Wilson CIs) |
| 15 | `unified_hybrid`                  | [15](15_unified_hybrid.md) | **active (canonical)** | Unified driver: MPC + score-bridge + retrieval/learned priors. **Headline 73.8 % pooled** |
| 16 | `value_baseline`                  | [16](16_value_baseline.md) | **scaffolded on main; eval deferred** | `V_φ` state-value control variate for pure MBD; Branch A merged 2026-05-25 (`25ca457`), headline eval cell still pending (see `docs/todos/future_2026_05_26.md` § A2) |
| 17 | `dp_hybrid`                       | [17](17_dp_hybrid.md) | **done** | Diffusion-Policy + score-bridge hybrid; Phase-D landed 2026-06-01 with **two GREEN cells at ep0149**: (K=10 λ=2.0) at 86.7 % +13.4 pp and (K=20 λ=0.5) at 83.3 % +10.0 pp.  K=40 at ep0049 deferred. |

## Lifecycle legend

- **done** — implemented, fully evaluated, result is committed and
  considered a permanent reference point. No new sweeps planned.
- **superseded → <N>** — implemented and evaluated, but the
  canonical implementation for new sweeps is now experiment N.
  The existing JSONs remain valid for archival reproducibility
  (Exp 15 includes `regression_test.py` that reproduces them
  within ±1/60).
- **scaffolded** — code is in place, a runner ships in the branch,
  but no full evaluation has been committed. Smoke tests pass.
- **partial** — partly implemented; some cells reproduce but the
  experiment is gated on a refactor or an upstream change.
- **WIP** — work in progress, currently the best result is a
  recorded negative result. Active development continues on a
  feature branch (not on main).
- **active (...)** — the experiment is the current home for one or
  more open research questions; new evaluations are landing now.

## Dependency / supersession graph

```
  01 demo_replay       02 random_baseline
        ↓                     ↓
        +─────── shared common.py (env + sim_state restore) ───────+
                                ↓
                       03 bc_baseline (BC π)
                                ↓
                       04 mbd_baseline (MBD with no prior)
                                ↓
                       05 hybrid_method (BC warm-start + MBD)  ──┐
                                ↓                                │
                       07 lambda_sweep (score-bridge λ ∇log p)   │
                                ↓                                │
                       09 sigma_sweep   10 kxreplans_grid        │
                                ↓        ↓                       │ superseded
              ┌─────────────────┴────────┘                       │ by 15
              ↓
              15 unified_hybrid (canonical: MPC + score-bridge + priors)
              ↑           ↑               ↑                ↑
              │           │               │                │
  Branch A: 16 V_φ    Branch B: 08    Branch C: this    Branch C': retrieval priors
   (control variate)   v_θ flow         (Branch C        (zero-train
   for pure MBD        prior            roadmap)         priors)
                       v1, v2)

  06 failure_taxonomy ── post-hoc analysis over any run_can JSON
  11 oracle_replan ──── mechanism check for MPC boost
  12 action_basis ───── alternate noise parameterisation
  13 square_task ────── cross-task generalisation
  14 seed_sweep ─────── multi-seed pool for confidence intervals
```

## How the canonical headline 73.8 % is reached

The unified driver (exp 15) running the May-19 Slate-E
configuration:

```
--prior=demo_1nn --lambda-prior=1 --sigma-max=0.25
--M=16 --K=10 --mpc-replans=1
--n-demos=20 --n-rollouts=3 --seeds=0,1,2,3 (pool)
```

Pools to **177 / 240 = 73.8 % (Wilson CI [67.8 %, 78.9 %])**.
See [`15_unified_hybrid.md`](15_unified_hybrid.md) for the full
slate timeline and where each result lives.

## Where to find what

- **Per-method headline numbers:** [`docs/results.md`](../results.md).
- **Chronological narrative (how decisions were made):**
  [`docs/run_log.md`](../run_log.md).
- **Cross-branch roadmap:** [`docs/plans.md`](../plans.md) +
  [`docs/branch_plan.md`](../branch_plan.md).
- **Project / MuJoCo / Robomimic setup:** [`docs/MuJoCo.md`](../MuJoCo.md).
- **MJX migration runbook:** [`docs/mjx_migration.md`](../mjx_migration.md).
- **Theory drafts:** [`docs/theory/`](../theory/).
- **Topic introduction + literature review:**
  [`docs/topic_introduction/`](../topic_introduction/),
  [`docs/literature_review/`](../literature_review/).

## Conventions used across all experiment docs

- Robomimic proficient-human (`ph`) Can task is the primary
  benchmark unless noted; protocol is `n_demos=20, n_rollouts=3,
  seed=0` (60 evaluations per cell).
- Cross-task generalisation is tested on `lift/ph`, `square/ph`,
  and (Slate H) `tool_hang/ph`.
- "MBD" = Model-Based Diffusion, the planner of
  [Pan et al. 2024](https://arxiv.org/abs/2407.01573).
- "BC" = behaviour cloning; the canonical baseline is exp 03's
  256-256 MLP unless stated otherwise.
- "Score-bridge" = adding `λ · ∇ log p_prior(x)` to the MBD
  per-step update; introduced in exp 07 and now part of exp 15.
- "Warm-start" = seeding `μ_K` with a deterministic BC rollout
  instead of zero actions; introduced in exp 05.
- "Retrieval prior" = use a stored demonstration (or a soft
  k-NN mixture of demonstrations) as the prior for the score-
  bridge — no training required. Introduced in Slate D of
  exp 15.
