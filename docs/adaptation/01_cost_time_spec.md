# Adaptation 1 — Cost-time adaptation: goal-swap and obstacle injection

*Branch: `feat/adapt-cost-time`. Companion specs:
`docs/adaptation/02_dynamics_spec.md` (dynamics adaptation,
`feat/adapt-dynamics`), `docs/adaptation/03_demo_scaling_spec.md`
(prior-bank scaling, `feat/adapt-demo-scaling`),
`docs/adaptation/04_incremental_prior_spec.md` (runtime bank swap,
`feat/adapt-incremental-prior`). Theory companion:
`docs/theory/01_soft_control.md` §1.*

This document specifies the **cost-time adaptation experiment** —
the cheapest of the three adaptation axes (cost / dynamics /
demo-bank). Maps onto the **$p^{\mathrm{model}}$** factor of the
composite score's soft-control target (`01_soft_control.md` eq. (3)):
the cost $J$ changes at inference, while $p^{\mathrm{data}}$ stays
fixed. **No retraining of any kind is required.**

## 1. What this measures

The hybrid's "OOD robustness" pitch (limitation #5 of
`presentation_brief_2026_05_22.md`) is currently unsubstantiated.
Cost-time adaptation is the cleanest possible substantiation
because:

- The **prior side** is unchanged — `demo_1nn` retrieves the same
  demo trajectories regardless of cost. So any improvement vs.
  BC under cost perturbation is attributable *entirely to the
  online MBD factor* of the hybrid.
- The **model side** auto-adapts — MBD's score
  $\nabla\!\log p^{\mathrm{model}} = -(1/\beta)\nabla J$ is
  recomputed from the deployed $J$ at every iteration. So the
  hybrid should track pure-MBD's adaptation profile up to the
  prior-induced bias.
- BC is a **clean negative control** — it has no awareness of $J$
  at all (it learned actions, not value), so its performance
  under cost perturbation should degrade monotonically with
  perturbation magnitude.

**Predicted plot.** $x$-axis: goal-pose displacement $\|\Delta x\|$
in $\{0, 0.05, 0.10, 0.20\}$ m. $y$-axis: success rate. Three
curves: BC, pure-MBD-no-prior, hybrid. Predicted shape:

| Method | $\Delta x = 0$ | growth in $\|\Delta x\|$ |
|---|---:|---|
| BC | 21.7 % | drops to ~0 % by $0.10$ m (demos memorised at $\Delta x = 0$) |
| MBD-no-prior | 0 % | flat at 0 % (still too few samples regardless of cost) |
| **Hybrid** | **75 %** | drops slower than BC; predicted ~50 % at $0.10$ m |

The hybrid's gap over BC should **widen with perturbation** —
the cleanest possible "the hybrid inherits MBD's adaptability"
plot.

## 2. Experiment cells

Operating point: canonical `demo_1nn` $\lambda = 1,\, \sigma_{\max} = 0.25,\,
M = 16,\, K = 10,\, R = 1$ on Can. Each cell: 20 demos × 3
rollouts = 60 evaluations, seed = 0 single-seed (pool later if a
gap is real).

| # | Cell | Hybrid | BC | MBD-no-prior | Total |
|---|---|---:|---:|---:|---:|
| C1.1 | $\Delta x = (0.00, 0.00, 0.00)$ (control) | 60 | 60 | 60 | 180 |
| C1.2 | $\Delta x = (0.05, 0.00, 0.00)$ (small) | 60 | 60 | 60 | 180 |
| C1.3 | $\Delta x = (0.10, 0.00, 0.00)$ (medium) | 60 | 60 | 60 | 180 |
| C1.4 | $\Delta x = (0.20, 0.00, 0.00)$ (large) | 60 | 60 | 60 | 180 |

**Wall-clock estimate.** Each cell ≈ 50 s/rollout × 60 = 50 min;
3 methods × 4 cells = 12 cells × 50 min = 10 h. Halves to 5 h
if BC and MBD-no-prior reuse a single `--goal-perturbation-xyz`
sweep run (they're cheap CPU paths). **Realistic budget: 5–7 h
wall on the test rig.**

Optional extension — obstacle injection (C2): instead of moving
the goal body, inject a small sphere obstacle at $(0.05, 0.00, 0.10)$
above the median demo path. Same three-method ladder, same
cells. Add only if C1 produces the predicted widening gap.

## 3. Implementation checklist

Before firing any cell, the following must land on this branch:

- [x] `--goal-perturbation-xyz DX DY DZ` flag in `run_can.py`.
- [x] Flag surfaced into output JSON `params.goal_perturbation_xyz`.
- [x] `env_perturbations.apply_goal_perturbation` **real impl**
      (commit `8a00b92`, 2026-05-25).
- [x] **Per-task body-name registry** in `env_perturbations.py`.
      Filled 2026-05-25 by REPL inspection:
      `can`→`bin2`, `lift`→None (height threshold), `square`→`peg1`,
      tool_hang / transport TODO.
- [x] **Body-translation logic** in `apply_goal_perturbation`.
      Implemented as `body_pos += xyz; sim.forward()`.
- [x] **Per-task supported-list.** Lift raises `ValueError`;
      unknown task raises `KeyError`. Verified by smoke
      (`/tmp/smoke_goal_pert.py`, 4/4 pass).
- [x] **Cost-function pass-through.** Traced (2026-05-25):
      `experiments/04_mbd_baseline/run_can.cost_fn` itself is
      goal-agnostic (it just returns `-rewards_batch`); the goal
      enters via the env-computed reward.  The actual issue was
      one layer below: each multiprocessing worker creates a fresh
      env via `make_env_for_dataset` and was never told about the
      perturbation, so workers' rewards (= MBD planning's
      `rewards_batch`) were computed against the canonical
      (unperturbed) goal.  Fixed by threading a `perturbations`
      dict through
      `run_can.main -> unified_plan_and_execute -> make_rollout_fn
      -> _worker_init`, which applies `apply_goal_perturbation`
      (and the dynamics stub) inside each worker after env
      creation.  See `experiments/04_mbd_baseline/worker.py:_worker_init`
      and `_apply_perturbations`.  Regression-tested by
      `experiments/04_mbd_baseline/test_worker_perturbation.py`
      (PART A: bin2_pos + cached attrs move by exactly the requested
      xyz; PART B: spawn-context pool rewards differ between
      canonical and perturbed initargs).

## 4. Pre-flight smoke

```bash
# 5 min, 2 demos x 1 rollout, K=4 -- verifies the env perturbation
# wires through to set_sim_state and the cost function.
python -m experiments.15_unified_hybrid.run_can \
    --task=can --prior=demo_1nn --lambda-prior 1.0 \
    --K=10 --M=16 --sigma-max=0.25 --mpc-replans=1 \
    --goal-perturbation-xyz 0.05 0.00 0.00 \
    --smoke --output-name-suffix=_C1_smoke
```

Should print `[exp15] Phase-1 unimplemented-flag notice` referencing
the stub (until the registry is filled), then raise
`NotImplementedError` from `apply_goal_perturbation`. Replace stub
with real impl, re-run, expect a single JSON in `results/` with
`params.goal_perturbation_xyz = [0.05, 0.0, 0.0]`.

## 5. Analysis plan

Per cell, three columns required:

- `success_rate` — primary metric.
- `mean_executed_cost` — secondary, smoother metric (already in
  per-rollout JSON).
- `mean_object_to_goal_distance_at_final_step` — diagnostic,
  measures how close the planner got even when binary success
  failed. Pull from the rollout's final obs.

Headline figure: a single 3-curve plot, $x$-axis $\|\Delta x\|$,
$y$-axis `success_rate` with Wilson 95 % CI bars.

### 5.1 What Theorem 1' predicts for this experiment

The Pareto bias–variance theorem
(`docs/theory/03_pareto_bias_variance.md` eq. (11)) decomposes
the hybrid's MSE into a bias term (bounded by the minimum of the
two endpoints' biases; `01_soft_control.md` §3.3) and a variance
term (bounded by $\rho^{-1}\!\times$ pure-MBD's variance;
`02_snis_bound.md` Theorem 1 + Corollary 1).

**Specialised to cost-time perturbation, with $\Delta x$ the
goal-pose displacement:**

| Term in Theorem 1' | Behaviour as $\|\Delta x\|$ grows | Why |
|---|---|---|
| BC bias | **grows monotonically** | demos memorised at $\Delta x = 0$ cannot generalise |
| MBD-no-prior bias | constant $O(M^{-1})$ | SNIS is consistent for any $J$; bias depends on $M$ only |
| **Hybrid bias** | **bounded by min(BC bias, MBD bias)** | inherits MBD's automatic re-targeting through the cost factor |
| MBD-no-prior variance | constant in $\|\Delta x\|$ | $V_w$ depends on proposal $q_k$ and cost $J$; the proposal is unanchored either way, $V_w$ stays large |
| **Hybrid variance** | **constant in $\|\Delta x\|$** | the anchored $V_w(\bar\tau,\beta)$ depends on the prior anchor $\bar\tau$, which is **unchanged by cost perturbation**. **The variance reduction $\rho$ is invariant under goal-pose perturbation.** |

The single most theory-relevant prediction is the bottom-right
cell: **the hybrid's variance term is invariant under cost
perturbation**, so any widening of the hybrid-vs-BC gap with
$\|\Delta x\|$ is **all attributable to the bias term** — i.e.,
to MBD's online cost-re-targeting flowing through the composite
score.

**Falsifies Theorem 1' if:** the hybrid's variance changes
visibly with $\|\Delta x\|$ (measured via seed-to-seed variance
of `success_rate` in a pooled cell). That would indicate
$\rho(\bar\tau)$ has implicit cost dependence we haven't
modelled, requiring a re-derivation of Corollary 1 of
`02_snis_bound.md` to admit a $J$-dependent prior anchor.

**Substantiates Theorem 1' if:** the hybrid-vs-BC succ gap
widens monotonically with $\|\Delta x\|$ while the
hybrid's per-cell standard error stays within the canonical
$\sim\!2\,$pp envelope (Slate E pooled). This is the
"clean adaptation" outcome.

## 6. Falsification protocol

The cleanest "adaptation claim is false" outcome:

- Hybrid degrades at the **same rate** as BC under perturbation.
  This would mean the prior pull is dominant over the MBD
  correction in the perturbation regime — re-run at smaller
  $\lambda$ to test if the optimal $\lambda$ shifts.
- Hybrid degrades **faster** than BC. Would indicate the
  perturbation breaks MBD's rollout-simulator alignment with the
  cost (e.g., the goal body moved but the cost still reads from a
  stale reference); a wiring bug, not a real result.

Either outcome is informative and easy to triage given the
`mean_object_to_goal_distance_at_final_step` diagnostic.

## 7. Status

- 2026-05-25: branch created, flag + stub + spec landed.
- 2026-05-25: body-name registry + translation logic +
  worker-propagation patch landed (commits `8a00b92`,
  `05edaf6`, `716630a`).  Regression test
  `test_worker_perturbation.py` PASS.
- 2026-05-25 23:11 PT - 2026-05-26 ~05:00 PT: slate fired
  in screen `slate_costtime`, 8 cells (~5.5 h wall, minus
  ~25 min SIGSTOP pause during v6 cleanup).  All 12 JSONs
  landed.
- 2026-05-26: **slate complete — negative result for the
  primary hypothesis.** Headline (single seed=0, Wilson 95% CIs):

  | $\Delta x$ (m) | hybrid | BC | pure MBD | hybrid-BC gap |
  |---:|---:|---:|---:|---:|
  | 0.00 | 73.3 % | 26.7 % | 6.7 %  | +46.7 pp |
  | 0.05 | 71.7 % | 26.7 % | 10.0 % | +45.0 pp |
  | 0.10 | 50.0 % | 21.7 % | 10.0 % | +28.3 pp |
  | 0.20 | 0.0 %  | 0.0 %  | 1.7 %  |   0.0 pp |

  - Spec §5.1 substantiation criterion ("gap widens
    monotonically with $\|\Delta x\|$"): **NOT MET.** Gap
    narrows monotonically.
  - Spec §6 falsification criterion 2 ("hybrid degrades
    faster than BC"): **TRIGGERED** ($z = 2.50$,
    $p \approx 0.012$ at $\Delta x = 0.10$).
    *But* not via a wiring bug — the regression test passes
    and the control cell reproduces canonical Slate E.
  - **Identified mechanism:** *fixed-anchor bias*. Can/ph
    obs does not contain the goal pose, so `demo_1nn`'s
    retrieved demo is identical across $\Delta x$. The
    prior anchor $\bar\tau$ is implicitly pinned to the
    canonical goal; as $\pi^\star$ moves, $\bar\tau$ stays,
    and the prior becomes biased away from the new optimum.
    Theorem 1' eq. (10) variance side holds, bias side dominates.
  - **What still holds from the spec:** hybrid invariance at
    $\Delta x \le 0.05$ m (within the demo bank's natural
    ~10 cm scatter); hybrid Pareto-dominates BC and pure-MBD
    at every working $\Delta x$.

- Full write-up:
  `experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/RESULT.md`.
  Follow-ups (Tier 1: multi-seed at $\Delta x = 0.10$; Tier 2:
  goal-conditioned retrieval) queued in
  `experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/NEGATIVE_RESULT_FOLLOWUP.md`.
  Headline plot: `cost_time.png` / `cost_time.pdf`.
- Aggregation: `aggregate_costtime.py` (Wilson CIs + 3-curve
  plot + AGGREGATE.json + aggregate.csv).
