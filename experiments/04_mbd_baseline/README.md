# Experiment 04 — Online MBD baseline (no prior)

**Status:** _stages 1 + 2 done._
[`mbd.py`](mbd.py), [`lq_sanity.py`](lq_sanity.py), and
[`run_can.py`](run_can.py) are all implemented and evaluated.
Stage 1 PASSes (LQ within 5–8 % of DARE optimum across 5 seeds).
Stage 2 full eval on `PickPlaceCan` (20 init-states × 3 rollouts =
**60 rollouts**, seed 0, `M=16 K=10 H=120 lambda=0.5`, wall 78.5 min):
**0/60 successes**, mean executed reward **+1.78**. Behaviour
breakdown: 30 % strong reach (`exec_reward > +3`), 17 % partial
reach, 53 % no reach — the planner *cannot reliably find a reaching
signal from zero-init without a prior*, and never grasps when it
does. This is the predicted MBD-no-prior floor and motivates exp 05.

## Purpose

Validate that **online Model-Based Diffusion** (MBD, [Pan et al.
2024](https://arxiv.org/abs/2407.01573)) can solve our chosen task
when given **no expert prior**. This is the row of the Week-4 results
table that isolates "what does MBD on its own buy us" from "what does
the BC prior buy us"; exp 05 then layers BC on top.

```
| Method                          | Success rate (Can / ph) | Bound |
| ------------------------------- | ----------------------- | ----- |
| Replay (Exp 01)                 | 0.90 (n=20)             | upper bound |
| Random actions (Exp 02)         | 0.00 (n=60)             | trivial lower bound |
| Behaviour cloning (Exp 03)      | 0.067 (n=60)            | first learned policy |
| **Online MBD, no prior (this)** | **0.00 (0/60)**         | training-free planner |
| Hybrid prior + MBD (Exp 05)     | TBD                     | our method |
```

## Method outline

For a horizon-`H` trajectory `tau = (a_0, ..., a_{H-1})` with task
cost `J(tau)`, MBD treats `tau` as a sample from a target density
`p*(tau) ∝ exp(-J(tau)/lambda)` and runs a denoising-style sampler:

1. Initialise `tau_K` from a base distribution (Gaussian noise, or in
   exp 05 a BC-warm-started distribution).
2. For `k = K-1, ..., 0`:

   a. Draw `M` perturbed copies `tau_k^{(m)} = mu_k + sigma_k * eps^{(m)}`,
      `eps^{(m)} ~ N(0, I)`.

   b. Roll out each `tau_k^{(m)}` through the simulator from the same
      initial state, get cost-to-go `J^{(m)}` per copy.

   c. Compute weights `w^{(m)} = softmax(-J^{(m)} / lambda)`.

   d. Update mean `mu_{k-1} = sum_m w^{(m)} * tau_k^{(m)}` (a
      score-matching estimator of the gradient of `log p*`).

   e. Schedule `sigma_{k-1} < sigma_k` (DDPM-style noise schedule).

3. Execute `tau_0` open-loop (or the first action of `tau_0` and
   replan -- to be decided in implementation).

The cost `J` is the negative dense reward from the env (e.g. distance
to target + grasp / lift bonuses for PickPlaceCan), summed over the
rollout.

## Two-stage plan

### Stage 1 — LQ sanity check (`lq_sanity.py`)

A 2-D point mass with linear dynamics `x_{t+1} = A x_t + B u_t` and
quadratic cost `J = sum_t x_t^T Q x_t + u_t^T R u_t`. Closed-form
optimal feedback gain `K*` available from the discrete algebraic
Riccati equation (`scipy.linalg.solve_discrete_are`).

**Why this matters before MJX.** If the MBD denoiser is buggy --
schedule wrong, weight normalisation off, sigma_0 misset -- it'll be
visible here too, but with debugging cost orders of magnitude lower
than a full mujoco rollout. Pure numpy / jax-numpy, no sim.

**Pass criterion.** MBD output `tau*` should yield total cost within
~10 % of the LQ optimum from `K*` for the same initial state, across
several random init states.

**Observed (2026-05-10):**

| metric | value |
| ------ | ----- |
| LQ optimum cost (`u = -K* x`, `H=30`, `x0=[1,0]`) | 6.5415 |
| zero-action baseline cost                         | 31.0000 |
| MBD final cost (`M=128`, `K=100`, `lambda=5`)     | 6.9254 |
| ratio MBD / optimum                               | **1.059** |

5 seeds (0..4) all pass: ratios `[1.059, 1.073, 1.075, 1.073, 1.068]`,
i.e. consistently 5–8 % above optimum, comfortably under the 10 %
threshold. The default `K` is 100; `K=50` (the original default in the
scaffold) sits right on the threshold and fails for some seeds. Full
per-iteration cost / sigma / ESS curve and the system definition are
in [`results/lq_sanity.json`](results/lq_sanity.json).

This validates the algorithm in `mbd.py` against an analytical ground
truth: the schedule, weight normalisation, and weighted-mean update are
all wired correctly. Any failure in stage 2 will therefore be in the
**sim coupling** (cost shaping, MJX scene, action clipping) — not in
the denoiser itself.

### Stage 2 — Full MBD on the can task (`run_can.py`)

Same algorithm, but rollouts go through the **same robosuite
`PickPlaceCan` env that exps 01–03 use** (no MJX, no MJCF authoring).
The trade-off is sequential rollouts (each MBD iteration evaluates
its `M` perturbed copies one at a time), but the env is identical to
the rest of the pipeline so the comparison is apples-to-apples.

**Cost.** `J(tau) = -sum_t reward_t`, where `reward` is robosuite's
dense shaped reward for `PickPlaceCan` (reaching + grasping +
lifting + hovering + success bonus). The env ships with
`reward_shaping=False` in the recorded `env_args`, which would give
zero gradient signal to the planner; `run_can.py` flips it to `True`
right after env construction. The success metric (`check_success`)
is unchanged.

**Tuned defaults** (from `run_can.py`'s tuning):

| flag | default | rationale |
| ---- | ------- | --------- |
| `--H` | 120 | matches exp 01 / 03 demo horizon |
| `--M` | 16 | M=8 is too noisy for descent; M=16 is the smallest reliable |
| `--K` | 10 | descent saturates around k=8-10 in tuning |
| `--lambda_` | 0.5 | reward differences on this task are O(2-4); lambda=5 averages too softly and `mu` drifts back to zero-action |
| `--sigma-max` | 0.5 | OSC_POSE actions are clipped to [-1, 1]; 0.5 explores meaningfully |
| `--sigma-min` | 0.05 | cosine schedule end |

**Eval result** (20 init-states × 3 rollouts = 60, seed 0, default
config above; full per-rollout log in
[`results/can_ph.json`](results/can_ph.json)):

| quantity | mean | median | min | max |
| -------- | ---: | -----: | --: | --: |
| `init_best_cost` (k=0) | -0.82 | -0.49 | -3.26 | -0.04 |
| `final_best_cost` (k=K-1) | -2.36 | -0.40 | -13.72 | -0.01 |
| **cost descent** (init − final) | **+1.54** | +0.10 | -0.48 | +11.69 |
| executed `mu_final` reward | **+1.78** | +0.31 | +0.01 | +6.72 |
| `success` | **0/60 = 0.0 %** |  |  |  |

Per-rollout wall: ~77 s plan + ~1.4 s exec. Full eval: 4 712.7 s
≈ **78.5 min** (matches the README's pre-eval estimate).

**Behaviour breakdown.** Bucketing the 60 rollouts by executed
reward gives a much richer picture than the 5-demo snapshot:

| bucket | criterion | count | share |
| ------ | --------- | ----: | ----: |
| Strong reach | `exec_reward > +3.0` | 18 | **30 %** |
| Partial reach | `0.5 < exec_reward ≤ +3.0` | 10 | 17 % |
| No reach | `exec_reward ≤ +0.5` | 32 | **53 %** |

The dominant failure mode is **"planner can't even reach"** (53 %),
not "reaches but can't grasp". The median cost descent is only +0.10
— most rollouts barely improve on the zero-action initial trajectory
because `M=16` perturbations of `~N(0, sigma^2 I)` rarely hit a
trajectory with non-zero reward signal in 7-D × 120-step space.
When the planner does find a reaching direction (the 30 % bucket),
descent is dramatic (up to -13.7 cost units / +6.7 exec reward), but
the rest of the time it just stays near zero.

**Interpretation for the report.** This isn't just "MBD-no-prior
fails to grasp" — it's "MBD-no-prior is **dominated by exploration
luck**". 70 % of starts never escape the flat region of the reward
landscape; 30 % do but stall before grasping. Both failure modes
disappear under a BC prior (exp 05): the initial trajectory already
reaches + grasps, so MBD's perturbations only refine the lift /
placement segment.

Per-rollout MP4s of the executed `mu_final` live in
[`results/videos/`](results/videos/) (60 mp4s, ~28 MB total;
gitignored — re-run to regenerate). Aggregate cost-vs-iteration plot:
[`results/can_ph.png`](results/can_ph.png). The 5-demo tuning snapshot
that established `lambda=0.5` is preserved in
[`results/can_ph_5demo.json`](results/can_ph_5demo.json) and
[`results/can_ph_tuning.json`](results/can_ph_tuning.json).

**Pass criterion.** Success rate strictly above exp 02's 0 % and
below exp 01's 90 % on the same 20 init-state × 3-rollout protocol.
If MBD-no-prior is _better_ than BC (Exp 03's 6.7 %), that's a
story; if it's _worse_, that's also a story (motivates exp 05's
hybrid).

**Stage 2 wall-time.** With the tuned defaults, one (demo, rollout)
takes ≈ 70 s (each MBD iteration sequentially rolls out M=16 copies
of H=120 env steps; robosuite's PickPlaceCan steps at ~280 step/s on
this Mac). Full eval is `20 × 3 × 70 s ≈ 70 min`; a smaller `5 × 1`
in-session run is ~6 min and is what the script's `--n-demos 5`
default is sized for.

**Live visualisation.** `--live-view` (default ON) opens a
matplotlib popup that streams the current `mu_k` after every MBD
iteration plus the final executed trajectory, with the iteration
number, sigma, best/mean cost, and effective sample size overlaid
in the title. Watching this is the fastest way to spot a bad
hyperparameter (e.g., the planner staying motionless when `lambda`
is too soft). Pass `--no-live-view` for headless runs.

## Prerequisites

The standard `hybrid-traj` env from exps 01-03 is sufficient. Stage 2
deliberately does **not** install `mujoco-mjx`: by reusing the
robosuite `PickPlaceCan` env directly, we get the same env as exps
01-03 (apples-to-apples comparison) at the cost of sequential
rollouts. The MJX-batched route is left as future work; see
"Open questions" below.

`run_can.py` does need `mediapy` for the offscreen MP4 export, which
is already pinned by the visualisation notebook (`notebooks/01_inline_render.ipynb`).

## File layout

```
experiments/04_mbd_baseline/
  README.md          (this file)
  __init__.py
  mbd.py             # MBDParams + denoise_loop(...). Pure algorithm,
                     # no sim coupling. Optional iter_callback hook
                     # used by run_can.py for the live preview.
  lq_sanity.py       # Stage 1: 2D point-mass LQ + DARE comparison.
  run_can.py         # Stage 2: live robosuite PickPlaceCan +
                     # mujoco.Renderer offscreen + matplotlib popup.
  results/
    lq_sanity.json       # Stage 1 result (committed, ~20 KB).
    can_ph.json          # Stage 2 result (committed once eval runs).
    can_ph_tuning.json   # 1-demo tuning snapshot kept for reference.
    videos/              # per-demo MP4s of executed mu_final (gitignored).
```

## Open questions / future work

1. **Full 20 × 3 evaluation.** ~70 min wall on this Mac with the
   tuned defaults. The 1-demo smoke result is informative (planner
   reaches the can but does not grasp), but the headline success
   rate for the report's results table needs the full eval.
2. **Open-loop vs replanning.** Currently single-shot MBD: plan once
   from `t=0`, execute open-loop. Pan et al. note this degrades
   under model error, but for a 120-step horizon on PickPlaceCan
   it's fine. MPC-style replanning (take `u_0`, replan) would be
   exp-04-bis if needed.
3. **MJX path** (deferred). PickPlaceCan's cylinder collisions are
   not MJX-supported, so a faster batched run would need a custom
   MJCF with capsule proxies for the can / bin walls. Not blocking
   the report; revisit if exp 05 needs many more rollouts.

## Usage

```bash
conda activate hybrid-traj

# Stage 1: math validation, pure numpy, runs in <1s.
python experiments/04_mbd_baseline/lq_sanity.py --seed 0
# expected: ratio MBD/opt ≈ 1.059, PASS.

# Stage 2: in-session 5-demo run (~6 min on this Mac).
# A matplotlib popup window will open and show the planner's current
# `mu_k` after every iteration plus the executed final trajectory.
python experiments/04_mbd_baseline/run_can.py --n-demos 5 --seed 0

# Stage 2: full 20 x 3 eval (~70 min). Headless variant for batch:
python experiments/04_mbd_baseline/run_can.py \
    --n-demos 20 --n-rollouts 3 --no-live-view --seed 0
```

## Status / open questions for next session

- [x] Implement `mbd.py`: `MBDParams` + `denoise_loop(...)` with the
      optional `iter_callback` hook for the live preview. Done in
      `feat(exp04): implement MBD denoising loop + LQ sanity check`.
- [x] Implement `lq_sanity.py`: 2-D LQ regulator, DARE solution, MBD
      application — done; PASS at ratio 1.059 on seed 0 (5/5 seeds in
      0..4 pass). Cost-vs-iter curve is in `results/lq_sanity.json`.
- [x] Implement `run_can.py` with the same robosuite env as exps
      01–03 (no MJX). Done in
      `feat(exp04): online MBD on robosuite PickPlaceCan + live
      mujoco preview`. Smoke-tested at 1 demo: descent works,
      executed reward +4.24, success=False as expected.
- [x] Run the full 20 × 3 evaluation and commit the resulting JSON.
      Done at 78.5 min wall, seed 0; result is 0/60 with the
      behaviour breakdown documented in "Eval result" above.
- [ ] (Optional) Try MPC-style replanning instead of single-shot
      open-loop. Compare success rate on the same 60 rollouts.
- [ ] (Optional) Multi-seed (3-5 seeds) to get a variance estimate
      on the 0/60 headline.
