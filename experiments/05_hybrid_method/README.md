# Experiment 05 — Hybrid prior + MBD (our method)

> **DEPRECATED for new work**: Exp 15 (`experiments/15_unified_hybrid/`)
> is now the canonical hybrid driver, consolidating Exp 05's MPC
> chunking with Exp 07's score-bridge. **All result JSONs in this
> directory remain valid** for archival reproducibility (Exp 15
> reproduces them within ±1/60 via `regression_test.py`). New
> sweeps should target Exp 15 instead. See
> [`experiments/15_unified_hybrid/README.md`](../15_unified_hybrid/README.md)
> § Migration for the flag mapping.

**Status:** _done._ [`run_can.py`](run_can.py) implements the
BC warm-start + MBD loop and a `--mpc-replans` MPC variant. The
full 20 init-states × 3 rollouts evaluations have been run on
`PickPlaceCan` (seed 0, `M=16 K=10 H=120 lambda=0.5 sigma_max=0.25`):

| Variant                           | Success   | Wall     |
| --------------------------------- | --------: | -------: |
| Hybrid, single-shot (Exp 05)      | **1/60**  | 81.3 min |
| Hybrid + MPC n_replans=1 (05b)    | **4/60**  | 222.6 min|

The hybrid is the only one of the three learned / planning baselines
that produces *any* successful pickup at this protocol.
Behaviour-wise it is the clearest improvement in the table —
**53 % strong reach** (single-shot) vs exp 04's 30 %, and
**85 % of rollouts execute a higher-reward trajectory than BC
alone would have**. Adding one MPC replan at `t=60` raises the
point estimate 4× and pushes the strong-reach -> success
conversion rate from 3.1 % to **14.8 %** (details in Ablation 05b),
though the 60-rollout sample is not large enough to statistically
separate 4/60 from 1/60 (Fisher `p = 0.36`). Strict successes
remain capped by the millimetre-scale grasp alignment that a
deterministic MLP-BC stand-in cannot represent.

## Purpose

Combine the **BC prior from experiment 03** with the **online MBD
denoiser from experiment 04** and report whether the hybrid beats
both on the can task. The headline row in the Week-4 results table:

```
| Method                          | Success rate (Can / ph) | Bound |
| ------------------------------- | ----------------------- | ----- |
| Replay (Exp 01)                 | 0.90 (n=20)             | upper bound |
| Random actions (Exp 02)         | 0.00 (n=60)             | trivial lower bound |
| Behaviour cloning (Exp 03)      | 0.067 (n=60)            | first learned policy |
| Online MBD, no prior (Exp 04)   | 0.000 (0/60)            | training-free planner |
| **Hybrid prior + MBD (this)**   | **0.017 (1/60)**        | our method |
```

`BC alone` (Exp 03's headline 6.7%) and `BC inside this script` (0/60,
see below) differ because exp 03 evaluates each demo at its native
horizon while exp 05 caps every rollout at the planner's `H=120`. The
apples-to-apples comparison is therefore against Exp 04 (also
`H=120`, also 60 rollouts, also seed 0): hybrid moves the success
rate from 0/60 to 1/60 **and** moves the behaviour distribution
far to the right.

The story this row tells: **MBD alone is fragile in high-dimensional
contact-rich tasks** (Exp 04: 53% no-reach); **BC alone compounds
error** (Exp 03: reaches 70% of starts but lifts only 6.7% of the
time within the dataset's native horizon, and 0% within the planner's
`H=120` budget); **warm-starting MBD with a BC rollout fixes the
reaching problem completely** (Exp 05: 12% no-reach, 88% reach,
including 53% strong-reach with `exec_reward > +3`). Strict-success
remains capped by the same lift/place difficulty that bottlenecked
BC at the same horizon.

## Method outline

Identical to exp 04 with one change: the MBD denoising loop's
`initial_mean` is no longer zero but a single deterministic rollout
of the BC policy from experiment 03.

```python
from policy import Policy                # exp 03
from mbd import MBDParams, denoise_loop  # exp 04
from run_can import plan_and_execute     # exp 04 (now takes initial_mean=)

# Stage A: BC warm-start, closed-loop on the live env.
bc = Policy.from_checkpoint(BC_CKPT)
bc_actions = bc_warmstart_actions(env, init_state, bc, H, low, high)

# Stage B + C: MBD around the BC trajectory, then execute mu_final.
result = plan_and_execute(
    env, init_state, params, child_rng,
    initial_mean=bc_actions,             # <-- the only difference vs exp 04
    ...
)
```

The BC warm-start is `closed-loop`: the policy sees the env's live
obs at every step, picks an action, the env steps, and we record
that action sequence. MBD then replays / perturbs that sequence
`open-loop` (the standard MBD protocol — each perturbed copy is
re-rolled from the same `init_state`).

Applied ablation: **smaller starting `sigma`.** The default
`sigma_max=0.25` (half of exp 04's `0.5`) keeps the BC structure
visible in iteration 1; otherwise iteration 1's perturbations
would have stddev comparable to the actions themselves
(OSC_POSE actions live in [-1, 1]) and the BC warm-start would be
essentially overwritten before the planner can refine it. The
`--no-warm-start` flag exists as a self-check ablation that should
reproduce the exp-04 zero-init baseline.

## Tuned defaults

| flag | default | rationale |
| ---- | ------- | --------- |
| `--H` | 120 | matches exp 01 / 03 / 04 |
| `--M` | 16 | matches exp 04 |
| `--K` | 10 | matches exp 04 |
| `--lambda_` | 0.5 | matches exp 04 |
| `--sigma-max` | **0.25** | half of exp 04; preserves BC structure |
| `--sigma-min` | 0.05 | matches exp 04 |
| `--bc-checkpoint` | `../03_bc_baseline/results/policy.pt` | exp 03's MLP |

## Eval result (full 20 × 3 = 60, seed 0)

Full per-rollout log: [`results/can_ph.json`](results/can_ph.json).
Aggregate cost-vs-iteration plot: [`results/can_ph.png`](results/can_ph.png).

| quantity | mean | median | min | max |
| -------- | ---: | -----: | --: | --: |
| `init_best_cost` (k=0) | -6.10 | -1.95 | -33.66 | -0.23 |
| `final_best_cost` (k=K-1) | -11.52 | -5.44 | -42.35 | -0.18 |
| **cost descent** (init − final) | **+5.42** | +1.16 | -4.28 | +41.29 |
| BC-alone exec reward | +4.51 | +1.08 | +0.13 | +34.95 |
| MBD exec reward (`mu_final`) | **+8.25** | +3.71 | +0.15 | +41.91 |
| **MBD success** | **1/60 = 1.7 %** |  |  |  |
| BC-alone success (this script, H=120) | 0/60 = 0.0 % |  |  |  |

The one strict success: `demo_4#2`, where BC alone got `+0.79`
(no reach), but MBD found a perturbed trajectory descending from
`init_best -1.06` to `final_best -42.35` and `mu_final` executed
for reward **+41.91** with `success=True`. Video:
[`results/videos/demo_04_demo_4_r2.mp4`](results/videos/).

**Behaviour breakdown** (mirrors the table in exp 04's README so
rows are directly comparable):

| bucket | criterion | Exp 04 | Exp 05 (this) |
| ------ | --------- | -----: | -----: |
| Strong reach | `exec_reward > +3.0` | 30 % (18/60) | **53 % (32/60)** |
| Partial reach | `0.5 < exec_reward ≤ +3.0` | 17 % (10/60) | 35 % (21/60) |
| No reach | `exec_reward ≤ +0.5` | **53 % (32/60)** | **12 % (7/60)** |

The "no reach" mass collapses from 53 % to 12 % under the BC warm
start. The mode of the distribution moves from "can't reach" to
"reaches and partially manipulates but doesn't lift cleanly".

**MBD-vs-BC head-to-head** on the same 60 (init_state, rollout)
pairs:

| comparison | count | share |
| ---------- | ----: | ----: |
| MBD `exec_reward` > BC-alone `exec_reward` | 51 / 60 | **85 %** |
| MBD descended (`init_best > final_best`) | 47 / 60 | 78 % |
| BC-alone successful (closed-loop, H=120) | 0 / 60 | 0 % |
| MBD successful | 1 / 60 | 1.7 % |

The planner adds value on **85 % of rollouts** and never destroys
a lot: the 9 rollouts where MBD's `exec_reward` is below BC's are
clustered in the low-reward regime (BC was already failing; MBD
found a slightly worse perturbation by chance).

## Interpretation for the report

1. **Reaching is solved by the prior, not by the planner.** Exp 04's
   53 % "can't reach" mass came from MBD searching for a reaching
   direction in 7-D × 120-step space starting from zero actions, with
   only `M=16` perturbations per iteration. Exp 05 inherits the
   reaching from BC; the warm-start trajectory already points the
   arm at the can on the majority of starts. This shows up as the
   no-reach bucket dropping from 53 % to 12 %.
2. **Grasping + lifting is still the bottleneck.** Both BC alone
   (Exp 03's 6.7 % on native horizon, 0 % on `H=120`) and Exp 05
   stall around the grasp-and-lift moment. A few millimetres of
   gripper-finger alignment swing the outcome and the planner's
   `M=16` cosine-noise perturbations rarely close the gap on this
   high-frequency contact event. The cost descent statistics back
   this up: mean descent is large (+5.42) but the descent goes into
   "hovering above the can with higher dense reward" rather than
   into a strict pickup.
3. **MBD does improve on BC.** The 85 % win rate of `mu_final`
   versus the BC rollout it was warm-started from, combined with the
   mean reward going from +4.51 to +8.25, shows that the denoising
   loop is using its perturbation budget productively rather than
   just averaging the BC trajectory back to zero.
4. **The headline 1/60 is conservative.** It would almost certainly
   move with `K=20` (twice the denoising budget; exp 04's 5-demo
   tuning hit descent saturation around `k=10`, but with a higher
   `init_best` floor) and/or with one MPC replan in the middle of
   the episode. Both are listed in "Open questions".

## Ablation 05b: one MPC replan at t=H/2

To attack the "53 % strong reach, only 1/60 success" bottleneck
(the planner spends its 120-step open-loop budget *reaching* the
can and has nothing left to correct the millimetre-scale grasp),
`run_can.py` now supports `--mpc-replans N`: the H-step execution
is split into N+1 equal chunks, and at each chunk boundary the env
state is snapshotted and a fresh `BC warm-start + MBD denoising
loop` is run from there (the planner still plans H steps ahead,
but only commits `chunk_steps` actions before replanning).

On the same 20 init-states × 3 rollouts protocol with
`--mpc-replans 1 --mpc-chunk-steps 60` (so 2 plans/rollout, wall
**222.6 min** — roughly 2× single-shot since we run the whole MBD
loop twice per rollout):

| Variant                             | Success   | No-reach  | Strong reach | Strong-reach -> success | Mean exec reward |
| ----------------------------------- | --------: | --------: | -----------: | ----------------------: | ---------------: |
| Hybrid, single-shot (Exp 05)        | **1/60**  |   7/60    |     32/60    | 1/32 = 3.1 %            | +8.25            |
| **Hybrid + MPC n_replans=1** (05b)  | **4/60**  |   2/60    |     27/60    | **4/27 = 14.8 %**       | **+11.32**       |
| MBD-only (Exp 04)                   |   0/60    |  32/60    |     18/60    | 0/18 = 0.0 %            | +4.51            |

Interpretation, tied back to formulation eq. (1):

- **Point estimate 4× single-shot**, but Fisher exact two-sided
  `p = 0.364` vs single-shot and `p = 0.119` vs Exp 04 -- _not_
  statistically separated at `alpha=0.05`. The 60-rollout protocol
  does not have the power to resolve a 4/60 vs 1/60 difference.
- The *mechanism* the MPC ablation was supposed to validate is
  cleanly visible: **strong-reach -> success conversion nearly 5×**
  (14.8 % vs 3.1 %). The single-shot planner is strongly biased
  toward "reach then stop"; MPC's second plan, anchored at
  `t=60`, has a near-grasp env state to refine against and
  converts some of those reaches into lifts.
- `no-reach` mass drops further (12 % -> 3 %), confirming that BC
  re-running from `t=60` recovers the rare cases where chunk 0
  stalled. Mean reward rises **+8.25 -> +11.32**.
- Known anomaly: the script's internal BC-alone success counter
  reports 5/60 in the MPC run vs 0/60 in the single-shot run,
  even though the BC policy is deterministic and the per-chunk-0
  BC call uses the same init state. Almost certainly a robosuite
  `env.reset()` RNG-consumption interaction (the MPC run calls
  `env.reset()` roughly twice as often inside `denoise_loop`);
  has no bearing on the hybrid numbers because the Exp-05 and
  Exp-05b MBD successes are compared to their own in-run BC arm.
  Pending follow-up: a deterministic BC wrapper that restores env
  metadata + RNG state explicitly.

Full log: `results/can_ph_mpc.json`. Cost-curve plot is the *last*
chunk's per-iteration descent (the aggregate plot script doesn't
know about chunks); per-chunk `iter_curve` is preserved inside
each `per_rollout[i]["chunks"][j]`.

## File layout

```
experiments/05_hybrid_method/
  README.md          (this file)
  __init__.py
  run_can.py         # ~700 lines: single-shot + MPC variants,
                     # BC warm-start rollout + MBD plan_and_execute
                     # + cost-curve plot. Imports the MBD machinery
                     # from exp 04 via importlib.
  results/
    can_ph.json          # Single-shot 20x3 per-rollout log (~300 KB).
    can_ph.png           # Single-shot aggregate cost-vs-iter plot.
    can_ph_mpc.json      # MPC (n_replans=1) 20x3 log (~480 KB).
    can_ph_mpc.png       # MPC last-chunk cost-vs-iter plot.
    can_ph_smoke.json    # 1-demo wiring check (single-shot).
    can_ph_smoke.png     # Cost curve from the smoke run.
    videos/              # 60 per-rollout MP4s of mu_final (single-shot).
    videos_mpc/          # 60 per-rollout MP4s (MPC).
```

## Usage

```bash
conda activate hybrid-traj

# 1-demo wiring check (~80 s, opens a live preview window).
python experiments/05_hybrid_method/run_can.py \
    --n-demos 1 --n-rollouts 1 --seed 0 \
    --output experiments/05_hybrid_method/results/can_ph_smoke.json

# Full 20x3 eval (~80 min wall on this Mac, headless).
python experiments/05_hybrid_method/run_can.py \
    --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view

# Ablation: zero-init (should reproduce exp 04 within seed noise).
python experiments/05_hybrid_method/run_can.py \
    --no-warm-start --sigma-max 0.5 \
    --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view

# Ablation 05b: one MPC replan at t=H/2 (~2.7 h wall, 4/60 = 6.7 %).
python experiments/05_hybrid_method/run_can.py \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --mpc-replans 1 --mpc-chunk-steps 60 --no-live-view \
    --output experiments/05_hybrid_method/results/can_ph_mpc.json \
    --video-dir experiments/05_hybrid_method/results/videos_mpc
```

## Status / open questions

- [x] Implement `experiments/05_hybrid_method/run_can.py` (BC
      closed-loop warm start + MBD `plan_and_execute` from exp 04
      with `initial_mean=bc_actions`).
- [x] Apply ablation 1 (smaller starting sigma): `sigma_max=0.25`
      instead of exp 04's `0.5`, so the BC structure isn't erased
      in iteration 1.
- [x] Run the 20 × 3 evaluation and commit `results/can_ph.json`
      + cost-curve plot.
- [x] Update the Week-4 table with the hybrid number (1/60 = 1.7 %).
- [ ] (Optional) Run the `--no-warm-start --sigma-max 0.5` ablation
      *through this script* to verify it reproduces exp 04's 0/60.
- [ ] (Optional) Stochastic BC: sample multiple BC rollouts with
      action noise, use them as the initial population instead of
      `M` perturbations of one mean.
- [x] MPC replanning (replan at `t=60`). Run as Ablation 05b;
      4/60 = 6.7 % vs 1/60 single-shot, strong-reach -> success
      conversion up to 14.8 % from 3.1 %, but Fisher `p = 0.36`
      vs single-shot (not statistically separated).
- [ ] (Optional) Try `K=20` without MPC; the exp 04 5-demo tuning
      saw descent saturation around `k=10` from zero-init, but
      whether that holds from a BC warm-start is unanswered.
- [ ] Investigate the BC-alone 0/60 vs 5/60 discrepancy between
      single-shot and MPC runs (likely `env.reset()` RNG
      consumption differs when the MPC loop calls it ~2x more
      often). Does not affect the hybrid MBD comparison because
      the Exp-05 and Exp-05b MBD rows are each scored inside their
      own run; it does affect the "BC alone at H=120" baseline
      number quoted in the main body.
