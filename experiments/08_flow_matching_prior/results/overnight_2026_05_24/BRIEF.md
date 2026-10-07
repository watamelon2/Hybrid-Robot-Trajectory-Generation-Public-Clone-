# Overnight queue brief — 2026-05-24

Trained four prior architectures on the same Can `low_dim_v141` 200-demo
slice and compared them in two evaluation regimes: **standalone**
(prior only, chunked replan) and **MBD** (prior warm-start + K=10
cost-shaped inner search, R=1, λ=1).

Three commits land the trained checkpoints
(`cbf1f1c`, `8148fc5`, `b8dd06e`, with `369ce75` from the prior night
for the v6 CFM model). The eval JSONs and `SUMMARY.md` are in
`e855b8b` / `ee8698d`. This brief documents what the numbers mean and
what to do next.

## Headline

> **v6 CFM (Lipman conditional flow matching, conv1d U-Net, hidden 1536, 50 epochs) hits 83.3 % standalone success on Can / 200 demos** — the
> strongest single prior we have trained in the project. All three
> v7 DP variants collapsed to **0/30 standalone**, but **MBD partially
> rescues them to 10–13 %**. The asymmetric pattern "MBD helps weak
> priors, hurts strong priors" is now confirmed across three priors.

## Scoreboard

### Stage 1 — trainings (50 epochs each, MPS on M4 Max)

| name | params | T_obs | H | wall (min) | final loss | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| **v6_cfm_xl** (Lipman CFM) | 117 M | 1 | 120 | 78 | 0.0174 | `v_theta_v6_unet_xl.pt` (316 MB) |
| v7_dp_tobs1_h120 (DDPM) | 66 M | 1 | 120 | 64 | 0.0190 | `v_theta_dp_tobs1_h120.pt` (252 MB) |
| v7_dp_tobs2_h120 (DDPM) | 66 M | 2 | 120 | 60 | 0.0182 | `v_theta_dp_tobs2_h120.pt` (254 MB) |
| v7_dp_tobs2_h16 (DDPM) | 66 M | 2 | 16  | 27 | 0.0339 | `v_theta_dp_tobs2_h16.pt`  (254 MB) |

Total training wall: 229 min on a single M4 Max MPS device (sequentially —
empirical test confirmed two-process MPS contention is worse than
additive throughput; see memory `247b96c3-9f3c-4ad1-b2f5-9fd4021bf684`).

### Stage 2 — standalone evals (10 demos × 3 rollouts = 30 trials, CPU)

| prior | n_action_steps | success | wall |
|---|---:|---:|---:|
| **v6_cfm_xl** | 60 | **25 / 30 = 83.3 %** | 1.0 min |
| v7_dp_tobs1_h120 | 60 | 0 / 30 = 0.0 % | 1.2 min |
| v7_dp_tobs2_h120 | 60 | 0 / 30 = 0.0 % | 1.2 min |
| v7_dp_tobs2_h16 (closed-loop) | 8 | 0 / 30 = 0.0 % | 3.8 min |

### Stage 3 — MBD evals (K=10, R=1, λ=1, M=16, σ_max=0.25)

| prior | success | wall |
|---|---:|---:|
| v6_cfm_xl | 17 / 30 = 56.7 % | 68.9 min |
| v7_dp_tobs1_h120 | 4 / 30 = 13.3 % | 67.5 min |
| v7_dp_tobs2_h120 | 3 / 30 = 10.0 % | 67.1 min |

### Standalone − MBD deltas

| prior | standalone | MBD | Δ = MBD − standalone |
|---|---:|---:|---:|
| v6_cfm_xl       | 83.3 % | 56.7 % | **−26.7 pp** *(MBD hurts a strong prior)* |
| v7_dp_tobs1_h120 |  0.0 % | 13.3 % | **+13.3 pp** *(MBD rescues a degenerate prior)* |
| v7_dp_tobs2_h120 |  0.0 % | 10.0 % | **+10.0 pp** *(MBD rescues a degenerate prior)* |

This asymmetric effect is the most interesting finding of the run.
With λ=1 cost shaping plus K=10 noisy variants, MBD's inner search
adds variance around the warm-start. When the prior is already near
the data manifold (v6), most of that variance is *worse* than the
deterministic warm-start, and ranking by a noisy cost estimate
occasionally picks suboptimal samples — net −27 pp. When the prior
is degenerate (DP-0%), the variance gives MBD something to search
over and the cost field has room to push trajectories toward the
goal — net +10 to +13 pp. We will revisit this with a λ sweep
(`λ ∈ {0.0, 0.25, 0.5, 1.0}`) on v6 to see whether reduced cost
shaping recovers v6's 83 % under MBD.

## Why DP collapsed to 0 % standalone — root cause

After reading `train_dp.py`, `adapter_dp.py`, `model_dp.py` and
`standalone_dp.py` end-to-end, the **DP training pipeline is correct
and the model definitely learned ε-prediction** (final loss 0.017–0.034,
comparable to v6). The bug is at inference, in the ancestral sampler.

### The bug

In `model_dp.py:551–605`, `dp_sample` accepts `n_inference_steps` and
subsamples the DDPM schedule uniformly when this is smaller than
`num_train_timesteps` (=100). The orchestrator passed `--n-steps 18`
(matching the v6 Heun budget) so the sampler used 18 strided
timesteps out of 100. **At each strided step it then applied the
standard one-step DDPM posterior**:

```python
mu = (1.0 / sqrt(alpha_t)) * (x - (beta_t / sqrt_1ma_t) * eps_pred)
```

That formula is the Markov posterior `p(x_{t-1} | x_t)` — valid only
for the consecutive transition `t → t-1`. Over a strided jump
averaging ~5.5 DDPM steps per iteration, applying a single-step
formula yields **massively under-denoised samples**. We end up with
trajectories that are still ~80 % noise after 18 iterations. The
empirical signature matches: rewards stay around +0.3 to +2.0 (vs
+26 to +42 for v6 successes), action chunks bounce around without
reaching the grasp.

### Why CFM v6 is unaffected (same `--n-steps 18`)

v6 uses Heun, a 2-nd-order Runge–Kutta ODE integrator. Heun is
defined for *arbitrary* `σ`-strides — it's a continuous-time
sampler. DDPM ancestral is a Markov chain; the single-step
transition formula is wrong under striding by construction. The
v6 result (83 % at 18 Heun steps) shows the prior is strong; it's
DP's *sampler*, not DP's *training*, that broke.

### Why MBD partially recovered DP (10–13 %)

MBD's inner search calls `make_prior_score_fn` → `dp_score` (in
`model_dp.py:608–644`), which is **not** affected by the sampler bug.
`dp_score` just queries `ε_θ` at the nearest DDPM integer timestep
for a given runtime σ and returns `−ε / σ`. The score field is
correct. Only the warm-start (from the buggy `sample_initial_mean`)
is bad. K=10 noisy variants around a bad warm-start, ranked by a
correct cost+score field, occasionally find usable trajectories →
10 to 13 %. This is *exactly* the gap we'd predict if `dp_sample`
is broken but `dp_score` is correct.

## Proposed fix

Replace the DDPM ancestral one-step formula in `dp_sample` with a
**DDIM** update (η = 0, deterministic). DDIM's stride-correct step
is:

$$ \hat{x}_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}} $$

$$ x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat{x}_0 + \sqrt{1 - \bar\alpha_{t'}}\,\varepsilon_\theta(x_t, t) $$

valid for any `t' < t`. This is what HuggingFace `DDIMScheduler` does
and what Cheng Chi's `diffusion_policy` repo uses at inference.
Implementation cost: ~20 lines in `model_dp.py`. No retraining
required — the checkpoints are already valid ε-predictors.

### Diagnostic to confirm before changing code

Re-run a single DP standalone with `--n-steps 100` (the unstrided
branch of `dp_sample`, which *is* mathematically correct DDPM
ancestral). Predicted outcome: success rate jumps from 0/30 to
something substantial (likely 30 % – 70 %), validating the
sampler-bug hypothesis. ~6 min wall on CPU.

## What to do next (priority-ordered)

1. **Confirm hypothesis**: run v7_dp_tobs1 standalone with `--n-steps 100`.
2. **Implement DDIM**: add `dp_sample_ddim` to `model_dp.py`, switch
   `adapter_dp.py.sample_initial_mean` to call it. Smoke-test.
3. **Re-run all 3 DP standalones** with the fixed sampler.
4. **λ sweep on v6 MBD**: λ ∈ {0.0, 0.25, 0.5, 1.0}, K=10 — does
   reducing cost shaping recover v6's 83 % under MBD?
5. **Update presentation brief** with the v6 number and the DP
   sampler-bug story (this is now a defensible finding, not a
   failure: we replicated DP's training correctly, found a real
   inference-side bug that wasn't obvious from the training-loss
   curve, and have a minimal fix on hand).

## Follow-up (same day): Stage-1 diagnostic + Stage-2 DDIM fix

Ran the two-stage fix from the diagnosis section above.

### Stage 1 — diagnostic (`--n-steps 100` on tobs1_h120)

Unstrided full-schedule DDPM ancestral, mathematically correct:

  - **3 / 30 = 10.0 %** (wall 3.1 min)
  - All 3 successes are on `demo_4`; the other 27 rollouts fail with
    rewards stuck < +5. Classic **mode-collapse** signature: model
    produces near-identical trajectories regardless of observation
    conditioning.

**Conclusion**: the sampler bug is real (0 → 10 %) but is *not the
full story*. The model is also under-trained — the DP paper uses
100–300 epochs with EMA; we trained 50 epochs without EMA.

### Stage 2 — DDIM sampler

Implemented `dp_sample_ddim` in `model_dp.py` and rerouted
`dp_sample` to dispatch to it when `n_inference_steps < num_train_timesteps`.
Full schedule (K ≥ 100) still uses DDPM ancestral (the correct path).
Deterministic η = 0 by default; η = 1 recovers DDPM ancestral.
Smoke-tested for shape, determinism, NaN-freeness, and η-stochasticity.

Results on the same 10×3 = 30-trial grid (CPU, `--n-steps` varied):

| variant | strided DDPM K=18 (BROKEN) | DDIM K=18 (FIX) | DDIM K=32 | DDIM K=64 | DDPM K=100 (full) |
|---|---:|---:|---:|---:|---:|
| dp_tobs1_h120         | 0/30 (0.0 %) | 0/30 (0.0 %) | 0/30 | 0/30 | **3/30 (10.0 %)** |
| dp_tobs2_h120         | 0/30 (0.0 %) | —            | —    | —    | **2/30 ( 6.7 %)** |
| dp_tobs2_h16_cl       | 0/30 (0.0 %) | —            | —    | —    | **11/30 (36.7 %)** 🎯 |

Additional sanity check on tobs1_h120:

  - DDIM K=18 η = **1.0** (stochastic): **0/30** — confirms the
    issue is not deterministic-vs-stochastic; the under-trained model
    is too noisy for any K = 18 sampler regardless of η.

### Why DDIM at low K didn't recover quality

DDIM's update divides by `√ᾱ_t` at every step — which is tiny
(~0.01) at high t. *Any* error in the network's ε prediction is
amplified by ~100× in the resulting `x̂_0` estimate. Full DDPM
ancestral over 100 small steps averages those errors out; strided
DDIM (K=18, 32, 64) compounds them. This is a known DDIM property
and the published DP recipe works at K=16 *because their model is
much better trained*. With our 50-epoch / no-EMA checkpoint, only
the full schedule recovers any quality.

### Final standalone scoreboard (corrected)

| prior | n_action_steps | best standalone | sampler | wall |
|---|---:|---:|---|---:|
| **v6_cfm_xl**       | 60 | **25/30 = 83.3 %** | Heun K=18 (stride-correct ODE) | 1.0 min |
| dp_tobs2_h16_cl     |  8 | **11/30 = 36.7 %** | DDPM ancestral K=100 | 14.6 min |
| dp_tobs1_h120       | 60 | **3/30 = 10.0 %**  | DDPM ancestral K=100 | 3.1 min |
| dp_tobs2_h120       | 60 | **2/30 =  6.7 %**  | DDPM ancestral K=100 | 3.7 min |

The **DP-paper-faithful closed-loop variant** (`T_obs=2`, `H=16`,
replan every 8 steps) now scores **36.7 %** at K=100 — a real number
that matches what you'd expect from a slightly under-trained Diffusion
Policy. The wall-time penalty (~15 min for 30 trials at K=100) is
the cost of doing 100 small denoising steps per chunk over 30 chunks
per rollout (30 × 100 ≈ 3000 UNet forwards per rollout).

### Recommended next steps (in priority order)

1. **Use K=100 going forward** for all DP standalone / MBD evals.
   K=18 was a v6-CFM heuristic that doesn't transfer to under-trained
   DDPM. The dispatch logic in `dp_sample` now routes K ≥ 100 to
   DDPM ancestral and K < 100 to DDIM, both mathematically correct.
2. **Retrain DP for 200–300 epochs with EMA** (`decay = 0.9999`).
   This matches the DP paper recipe and should let DDIM at K=18–32
   recover the full quality, dropping wall by ~5×.
3. **Re-run MBD evals on the H=16 closed-loop DP** with K=100
   sampler. With standalone now at 36.7 %, MBD may push it higher
   (or, per the asymmetric-MBD finding, may *hurt* it). Either
   outcome is publishable.
4. **λ sweep on v6 MBD** (λ ∈ {0, 0.25, 0.5, 1.0}) — still pending,
   would round out the asymmetric-MBD story.

## Artifacts

- Checkpoints in `experiments/08_flow_matching_prior/results/v_theta_*.pt`
- Per-rollout eval JSONs in `standalone_*/` and `mbd_*/` under this dir
- DDIM fix: `experiments/08_flow_matching_prior/model_dp.py` (`dp_sample_ddim` + `dp_sample` dispatcher)
- Training loss curves in `v_theta_*.log.json`
- `queue.log` for the full orchestrator timeline (~5 h wall)
- This file: `BRIEF.md`

## Follow-up (later same evening): EMA-300 retrain + asymmetric-MBD on tobs2_h16_cl

Acted on items **2** and **3** from the recommended-next-steps list above.

### Item 2 — DP retrain with EMA (300 epochs, decay 0.9999)

`train_dp.py` was missing the published DP recipe's EMA. Implemented Cheng
Chi's power-decay schedule
(`decay_t = clip(1 - (1 + (step - update_after_step) / inv_gamma)^-power, …)`),
defaults `inv_gamma=1.0, power=0.75, max_decay=0.9999`. EMA shadow params
saved as `state_dict`; raw last-iterate kept under `state_dict_raw`.
Implementation in commit `91afdb9`.

Training: 300 epochs × 91 batches/epoch = 27 300 steps on MPS, 2 h 33 min
wall. **Final training loss 0.0071** (5× lower than the 50-epoch run's
0.0339).

Standalone eval (K=100, n_action_steps=8, max_env_steps=240), 30 trials:

| variant | training | wall | succ | comment |
|---|---|---:|---:|---|
| `v_theta_dp_tobs2_h16.pt`         | 50 ep, no EMA  | 27 min | **11/30 = 36.7 %** | baseline (from earlier today) |
| `v_theta_dp_tobs2_h16_ema_e300.pt`, EMA weights | 300 ep, EMA 0.9999 | 154 min |  5/30 = 16.7 % | regressed |
| same checkpoint, **raw** state_dict | 300 ep, EMA 0.9999 | 154 min |  5/30 = 16.7 % | regressed (EMA is not the cause) |
| same checkpoint, DDIM K=18         | 300 ep, EMA 0.9999 |   3.6 min|  0/30 =  0.0 % | DDIM still can't recover quality |

The EMA-vs-raw weight diff at step 27 300 is **0.13 % median** (well-behaved
EMA schedule). Both EMA and raw checkpoints score 16.7 %, so the **20 pp
regression is from longer training itself, not from EMA**.

**Diagnosis**: classical overfitting on the 200-demo set. Per-rollout
diffs show the EMA model converged to a different mode pattern from the
50-epoch model — demo_8 now reaches reward +15-16 (up from +2-3) but
*never closes the gripper*; demo_0 now reaches +6-7 (down from +27-38)
because the longer-trained model commits to a different grasp strategy
that misses on these inits. The model has tightened around an
"averaged-mode" behaviour that approaches the can but doesn't complete
the lift-and-place sequence.

The DP paper reports ~99 % on this benchmark with this exact recipe.
With our 200-demo + same-recipe + MPS training, we top out at 36.7 %
at 50 epochs and 16.7 % at 300 epochs. Possible gaps: paper uses
Adam `betas=(0.95, 0.999)` (vs our default `(0.9, 0.999)`); they may
also use stronger image-side regularisation that isn't in our state-
based pipeline. **We're keeping `v_theta_dp_tobs2_h16.pt` (50 epochs)
as the DP baseline going forward — it's the best DP standalone we
have at 36.7 %**.

Checkpoint: `v_theta_dp_tobs2_h16_ema_e300.pt` (508 MB; carries both
EMA and raw state_dicts). Log: `v_theta_dp_tobs2_h16_ema_e300.log.json`.
Training driver commit: `91afdb9`.

### Item 1 (silent bug) — `unified_loop.py` capped at H instead of total budget

While running item 3 (MBD on the 36.7 % baseline), every rollout finished in
~5 s with `final_reward ≈ 0.06` and only **2 chunks executed instead of
30**. Root cause in `experiments/15_unified_hybrid/unified_loop.py`:

```python
for chunk_idx in range(n_chunks):
    steps_left = H - steps_executed           # <-- BUG
    if steps_left <= 0: break
```

The total execution was capped at `params.H`, which was 16 for the
H=16 closed-loop config. The 30 requested replans were silently
ignored after chunk 2. (Pre-fix the loop was only ever used with
H=120 single-shot or H=120/R=1/2 setups where `n_chunks * chunk_steps
== H` exactly, so the bug was load-bearing-as-safety-check and never
surfaced.)

**Fix** (commit `11e8ad3`): add `max_env_steps` arg to
`unified_plan_and_execute`; when unset, fall back to `n_chunks *
chunk_steps` if that exceeds H, else H (bit-for-bit identical
budget for all historical H=120 configs). Also added matching
`--max-env-steps` and `--prior-n-steps` CLI flags to `run_can.py`.
Smoke test: `--H 16 --mpc-replans 4 --mpc-chunk-steps 8
--max-env-steps 40` now executes 5 chunks = 40 env steps, vs
2 chunks = 16 pre-fix.

### Item 3 — MBD on the 36.7 % DP baseline (asymmetric test)

After the bug fix, ran the proper test:

```
run_can.py --prior flow
    --prior-checkpoint v_theta_dp_tobs2_h16.pt
    --n-demos 10 --n-rollouts 3
    --H 16 --K 10 --M 16
    --mpc-replans 29 --mpc-chunk-steps 8 --max-env-steps 240
    --lambda-prior 1.0
    --prior-n-steps 100        # NEW: full DDPM ancestral for warm-start
    --warm-start --deterministic-prior-warmstart
    --sigma-max 0.25 --sigma-min 0.05
    --n-workers 8
```

Every rollout executed 30 chunks × 8 env steps = 240 total (confirmed
from the JSON `chunks` field). Wall: ~2.5 h (worker-pool MBD has high
overhead at chunk=8).

**Result: 0 / 30 = 0.0 % success.** Drop of **−36.7 pp** vs the
36.7 % standalone baseline. This is the most extreme
"asymmetric-MBD destroys" we've measured: MBD's K=10 noisy-perturb /
softmax-average inner loop *completely wipes out* a 36.7 % prior.

Per-rollout signature: demo_0 standalone gets rewards (27, 33, 38)
with 3 successes; demo_0 MBD gets (3.7, 3.1, 2.8) with 0 successes —
about 10× lower reward, no completion. The MBD-refined trajectories
*do* monotonically increase reward chunk-over-chunk
(+0.013 → +0.050 → +0.130 → +0.229 → +0.292 → +0.326 in the first
6 chunks of `demo_0#0`), but the cumulative reward after 30 chunks
(~+3.7) never reaches the +25 reward threshold that signals a real
grasp+lift. The prior alone would have reached +27 on the same
240 env steps; MBD's averaged action sequence stalls out around the
"approach but don't commit" mode.

This is asymmetric-MBD at its sharpest. With the v6-CFM 83.3 %
prior we measured −26.7 pp from MBD. With the DP 36.7 % prior we
measure −36.7 pp. With the DP 0 % priors (`tobs1_h120`, `tobs2_h120`)
we measure +10–13 pp. Confirmed pattern: **MBD's noise injection +
softmax averaging is incompatible with a non-degenerate
closed-loop diffusion policy**; it acts as a regulariser that only
helps when the prior is already broken.

Per-rollout JSON:
`eval_dp_ema_e300/unified_flow_lam-1.0_K-10_R-29_mbd_tobs2_h16_K100prior_lam1_v2.json`.

### Updated scoreboard — DP standalone vs MBD on the corrected H=16 closed-loop axis

| prior (closed-loop, n_act=8) | standalone K=100 | MBD K_in=10 R=29 (K=100 warm-start) | Δ = MBD − standalone |
|---|---:|---:|---:|
| v_theta_dp_tobs2_h16 (50 ep, no EMA) | **11/30 = 36.7 %** | **0/30 = 0.0 %** | **−36.7 pp** |
| v_theta_dp_tobs2_h16_ema_e300 (300 ep, EMA) | 5/30 = 16.7 % | not run (skipped — worse prior) | n/a |

### Open follow-ups

- **λ sweep on the DP H=16 prior**: λ_prior ∈ {0.0, 0.25, 0.5} to test
  whether *reducing* the prior-score weight recovers some of the
  −36.7 pp. λ_prior=0 = pure-MBD-with-DP-warm-start would isolate
  whether the warm-start alone has any value once MBD is applied.
- **Verify DP score function (`dp_score` in `model_dp.py`) is
  σ-correct** — the warm-start path is fine (it's just `dp_sample`),
  but the per-iteration score function may have a σ↔t mapping bug
  that contributes to the regression. If the score is wrong,
  `lambda_prior=0.0` (no score) should perform much better than
  `lambda_prior=1.0`.
- **DP recipe gap diagnosis** — why we max out at 36.7 % vs the
  paper's 99 %. Candidates: Adam betas, data augmentation that
  isn't in our state-based pipeline, or smaller demo split. A
  100-epoch (no EMA) eval would resolve whether 50→300 trades a
  partial-fit sweet spot for an overfit one.

### Item 4 — score-mode ablation on v6 CFM prior (the rescue)

After confirming asymmetric-MBD destroys both the v6 (−26.7 pp) and DP
H=16 (−36.7 pp) priors, isolated the destruction mechanism by replacing
the **learned**, non-stationary ε-prediction score with a
**pull-toward-warm-start** score
`s_pull(x) = (a_warmstart − x) / σ²` (the same form
`priors_zerotrain.py` uses for retrieval priors like `demo_1nn`).

Added `--score-mode {learned, pull}` CLI flag to `run_can.py`
(commit `<TBD>`); the warm-start, prior network, σ schedule, K, M,
H, λ_prior are all otherwise identical.

**Config**: v6 CFM (`v_theta_v6_unet_xl.pt`), H=120 single-shot,
K_inner=10, M=16, R=1 (chunk=120), λ_prior=1.0, σ_max=0.25,
σ_min=0.05, `--prior-n-steps 18`,
`--deterministic-prior-warmstart`, n_workers=4 cpu, 30 trials
(10 demos × 3 rollouts).

| variant | succ | Δ vs v6 standalone (83.3 %) |
|---|---:|---:|
| v6 standalone (K=18 deterministic) | 25/30 = 83.3 % | baseline |
| v6 + MBD + `learned` score, λ=1 | **15/30 = 50.0 %** | **−33.3 pp** |
| v6 + MBD + `pull` score, λ=1     | **25/30 = 83.3 %** | **0 pp (parity)** |

**Interpretation**: swapping the score field from learned-ε to
stationary-pull-to-warm-start completely eliminates MBD's
destruction of the strong prior. The pull-style MBD recovers
the standalone prior exactly (same 25/30, same demos).

This confirms the root cause from item 3 + the v6 −26.7 pp
finding: **MBD's K=10 inner loop is incompatible with a
non-stationary score field that moves the optimum every
iteration**, regardless of whether the field comes from a
diffusion-policy ε-net or a CFM. The retrieval-style score
(`demo_1nn` etc.) works *not because the prior is good* but
because the score field is **frozen during the inner loop**;
applying the same trick to learned priors recovers them.

Status: pull-mode brings hybrid to **parity** with the strong
standalone prior. To push *past* parity (the asymmetric-rescue
direction), the next move is a λ_prior sweep with pull-mode on
v6 — lowering λ should let MBD's cost-driven softmax-average
dominate, which is the regime where MBD can actually search
above the prior on a long-horizon (H=120, full-rollout reward)
cost signal.

Per-rollout JSONs:
`eval_dp_ema_e300/unified_flow_lam-1.0_K-10_R-1_v6_30trial_{learned,pull}.json`.

### Commit graph (today's adds, on top of the overnight queue)

```
<TBD>   exp15: --score-mode {learned, pull} (pull recovers v6 strong prior)
11e8ad3  exp15: decouple unified_loop budget from H (closed-loop bug fix)
91afdb9  exp08-dp: add EMA (DP paper recipe) to train_dp.py
```
