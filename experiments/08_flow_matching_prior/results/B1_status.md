# B1 training status (feat/principled-score)

**Run date**: 2026-05-14 20:33 PDT.
**Branch**: `feat/principled-score`.
**Plan reference**: [`docs/plan_principled_score.md`](../../../docs/plan_principled_score.md) § Task B1.

## What was run

```
PYTHONPATH=. /Users/szh/anaconda3/envs/hybrid-traj/bin/python \
    -m experiments.08_flow_matching_prior.train \
    --task=can --n-demos=20 --horizon=120 \
    --epochs=2000 --batch-size=64 --lr=2e-4 \
    --hidden=512 --n-layers=3 \
    --seed=0 --device=cpu \
    --output=experiments/08_flow_matching_prior/results/v_theta.pt \
    --log=experiments/08_flow_matching_prior/results/train_log.json
```

## Results

- Wall: **7.1 s** on M4 Max CPU (plan doc's 10-30 h estimate was wildly
  off; the model is too small and the dataset too small for training
  to be the bottleneck).
- Final FM loss: **0.9047** (started at 1.2554).
- Checkpoint: `v_theta.pt` (5.4 MB, **gitignored** per `*.pt` rule;
  rebuildable from this log + the command above).
- Per-epoch loss log: `train_log.json` (114 KB, **tracked**).

## B1.1 LQ sanity test

Passes:
```
[lq] final FM loss: 1.1016
[lq] sample radii: mean=0.881 std=0.242  (target ~1.0, ~0.0)
[lq] lambda=0 backward-compat: OK
[lq] composite (lambda=1) final point: [0.4016, -0.1436]  radius=0.427
[lq] OK
```

## B2.1 smoke (FlowPrior in Exp 07)

```
python -m experiments.07_lambda_sweep.run_can \
    --prior=flow --prior-checkpoint=…/v_theta.pt --smoke
```

Runs end-to-end (no NaN, no crash). 0/2 success at smoke scale,
which is uninformative -- smoke uses `K=4` denoising steps and
`n_demos=2`. **The pipeline integration works**; whether the prior
is *useful* is the B3 question, gated behind the issue below.

## Open issue — FlowPrior sample magnitudes are ~1000x action-space scale

Sampling from `FlowPrior.sample_initial_mean(s0=zeros)` produces
trajectories with element range `[-2924, 3151]` and L2-per-step
≈ 1670. The robomimic `can/ph` action space is bounded `[-1, 1]^7`,
so a single-step action vector should have norm ≤ √7 ≈ 2.65.
The sampled trajectories exceed this by ~600x in mean and ~1000x
in worst case.

Pairwise L2 between 8 same-s0 samples: mean 40, max 42. This is
"different" but the differences are dominated by the diverging
magnitude, not by meaningful trajectory variation. **The
multimodality test in `docs/plan_principled_score.md` § B1.4 is
NOT confidently passed** -- the L2 number nominally clears the
0.1-per-step bar but the underlying samples aren't in the valid
action manifold.

### Hypotheses (untested)

1. **ODE integration step size too large**. The
   `sample_trajectory` function uses a fixed number of Euler steps;
   if the time-step is too coarse, integration diverges.
2. **Conditioning extrapolation**. The test used `s0 = zeros`,
   which after normalisation is `-obs_mean / obs_std` -- far from
   any training s0. The model may extrapolate badly off-distribution.
3. **Output not bounded**. `VelocityField` has no `tanh` head or
   action-bound clamp. Compare to exp 03's MLP-BC which is
   trained against the bounded action targets directly.
4. **Insufficient training**. Final FM loss 0.9047 is in the same
   ballpark as the LQ-sanity converged loss (1.10), but
   convergence is dimension-dependent; FM loss values aren't
   directly comparable across problems.

### Decision

**Do not run B3 (6 h compute) until this is investigated.**
Three things to try, ordered by cheapness:

1. Sample with a *real* `s0` from the training set (rules out
   hypothesis 2). ~5 min.
2. Increase ODE solver steps (`config.n_sample_steps` or
   equivalent). ~10 min if the param exists.
3. Retrain with a `tanh` head on the action output (rules out
   hypothesis 3). ~1 h coding + 7 s training.

If all three fail to bring sample magnitudes into `[-1, 1]^7`,
fall back to Branch B's Task B4 (Gaussian-head BC) as the
half-step principled-score fix.

## Next session for this branch

After Branch C Phase 1 lands and we have the unified driver:

1. **Investigate FlowPrior magnitudes** (see above; 1-2 h
   debugging max).
2. **B3 lambda-sweep on v_theta prior** through unified driver
   (~6 h compute).
3. **B4 Gaussian-head BC** if v_theta still underperforms.
