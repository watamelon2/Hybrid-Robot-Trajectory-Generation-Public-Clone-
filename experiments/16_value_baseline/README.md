# Experiment 16 — `V_φ` state-value baseline as MBD control variate

**Branch A — Task A2** of
[`docs/branch_plan.md`](../../docs/branch_plan.md) on
`feat/stronger-mbd-signal`.

## Hypothesis

Pure MBD (`λ_prior=0`, no BC anchor) on `can/ph` is **0 / 60** at every
configuration tested. Two non-exclusive hypotheses:

- **H1 (capacity):** the MBD denoiser cannot represent the policy
  manifold without a BC warm start.
- **H2 (variance):** the per-step cost is too high-variance for
  MBD's softmax-over-cost weighting to make progress without an
  anchor.

If H2 is right, subtracting a learned **state-value baseline**
`V_φ(s_t)` from per-step costs reduces the variance of the softmax
weighting without changing its expectation (the standard
REINFORCE / A2C / PPO baseline trick). If H2 is wrong, `V_φ` should
have no measurable effect on pure-MBD.

## How `V_φ` is wired into MBD

The per-trajectory cost inside `denoise_loop` becomes

```
c_i_new  =  c_i  -  sum_t V_φ(s_{i,t})
```

with `c_i = -reward_i` as before. The softmax over `c_i_new`
inherits the same expected behaviour but with lower variance,
provided `V_φ` is well-fit. The integration happens at the
`cost_fn` boundary — see `adapter.py`.

The unified driver
(`experiments/15_unified_hybrid/run_can.py`) routes the
`--value-baseline-ckpt` flag here:
when set, it (a) switches `make_rollout_fn` to
`exp 04`'s `make_rollout_fn_with_obs` variant (which collects
per-step `OBS_KEYS` observations alongside the episode reward),
and (b) wraps the base `cost_fn` with
`wrap_cost_fn_with_value_baseline(...)`.

## Files

| File                | Role                                                  |
|---------------------|-------------------------------------------------------|
| `value_model.py`    | `V_φ` MLP class + `ValuePhi` load/save wrapper.       |
| `train.py`          | TD(0) / MC regression on BC-policy rollouts.          |
| `adapter.py`        | `wrap_cost_fn_with_value_baseline` for the planner.   |
| `results/V_phi.pt`  | Trained model + obs stats + meta. Output of train.py. |

We renamed `model.py` to `value_model.py` to avoid clashing with
`experiments/08_flow_matching_prior/model.py` if anyone ever
combines `--prior=flow` with `--value-baseline-ckpt` (which would
push both modules onto `sys.path`).

## How to train

```bash
conda activate hybrid-traj

# Defaults: 500 BC rollouts, 100 epochs, no action noise, seed 0.
# ~3 min for data collection + ~30 s of training on M4 Max CPU.
python experiments/16_value_baseline/train.py \
    --bc-checkpoint experiments/03_bc_baseline/results/policy.pt \
    --output experiments/16_value_baseline/results/V_phi.pt
```

Diagnostics printed at end of training: `RMSE/|mean(R)|`. If this
exceeds 0.5, the regression has not fit the cost-to-go reliably;
see the branch plan's risk-mitigation notes (raise `--n-trajs` to
2000, or add `--action-noise 0.1` for better state-action coverage).

## How to evaluate (the acceptance gate)

The eval cell, per `docs/branch_plan.md`'s A2 spec, is **pure MBD
with `V_φ` as a control variate** at the regime where pure MBD is
currently 0/60:

```bash
python experiments/15_unified_hybrid/run_can.py \
    --K 10 --mpc-replans 0 --lambda-prior 0 \
    --sigma-max 0.05 --M 64 --prior bc \
    --no-warm-start \
    --value-baseline-ckpt experiments/16_value_baseline/results/V_phi.pt \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --n-workers 2 \
    --output-dir experiments/16_value_baseline/results/ \
    --output-name-suffix "_pure_mbd_with_vphi"
```

Acceptance bar from
[`docs/branch_plan.md` § A2](../../docs/branch_plan.md):

- **H2 supported (success):** ≥ **5 / 60 = 8.3 %**. Distinguishes
  "control variate helped" from "control variate didn't help" by
  Wilson 95 % CI.
- **H1 supported (failure):** ≤ **3 / 60 = 5.0 %**. Pure MBD with
  V_φ still essentially zero ⇒ accept H1 (model capacity is the
  bottleneck, not variance); document as a negative result.

The eval takes ~2 h on M4 Max with `n-workers=2`. To avoid
oversubscription, run after A1.4's σ-sweep cells finish.

## Open questions

Carried over from `docs/branch_plan.md` and from the source
archive plan
(`docs/archive/plan_stronger_mbd_signal.md`):

1. **Training data scale.** 500 trajectories × ~120 steps =
   60 k transitions might be too few for a 2-layer MLP to converge
   cleanly. If TD residuals plateau at high variance (`RMSE /
   |mean(R)| > 0.5`), scale to 2 000 trajectories (still under
   1 h M4 Max compute).
2. **State-action coverage.** The BC-policy rollouts might not
   cover the state-action distribution that MBD explores. If
   `V_φ` is OOD during pure-MBD inference, the control variate
   adds variance rather than reducing it. Mitigation: `train.py
   --action-noise 0.1` warm-starts the rollouts with small
   tanh-Gaussian noise for coverage. Try it if H1-vs-H2 verdict
   looks unclear.
