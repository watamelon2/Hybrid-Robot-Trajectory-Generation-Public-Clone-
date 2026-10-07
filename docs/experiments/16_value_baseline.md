# Experiment 16 — `V_φ` state-value baseline (Branch A control variate)

**Lifecycle:** **scaffolded on main; eval deferred**.  The A2
task in `feat/stronger-mbd-signal` (Branch A) shipped the
scaffolding -- the `V_φ` training script, the
`--value-baseline-ckpt` flag on the unified driver, and the
exp 16 dir -- and `feat/stronger-mbd-signal` was merged into
main on 2026-05-25 (`25ca457`).  The headline eval cell
(pure-MBD at K=0 λ=0 σ=0.05 M=64 with `--value-baseline-ckpt`)
is still pending; see
[`docs/todos/future_2026_05_26.md`](../todos/future_2026_05_26.md)
§ A2 for the gate ("≥ 5 / 60 succ to support H2 over H1")
and the ~3 h wall budget.

**In-tree deep doc:**
[`experiments/16_value_baseline/README.md`](../../experiments/16_value_baseline/README.md).

## Hypothesis

Pure MBD on `can/ph` is **0 / 60** at every configuration tested
(see [`04_mbd_baseline.md`](04_mbd_baseline.md)). Two non-exclusive
explanations:

- **H1 — capacity**: the MBD denoiser cannot represent the policy
  manifold without a BC warm-start. Reaching is intrinsically
  hard from zero-init in 7-D × 120-step space with only `M=16`
  perturbations.
- **H2 — variance**: the per-step cost is too high-variance for
  MBD's softmax-over-cost weighting to make progress without an
  anchor.

If H2 is right, subtracting a learned **state-value baseline**
`V_φ(s_t)` from per-step costs reduces the variance of the softmax
weighting without changing its expectation (the standard
REINFORCE / A2C / PPO control-variate trick). If H2 is wrong, V_φ
should have no measurable effect on pure MBD.

## How V_φ is wired into MBD

The per-trajectory cost inside `denoise_loop` becomes:

```
c_i_new  =  c_i  −  Σ_t V_φ(s_{i, t})
```

with `c_i = −reward_i` as before. The softmax over `c_i_new`
inherits the same expectation but with lower variance, provided
V_φ is well-fit. The integration happens at the `cost_fn`
boundary — see [`adapter.py`](../../experiments/16_value_baseline/adapter.py).

The unified driver
([`experiments/15_unified_hybrid/run_can.py`](../../experiments/15_unified_hybrid/run_can.py))
routes the `--value-baseline-ckpt` flag here:

1. Switches `make_rollout_fn` to exp 04's
   `make_rollout_fn_with_obs` variant (which collects per-step
   `OBS_KEYS` observations alongside the episode reward).
2. Wraps the base `cost_fn` with
   `wrap_cost_fn_with_value_baseline(...)`.

This is the exact wiring that produced the
`active_make_rollout_fn / active_cost_fn` conflict resolved
in the May-25 main merge.

## Files

| Path | Role |
|------|------|
| [`experiments/16_value_baseline/value_model.py`](../../experiments/16_value_baseline/value_model.py) | `V_φ` MLP + `ValuePhi` load/save wrapper. |
| [`experiments/16_value_baseline/train.py`](../../experiments/16_value_baseline/train.py) | TD(0) / MC regression on BC-policy rollouts. |
| [`experiments/16_value_baseline/adapter.py`](../../experiments/16_value_baseline/adapter.py) | `wrap_cost_fn_with_value_baseline` for the planner. |
| `experiments/16_value_baseline/results/V_phi.pt` | Trained model + obs stats + meta. |
| `experiments/16_value_baseline/results/V_phi.meta.json` | Training-config + RMSE metadata. |

Note: we renamed `model.py` → `value_model.py` to avoid clashing
with `experiments/08_flow_matching_prior/model.py` if anyone ever
combines `--prior=flow` with `--value-baseline-ckpt` (which would
push both modules onto `sys.path`).

## Train V_φ

```bash
conda activate hybrid-traj

# Defaults: 500 BC rollouts, 100 epochs, no action noise, seed 0.
# ~3 min data collection + ~30 s training on M-series CPU.
python experiments/16_value_baseline/train.py \
    --bc-checkpoint experiments/03_bc_baseline/results/policy.pt \
    --output experiments/16_value_baseline/results/V_phi.pt
```

Training prints `RMSE/|mean(R)|` as a diagnostic. If this exceeds
**0.5**, the regression has not fit the cost-to-go reliably; raise
`--n-trajs` to 2000 (still under 1 h M4 Max compute) or add
`--action-noise 0.1` for better state-action coverage. See
[`docs/branch_plan.md`](../branch_plan.md) § A2 for the risk-
mitigation notes.

## Evaluation (the acceptance gate)

Per `docs/branch_plan.md`'s A2 spec, eval is **pure MBD with V_φ as
a control variate** at the regime where pure MBD is currently 0/60:

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

Eval takes ~2 h on M4 Max with `n-workers=2`.

### Acceptance bar

| Outcome | Interpretation                                                                                   |
|---------|--------------------------------------------------------------------------------------------------|
| ≥ **5 / 60 = 8.3 %** | **H2 supported**: control variate helped. Distinguishes "helped" from "didn't help" by Wilson 95 % CI. |
| ≤ **3 / 60 = 5.0 %** | **H1 supported**: pure MBD with V_φ still essentially zero ⇒ model capacity is the bottleneck, not variance. Document as a negative result. |

## Cross-references

- **Hypothesis source**:
  [`04_mbd_baseline.md`](04_mbd_baseline.md) (the 0 / 60 result
  this experiment tries to crack).
- **Consumer**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md) routes
  `--value-baseline-ckpt`. The wiring lives in
  [`run_can.py`](../../experiments/15_unified_hybrid/run_can.py)
  ~ line 539 (`active_make_rollout_fn` / `active_cost_fn`).
- **Branch context**:
  Branch A roadmap in [`docs/branch_plan.md`](../branch_plan.md).
- **Related**:
  [`05_hybrid_method.md`](05_hybrid_method.md) (BC anchor is the
  alternative variance-reduction mechanism — anchor instead of
  control variate).

## Open questions

1. **Training data scale**. 500 trajectories × ~120 steps =
   60 k transitions might be too few for a 2-layer MLP to
   converge cleanly. If TD residuals plateau at high variance
   (`RMSE / |mean(R)| > 0.5`), scale to 2 000 trajectories
   (~1 h M4 Max).
2. **State-action coverage**. BC-policy rollouts might not cover
   the state-action distribution that MBD explores. If V_φ is
   OOD during pure-MBD inference, the control variate adds
   variance rather than reducing it. Mitigation:
   `train.py --action-noise 0.1` warm-starts the rollouts with
   small tanh-Gaussian noise for coverage.
3. **Sensitivity to the eval cell**. The acceptance bar is at
   the `--sigma-max=0.05 --M=64 --no-warm-start` cell, which is
   the cell where exp 09 found a 50 % single-seed peak under the
   *BC-warm-started* hybrid. Pure MBD at the same cell (no warm
   start) might still be 0/60 if H1 dominates regardless of
   variance. The script also supports `--lambda-prior > 0`
   for an apples-to-orange comparison.
