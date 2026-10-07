# Experiment 07 — λ-sweep (per-step composite score)

> **DEPRECATED for new work**: Exp 15 (`experiments/15_unified_hybrid/`)
> consolidates this driver's score-bridge with Exp 05's MPC chunking.
> The `lambda_bc_*.json` files in this directory remain the
> canonical Tier-1b BC-prior λ-sweep result and are reproduced by
> Exp 15 within ±1/60. New λ-sweeps (including the v_theta sweep)
> should target Exp 15 with `--mpc-replans=0`. See
> [`experiments/15_unified_hybrid/README.md`](../15_unified_hybrid/README.md)
> § Migration.

**Status:** _scaffolded._ Runner ships in this branch; needs
`v_theta.pt` from exp 08 to fill in the `--prior flow` rows. The
`--prior bc` mode runs end-to-end against exp 03's existing MLP-BC
checkpoint and is the recommended wiring smoke test.

This is **Tier 1b** in `docs/plans.md`: wire the
`λ ∇log p_data(x_k)` term at every denoising step `k`, sweep λ, and
verify the endpoints match the existing rows.

## Endpoints (all reproduced by this script)

| flag combo                                    | row in the table        |
| --------------------------------------------- | ----------------------- |
| `--prior bc --no-warm-start --lambda-prior 0` | Exp 04 (MBD only)       |
| `--prior bc --warm-start    --lambda-prior 0` | Exp 05 (BC warm-start)  |
| `--prior flow --warm-start --lambda-prior 0` | Exp 05 with `v_θ` warm-start, no per-step prior |
| `--prior flow --warm-start --lambda-prior <0..>` | Tier 1b composite update |
| `--prior flow --lambda-prior inf`           | Pure prior ascent        |

## Running

Smoke (a couple of minutes, headless):

```bash
python experiments/07_lambda_sweep/run_can.py --smoke
```

Full sweep on can/ph (target table column for the paper):

```bash
# 1. Train the prior first (exp 08).
python experiments/08_flow_matching_prior/train.py \
    --task can --n-demos 20 --horizon 120

# 2. Sweep lambda. Each lambda value writes its own JSON.
python experiments/07_lambda_sweep/run_can.py \
    --prior flow \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta.pt \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --lambda-prior 0 0.1 0.5 1.0 5.0 inf \
    --no-live-view
```

## Outputs

```
results/
  lambda_0.0.json          # per-lambda full per-rollout log
  lambda_0.1.json
  ...
  lambda_inf.json
  summary.json             # one-row-per-lambda success-rate summary
```

`experiments/06_failure_taxonomy/run.py --inputs results/lambda_*.json`
will produce the per-bucket taxonomy table for the same sweep.

## Implementation notes

- λ values are ordered so the same `(demo, rollout_idx)` sees
  identical perturbation noise across λ values (paired
  comparison; the variance-reduction makes the curve much
  cleaner with `n_rollouts=3`).
- `--prior bc` reinterprets the BC policy as a deterministic
  velocity field that points `μ_t` at `a_BC(s_t)`. It is **not** a
  faithful score; it exists so you can verify the wiring and
  endpoint behaviour without waiting on the flow trainer.
- The MBD softmax temperature (`--lambda_`) is unchanged from
  exp 04/05 (`0.5`); the prior weight is the new
  `--lambda-prior` argument.
