# Experiment 07 — λ-sweep (per-step score-bridge)

**Lifecycle:** **superseded → [`15_unified_hybrid`](15_unified_hybrid.md)**.
The `lambda_bc_*.json` files remain the canonical Tier-1b BC-prior
λ-sweep result; exp 15 reproduces them within ±1/60 at
`--lambda-prior` matching this experiment's `--lambda-prior`.

**In-tree deep doc:**
[`experiments/07_lambda_sweep/README.md`](../../experiments/07_lambda_sweep/README.md).

## Purpose

Add the **per-step score-bridge term** `λ · ∇ log p_prior(x_k)` at
every denoising step `k`, sweep λ, and verify the endpoints match
the existing rows (λ=0 reproduces exp 05; λ=∞ reduces to pure
prior ascent).

This is **Tier 1b** in [`docs/plans.md`](../plans.md).

### Endpoints (all reproduced by this script)

| flag combo                                       | Reproduces                                |
|--------------------------------------------------|-------------------------------------------|
| `--prior bc --no-warm-start --lambda-prior 0`    | Exp 04 (MBD only)                         |
| `--prior bc --warm-start    --lambda-prior 0`    | Exp 05 (BC warm-start)                    |
| `--prior flow --warm-start --lambda-prior 0`     | Exp 05 with `v_θ` warm-start, no per-step prior |
| `--prior flow --warm-start --lambda-prior 0.1..5` | Tier-1b composite update                  |
| `--prior flow --lambda-prior inf`                | Pure prior ascent                         |

## Headline result

| Variant                                          | Success         | Source |
|--------------------------------------------------|-----------------|--------|
| `--prior bc --lambda-prior 5`, single-shot, seed 0 | **20 / 60 = 33.3 %** | `experiments/07_lambda_sweep/results/_archive/lambda_bc_5.0.json` |

The 33.3 % at λ=5 is the **stronger** of the two pre-canonicalisation
hybrids (exp 05 single-shot was 1 / 60 = 1.7 %). The 18.3 % vs 33.3 %
gap between exps 05/10 and exp 07 on the same K=10 cell drove the
**BC-swap re-run analysis** (see
[`docs/bc_swap_rerun_analysis.md`](../bc_swap_rerun_analysis.md))
and motivated the consolidation into exp 15.

## Reproduce

```bash
# Smoke (~2 min).
python experiments/07_lambda_sweep/run_can.py --smoke

# Full sweep on Can / ph.
python experiments/07_lambda_sweep/run_can.py \
    --prior bc --warm-start \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --lambda-prior 0 0.1 0.5 1.0 5.0 inf \
    --no-live-view
```

Or use the canonical successor (exp 15):

```bash
python experiments/15_unified_hybrid/run_can.py \
    --prior bc --lambda-prior 5 --mpc-replans 0 \
    --K 10 --M 16 --sigma-max 0.25 \
    --n-demos 20 --n-rollouts 3 --seed 0
```

Both reach the same 33.3 % (± 1/60).

## Files

| Path | Role |
|------|------|
| [`experiments/07_lambda_sweep/run_can.py`](../../experiments/07_lambda_sweep/run_can.py) | Driver with `--lambda-prior` and `--prior {bc, flow}` flags. |
| [`experiments/07_lambda_sweep/analyze.py`](../../experiments/07_lambda_sweep/analyze.py) | Per-λ summary + cost-vs-iteration plotter. |
| `experiments/07_lambda_sweep/results/_archive/lambda_bc_*.json` | Per-λ result JSONs at the BC prior. |

## Cross-references

- **Predecessors**:
  [`04_mbd_baseline.md`](04_mbd_baseline.md),
  [`05_hybrid_method.md`](05_hybrid_method.md).
- **Canonical successor**:
  [`15_unified_hybrid.md`](15_unified_hybrid.md).
- **Flow-prior path**:
  [`08_flow_matching_prior.md`](08_flow_matching_prior.md) trains
  the `v_θ` model that `--prior flow` consumes.
- **Re-run analysis**:
  [`docs/bc_swap_rerun_analysis.md`](../bc_swap_rerun_analysis.md)
  documents why exp 07 and exp 05/10 disagreed.

## Implementation notes

- **Paired comparison**: λ values are ordered so the same
  `(demo, rollout_idx)` sees identical perturbation noise across
  λ values. Cleaner curve with `n_rollouts=3`.
- **`--prior bc` is not a faithful score**: it reinterprets the
  BC policy as a deterministic velocity field that points μ at
  `a_BC(s_t)`. Exists for wiring smoke tests / endpoint
  reproduction.
- **MBD softmax temperature** (`--lambda_`) is unchanged from
  exp 04/05 (`0.5`); the new flag is `--lambda-prior`.
