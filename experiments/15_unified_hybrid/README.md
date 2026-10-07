# Exp 15: unified hybrid driver

Single canonical implementation of the BC-prior + MBD hybrid that
supersedes:

- **Exp 05** (`05_hybrid_method/run_can.py`) -- BC warm-start +
  MPC chunking, no per-step score correction.
- **Exp 07** (`07_lambda_sweep/run_can.py`) -- BC or flow warm-
  start + per-step score-bridge, single-shot only.
- **Exp 10** (`10_kxreplans_grid/run.sh`) -- K x R pareto sweep
  using exp 05's driver.

The fusion is structurally simple: take exp 05's MPC outer loop
and inside each chunk, build a fresh per-step `score_fn` from the
prior (exp 07's mechanism) and pass it to `denoise_loop` with
`lambda_prior`. The diagonal is exposed by two flags:

| `--mpc-replans` | `--lambda-prior` | Behaviour                                          |
|-----------------|------------------|----------------------------------------------------|
| `0`             | `0`              | Pure single-shot warm-start (exp 04 baseline).     |
| `0`             | `>0`             | Single-shot score-bridge (exp 07).                 |
| `>0`            | `0`              | MPC chunking, warm-start-only (exp 05 / exp 10).   |
| `>0`            | `>0`             | MPC chunking with score-bridge (NEW, Phase 2+).    |

The fourth row is the configuration that has never been
benchmarked and is the headline question for Branch C.

## Files

- [`run_can.py`](run_can.py) -- CLI + main loop + IO.
- [`unified_loop.py`](unified_loop.py) -- the actual MPC + score-
  bridge plan-and-execute function.
- [`regression_test.py`](regression_test.py) -- Phase-1 acceptance
  gate (asserts exp 07 lambda=5 and exp 10 K=10 R=2 reproducible
  within +/- 1/60).

## Usage

Headline single-shot, BC prior, lambda=5 (matches exp 07's
33.3 % result):

```bash
python -m experiments.15_unified_hybrid.run_can \
    --prior=bc --lambda-prior=5 \
    --mpc-replans=0 \
    --K=10 --M=16 --sigma-max=0.25 \
    --n-demos=20 --n-rollouts=3 --seed=0
```

MPC chunking, lambda=0 (warm-start-only, matches exp 10 K=10 R=2's
18.3 %):

```bash
python -m experiments.15_unified_hybrid.run_can \
    --prior=bc --lambda-prior=0 \
    --mpc-replans=2 \
    --K=10 --M=16 --sigma-max=0.25 \
    --n-demos=20 --n-rollouts=3 --seed=0
```

Flow-matching prior (requires `experiments/08_flow_matching_prior/results/v_theta.pt`):

```bash
python -m experiments.15_unified_hybrid.run_can \
    --prior=flow --lambda-prior 0 0.5 1 5 \
    --mpc-replans=0 \
    --K=10 --M=16 --sigma-max=0.25 \
    --n-demos=20 --n-rollouts=3 --seed=0
```

End-to-end smoke (~30 s, 2 demos x 1 rollout x 2 lambdas):

```bash
python -m experiments.15_unified_hybrid.run_can --smoke
```

## Output schema

Each invocation writes one JSON per lambda-prior value, named:

```
unified_{prior}_lam-{lam}_K-{K}_R-{R}{suffix}.json
```

Schema:

```json
{
  "experiment": "15_unified_hybrid/run_can",
  "prior": "bc",
  "prior_checkpoint": "...",
  "lambda_prior": 5.0,
  "warm_start": true,
  "mpc_replans": 0,
  "mpc_chunk_steps": 120,
  "n_total": 60,
  "n_success": 20,
  "success_rate": 0.333,
  "params": {"H": 120, "M": 16, "K": 10, ...},
  "seed": 0,
  "wall_s": 7283.4,
  "per_rollout": [
    {
      "demo": "demo_0",
      "rollout_idx": 0,
      "lambda_prior": 5.0,
      "final_reward": ...,
      "success": true,
      "plan_elapsed_s": ...,
      "chunks": [...]
    }
  ]
}
```

Plus a cross-lambda summary at
`unified_summary_{prior}_K-{K}_R-{R}{suffix}.json`.

## Phase 1 acceptance gate

The acceptance criterion lives in [`regression_test.py`](regression_test.py).
It asserts:

| Cell                                            | Reference  | Tolerance |
|-------------------------------------------------|------------|-----------|
| `--lambda-prior=5 --mpc-replans=0`              | 20/60      | +/- 1/60  |
| `--lambda-prior=0 --mpc-replans=2`              | 11/60      | +/- 1/60  |

To run the full test (n=60 per cell, ~3 h total):

```bash
python -m experiments.15_unified_hybrid.regression_test
```

Smoke version (n=2 per cell, ~3 min, wiring check only):

```bash
python -m experiments.15_unified_hybrid.regression_test --smoke
```

If the full test fails for either cell, the consolidation has a
semantic bug and Phase 1 is NOT complete. Do not merge to main
until both pass.

## Phase 1 vs Phases 2-5

This Phase 1 deliverable is a strict superset of exp 05's and
exp 07's drivers but does NOT add any new algorithmic features.
Phases 2-5 (detailed in [`docs/archive/plan_hybrid_coupling.md`](../../docs/archive/plan_hybrid_coupling.md))
add:

- **Phase 2**: K x R verification grid at lambda=5 (9 cells).
- **Phase 3**: adaptive lambda(sigma) schedule along the
  denoising chain.
- **Phase 4**: truncated denoising (K < 10).
- **Phase 5**: BC-uncertainty-gated replanning trigger.

The Phase 1 CLI already declares no-op flags for Phases 3-5
(`--lambda-schedule`, `--K-truncate`, `--replan-trigger`) and
for Branches A and B's deliverables (`--value-baseline-ckpt`,
`--shaped-cost`, `--prior=gaussian_bc`). They each print a
notice when set so a future user knows the flag is recognised
but not yet implemented.

## Migration from exp 05 / 07 / 10

| Old command                                    | New equivalent                                              |
|------------------------------------------------|-------------------------------------------------------------|
| `exp 05 --sigma-max 0.25`                       | `exp 15 --lambda-prior=0 --mpc-replans=0` (single-shot)     |
| `exp 05 --sigma-max 0.25 --mpc-replans=2`       | `exp 15 --lambda-prior=0 --mpc-replans=2`                   |
| `exp 07 --prior=bc --lambda-prior 5`            | `exp 15 --prior=bc --lambda-prior=5 --mpc-replans=0`        |
| `exp 07 --prior=flow --lambda-prior 5`          | `exp 15 --prior=flow --lambda-prior=5 --mpc-replans=0`      |
| `exp 10 K=10 R=2`                               | `exp 15 --lambda-prior=0 --mpc-replans=2 --K=10`            |

The old runners are kept for archival reproducibility but have
deprecation banners directing new work to this driver.
