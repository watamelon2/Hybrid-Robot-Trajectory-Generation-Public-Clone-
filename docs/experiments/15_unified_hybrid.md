# Experiment 15 — Unified hybrid driver (canonical)

**Lifecycle:** **active (canonical)**. The canonical implementation
of the BC-prior + MBD hybrid. Supersedes exps 05, 07, 09, 10.
All new sweeps target this driver.

**In-tree deep doc:**
[`experiments/15_unified_hybrid/README.md`](../../experiments/15_unified_hybrid/README.md).

## Headline

| Cell | Success | Source |
|------|---------|--------|
| `--prior=demo_1nn --lambda-prior=1 --sigma-max=0.25 --K=10 --R=1 --M=16`, 4-seed pool | **177 / 240 = 73.8 %** (Wilson CI [67.8 %, 78.9 %]) | Slate E, 2026-05-19 |
| Same cell, single seed (peak) | 45 / 60 = 75.0 % | Slate D, 2026-05-18 |
| Cross-task `Lift / ph` λ=1 | **60 / 60 = 100.0 %** | Slate G G1, 2026-05-21 |
| Cross-task `Square / ph` σ=0.30 | 5 / 60 = 8.3 % | Slate G-aux AUX1 |
| Cross-task `tool_hang / ph` λ=1 H=300 (sparse reward) | 0 / 40 (sparse-reward bug) | Slate H H1 |

The **73.8 % pooled headline** is the canonical result of the
project as of the May-22 presentation. See
[`docs/results.md`](../results.md) for the full table, including
the BC and pure-MBD baselines used as anchors.

## What the driver consolidates

Single canonical implementation that supersedes:

- **Exp 05** (`05_hybrid_method/run_can.py`) — BC warm-start + MPC
  chunking, no per-step score correction.
- **Exp 07** (`07_lambda_sweep/run_can.py`) — BC or flow warm-start
  + per-step score-bridge, single-shot only.
- **Exp 09** (`09_sigma_sweep/run.sh`) — σ-max sweep through exp 05's
  driver.
- **Exp 10** (`10_kxreplans_grid/run.sh`) — K × R Pareto sweep
  through exp 05's driver.

The fusion is structurally simple: take exp 05's MPC outer loop,
and inside each chunk build a fresh per-step `score_fn` from the
prior (exp 07's mechanism), pass it to `denoise_loop` with
`lambda_prior`.

The diagonal is exposed by two flags:

| `--mpc-replans` | `--lambda-prior` | Behaviour                                   |
|-----------------|------------------|---------------------------------------------|
| `0`             | `0`              | Pure single-shot warm-start (exp 04 baseline). |
| `0`             | `>0`             | Single-shot score-bridge (exp 07).          |
| `>0`            | `0`              | MPC chunking, warm-start-only (exp 05 / 10). |
| `>0`            | `>0`             | **MPC chunking with score-bridge (new; the headline configuration).** |

## Priors supported

The `--prior` flag selects which density's score is used in the
per-step update:

| `--prior` | What it is | Training required |
|-----------|------------|-------------------|
| `bc` | Deterministic velocity field from exp-03 BC policy; not a faithful score but works as a smoke test and was the May-14 headline 33.3 %. | Yes (exp 03 MLP). |
| `flow` | Trained `v_θ(x, t)` from exp 08. Best so far: 16.7 % (v_θ v2). | Yes (exp 08). |
| `demo_1nn` | **1-NN retrieval** from the demo bank: at each step, return the recorded action of the nearest demo state. **Headline prior.** | No (just reads the dataset). |
| `demo_knn` | Soft k-NN mixture; β=∞ reduces to demo_1nn. | No. |
| `demo_softnn` | softmax(-d/β) over the k nearest demo states; β=5 was best on Slate G2. | No. |
| `demo_random` | Random demo (control). | No. |
| `demo_mean` | Mean action of the k-NN cluster (a smoothed retrieval). | No. |
| `gaussian_bc` | Per-step Gaussian centred at BC action (placeholder). | Yes. |
| `contrastive_*` | Contrastive obs-embedding-based retrieval (Slate I). | Yes (encoder). |

## Slate timeline (committed on main)

| Date | Slate | Outcome |
|------|-------|---------|
| 2026-05-17 | Z | Zero-train retrieval priors first probed (demo_1nn + bridge = 63.3 % NEW PEAK at the time). |
| 2026-05-17 | Z-followup | 4-seed pool at λ=5 σ=0.25 → 64.2 % CI [57.9, 70.0]. |
| 2026-05-18 | D | `lambda` finer-grained sweep → demo_1nn λ=1 = 75.0 % NEW PEAK. σ-invariance to 1.0 documented. |
| 2026-05-19 | E | **4-seed pool at the new λ=1 cell → 177/240 = 73.8 %, CI [67.8, 78.9]. Canonical headline.** |
| 2026-05-20 | F | Mode-selection ablations; Square cross-task fails (mechanism-limited). |
| 2026-05-21 | G | 3 new tasks, λ-sweep on Lift → 100 %; A1/A2/A3 priors. |
| 2026-05-21 | G-aux | Square σ-tail rescue; peak at σ=0.30 = 8.3 %. |
| 2026-05-21 | H | Cross-task validation: Lift A1 hist=1/5/10 = 100/100/98.3 %, Can A1 = 75/68.3/66.7 %. tool_hang H1 = 0/40 (sparse-reward bug). |
| 2026-05-22 | I | C1 contrastive obs-embedding prior on Square + Can (paused mid-run). |
| 2026-05-22 | J | Queued: tool_hang + transport with dense-reward fix. Not yet run. |

## Reproduce the 73.8 % headline

```bash
conda activate hybrid-traj

# Slate E reproduction (4 seeds × 60 rollouts = 240; ~6 h on M-series with --n-workers 2).
for seed in 0 1 2 3; do
  python experiments/15_unified_hybrid/run_can.py \
      --prior demo_1nn --lambda-prior 1 --sigma-max 0.25 \
      --K 10 --M 16 --mpc-replans 1 \
      --n-demos 20 --n-rollouts 3 --seed $seed \
      --n-workers 2 --no-live-view \
      --output experiments/15_unified_hybrid/results/headline_seed${seed}.json
done

# Aggregate.
python experiments/15_unified_hybrid/run_can.py --aggregate \
    --inputs experiments/15_unified_hybrid/results/headline_seed*.json \
    --output experiments/15_unified_hybrid/results/headline_AGGREGATE.json
```

## Smoke and regression tests

```bash
# End-to-end smoke (~30 s, 2 demos × 1 rollout × 2 lambdas).
python experiments/15_unified_hybrid/run_can.py --smoke

# Phase 1 acceptance gate: asserts λ=5 → 20/60 and λ=0 R=2 → 11/60
# within ±1/60. Full test ~3 h.
python experiments/15_unified_hybrid/regression_test.py

# Smoke regression (2 rollouts per cell, ~3 min).
python experiments/15_unified_hybrid/regression_test.py --smoke
```

## Output schema

Each invocation writes one JSON per `--lambda-prior` value:

```
unified_{prior}_lam-{lam}_K-{K}_R-{R}{suffix}.json
```

Plus a cross-lambda summary at
`unified_summary_{prior}_K-{K}_R-{R}{suffix}.json`.

Schema (one rollout):

```json
{
  "experiment": "15_unified_hybrid/run_can",
  "prior": "demo_1nn",
  "prior_checkpoint": "<dataset_path>",
  "lambda_prior": 1.0,
  "warm_start": true,
  "mpc_replans": 1,
  "mpc_chunk_steps": 60,
  "params": {"H": 120, "M": 16, "K": 10, "sigma_max": 0.25, ...},
  "n_total": 60,
  "n_success": 45,
  "success_rate": 0.75,
  "seed": 0,
  "wall_s": 5300.4,
  "per_rollout": [
    {
      "demo": "demo_0", "rollout_idx": 0,
      "final_reward": ..., "success": true,
      "plan_elapsed_s": ..., "chunks": [...]
    }
  ]
}
```

## Files

| Path | Role |
|------|------|
| [`experiments/15_unified_hybrid/run_can.py`](../../experiments/15_unified_hybrid/run_can.py) | CLI + main loop + IO + prior loader. The ONLY runner you need for new sweeps. |
| [`experiments/15_unified_hybrid/unified_loop.py`](../../experiments/15_unified_hybrid/unified_loop.py) | `unified_plan_and_execute` — the actual MPC + score-bridge loop. |
| [`experiments/15_unified_hybrid/priors_zerotrain.py`](../../experiments/15_unified_hybrid/priors_zerotrain.py) | Retrieval-prior implementations (1-NN, k-NN, soft-NN, demo-mean, random-demo) + task-aware `OBS_KEYS`. |
| [`experiments/15_unified_hybrid/train_contrastive_encoder.py`](../../experiments/15_unified_hybrid/train_contrastive_encoder.py) | C1 contrastive obs-embedding training (Slate I). |
| [`experiments/15_unified_hybrid/env_dense_rewards.py`](../../experiments/15_unified_hybrid/env_dense_rewards.py) | Dense-reward shaping shim for tool_hang + transport (Slate J prep). |
| [`experiments/15_unified_hybrid/regression_test.py`](../../experiments/15_unified_hybrid/regression_test.py) | Phase-1 acceptance gate (reproduces exp 07 / exp 10 within ±1/60). |
| `experiments/15_unified_hybrid/slate_{d,e,f,g,h,i,j}_*.sh` | Slate launcher shells (one per evening's run). |
| `experiments/15_unified_hybrid/results/slate_{d,e,f,g,g_aux,h,i,j}_*/` | Per-slate result subdirs (JSONs + AGGREGATE.json + BRIEF.md). |

## Cross-references

- **Predecessors (now reproduced by this driver)**:
  [`04_mbd_baseline.md`](04_mbd_baseline.md),
  [`05_hybrid_method.md`](05_hybrid_method.md),
  [`07_lambda_sweep.md`](07_lambda_sweep.md),
  [`09_sigma_sweep.md`](09_sigma_sweep.md),
  [`10_kxreplans_grid.md`](10_kxreplans_grid.md).
- **Optional priors**:
  [`08_flow_matching_prior.md`](08_flow_matching_prior.md) for
  `--prior=flow`.
- **Companion experiments still useful on top of exp 15**:
  [`06_failure_taxonomy.md`](06_failure_taxonomy.md) (post-hoc),
  [`14_seed_sweep.md`](14_seed_sweep.md) (Wilson CIs).
- **V_φ control variate**:
  [`16_value_baseline.md`](16_value_baseline.md) plugs into
  `--value-baseline-ckpt`.
- **Bias-variance Pareto analysis** (added 2026-06-01 via
  `feat/adapt-pareto-analysis`, commit `aaefba2`): the
  [`proxy_bias_variance.py`](../../experiments/15_unified_hybrid/proxy_bias_variance.py)
  cell-level proxy decomposition verifies T2.1 strictly Pareto-
  dominates T2.3 on the T2 4-knob robustness slate; companion
  scaffolding for the per-step measurement is
  [`measure_bias_variance.py`](../../experiments/15_unified_hybrid/measure_bias_variance.py)
  (deferred to a journal version).  Theory anchor:
  [`docs/theory/08_asymmetric_robustness_model.md`](../theory/08_asymmetric_robustness_model.md);
  writeup: [`docs/results_writeups/pareto_analysis.md`](../results_writeups/pareto_analysis.md).
- **DP hybrid (exp 17, Phase D)** (added 2026-06-01 via
  `feat/dp-phase-c`, commit `8e97ca3`):
  [`17_dp_hybrid.md`](17_dp_hybrid.md) uses this driver with
  `--prior=flow --prior-checkpoint=<v3b>` to land **two GREEN
  cells at ep0149** (K=10 λ=2.0 = 86.7 %, K=20 λ=0.5 = 83.3 %).
- **Reading**:
  [`docs/results.md`](../results.md) — headline tables;
  [`docs/run_log.md`](../run_log.md) — chronological narrative;
  [`docs/slate_f_final_reading.md`](../slate_f_final_reading.md),
  [`docs/theory_post_slate_e.md`](../theory_post_slate_e.md).

## Migration from exp 05 / 07 / 09 / 10

| Old command | Exp-15 equivalent |
|-------------|-------------------|
| `exp 05 --sigma-max 0.25` | `exp 15 --lambda-prior=0 --mpc-replans=0` |
| `exp 05 --sigma-max 0.25 --mpc-replans=2` | `exp 15 --lambda-prior=0 --mpc-replans=2` |
| `exp 07 --prior=bc --lambda-prior 5` | `exp 15 --prior=bc --lambda-prior=5 --mpc-replans=0` |
| `exp 07 --prior=flow --lambda-prior 5` | `exp 15 --prior=flow --prior-checkpoint=...v_theta_v2.pt --lambda-prior=5 --mpc-replans=0` |
| `exp 09 sigma sweep` | `for s in ...; do exp 15 --sigma-max $s ...; done` |
| `exp 10 K=10 R=2` | `exp 15 --lambda-prior=0 --mpc-replans=2 --K=10` |

The legacy runners are kept for archival reproducibility but
have deprecation banners directing new work here.

## WIP / open questions

- **Square cross-task** is mechanism-limited (best ~8.3 %). The
  Slate I / C1 contrastive prior was paused mid-run; see
  `feat/zero-train-priors` Slate-I results and exp 15's
  `results/slate_i_2026_05_22/` for the in-progress data.
- **`tool_hang` + `transport`** are blocked on the dense-reward
  shim landing (commit `18af830` added it; Slate J queue is
  ready to fire).
- **Learned-prior beat retrieval-prior** is still open. See
  `feat/flow-v3-karras` (off main) for Slate K's parity-ceiling
  diagnostic.
- **`--lambda-schedule`, `--K-truncate`, `--replan-trigger`**
  are CLI flags that print a notice when set but are not yet
  wired (Phases 3, 4, 5 of Branch C's original roadmap).
