# Calibration-of-guidance diagnostics — instrumentation spec (2026-06-11)

Companion to `docs/todos/next_direction_2026_06_11.md` (the
"calibration of test-time guidance" study) and the data-recovery
analysis `vast_dp_grid_recovery_analysis_2026_06_11.md`.

Goal: for each rollout, log the signals that predict *whether and how
much* test-time cost guidance helps a frozen prior, so we can (a) regress
hybrid lift on them across the valid prior ladder and (b) fit a tiny
"guide? at what lambda?" decision rule.

## Tier 1 — already logged (no new compute): use the extractor

`experiments/15_unified_hybrid/extract_calibration_features.py` reads
existing `unified_*.json` and emits one tidy row per rollout with:
`success`, `final_reward`, `n_chunks`, `total_elapsed_s`,
`used_score_fn`, `mean_cost_drop` (= mean over chunks of
init_best_cost − final_best_cost), `mean_final_best_cost`, `mean_ess`,
`min_ess`, `mean_gate_frac`, `mean_ldem_minus_lmod`.

Validated on `can_dp_calibration` (valid, nw>=2):

| lambda | success | mean_cost_drop | mean_ess | med_s |
|---:|---:|---:|---:|---:|
| 0.0 | 70% | +0.0044 | 15.7 | 364 |
| 0.5 | 50% | +0.0093 | 15.7 | 404 |
| 1.0 | 67% | **−0.0106** | 15.7 | 427 |

`mean_cost_drop` going **negative at lambda=1** is a clean
over-anchoring readout: the prior pull is dragging the MBD-achieved plan
*away* from low task cost. This is exactly the reward-over-optimization
boundary the study is about, and it is recoverable from data already on
disk for every valid regime (`demo_1nn`, DP ep0149/ep0199, Square-DP, BC
sweeps, mass/friction adaptation).

Run the regression now (sign-of-lift ~ Tier-1 signals) before adding any
new logging.

## Tier 2 — needs a small `unified_loop.py` hook (one cheap run)

These are NOT in the current logs and must not be faked:

1. **kNN distance to demo bank** `knn_dist`: L2 from the chunk-0
   `s0_vec` to its nearest bank obs. Only meaningful for retrieval-bank
   priors; for flow/BC priors, log NaN. Hook point:
   `unified_loop.unified_plan_and_execute` already builds `s0_vec`
   (line ~438) and the retrieval priors load a bank in
   `priors_zerotrain.load_demo_bank` — expose the min distance there and
   thread it into `chunk_log_entry`.
2. **kNN action disagreement** `knn_action_disp`: spread of the top-k
   bank actions at `s0_vec` (proxy for prior multimodality / coverage).
   Same hook as (1).
3. **warm-start action variance across M** `warmstart_var`: std of the
   M MBD perturbation seeds' first-action around the warm-start mean —
   proxy for prior confidence. Compute inside `denoise_loop` at k=0 and
   surface via the existing `log` dict.

All three are O(1) per chunk and gate-able behind a
`--log-calibration-diagnostics` flag (default off, bit-for-bit
reproducible when off), mirroring the existing `--log-gate-diagnostic`
pattern. One instrumented pass over the valid ladder (nw>=2) backfills
them; the extractor then picks up the new chunk fields automatically.

## Analysis plan (report-scoped)

1. Tier-1 regression: logistic `success ~ lambda + mean_cost_drop +
   min_ess + prior_quality` pooled across regimes; report sign-of-lift
   accuracy and the over-anchoring coefficient.
2. Add Tier-2 (`knn_dist`, disagreement) and test whether off-bank
   distance predicts where guidance helps vs hurts (the retrieval
   flagship case study).
3. Combination-rule contrast (2-3 cells, nw>=2): additive vs
   product/gate/temper at matched strength, using `mean_cost_drop` sign
   and `gate_frac` as the over-optimization proxy.
4. Fit + report the tiny decision rule on held-out cells.

Evaluation rigor: anytime-valid sequential testing (SAVI) with
partial-credit progress for guided-vs-unguided, per the next-direction
doc.
