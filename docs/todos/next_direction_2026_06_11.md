# Next experiment direction — final-report decision (2026-06-11)

Decision doc derived from: a full network deep-research refresh + a 6-round
novelty assessment + the completed remote offload. Full literature pass lives in
the followup repo at
`HRTG-square-dp-followup/docs/deep-research/research_directions_2026_06_11/`
(see `phase6_report/report.md`).

## Update (2026-06-11, post data-recovery)

The remote roots were recovered to local and analyzed. **Correction:** the
"140/140 DP λ/σ/seed grid" (`vast_safe_dp_2026_06_10`) referenced below as a data
source is **INVALID** — it ran with `--n-workers 1` and is corrupted (Square
0/30 everywhere, Can ≤26.7%; same Can ep0199 σ0.1 cell = 70% with proper workers
vs 6.7% here). See
`HRTG-square-dp-followup/docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md`.
Consequences for the steps below: **drop the 140/140 grid from the calibration
map**, or re-run it with `--n-workers ≥ 2` (the one justified re-compute). The
calibration map still has enough valid prior-quality rungs without it:
`demo_1nn` 73.8%, DP ep0149 hybrid 86.7%, DP ep0199 standalone 76-80% +
`can_dp_calibration` 70/50/66.7%, Square-DP 70%, BC Can λ-sweep 35-52%,
mass/friction adaptation (λ1) 17-40%. Add an `--n-workers ≥ 2` guard + a
λ0-vs-standalone sanity check to any new offload.

## Update (2026-06-13, re-run COMPLETE — grid restored)

The 140/140 grid was recomputed with `--n-workers 4` as
`vast_safe_dp_nwfix_2026_06_12` (valid; corruption guard now enforced in
`run_can.py::_guard_nworkers`, with `check_unified_sanity.py` as the λ0-vs-standalone
gate). So the 140/140 grid is **back in the calibration map** (no longer dropped).
Headlines: Square DP `ep0199` λ0 43.3% → λ1 **86.7%** (CI-separated lift) → λ2 66.7%
(inverted-U / over-anchoring); Can DP `ep0199` σ0.10 λ0 56.7% → λ2 **80.0%**; and a
Square cross-prior ladder (ep0049/0099/0149/0199) showing guidance benefit is
**non-monotonic in prior quality** — the strongest checkpoint gains most, mid
checkpoints over-anchor. Numbers in `current_claims.md`.

## Decision

> **Run the "calibration of test-time guidance" study: an empirical, matched-seed
> map of *when* and *how strongly* test-time cost guidance helps a frozen robot
> prior, and *why* product/gate/temper rules over-optimize — with the
> zero-training retrieval prior as the flagship case study and anytime-valid
> testing as the evaluation method.**

This finishes the report story because it converts HRTG's scattered positives and
its large negative-results bank into a single, literature-grounded thesis, using
data already on disk plus light new logging (low compute, low risk).

## Why now (frontier + repo state)

- The 2025-2026 frontier is dominated by **test-time guidance/steering of frozen
  generative policies** (DynaGuide, ADPro, GRITS, PCD, Fisher-Preserving). HRTG's
  `lambda_prior` cost-guided denoising is a member of this family.
- A 2026 theory thread shows such guidance is often **miscalibrated / reward-
  over-optimizing** (Calibrated Test-Time Guidance; Test-time Alignment w/o reward
  over-optimization; PGG shows guidance can destabilize control). HRTG holds the
  matched-seed *manipulation* evidence for this boundary.
- The completed offload supplies the prior-quality ladder needed: zero-train
  `demo_1nn`, DP `ep0149`/`ep0199`, Square-DP, the 140/140 DP λ/σ/seed grid, and
  static λ sweeps.

## Novelty guardrail (harsh-critic verdict)

A 6-round novelty check found **calibrated/predictive test-time guidance for
diffusion policies is NOT a novel method** — direct neighbors exist:
**LRT-Diffusion** (risk-calibrated LLR gating, offline RL), **PPGuide**
(performance-predictive steering), **DynaGuide** (external-model steering).

**So do NOT claim a new guidance/gating algorithm.** The defensible
class-project/workshop contribution is the **empirical cross-prior-type
calibration map** on Robomimic (retrieval vs DP-ep0149 vs DP-ep0199 vs Square-DP),
plus the **zero-training retrieval-score-prior + finite-coverage bank curve** —
which none of the neighbors cover. Frame as an empirical study + diagnostic; cite
the neighbors as related work. (`.../novelty/novelty_report.json`.)

## The experiment (concrete, report-scoped)

1. **Instrument** `experiments/15_unified_hybrid/run_can.py` to log per rollout:
   k-NN distance to bank, k-NN action disagreement, prior denoising/score
   consistency, short-horizon rollout drift, standalone (λ=0) success → persist
   into the `per_lambda` rows of `unified_summary_*.json`.
2. **Re-analyze** existing regimes (zero new compute): regress hybrid lift on the
   Step-1 signals across `demo_1nn`, DP `ep0149`/`ep0199`, Square-DP, the 140/140
   grid, and static λ sweeps.
3. **Combination-rule contrast** (2-3 cells): additive vs product/gate/temper at
   matched strength + a reward-over-optimization proxy (prior-likelihood drop /
   off-manifold drift).
4. **Fit a tiny decision rule**: "given prior diagnostics, guide? at what λ?" and
   report sign-of-lift prediction accuracy on held-out cells.
5. **Evaluation rigor**: anytime-valid sequential testing with partial-credit task
   progress (SAVI) for guided-vs-unguided, to spend fewer rollouts.

Flagship case study: retrieval as a zero-training test-time score prior;
differentiate from DARP (which trains a kNN-conditioned model); report the
in-bank vs out-of-bank bank-size split.

## Out of scope for the report (future work, literature-justified)

- Phase/contact decomposition for `tool_hang`/transport (HDP, RDP, ARCH, SPIRE):
  high impact but needs new observability + heavy compute.
- New hyperparameter sweeps/seed pools beyond steps 3-5 (standing project rule).
- Full TAMP integration; large heterogeneous retrieval banks before locality
  filtering.

## Pointers

- Decision detail + bibliography: `…/research_directions_2026_06_11/phase6_report/`.
- Synthesis + gaps/hypotheses: `…/phase5_synthesis/`.
- Current claims: `docs/current_claims.md`.
- Prior research pass (still valid): `…/research_directions_2026_06_10/`.
