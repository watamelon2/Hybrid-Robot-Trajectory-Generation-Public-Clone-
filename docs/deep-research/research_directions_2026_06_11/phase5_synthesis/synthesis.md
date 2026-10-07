# Phase 5 — Synthesis (2026-06-11 refresh)

## Core synthesis

The 2025-2026 frontier and HRTG's evidence converge on one theme:

> **Test-time cost guidance of a frozen generative prior is now a recognized design
> space (DynaGuide, ADPro, GRITS, PCD, DAS), and a 2026 theory thread shows such
> guidance is frequently *miscalibrated / over-optimizing*. HRTG holds matched-seed
> manipulation evidence of exactly this boundary. The finishing contribution is a
> practical *calibration/diagnostic*: predict when, and how strongly, to guide a
> robot prior — and explain why product/gate rules fail.**

This unifies HRTG's positives (additive bridge wins on Can/Square-DP) and its large
bank of negatives (product/gate/temper/adaptive-σ/anneal lose) under a single story
that is now literature-grounded rather than ad hoc.

## How the finished experiments change the picture

The completed remote offload supplies the matched-seed regimes a calibration study
needs, already on disk:

- **Prior-quality ladder.** zero-train `demo_1nn` (73.8% Can), DP `ep0149`
  (hybrid-lift, 86.7%), DP `ep0199` (standalone 76-80%), Square-DP (66.7%→70.0%),
  plus the 140/140 DP shard grid across λ/σ/seed.
- **Combination-rule contrast.** additive score-bridge vs product/gate/temper, with
  consistent additive wins (memory: score-bridge >> tempered at matched M, n=90).
- **Saturation curve.** bank-size 5→200 with a knee near 20-50 (finite coverage).
- **Negative long-horizon set.** transport sparse-extra cells 0/5; `tool_hang` 0/10
  across raw/dense/H/σ/demo-phase — strong evidence guidance alone is insufficient
  without phase observability.

What is *missing* is not more cells but **per-rollout prior-quality logging** to turn
"additive wins / product loses" into a *predictor*.

## Directions, ranked for finishing the final report

| Rank | Direction | Finish-the-story value | Compute/risk | Literature anchor |
|---:|---|---|---|---|
| **1** | **Calibration of test-time guidance for manipulation priors** (when/how-strongly to guide; why product/gate over-optimize) | **Highest** — unifies all positives+negatives into one thesis; reuses data on disk | **Low** — mostly logging + re-analysis, few new cells | [@calibrated2026; @das2025; @bagaria2025dynaguide; @pgg2025; @anthology2025] |
| 2 | Retrieval-as-test-time-score-prior + finite-coverage diagnostics | High — strongest single result (73.8%) + bank curve; clean DARP differentiation | Low-med | [@darp2026; @curation2025; @ndadp2025] |
| 3 | Phase/contact decomposition for `tool_hang`/Square residuals | High impact, but report-risky | **High** — new instrumentation + many rollouts | [@hdp2025; @xue2025reactive; @sun2024arch; @zhou2024spire] |
| 4 | Contact-rich residual failure diagnosis for Square | Medium — needs richer traces | Med | [@xue2025reactive; @tacdiffusion2024] |
| 5 | Evaluation-rigor upgrade (anytime-valid + partial credit) | Medium — strengthens *all* claims cheaply | Low | [@savi2026] |

## Recommended composition for the report

Make **Direction 1 the headline finishing experiment**, fold **Direction 2** in as the
flagship case study (retrieval prior), and **adopt Direction 5 as the evaluation
method** for the guided-vs-unguided comparison. Keep Direction 3 as clearly-labeled
future work (literature-justified, but out of scope for the final report's compute).

This yields a single coherent claim — *"we map the calibration boundary of test-time
guidance for robot priors"* — backed entirely by data already collected, plus light
new logging, and positioned against the newest 2025-2026 work.
