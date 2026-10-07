# Phase 5 — Gaps and hypotheses (2026-06-11 refresh)

## Gap 1 — No practical calibration criterion for test-time guidance in control

The 2026 theory shows common test-time guidance is miscalibrated / over-optimizing
[@calibrated2026; @das2025], and DynaGuide/ADPro leave guidance strength as a manual
knob [@bagaria2025dynaguide; @adpro2025]. Nobody gives a *predictor* of when, and how
strongly, to guide a **robot manipulation** prior.

**Hypothesis (H1).** Hybrid lift is a function of measurable prior-quality signals:
positive when the prior is coherent (low k-NN action disagreement / high
denoising-score consistency) with residual headroom (moderate standalone success),
and negative/over-optimizing when the combination rule maximizes reward greedily
(product/gate) on an incoherent prior. A simple regression on
{standalone success, k-NN distance, action disagreement, score consistency, rollout
drift} predicts the sign and rough magnitude of lift across HRTG's regimes.

## Gap 2 — Combination-rule effect dominates the λ-value effect, but is untheorized

HRTG repeatedly finds additive score-bridge ≫ product/gate/temper at matched
strength, mirroring DAS's "tempered beats greedy reward-max" [@das2025] and PGG's
"guidance can destabilize continuous control" [@pgg2025].

**Hypothesis (H2).** The combination rule is the primary knob and λ is secondary;
product/gate fail because they multiply/threshold an approximate prior score,
amplifying its miscalibration, whereas additive bridging preserves prior coherence.

## Gap 3 — Retrieval as a *zero-training* test-time score prior is still unique

DARP [@darp2026] trains a difference-aware kNN model; Behavior Retrieval / skill
retrieval use retrieval to augment *training* data [@du2023behavior; @nasiriany2022learning].
HRTG uses retrieval with no training as a score/pull target inside MBD.

**Hypothesis (H3).** `demo_1nn` success tracks finite mode coverage: it rises until the
bank covers the task's initial-state modes (knee ~20-50) and plateaus; in-bank initial
states succeed more than out-of-bank ones, predictable from k-NN distance.

## Gap 4 — Sparse long-horizon tasks need phase observability, not stronger guidance

`tool_hang`/transport remain ~0 across all static guidance variants; HDP shows
high-level contact/subgoal guidance is what unlocks contact-rich tasks [@hdp2025], and
RDP/TacDiffusion show contact observability matters [@xue2025reactive; @tacdiffusion2024].

**Hypothesis (H4).** `tool_hang` stays near zero until target/hook/hole observables
enter the prior state or MBD cost; decomposing into short phase-local guided segments
will outperform one long sparse objective.

## Gap 5 — HRTG's evaluation underuses its data

Binary success at n=30 with Wilson CIs is weaker than necessary. SAVI [@savi2026]
enables anytime-valid sequential tests over partial-credit metrics.

**Hypothesis (H5).** Guided-vs-unguided lift can be established at the same confidence
with materially fewer rollouts using anytime-valid sequential testing on task-progress
partial credit (e.g., Square approach/grasp/align/insert), freeing compute.

## Gap 6 — Coverage diagnostics exist in theory but aren't wired into HRTG

kNN-MI demonstrates principled demo-quality scoring [@curation2025]; HRTG logs only
success/wall-time, not per-rollout retrieval-quality signals.

**Hypothesis (H6).** Adaptation benefit (few new demos) is predicted by how much they
reduce k-NN distance in the failure region, not by total bank size.
