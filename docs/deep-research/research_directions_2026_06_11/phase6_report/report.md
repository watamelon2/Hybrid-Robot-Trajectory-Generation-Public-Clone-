# Next experiment direction to finish the HRTG final-report story

**Date:** 2026-06-11
**Inputs:** repo docs + branches, the completed remote offload, and a refreshed
6-phase literature pass (`research_directions_2026_06_11/`).
**Companion:** updates the 2026-06-10 pass; folds in the now-finished experiments and
the 2025-2026 frontier.

## TL;DR decision

> **Run the "calibration of test-time guidance" experiment: a matched-seed study that
> predicts *when* and *how strongly* cost guidance helps a frozen robot prior, and
> explains *why* product/gate/temper rules over-optimize.**
> It reuses data already on disk, needs only light new per-rollout logging, and is the
> single direction that unifies every positive and negative result HRTG has into one
> literature-grounded thesis.

Fold the **retrieval prior** in as the flagship case study, and adopt **anytime-valid
evaluation** as the comparison method. Keep phase/contact decomposition for
`tool_hang` as clearly-labeled future work (high impact, too compute-risky for the
report).

## Why this finishes the story

HRTG's results currently read as a pile of wins and losses:

- Wins: zero-train retrieval + additive bridge 73.8% Can; DP `ep0149` hybrid-lift
  86.7%; Square-DP 70.0%; bank-size saturation; cheap bank-adaptation.
- Losses: product/gate/temper, adaptive σ, λ-annealing, demo-phase cost; `tool_hang`
  and transport ~0.

The 2025-2026 literature gives these a single frame. Test-time guidance of a frozen
generative policy is now its own subfield (DynaGuide [@bagaria2025dynaguide], ADPro
[@adpro2025], GRITS [@grits2025], PCD [@pcd2025], Fisher-Preserving [@fisher2026]), and
a 2026 theory thread shows such guidance is **frequently miscalibrated and prone to
reward over-optimization** [@calibrated2026; @das2025; @anthology2025], with
independent control evidence that more guidance can destabilize [@pgg2025]. None of
these provide a **practical predictor** of when to guide a *robot manipulation* prior.

HRTG is uniquely positioned to supply that predictor: it already has multiple
prior-quality regimes under matched seeds and a large, honest negative-results bank.
Reframing the negatives as **evidence for a calibration boundary** converts HRTG's main
weakness (no single SOTA-beating headline) into its contribution.

## The experiment (concrete)

**Claim to establish.** Hybrid lift is predictable from measurable prior-quality
signals, and the combination rule — not the λ value — is the primary determinant of
whether guidance helps or over-optimizes.

**Step 1 — Instrument (new logging, no new physics).** Extend `run_can.py` to log,
per rollout: k-NN distance to the retrieval bank, k-NN action disagreement,
denoising/score consistency of the prior, short-horizon rollout drift, and standalone
(λ=0) success. Persist into the existing `unified_summary_*.json` `per_lambda` schema.

**Step 2 — Re-analyze existing data (zero new compute).** Across regimes already on
disk — `demo_1nn`, DP `ep0149`, DP `ep0199`, Square-DP, the 140/140 DP λ/σ/seed grid,
and the static λ sweeps — regress hybrid lift on the Step-1 signals. Expect: positive
lift at coherent-but-unsaturated priors; ≤0 at saturated or incoherent priors.

**Step 3 — Combination-rule contrast (few targeted cells).** On 2-3 representative
cells, compare additive score-bridge vs product/gate/temper at matched strength, and
log a reward-over-optimization proxy (prior-likelihood drop / off-manifold drift).
Hypothesis: product/gate maximize the approximate cost greedily and collapse, mirroring
DAS's tempered-vs-greedy result [@das2025] and the calibration failure [@calibrated2026].

**Step 4 — Predictor + decision rule.** Fit a tiny logistic/linear model:
"given prior diagnostics, should we guide, and at what λ?" Report its accuracy at
predicting the sign of lift on held-out cells. This is the deliverable: a calibration
rule, not another sweep.

**Step 5 — Evaluation rigor.** Use anytime-valid sequential testing with partial-credit
task progress [@savi2026] for the guided-vs-unguided comparison to spend fewer rollouts.

## Flagship case study: retrieval as a zero-training test-time score prior

Differentiate cleanly from the closest neighbor, DARP [@darp2026], which *trains* a
difference-aware kNN model: HRTG uses retrieval with **no training** as a score/pull
target inside MBD, and reports the **finite-coverage bank-size curve**. Use the kNN-MI
quality idea [@curation2025] for the coverage diagnostic and split bank-size results by
in-bank vs out-of-bank initial states.

## Novelty positioning (harsh-critic verdict)

A dedicated 6-round novelty pass (`novelty/novelty_report.json`) found that
**calibrated/predictive test-time guidance for diffusion policies is NOT a novel
method** — there are direct 2025-2026 neighbors:

- **LRT-Diffusion** [@lrtdiffusion2025] — risk-calibrated log-likelihood-ratio gating of
  guidance for diffusion policies (offline RL, D4RL MuJoCo).
- **PPGuide** [@ppguide2026] — steering diffusion policies with a learned performance
  predictor.
- **DynaGuide** [@bagaria2025dynaguide] — external-model steering of a frozen DP.

**Implication:** HRTG must **not** claim a new guidance/gating algorithm. The
defensible, non-overlapping contribution for a *class final report / workshop* is the
**empirical cross-prior-type calibration map**: under matched seeds on Robomimic,
relate hybrid lift to prior-quality diagnostics across *qualitatively different* priors
(zero-training retrieval vs trained DP at `ep0149`/`ep0199` vs task-specific Square-DP),
and explain product/gate over-optimization. None of the neighbors compare heterogeneous
prior types, and none use retrieval as a zero-training test-time score prior with a
finite-coverage law. Frame the work as an **empirical study + diagnostic**, cite the
neighbors as related work, and lead with the retrieval case study.

## Scope guardrails (what NOT to do for the report)

- No new hyperparameter sweeps or seed pools beyond Steps 3-5 (consistent with the
  standing project rule: settle the method/analysis before sweeping).
- No phase/contact `tool_hang` build-out — cite HDP [@hdp2025], RDP [@xue2025reactive],
  ARCH/SPIRE [@sun2024arch; @zhou2024spire] as future work; it needs new observability
  and heavy compute.
- No full TAMP integration; no large heterogeneous retrieval banks before locality
  filtering is validated.

## Suggested timeline (report-scoped)

| Block | Output |
|---|---|
| 1 | Add per-rollout prior-quality logging to `run_can.py`; backfill on a few cells. |
| 2 | Re-analyze existing regimes; produce the lift-vs-diagnostics figure + regression. |
| 3 | Combination-rule contrast cells + over-optimization proxy figure. |
| 4 | Fit/report the calibration decision rule; write the retrieval case-study + bank-coverage figure; apply anytime-valid testing. |

## Deliverables produced in this pass

- `paper_db_raw.jsonl` (167 records), `paper_db.jsonl` (top 70).
- `phase1_frontier/frontier.md`, `phase2_survey/survey.md`.
- `phase3_deep_dive/selection.md`, `deep_dive.md` (12 papers).
- `phase4_code/code_repos.md`.
- `phase5_synthesis/synthesis.md`, `gaps.md`.
- `phase6_report/references.bib` (18 arXiv-backed + 9 foundational).

## Caveats

Planning-grade review. Several records are arXiv/S2 API metadata; venue and
peer-review status (T-RO, ICLR, ICRA, RA-L tags) and exact numbers must be verified
from PDFs before camera-ready. Future-dated arXiv IDs reflect the environment clock.
