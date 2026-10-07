# Research directions for Hybrid Robot Trajectory Generation

**Date:** 2026-06-10

**Purpose.** Identify project-specific follow-up research directions by combining repo results with a targeted literature search/review.

**Main recommendation.** Position the project around **test-time prior geometry for robot trajectory optimization**:

> Retrieval or diffusion priors provide local action geometry; MBD/test-time optimization corrects the prior when the task cost and state representation make the correction meaningful.

This framing explains the current successes and failures better than a generic “hybrid beats BC” story.

## Project state from the repo

| Empirical anchor | Current readout |
|---|---|
| Can zero-training headline | `demo_1nn` retrieval + score-bridge MBD reaches 177/240 = 73.8%. |
| Bank-size scaling | Can retrieval gains arrive by ~20 demos and plateau through 50–200 demos. |
| Square | `square_dp` prior reaches 66.7% standalone and 70.0% hybrid; representation matters. |
| Square residual failures | The traced `λ=1.0` run was 17/30 and the traced `λ=0.5` run was 19/30; failures are mostly post-contact/alignment-like, not approach. |
| tool_hang | Static MBD tuning variants remain ineffective; first trace exposes only `robot0_eef_pos` and `tool_pos`. |
| Can adaptation | Bank augmentation can beat BC retraining under mass shift in wall-clock and success. |

## Literature grounding

- Diffusion Policy is the canonical action-diffusion prior for robot manipulation [@chi2023diffusion].
- Diffusion planning / trajectory optimization work supports treating diffusion models as test-time trajectory generators, not only closed-loop policies [@yan2024diffuser; @yun2024guided].
- Retrieval work selects useful prior data for few-shot imitation [@du2023behavior; @nasiriany2022learning], but your repo’s inference-time retrieval prior is a distinct angle.
- Long-horizon sparse/contact tasks increasingly use hierarchy, subgoals, or TAMP decomposition [@correia2023hierarchical; @sun2024arch; @zhou2024spire].
- Contact-rich manipulation literature suggests reactive/contact-aware representations matter for residual insertion failures [@levine2015learning; @xue2025reactive; @he2024foar].

## Recommended research directions

### 1. Phase-conditioned MBD for long-horizon sparse tasks

**Question.** Can `tool_hang` and `transport` be solved by decomposing them into short phase-local MBD problems?

**Why this is first.** The repo has already falsified many static tuning variants. Literature supports hierarchy for sparse long-horizon settings, and the current trace shows exactly what is missing: target/hole observability.

**Next implementation.**

1. Extend `_compact_step_obs` for `tool_hang` to capture target/hook/hole keys.
2. Write a `tool_hang` phase classifier analogous to `analyze_square_failure_phases.py`.
3. Add a phase-local runner: approach tool → lift tool → align hook → hang/insert.
4. Compare to monolithic `demo_1nn` score-bridge with identical seeds.

**Paper angle.** “Sparse long-horizon failure is not fixed by stronger priors alone; phase observability and local objectives are required.”

### 2. Retrieval prior coverage theory and diagnostics

**Question.** Why does a zero-training nearest-neighbor prior work so well on Can, and when should it fail?

**Why this is strong.** It builds directly on the 73.8% headline and the bank-size saturation curve.

**Next implementation.**

1. Split bank-size results by in-bank/out-of-bank initial-state proximity.
2. Log nearest-neighbor distance and k-NN action disagreement during rollouts.
3. Test object-relative and phase-conditioned retrieval metrics.
4. Fit a simple coverage/readiness predictor for rollout success.

**Paper angle.** “Retrieval-guided MBD behaves like finite mode coverage; bank editing and local action consensus predict success.”

### 3. Contact-rich residual failure diagnosis for Square

**Question.** What exactly remains unsolved after Square-DP reaches 70%?

**Why this matters.** Square is no longer a pure failure, but the trace suggests unresolved post-contact/alignment failures.

**Next implementation.**

1. Add richer Square trace keys: peg/nut relative pose, gripper state, contact proxy, final alignment error.
2. Compare standalone DP, traced `λ=0.5`, traced `λ=1.0`, and `λ=2.0` failure distributions.
3. Re-run only after instrumentation changes; the current traced `λ=0.5` and `λ=1.0` reruns are already sufficient to show post-contact/alignment failures.

**Paper angle.** “Task-specific representation unlocks Square; residual failures are contact/alignment observability failures.”

### 4. Calibration rules for learned-prior/MBD coupling

**Question.** Can we predict when MBD helps a learned prior without sweeping `λ`?

**Why this is natural.** Can DP and Square-DP now provide multiple prior-quality regimes.

**Next implementation.**

1. Compute checkpoint diagnostics: standalone success, action variance, denoising consistency, rollout drift.
2. Correlate diagnostics with hybrid lift across matched seeds.
3. Test uncertainty/adaptive-λ only after the fixed diagnostic is predictive.

**Paper angle.** “MBD helps coherent-but-unsaturated priors; it does not magically repair incoherent priors or add much to saturated ones.”

### 5. Fast adaptation through bank editing

**Question.** Can a few new demos adapt the controller faster and more robustly than retraining?

**Why this is practical.** The mass-shift T2 result already supports this, and retrieval literature provides a broader context for selecting prior data.

**Next implementation.**

1. Add friction/object perturbation axes beyond mass.
2. Compare add-only, replace-only, and reweighted banks.
3. Use nearest-neighbor distance reduction in failure regions as the adaptation predictor.

**Paper angle.** “For test-time retrieval priors, adaptation is local data structure editing, not network fine-tuning.”

## Suggested 4-week execution plan

| Week | Goal | Concrete outputs |
|---:|---|---|
| 1 | Finish diagnostics | Extend Square and `tool_hang` trace keys; start from the completed `λ=0.5` and `λ=1.0` Square trace summaries. |
| 2 | Build phase-local sparse runner | `tool_hang` phase classifier and first approach/lift/align probes. |
| 3 | Retrieval diagnostics | Add k-NN distance/action-disagreement logging; analyze bank-size and T2 results. |
| 4 | Calibration/adaptation writeup | Correlate DP checkpoint quality with hybrid lift; draft research-note figures. |

## What not to prioritize

- More product/gate/temper-p sweeps for `tool_hang` without new phase observability.
- More Square λ sweeps before richer residual-failure traces are analyzed.
- Full external TAMP integration before a minimal phase-local MBD prototype.
- Larger heterogeneous retrieval banks before task-family/locality filters are validated.

## Deliverables produced in this research pass

- `paper_db_raw_supplemented.jsonl` — merged raw search database.
- `paper_db.jsonl` — ranked top-80 database.
- `paper_db_curated.jsonl` — 45 selected project-relevant records.
- `phase1_frontier/frontier.md` — frontier scan.
- `phase2_survey/survey.md` — landscape survey.
- `phase3_deep_dive/selection.md` and `deep_dive.md` — selected-paper notes.
- `phase4_code/code_repos.md` — code ecosystem.
- `phase5_synthesis/synthesis.md` and `gaps.md` — synthesis and gap analysis.
- `phase6_report/references.bib` — generated bibliography.

## Caveats

This is a planning-grade literature review. Some recent entries are preprints or API metadata records with incomplete abstracts. Before final paper writing, re-read the exact PDFs for the papers cited in the main narrative and validate venue/status.
