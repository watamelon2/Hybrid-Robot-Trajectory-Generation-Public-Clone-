# Phase 5 — Synthesis

## Core synthesis

The repo evidence and literature converge on one central research theme:

> **Use test-time optimization to exploit local prior geometry, and study when that prior geometry is coherent enough to help.**

This decomposes the project into five concrete directions.

## Direction 1 — Nonparametric retrieval as an inference-time score prior

**Current evidence.** The Can headline is `demo_1nn` retrieval + score-bridge MBD: 177/240 = 73.8%. Bank-size scaling reaches most of the gain by 20 demos and plateaus around 50–200 demos.

**Literature anchor.** Behavior Retrieval and skill-retrieval work use prior data to select or augment training data [@du2023behavior; @nasiriany2022learning]. Your project is different: retrieval acts directly at inference time as a score/pull target.

**Research question.** When does nearest-neighbor action retrieval provide a useful local score field for trajectory optimization?

**Minimal experiments.**

1. In-bank vs out-of-bank initial-state split for the bank-size curve.
2. Phase-conditioned nearest-neighbor metrics: full-state vs object-relative vs gripper/object phase features.
3. Counterfactual retrieval quality metrics: nearest distance, action disagreement among k-NN, and success probability.

**Potential contribution.** A finite-coverage theory and diagnostic toolkit for retrieval-guided MBD.

## Direction 2 — Phase-conditioned sparse-task control for tool_hang / transport

**Current evidence.** `tool_hang` remains at 0% under raw sparse reward, dense shaping, horizon increases, sigma sweeps, product/gate/temper variants, and demo-phase pseudo-cost. The key-discovery probe found only `robot0_eef_pos` and `tool_pos` in compact traces.

**Literature anchor.** HDT, ARCH, SPIRE, and skill-retrieval work all support using hierarchy/subgoals for sparse or contact-rich long-horizon tasks [@correia2023hierarchical; @sun2024arch; @zhou2024spire; @nasiriany2022learning].

**Research question.** Can a small set of geometric phases turn sparse long-horizon MBD into a sequence of short, solvable local optimizations?

**Minimal experiments.**

1. Extend trace extraction to include target/hole/hook keys for `tool_hang`.
2. Define phase predicates: approach tool, grasp/lift tool, align hook, insert/hang.
3. Run phase-local MBD probes with the existing `demo_1nn` prior and compare to the monolithic baseline.

**Potential contribution.** A lightweight hierarchy for sparse manipulation that preserves the repo’s test-time-control framing.

## Direction 3 — Residual failure analysis for contact-rich insertion

**Current evidence.** Square-DP with task-specific representation reaches 70% in the original sweep, but traced reruns achieved 56.7% at `λ=1.0` and 63.3% at `λ=0.5`.  The traced `λ=0.5` taxonomy found 19 successes, 4 `align_or_insert` failures, and 7 `post_contact_unclear` failures; the traced `λ=1.0` taxonomy found 17 successes, 3 `align_or_insert` failures, and 10 `post_contact_unclear` failures.  The failures are not approach failures.

**Literature anchor.** Contact-rich manipulation often benefits from primitive structure, force/tactile/reactive feedback, or contact-aware control [@levine2015learning; @xue2025reactive; @he2024foar].

**Research question.** Are Square failures due to prior representation, contact observability, alignment geometry, or stochastic execution?

**Minimal experiments.**

1. Add compact peg/nut relative pose and contact-state proxies to traces.
2. Compare failure distributions across traced `λ=0.5`, traced `λ=1.0`, `λ=2.0`, and standalone DP.
3. Re-run Square only after the trace instrumentation is richer.

**Potential contribution.** A contact-phase diagnostic showing when MBD correction helps a learned diffusion prior and when it cannot fix missing state features.

## Direction 4 — Calibrated learned-prior/MBD coupling

**Current evidence.** Can DP ep0199 correction cells reached 63.3% and 70.0%; Square-DP improves only modestly from standalone 66.7% to hybrid 70.0%. The repo hypothesis is that MBD helps when the prior is coherent but has headroom.

**Literature anchor.** Diffusion Policy establishes strong action-prior learning [@chi2023diffusion], IBC motivates implicit/energy-style action models [@florence2021implicit], and guided diffusion planning motivates test-time guidance [@yun2024guided].

**Research question.** Can we predict the correct MBD coupling weight from prior calibration diagnostics instead of sweeping `λ` and `σ`?

**Minimal experiments.**

1. For each DP checkpoint, compute standalone success, action log-likelihood/proxy score, denoising consistency, and k-step rollout drift.
2. Correlate these diagnostics with hybrid lift under matched seeds.
3. Test adaptive `lambda_prior` based on prior uncertainty or action disagreement only after fixed diagnostics are validated.

**Potential contribution.** A principled criterion for when to hybridize a learned prior.

## Direction 5 — Fast adaptation by bank editing rather than retraining

**Current evidence.** Can mass-shift results show 5 adapted demos added to the retrieval bank can dominate BC retraining in wall-clock and success in the bite zone.

**Literature anchor.** Behavior Retrieval and large robot datasets highlight the value of selecting relevant prior data [@du2023behavior; @khazatsky2024droid].

**Research question.** Can local bank editing provide a general fast-adaptation mechanism under dynamics, object, or task perturbations?

**Minimal experiments.**

1. Extend the T2 adaptation grid to friction/object-shape perturbations.
2. Compare add-only, replace-only, and weighted retrieval banks.
3. Add retrieval-locality diagnostics to predict when 5 demos suffice.

**Potential contribution.** A practical adaptation method: change the bank, not the policy.

## Ranking for this project

| Rank | Direction | Why now | Risk |
|---:|---|---|---|
| 1 | Phase-conditioned sparse-task control | Directly attacks the largest remaining failure (`tool_hang`) and is strongly supported by hierarchy literature. | Requires richer observations and careful phase predicates. |
| 2 | Retrieval prior coverage theory | Builds on the strongest existing result and bank-size curve. | Theory may be modest unless paired with diagnostics. |
| 3 | Contact-rich residual diagnostics | Needed before claiming Square is robustly solved. | Might reveal missing simulator/contact observability beyond current code. |
| 4 | Learned-prior/MBD calibration | Turns DP results into a general method question. | Requires many matched-seed cells. |
| 5 | Bank-edit fast adaptation | Strong practical story from T2 results. | May be viewed as empirical unless grounded in coverage/locality theory. |
