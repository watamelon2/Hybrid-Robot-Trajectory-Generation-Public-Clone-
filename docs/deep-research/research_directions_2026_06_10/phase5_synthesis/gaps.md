# Phase 5 — Gaps and hypotheses

## Gap 1 — Retrieval as an online score prior is underexplored

Most retrieval papers found in the search use retrieval to select or augment data before training [@du2023behavior; @nasiriany2022learning]. The repo uses retrieval inside the controller at inference time. This leaves room for a paper contribution if the project can formalize coverage, locality, and action-consensus diagnostics.

**Hypothesis.** `demo_1nn` succeeds when the local neighbor set is both close in task-relevant coordinates and low-disagreement in action space.

## Gap 2 — Sparse long-horizon manipulation needs phase observability

The literature supports hierarchy, but the current `tool_hang` compact trace lacks target/hole keys. Without phase observability, a classifier cannot separate alignment from grasp/lift failures.

**Hypothesis.** `tool_hang` will remain near zero until target/hook observables enter either the prior state or the MBD cost.

## Gap 3 — Contact-rich residual failures are not diagnosed by scalar success alone

Square-DP reaches a promising peak, but residual failures cluster after contact in the compact trace. The traced `λ=0.5` rerun reached 19/30 with 4 `align_or_insert` and 7 `post_contact_unclear` failures; the traced `λ=1.0` rerun reached 17/30 with 3 `align_or_insert` and 10 `post_contact_unclear` failures. Contact-rich papers suggest tactile/force/reactive features may matter [@xue2025reactive; @he2024foar].

**Hypothesis.** Square failures at the DP peak are mostly alignment/contact-state failures, not approach failures.

## Gap 4 — Learned-prior coupling lacks calibration criteria

Diffusion/implicit-policy literature provides strong priors, but it does not directly answer when to add MBD guidance. The repo has the right matched-seed setup to test this.

**Hypothesis.** Hybrid lift is maximized at intermediate prior quality: weak priors give incoherent scores; saturated priors leave little headroom; moderate priors benefit most.

## Gap 5 — Fast adaptation lacks a retrieval-locality diagnostic

The T2 result is compelling, but the method needs a predictor for when a few demonstrations will help.

**Hypothesis.** Adaptation benefit is predicted by how much new demonstrations reduce nearest-neighbor distance in the failure region, not by total bank size.

## Gap 6 — Large-bank scaling could hurt without task-family filtering

DROID-scale data could expand bank coverage [@khazatsky2024droid], but cross-task retrieval in the repo failed. Larger heterogeneous banks may increase bad nearest neighbors unless task/family filters or embeddings are learned.

**Hypothesis.** Retrieval-bank scaling is useful only after gating by task family, phase, or object-relative representation.
