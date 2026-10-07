# Phase 3 — Paper selection

Selected 12 papers for deep reading, chosen to (a) pin HRTG's position in the
test-time-guidance neighborhood, (b) test the novelty of a "when does guidance
help a manipulation prior" calibration study, and (c) inform the contact/phase
direction for the hardest tasks.

| # | Paper | Why selected | Threat-to-novelty? |
|---:|---|---|---|
| 1 | DynaGuide [@bagaria2025dynaguide] | Closest method: steering off-the-shelf DP via external model during denoising. | High — must differentiate. |
| 2 | ADPro [@adpro2025] | Test-time DP adaptation, alternative combination rule (manifold + init). | Medium-high. |
| 3 | Calibrated Test-Time Guidance [@calibrated2026] | Formalizes guidance miscalibration; theory hook for HRTG negatives. | Low (theory, non-robotics) — strong ally. |
| 4 | Test-time Alignment w/o Reward Over-optimization [@das2025] | Names the over-optimization failure mode HRTG observes. | Low — ally. |
| 5 | DARP — Difference-Aware Retrieval Policies [@darp2026] | Closest retrieval-prior neighbor. | High for the retrieval claim. |
| 6 | Hierarchical Diffusion Policy / Contact Guidance [@hdp2025] | Template for phase/contact decomposition of hard tasks. | Medium for the sparse-task claim. |
| 7 | Reactive Diffusion Policy [@xue2025reactive] | Contact-rich residual failures need richer observability. | Low. |
| 8 | Beyond Binary Success (SAVI) [@savi2026] | Evaluation-rigor upgrade for n=30 cells. | None — tool. |
| 9 | Diffusion Policy [@chi2023diffusion] | The canonical prior HRTG guides. | None — foundation. |
| 10 | Neural-Dynamics-Augmented DP [@ndadp2025] | Few-shot coverage + dynamics; coverage angle. | Medium for adaptation. |
| 11 | Robot Data Curation w/ kNN-MI [@curation2025] | Quantifies demo quality; bank-coverage diagnostic. | Low — ally. |
| 12 | Policy Gradient Guidance [@pgg2025] | Independent "guidance can destabilize control" evidence. | None — ally. |

**Reading method.** Abstracts retrieved via arXiv/S2 API; section structure
(related work + experiments) pulled from ar5iv for DynaGuide and ADPro. Notes
below combine those with domain knowledge. Full PDFs should be re-read before
camera-ready citation (see caveats).
