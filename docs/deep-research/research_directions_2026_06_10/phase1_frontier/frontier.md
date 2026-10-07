# Phase 1 — Frontier scan

**Topic.** Research directions for Hybrid Robot Trajectory Generation after the June-2026 results.

**Repo grounding.** The project currently has four empirical anchors:

1. zero-training `demo_1nn` retrieval + score-bridge MBD is the Can headline at 177/240 = 73.8%;
2. Square improves when the prior representation becomes task-specific (`square_dp` DP prior: 20/30 standalone and 21/30 hybrid peak);
3. long-horizon sparse tasks such as `tool_hang` are not solved by more static product/gate/temper tuning;
4. adaptation results favor cheap demo-bank augmentation over BC retraining in the Can mass-shift bite zone.

**Search corpus.** API searches wrote raw JSONL under `phase1_frontier/search_results/` and `phase2_survey/search_results/`. The supplemented raw database contains 295 unique records. `paper_db.jsonl` keeps 80 ranked records and `paper_db_curated.jsonl` keeps 45 project-relevant anchor records.

## Frontier trends that matter for this project

| Trend | Representative papers | Why it matters here |
|---|---|---|
| Diffusion policies are becoming the default strong visuomotor prior. | `Diffusion Policy` [@chi2023diffusion], `3D Diffusion Policy` [@ze2024diffusion], `Consistency Policy` [@prasad2024consistency] | Your Can/Square DP results should be positioned as *test-time hybridization of a strong action-prior*, not as yet another BC baseline. |
| Diffusion is increasingly used as a planner / trajectory optimizer, not only as a closed-loop policy. | `M2Diffuser` [@yan2024diffuser], guided trajectory generation [@yun2024guided], `PRESTO` [@seo2024presto] | This directly overlaps your MBD score-bridge loop: test-time denoising can be framed as prior-guided trajectory optimization. |
| Long-horizon manipulation work is shifting toward hierarchy, skills, or TAMP interfaces. | `ARCH` [@sun2024arch], `SPIRE` [@zhou2024spire], `MOSAIC` [@mishani2025mosaic], `Hierarchical Decision Transformer` [@correia2023hierarchical] | Your `tool_hang` failures support leaving static MBD tuning and moving to phase/subgoal decomposition. |
| Retrieval from prior data is an active few-shot mechanism, but underexplored as a *test-time score prior*. | `Behavior Retrieval` [@du2023behavior], skill retrieval [@nasiriany2022learning], nearest-neighbor guided diffusion [@van2025dmc] | Your strongest result is retrieval-as-prior with no training; the novelty lies in using retrieval to shape MBD, not just to augment training data. |
| Contact-rich manipulation is moving toward tactile/force-aware and reactive policies. | contact-rich GPS [@levine2015learning], `Reactive Diffusion Policy` [@xue2025reactive], `FoAR` [@he2024foar] | Square/tool_hang residual failures are likely contact-phase failures; richer contact/alignment traces may matter more than more λ sweeps. |
| Large robot datasets and foundation-action models are shifting the scaling question. | DROID [@khazatsky2024droid], VLA surveys [@shao2025large] | A natural extension is replacing the small Robomimic bank with retrieval from larger heterogeneous banks, but only after validating task-family locality. |

## Frontier implication

The highest-leverage next direction is not a bigger hyperparameter sweep. The literature frontier suggests a sharper thesis:

> **Hybrid test-time control is useful when the prior supplies local, phase-correct action geometry, while MBD supplies online correction.**

That thesis can be tested with five research threads: retrieval coverage theory, phase-conditioned sparse-task control, residual failure traces for contact tasks, calibrated learned-prior/MBD coupling, and fast adaptation via local bank editing.
