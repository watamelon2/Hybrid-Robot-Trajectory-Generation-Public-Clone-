# Phase 3 — Deep-dive selection

The selected papers were chosen to cover the mechanisms directly implicated by the repo results: action diffusion priors, diffusion/planning hybrids, retrieval from prior data, long-horizon hierarchy, and contact-rich manipulation.

| # | Paper | Reason for selection | Project theme |
|---:|---|---|---|
| 1 | Diffusion Policy [@chi2023diffusion] | Canonical action-diffusion visuomotor prior; direct baseline/ancestor for Square-DP. | learned prior representation |
| 2 | M2Diffuser [@yan2024diffuser] | Diffusion as trajectory optimization in mobile manipulation. | diffusion-guided planning |
| 3 | Guided trajectory generation with diffusion models [@yun2024guided] | Connects diffusion models to offline model-based optimization. | test-time trajectory optimization |
| 4 | Implicit Behavioral Cloning [@florence2021implicit] | Energy/implicit policy framing for multimodal actions; conceptual bridge to score-like policies. | score/energy priors |
| 5 | Behavior Retrieval [@du2023behavior] | Few-shot task data queries relevant behaviors from broader datasets. | retrieval / data selection |
| 6 | Learning and Retrieval from Prior Data for Skill-based Imitation Learning [@nasiriany2022learning] | Skill extraction plus retrieval-based augmentation. | retrieval + temporal abstraction |
| 7 | Hierarchical Decision Transformer [@correia2023hierarchical] | Uses high-level subgoals for sparse/long-horizon imitation/RL. | subgoals for sparse tasks |
| 8 | ARCH [@sun2024arch] | Hierarchical hybrid learning for long-horizon contact-rich assembly. | contact-rich hierarchy |
| 9 | SPIRE [@zhou2024spire] | TAMP decomposes long-horizon manipulation into smaller learning problems. | TAMP + IL/RL decomposition |
| 10 | Reactive Diffusion Policy [@xue2025reactive] | Slow-fast visual-tactile policy for contact-rich manipulation. | residual contact failures |
| 11 | DROID [@khazatsky2024droid] | Large in-the-wild manipulation dataset; relevant to future retrieval-bank scaling. | scaling priors |
| 12 | Consistency Policy [@prasad2024consistency] | Distills visuomotor policies for faster inference. | compute-efficient priors |

**Evidence note.** This pass is grounded in API metadata/abstracts and known paper-level claims where available. It is suitable for research-direction planning, but paper-writing claims should be revalidated with full PDFs before final citation use.
