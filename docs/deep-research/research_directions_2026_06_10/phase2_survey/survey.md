# Phase 2 — Survey landscape

## Database summary

- **Raw supplemented database:** `paper_db_raw_supplemented.jsonl`, 295 unique records after merge.
- **Ranked database:** `paper_db.jsonl`, top 80 by project-specific affinity score.
- **Curated anchor database:** `paper_db_curated.jsonl`, 45 records selected for relevance to the project.
- **BibTeX:** `phase6_report/references.bib`, 45 generated entries.

The automated ranking is useful for broad discovery but imperfect: it over-ranks some non-robotics diffusion-planning papers and under-ranks classic retrieval/nearest-neighbor imitation work. The curated set corrects this by forcing inclusion of known anchor papers.

## Theme map

| Theme | What the literature says | Project connection | Directional consequence |
|---|---|---|---|
| Action diffusion policies | Diffusion Policy represents visuomotor policies as conditional denoising processes over actions and reports strong results across manipulation benchmarks [@chi2023diffusion]. Follow-ups accelerate or extend this family through 3D representations [@ze2024diffusion] and consistency distillation [@prasad2024consistency]. | Your Square-DP result shows representation-specific DP priors can unlock a contact-rich task that retrieval struggled with. | Treat DP as a strong but fallible prior; study when MBD improves it and when it over-anchors. |
| Diffusion as trajectory optimization | M2Diffuser and related work use diffusion for trajectory optimization/planning in mobile manipulation or model-based optimization [@yan2024diffuser; @yun2024guided]. | Your score-bridge loop already resembles test-time prior-guided trajectory optimization. | Reframe the method as *retrieval/diffusion-guided MBD* and compare to diffusion planner baselines rather than only BC. |
| Retrieval and prior-data selection | Behavior Retrieval selects relevant behaviors from unlabeled datasets using few task examples [@du2023behavior]. Skill-based imitation retrieves / augments prior data to improve novel-task learning [@nasiriany2022learning]. | Your method retrieves one nearest demonstration action at every state and uses it as a score/pull target. | Novel direction: retrieval at inference time as a nonparametric score prior, with coverage diagnostics and bank-edit adaptation. |
| Long-horizon hierarchy | HDT uses subgoals for sparse/long-horizon settings [@correia2023hierarchical]. ARCH and SPIRE decompose long-horizon contact-rich tasks into primitives, planning, imitation, and RL components [@sun2024arch; @zhou2024spire]. | `tool_hang` remains unsolved after static MBD tuning; the first key-discovery trace exposes only `robot0_eef_pos` and `tool_pos`. | Move to phase predicates / subgoal-local MBD rather than more product/gate tuning. |
| Contact-rich assembly and insertion | Contact-rich manipulation often needs primitive structure, force/tactile/reactive feedback, or careful contact modeling [@levine2015learning; @xue2025reactive; @he2024foar]. | Square residual failures are post-contact or alignment-like; tool_hang lacks target-alignment trace keys. | Add contact/alignment observables before proposing new algorithmic variants. |
| Fast adaptation and data scaling | Large datasets like DROID broaden the available prior data [@khazatsky2024droid], while imitation-bootstrapped RL and behavior retrieval address adaptation/data efficiency [@hu2024imitation; @du2023behavior]. | Your Can mass-shift results show cheap bank augmentation can dominate BC retraining in the bite zone. | Formalize bank editing as a fast adaptation mechanism and test it under larger task/dynamics shifts. |

## Survey conclusion

The most defensible next research direction is a **test-time prior geometry** story:

1. retrieval priors work when nearest-neighbor coverage gives local action geometry;
2. diffusion priors work when the observation schema makes the score field coherent;
3. MBD helps when the prior is coherent but has headroom;
4. hierarchy is needed when sparse tasks require temporally separated contact phases.

This story is narrower and stronger than claiming a universal hybrid controller. It aligns with the current experimental evidence and with the literature frontier.
