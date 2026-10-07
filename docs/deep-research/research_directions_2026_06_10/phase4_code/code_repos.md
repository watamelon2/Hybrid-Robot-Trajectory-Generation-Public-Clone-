# Phase 4 — Code and tools landscape

This phase records implementation ecosystems relevant to the project. Repository metadata should be refreshed before a publication deadline; entries below are enough to guide near-term engineering choices.

| Repo / tool | URL | Relevance | How it informs this project |
|---|---|---|---|
| Diffusion Policy | `https://github.com/real-stanford/diffusion_policy` | Reference implementation for action-diffusion visuomotor policies [@chi2023diffusion]. | Compare architecture/inference defaults against `experiments/08_flow_matching_prior` and Square-DP training. |
| RoboMimic | `https://github.com/ARISE-Initiative/robomimic` | Benchmark suite and datasets used by this project. | Keep using Robomimic tasks for controlled ablations; cite benchmark context separately from method claims. |
| robosuite | `https://github.com/ARISE-Initiative/robosuite` | Simulation environment backing Robomimic manipulation tasks. | Needed for richer observations in Square/tool_hang failure-phase tracing. |
| DROID | `https://github.com/droid-dataset/droid` | Large in-the-wild manipulation data ecosystem [@khazatsky2024droid]. | Future retrieval-bank scaling direction after task-family locality is validated. |
| D4RL | `https://github.com/Farama-Foundation/D4RL` | Offline RL benchmark used by IBC/HDT-style papers. | Useful for positioning but less central than Robomimic for the current manipulation experiments. |
| TAMP / planning ecosystems | Example: `https://github.com/caelan/pddlstream` | Task-and-motion planning infrastructure related to SPIRE-style decomposition. | Optional future path if geometric phase predicates become symbolic task plans. |
| Diffusion-planning repos | Paper-specific repos for M2Diffuser / PRESTO / guided diffusion planning, if available. | Useful for implementing diffusion-planner baselines. | Search and pin exact repos before implementing a baseline; avoid speculative dependency creep now. |

## Practical engineering recommendation

Do not immediately port a full external TAMP or diffusion-planner codebase. The repo already has a working unified driver and DP prior path. The next code should be incremental:

1. extend trace extraction for Square/tool_hang target/contact keys;
2. implement phase-local objective wrappers inside the existing MBD loop;
3. add retrieval diagnostics and bank locality metrics;
4. only then compare against external diffusion-planning baselines.
