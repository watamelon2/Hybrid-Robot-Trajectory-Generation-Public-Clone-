# Phase 4 — Code & tools

Repositories/project pages relevant to the test-time-guidance neighborhood and to
HRTG's planned calibration experiment. URLs are from the papers' abstracts/pages;
**star counts and last-commit dates are not verified live** (offline metadata
policy) — verify before camera-ready.

| Project | URL | Lang/stack | Relevance to HRTG |
|---|---|---|---|
| Diffusion Policy (Chi et al.) | `github.com/real-stanford/diffusion_policy` | PyTorch | Canonical action-diffusion prior; reference impl HRTG's DP prior derives from. |
| robomimic | `github.com/ARISE-Initiative/robomimic` | PyTorch | Benchmark/datasets (`can/ph`, `square/ph`, `tool_hang`, `transport`) used throughout HRTG. |
| robosuite | `github.com/ARISE-Initiative/robosuite` | MuJoCo | Simulator backend for HRTG rollouts. |
| DynaGuide | `dynaguide.github.io` (project page) | PyTorch (DP + external dynamics model) | Closest method: external-model guidance of frozen DP during denoising. Reference for the steering baseline HRTG differentiates from. |
| DARP | `weirdlabuw.github.io/darp-site` | PyTorch | Closest retrieval-prior neighbor; semi-parametric kNN IL. Differentiation target for HRTG's zero-training retrieval claim. |
| DAS (Test-time Alignment) | `github.com/krafton-ai/DAS` | PyTorch | SMC/tempered test-time alignment; reference for the "reward over-optimization" failure mode and a tempered alternative to product/gate. |
| Neural-Dynamics-Augmented DP | `dynamics-dp.github.io` (project page) | PyTorch | Few-shot DP + dynamics coverage; coverage/adaptation comparison. |
| Reactive Diffusion Policy | project page (TactAR/RDP) | PyTorch | Contact/tactile slow-fast DP; motivates richer Square/`tool_hang` contact traces. |

## HRTG's own implementation (the experiment platform)

| Component | Path | Role |
|---|---|---|
| Unified hybrid driver | `experiments/15_unified_hybrid/run_can.py` | Runs prior + MBD with `--lambda-prior`, `--combine-mode {score,tempered}`, `--prior {bc,demo_1nn,flow_*}`, `--n-workers`, seed/rollout controls. The exact knob surface a calibration study needs. |
| MBD combine rules | `experiments/04_mbd_baseline/mbd.py` | `tempered_combine` + `denoise_loop` `combine_mode` (score vs tempered) — the additive vs product/gate/temper rules under test. |
| DP prior checkpoints | `experiments/08_flow_matching_prior/`, `experiments/17_dp_hybrid/` | `ep0149` (hybrid-lift) vs `ep0199` (standalone) regimes. |
| Failure-phase analyzer | `analyze_square_failure_phases.py` (square_dp trace) | Existing phase taxonomy; template for `tool_hang` phase classifier and for partial-credit metrics. |
| Result schema | `unified_summary_*.json` (`per_lambda` rows: `n_success`, `n_total`, `success_rate`, `wall_s`) | Per-lambda matched-seed records — directly consumable for a calibration regression. |

## Tooling gap identified

- **No anytime-valid evaluation harness yet.** SAVI-style sequential testing
  [@savi2026] is not implemented in HRTG. A small wrapper over the existing
  `per_lambda` summaries (success + partial-credit task progress) would let the
  final report compare guided vs unguided with fewer rollouts — valuable given the
  finite Vast budget.
- **No prior-quality diagnostic logger.** HRTG records success/`wall_s` but not the
  per-rollout prior-quality signals (kNN distance, k-NN action disagreement,
  denoising/score-consistency, rollout drift) needed to *predict* hybrid lift. This
  is the main new instrumentation the calibration experiment requires.
