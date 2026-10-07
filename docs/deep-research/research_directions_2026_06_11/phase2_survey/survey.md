# Phase 2 — Survey (2026-06-11 refresh)

**Corpus.** `paper_db.jsonl` keeps the 70 highest-relevance records from the 167
deduplicated frontier hits. Records carry a `_score` (domain-keyword relevance +
recency). The survey clusters them into six themes that map onto HRTG's decision.

> Status note: many records are arXiv API metadata; venue strings from Semantic
> Scholar (e.g. "IEEE Transactions on Robotics", "ICLR", "ICRA", "RA-L") are noted
> where present, but exact venue/peer-review status must be re-verified at write-up
> time. Future-dated arXiv IDs (26xx.*) reflect the environment clock (2026).

## Cluster A — Test-time guidance of pretrained policies (the core neighborhood)

This is HRTG's home cluster. All of these steer a *fixed, pretrained* generative
policy at inference time with an external signal, no policy retraining.

- **DynaGuide** [@bagaria2025dynaguide] — guides action diffusion with a *separate*
  dynamics model during denoising; works with off-the-shelf DP; steers to multiple
  objectives and stays robust under low-quality objectives (70% steering success on
  CALVIN, 5.4× over goal-conditioning with low-quality objectives).
- **GRITS** [@grits2025] — a differentiable spillage predictor as guidance steering
  DP sampling toward safer trajectories while preserving success.
- **Projected Coupled Diffusion** [@pcd2025] — test-time coupled guidance + a hard
  projection step per denoising step (object manipulation, multi-robot motion).
- **Fisher-Preserving Guidance** [@fisher2026] — training-free guidance that limits
  off-manifold Fisher drift; uses official Diffusion Policy weights on PushT.
- **Policy Gradient Guidance** [@pgg2025] — ports classifier-free guidance to policy
  gradients; finds guidance can *destabilize* continuous control (a "when guidance
  hurts" data point).

## Cluster B — Test-time adaptation / inference-compute control of DP

- **ADPro** [@adpro2025] — manifold-constrained denoising + analytic init; no
  retraining; an *alternative combination rule* to HRTG's additive bridge.
- **Neural-Dynamics-Augmented DP** [@ndadp2025] — robust DP in a local support
  region from 1-30 demos + dynamics model to expand coverage (few-shot/coverage).
- **DA-SIP** [@dasip2025] — difficulty-aware adaptive inference budget for
  diffusion/flow policies (2.6-4.4× compute reduction).
- **ManiCM** [@manicm2024] / **Consistency Policy** [@prasad2024consistency] —
  consistency-distilled fast DP inference (efficiency appendix relevance).

## Cluster C — Calibration / over-optimization of guidance (the theory hook)

- **Calibrated Test-Time Guidance for Bayesian Inference** [@calibrated2026] —
  shows common test-time guidance does **not** recover the true posterior and
  identifies the structural approximations responsible; proposes consistent
  estimators. This is the formal statement of HRTG's empirical calibration gap.
- **Test-time Alignment without Reward Over-optimization** (ICLR 2025) [@das2025] —
  SMC/tempering to hit a reward-aligned target without over-optimizing; names the
  failure mode that HRTG's product/gate variants exhibit.
- **Connections: RL-feedback ↔ test-time scaling ↔ diffusion guidance** [@anthology2025]
  — equivalences between guidance, soft best-of-N, and reward-directed sampling.

## Cluster D — Retrieval / nonparametric priors for imitation

- **DARP — Difference-Aware Retrieval Policies** [@darp2026] — semi-parametric kNN
  retrieval IL, 15-46% over BC; *trains* a model on neighbors + distance vectors.
  Closest neighbor to `demo_1nn`, but HRTG uses retrieval with **zero training** as
  a score target inside MBD.
- **Robot Data Curation with kNN-MI** [@curation2025] — kNN mutual-information
  estimator scores demonstration quality (supports a bank-coverage diagnostic).
- **Behavior Retrieval** [@du2023behavior], **skill retrieval** [@nasiriany2022learning]
  — retrieval used to select/augment *training* data (the training-time contrast).
- **Latent NN-guided sparse control** [@nnsparse2023] — reward agents for staying
  near demonstrated trajectories in an embedding space (sparse-task analogue).

## Cluster E — Hierarchy / contact guidance for long-horizon, contact-rich tasks

- **Hierarchical Diffusion Policy via Contact Guidance** (T-RO) [@hdp2025] —
  high-level contact-point prediction guides low-level action diffusion; +20.8% over
  DP, strongest on contact-rich tasks. Template for `tool_hang`/Square residuals.
- **Reactive Diffusion Policy** [@xue2025reactive], **TacDiffusion** [@tacdiffusion2024],
  **DP + Residual RL** [@dprrl2025] — contact/force/tactile feedback for insertion
  and force-sensitive tasks; argues residual insertion failures need richer
  contact observability, not bigger λ.
- **ARCH** [@sun2024arch], **SPIRE** [@zhou2024spire], **HDT** [@correia2023hierarchical]
  — subgoal/TAMP decomposition for sparse long-horizon control.

## Cluster F — Evaluation rigor

- **Beyond Binary Success** [@savi2026] — safe anytime-valid sequential inference for
  robot policy comparison; up to 70% fewer rollouts. A cheap rigor upgrade for
  HRTG's n=30 cells and a way to compare guided vs unguided under partial-credit
  metrics.

## What changed vs the 2026-06-10 survey

- The "test-time guidance of pretrained policies" cluster is now **explicit and
  crowded** (DynaGuide, GRITS, PCD, Fisher, ADPro), confirming HRTG's framing is
  current rather than niche.
- A **calibration/over-optimization theory thread** (Calibrated Guidance 2026; DAS
  ICLR'25) appeared that did not feature in the prior pass — this is the new,
  strongest hook for HRTG's negative-results bank.
- **DARP (2026)** is a newly-surfaced near-neighbor for the retrieval prior and must
  be cited for novelty differentiation.
- **Beyond Binary Success (2026)** offers a concrete evaluation-rigor upgrade.
