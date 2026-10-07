# MJX Level-1 scope matrix

This file records the expanded MJX future-work scope. The key constraint remains unchanged: MJX is a planner/candidate-rollout accelerator, while robosuite certifies final success.

## Scope principles

1. Preserve the 7-D `OSC_POSE` action space.
2. Keep robosuite as the canonical evaluator.
3. Treat MJX rewards/costs as planning diagnostics until validated.
4. Add tasks in phases, not as a single full-framework rewrite.
5. Require paired robosuite certification before reporting MJX-planned results as experimental evidence.

## Phase 0: existing MJX evidence

Existing status:

- Simplified Can MJX backend exists in `experiments/04_mbd_baseline`.
- JAX OSC controller parity is strong.
- Colab L4 run completed 60 rollouts.
- Measured speedup is **13.8x**.

Report role:

- Engineering appendix.
- Future-work motivation.
- Not canonical task success.

## Phase 1: Can Level-1 planner + certifier

Goal:

- Use MJX for candidate rollout scoring.
- Replay selected `mu_final` in robosuite.
- Record robosuite success as the final metric.

Required pieces:

- `exp15` backend seam.
- Can state/init bridge.
- Can observation bridge.
- Robosuite certification replay.
- Paired validation against robosuite-planned baseline.

Acceptance gate:

- 30-60 paired Can trials where MJX-planned / robosuite-certified success is comparable enough to justify continued use.

## Phase 2: Can retrieval / bank-size / coverage

Goal:

- Support `demo_1nn`, `demo_knn`, `demo_lwr`, and related retrieval variants under the MJX planner.
- Preserve existing retrieval-bank semantics.

Required pieces:

- MJX-compatible feature extraction for nearest-neighbor retrieval.
- Fresh-seed state bridge.
- Bank-size experiment wrapper.

Acceptance gate:

- Fresh-seed qualitative trend matches robosuite: small banks under-cover, 20-50 demos improve sharply, larger banks plateau.

## Phase 3: Can-DP checkpoint/headroom cells

Goal:

- Run selected Can-DP checkpoint/headroom experiments with MJX as the planner backend.

Required pieces:

- `can_dp` observation schema parity.
- Obs-history buffer support.
- DP prior invocation without excessive CPU/GPU transfer overhead.
- Robosuite certification for every selected action sequence.

Acceptance gate:

- Headroom conclusion is stable: moderate checkpoints can benefit from MBD, saturated checkpoints have less room.

## Phase 4: Square-DP planning prototype

Goal:

- Evaluate whether MJX can accelerate Square-DP planning without changing the scientific meaning of the task.

Required pieces:

- Square MJX scene.
- SquareNut / peg geometry and contact approximation notes.
- Square observation schema.
- Square reward/cost and success proxy.
- Paired robosuite certification.

Acceptance gate:

- MJX cost ranking correlates with robosuite outcome well enough to justify selected Square-DP acceleration.

Risk:

- High. Square is contact/geometry sensitive, so simplification may change the task.

## Phase 5: Can adaptation under mass/friction

Goal:

- Use MJX to accelerate selected Can adaptation sweeps.

Required pieces:

- Mass/friction perturbation controls in MJX.
- Explicit mapping from robosuite perturbation flags to MJX model parameters.
- Paired validation on representative perturbations.

Acceptance gate:

- Perturbation semantics match robosuite sufficiently for trend-level claims.

## Phase 6: sparse-task prototypes

Goal:

- Explore whether MJX can support tool_hang / transport planning prototypes.

Required pieces:

- New MJX scenes.
- Multi-object contact setup.
- Task-specific success predicates.
- Dense/subgoal/value-style reward proxies.
- Demonstration/state compatibility plan.

Acceptance gate:

- The prototype produces interpretable task-phase behavior and survives robosuite certification on small paired runs.

Risk:

- Very high. This is research infrastructure, not final-report work.

## Phase 7: MJX-native training/evaluation

Goal:

- Only if the project intentionally becomes a separate MJX benchmark effort.

Required pieces:

- MJX-native reset/distribution definitions.
- MJX-native dataset generation.
- MJX-native training loops.
- Separate reporting from robosuite SOTA.

Acceptance gate:

- Results are explicitly labeled as MJX benchmark results, not robosuite benchmark results.

## Recommended near-term cut

For the current project, stop at:

1. Phase 1 as implementation target.
2. Phase 2 / Phase 3 as planned extensions.
3. Phase 4+ as documented future scope only.

This gives the report a credible MJX roadmap without requiring a risky full-framework rewrite.
