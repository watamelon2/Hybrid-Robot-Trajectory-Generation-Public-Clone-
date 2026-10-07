# MJX Level-1 implementation checklist

## Phase 1: Can planner + robosuite certifier

- Add an exp15-compatible MJX rollout factory.
- Keep action shape `(H, 7)` and `OSC_POSE` semantics.
- Implement Can robosuite-state to MJX `qpos/qvel` bridge.
- Implement Can MJX/robosuite observation bridge for retrieval and flow priors.
- Reuse existing `experiments/04_mbd_baseline/mjx_rollout.py` where possible.
- Certify selected `mu_final` in robosuite using `ReplayRobosuiteCertifier`.
- Save both MJX planner diagnostics and robosuite final metrics.
- Run 30-60 paired Can validation trials.

## Phase 2: Can retrieval / coverage

- Add retrieval feature bridge for `demo_1nn`, `demo_knn`, and `demo_lwr`.
- Support fresh-seed state bridge.
- Run bank-size checks for `20`, `50`, and `200` demos.
- Verify qualitative bank-size trend against robosuite.

## Phase 3: Can-DP

- Add `can_dp` observation schema.
- Add observation-history buffer support.
- Add DP prior bridge with private-generator seed handling.
- Track CPU/GPU transfer overhead separately from MJX rollout time.
- Verify checkpoint/headroom conclusion under robosuite certification.

## Phase 4: Square-DP prototype

- Create Square MJX scene only after Can gates pass.
- Record geometry/contact simplifications explicitly.
- Add Square observation schema and reward proxy.
- Validate cost-rank correlation before using any Square MJX acceleration.

## Phase 5: Can adaptation

- Map robosuite mass/friction perturbation flags to MJX model parameters.
- Validate representative perturbations against robosuite trends.

## Phase 6: sparse tasks

- Treat tool_hang and transport as research infrastructure.
- Add task scenes, success predicates, and subgoal/value reward proxies before running sweeps.

## Phase 7: MJX-native benchmark

- Only start if explicitly separating MJX results from robosuite SOTA.
- Requires MJX-native datasets, resets, rewards, success predicates, training loops, and reporting.
