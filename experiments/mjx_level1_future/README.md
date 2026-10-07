# MJX Level-1 future-direction scaffold

Purpose: define a staged future-work scaffold for using MJX as a planner accelerator while keeping robosuite as the canonical evaluator.

This is intentionally **not** a replacement for the current robosuite experiment pipeline. It is a broader Level-1 scope plan: MJX can grow from Can acceleration into Can-DP, Square-DP, adaptation, and eventually sparse-task prototypes, but every promoted result must still be robosuite-certified.

## Chosen level

**Level 1: MJX planner, robosuite certifier.**

- MJX evaluates candidate MBD rollouts quickly.
- The final action sequence is replayed in robosuite.
- Reported success comes from robosuite.
- Existing robomimic datasets, OSC action space, DP/BC/retrieval priors, and SOTA comparisons remain valid.

## Non-negotiable consistency constraint

Keep the robosuite-compatible **7-D `OSC_POSE` action space**.

The action path should be:

```text
policy / prior outputs OSC_POSE action chunks
        -> JAX OSC controller
        -> MJX motor torques
        -> mjx.step candidate rollout
        -> selected mu_final replayed in robosuite
        -> robosuite success / reward reported
```

Do **not** switch to raw torque or joint actions for the Level-1 path. That would require new demos, retraining, and new baselines.

## Phased scope

The scaffold has a wider roadmap than the first implementation sprint.

| Phase | Scope | Status | Promotion gate |
|---|---|---|---|
| 0 | Existing simplified Can MJX backend in `experiments/04_mbd_baseline` | available evidence | keep as appendix/future-work result |
| 1 | Can Level-1 planner + robosuite certifier | first implementation target | paired Can validation passes |
| 2 | Can retrieval / bank-size / coverage | after Phase 1 | fresh-seed Can trend matches robosuite qualitatively |
| 3 | Can-DP checkpoint/headroom cells | after retrieval schema is stable | DP obs-history parity and robosuite-certified success |
| 4 | Square-DP planning prototype | later scope | Square scene/reward parity and paired validation |
| 5 | Can adaptation under mass/friction | later scope | perturbation semantics match robosuite intent |
| 6 | tool_hang / transport sparse-task prototypes | deferred research scope | task scenes, success predicates, and subgoal/value validation exist |
| 7 | MJX-native training/evaluation | separate project | only if explicitly reported as MJX benchmark results |

Initial implementation target remains **Can + Can-DP**, but the scaffold now records the larger future scope so later work can extend cleanly instead of redesigning the interface.

## Required future components

1. **Exp15 backend seam**
   - Add an MJX candidate-rollout backend compatible with `experiments/15_unified_hybrid/unified_loop.py`.
   - Preserve the current robosuite path as default.

2. **Robosuite certification replay**
   - After MJX selects `mu_final`, replay that exact action sequence in robosuite.
   - Record robosuite reward, success, and trace fields as the final metric.

3. **State/init bridge**
   - Map robosuite HDF5/fresh-seed starts into the MJX scene as closely as possible.
   - Record when the mapping is approximate.

4. **Observation bridge**
   - Build the same obs schemas expected by retrieval and DP priors.
   - Initial target schemas: `can` and `can_dp`.

5. **Prior support**
   - Minimum: `demo_1nn` and `flow` with `score-mode=pull`.
   - Later: retrieval variants and DP checkpoint grids.

6. **Validation gate**
   - Run paired Can cells comparing:
     - robosuite-planned / robosuite-evaluated,
     - MJX-planned / robosuite-certified.
   - Do not use MJX-planned numbers in main report tables until paired validation is acceptable.

7. **Task scope registry**
   - Track each MJX task by phase, required scene work, obs schema, prior support, and validation gate.
   - Do not allow a task to appear in final result tables before its gate is satisfied.

8. **Scene/reward parity notes**
   - For each task, record which robosuite components are approximated in MJX.
   - Report MJX-only cost/reward as planner diagnostics, not canonical success.

9. **Compute/back-end policy**
   - Keep CPU robosuite as the certification path.
   - Use MJX/JAX only where batch planning dominates cost.
   - Track compilation time separately from rollout/evaluation time.

## Scaffolded modules

| File | Role |
|---|---|
| `level1_contract.py` | Core planner/certifier protocols and `run_level1_trial` orchestration |
| `task_registry.py` | Phase/task registry for Can, Can-DP, Square-DP, adaptation, and sparse-task scope |
| `planner_adapter.py` | Dry-run planner plus future Level-1 MJX planner adapter seam |
| `certifier.py` | Dry-run certifier plus generic robosuite replay certifier seam |
| `state_bridge.py` | Robosuite-init to MJX-state bridge interface |
| `obs_bridge.py` | MJX/robosuite observation bridge interface |
| `scene_registry.py` | Scene availability and parity metadata for each task phase |
| `reward_proxy.py` | Reward/cost proxy seam for future MJX task costs |
| `task_scaffolds.py` | First-class Square-DP, tool_hang, and transport MJX scaffold metadata |
| `success_predicates.py` | Success-predicate seam for future task-specific MJX checks |
| `exp15_seam.py` | Non-invasive adapter seam matching the exp15 planner inputs |
| `validation.py` | Validation gate records and component-readiness checks |
| `dry_run.py` | Dependency-light CLI for checking the scaffold |
| `scope_matrix.md` | Staged scope and acceptance gates |
| `implementation_checklist.md` | Phase-by-phase implementation checklist |

## Out of scope for the first implementation sprint

- Full MJX-native training.
- Raw torque-action policy retraining.
- Square MJX scene implementation.
- tool_hang / transport MJX scene implementation.
- Replacing robosuite as the final evaluator.
- Reporting MJX-only success as canonical task success.

Transport is the one expected exception to 7-D action shape: it should still
use `OSC_POSE` semantics, but as two concatenated 7-D arm commands
(`OSC_POSE_DUAL_14D`).

## Success criteria for promoting beyond future work

Promote Level 1 from future direction to experimental backend only if:

1. Can paired validation shows MJX-planned trajectories survive robosuite certification.
2. The final success metric is always computed by robosuite.
3. The speedup is large enough to justify backend complexity.
4. The report clearly distinguishes robosuite-planned vs MJX-planned/robosuite-certified rows.

## Current status

Existing MJX Gate-4 evidence is enough for an appendix/future-work claim:

- Simplified Can MJX backend exists in `experiments/04_mbd_baseline`.
- JAX OSC controller parity is excellent.
- Colab L4 run completed 60 rollouts.
- Measured speedup is **13.8x**.
- The path is not yet canonical because it uses a simplified scene and lacks robosuite success certification.
