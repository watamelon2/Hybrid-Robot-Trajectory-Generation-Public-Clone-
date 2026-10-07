# tool_hang value/subgoal decomposition scaffold — 2026-06-09

## Motivation

The closed MBD gated/tempered follow-ups show that tool_hang is not fixed by static product, max-gate, finite-p tempering, adaptive demo width, or demo-phase reward alone. The next scaffold should test whether long-horizon sparse success can be decomposed into short, denser geometric phases.

## Candidate phases

| Phase | Intended predicate | Candidate observations |
|---|---|---|
| approach_tool | EEF reaches the tool body | `robot0_eef_pos`, `Tool_pos`, `tool_pos` |
| grasp_tool | gripper/tool contact becomes stable | reward trace, tool displacement after close approach |
| lift_tool | tool leaves table height | `Tool_pos`, `tool_pos` z coordinate |
| align_hook | tool/hook approaches target/hole | `tool_hole_pos`, `target_pos` |
| hang_insert | terminal success or near-success | env success flag, reward jump |

## Minimal implementation plan

1. Use `--save-step-trace` on tiny manual probes to verify which tool_hang observation keys exist in robosuite outputs.
2. Write a phase classifier analogous to `analyze_square_failure_phases.py` once the observed keys are confirmed.
3. Replace one long H=300 objective with a chained set of phase-local probes, each using the existing `demo_1nn` warm-start and additive score mode.
4. Compare only against the existing additive score-bridge baseline. Do not reopen product/gate/temper tuning unless phase-local score mode unexpectedly works.

## Manual smoke command

Use `slate_tool_hang_subgoal_scaffold_2026_06_09.sh`. By default it prints commands only. Set `RUN=1` to launch a tiny key-discovery run.

## Acceptance for moving beyond scaffold

- JSON contains non-empty `chunks[*].step_trace`.
- At least one stable tool position key is present across steps.
- A phase classifier can separate approach/lift/align failures without using video.

## 2026-06-09 key-discovery result

The first tiny probe completed one rollout with 120 traced steps and 0/1
success.  The compact observation trace contained `robot0_eef_pos` and
`tool_pos` at every step.  It did not contain `target_pos`, `tool_hole_pos`,
or a similar target-alignment key, so approach and lift predicates can be
defined from the current trace, but align/hang predicates need a richer
observation extractor or task-specific simulator access.
