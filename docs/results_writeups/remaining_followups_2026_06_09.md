# Remaining follow-up scaffolds — first execution

**Slate:** `scaffold/remaining-work-2026-06-09` follow-up queue
**Status:** completed 2026-06-09.
**Queue log:** `experiments/15_unified_hybrid/results/remaining_scaffold_runs_2026_06_09/queue.log`

## Summary

The first execution of the remaining-work scaffolds completed without active processes left behind.  The run covered three targeted items from the post-presentation roadmap: a tiny `tool_hang` key-discovery probe, Square-DP failure tracing, and the Can DP `ep0199` checkpoint correction cells.

| Cell | Success | Rate | Wall time | Output |
|---|---:|---:|---:|---|
| `tool_hang` key discovery, `demo_1nn`, `λ=1`, H=120 | 0/1 | 0.0% | 120.7 s | `experiments/15_unified_hybrid/results/tool_hang_subgoal_scaffold_2026_06_09/unified_demo_1nn_lam-1.0_K-10_R-3_tool_hang_key_discovery.json` |
| Square-DP trace, pull, `λ=1`, H=16, R=49 | 17/30 | 56.7% | 24595.4 s | `experiments/15_unified_hybrid/results/square_dp_failure_trace_2026_06_09/unified_flow_pull_lam-1.0_K-10_R-49_square_dp_lam1_trace_test30.json` |
| Can DP correction D1, `ep0199`, pull, `λ=1`, `σ_max=0.10` | 19/30 | 63.3% | 3261.0 s | `experiments/17_dp_hybrid/results/ep0199_checkpoint_correction_2026_06_09/unified_flow_pull_lam-1.0_K-10_R-29_D1_ep0199_lam1_K10_smax10.json` |
| Can DP correction D2, `ep0199`, pull, `λ=10`, `σ_max=0.25` | 21/30 | 70.0% | 3275.4 s | `experiments/17_dp_hybrid/results/ep0199_checkpoint_correction_2026_06_09/unified_flow_pull_lam-10.0_K-10_R-29_D2_ep0199_lam10_K10_smax25.json` |

## Square-DP failure trace

The traced `λ=1` rerun reached 17/30 = 56.7%, below the earlier Square-DP co-peak of 21/30 = 70.0% at `λ=0.5` and `λ=1.0`.  The command matched the earlier Square-DP hybrid configuration and added `--save-step-trace`; the long wall time includes the overnight pause/resume interval.

The generated analyzer output is `experiments/15_unified_hybrid/results/square_dp_failure_trace_2026_06_09/square_failure_phase_summary.json`.

| Phase label | Count | Readout |
|---|---:|---|
| `success` | 17 | Completed insertions. |
| `align_or_insert` | 3 | High reward and close contact, but no terminal success. |
| `post_contact_unclear` | 10 | EEF reached the nut and the nut often moved/lifted, but the compact trace was insufficient to distinguish release, alignment, or insertion miss. |

The useful negative result is that failures do not look like pure approach failures under the compact trace.  Most failed rollouts reach object contact or near-contact, so the next Square follow-up should either trace the other co-peak (`λ=0.5`) or add richer peg/nut-relative alignment observables before proposing a new controller variant.

## tool_hang key discovery

The tiny probe intentionally used only one rollout.  It failed, but it confirmed that `--save-step-trace` is active for `tool_hang` and captured 120 step records.  The compact trace contained:

| Observation key | Count |
|---|---:|
| `robot0_eef_pos` | 120 |
| `tool_pos` | 120 |

No `target_pos`, `tool_hole_pos`, or similar target-alignment key appeared in the compact observations.  That means a tool_hang phase classifier can start with approach/lift predicates from `robot0_eef_pos` and `tool_pos`, but target/hole alignment will require either a richer observation extractor or task-specific simulator access.

## DP checkpoint correction

The corrected `ep0199` Can cells are strong:

- D1 (`λ=1`, `σ_max=0.10`) gives 19/30 = 63.3%.
- D2 (`λ=10`, `σ_max=0.25`) gives 21/30 = 70.0%.

These are the canonical checkpoint-corrected numbers for the D1/D2 cells and should replace any stale interpretation based on `ep0149` for these settings.

## Bank-size figure artifact

The bank-size saturation plotting scaffold generated `docs/results_writeups/bank_size_saturation.pdf` from the completed June-07 JSONs.  This is a paper-style rendering of the finite-coverage curve described in `docs/results_writeups/bank_size_scaling.md`.

## Immediate follow-ups

1. Run a `λ=0.5` Square-DP trace, because `λ=0.5` was the other 70.0% co-peak and may give the intended 9-failure taxonomy.
2. Extend the compact trace or add a task-specific key dump for `tool_hang` before trying to classify align/hang failures.
3. Use the `ep0199` D1/D2 correction numbers in the trained-prior section and retire the stale `ep0149` comparison for those cells.
