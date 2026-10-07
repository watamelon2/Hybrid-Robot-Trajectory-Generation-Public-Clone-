# Vast A40 remote offload closeout

**Status:** completed and cleaned up 2026-06-11.
**Remote instance:** Vast.ai instance `40441157`, 2x NVIDIA A40.
**Remote root:** `/workspace/hrtg-cloud/HRTG-square-dp-followup`.
**Remote venv:** `/workspace/hrtg-venv`.

## Summary

The June-10 remote offload finished successfully.  The cloud instance ran the validation-gated DP production queue, the static offload queue, and the longest local remainder lanes.  The final cleanup check found no active local screens, no active local `run_can.py`, no remote tmux sessions, and no active remote experiment processes.

The final state is:

| Component | Completion evidence | Status |
|---|---:|---|
| Local `coverage_fresh` | 16/16 summaries | complete |
| Local `retrieval_variants` | 16/16 summaries | complete |
| Remote DP production | 140/140 summaries and `queue_completed_success` | complete |
| Remote adaptation/friction remainder | 18/18 summaries | complete |
| Remote `tr_demo_phase_trace` | 1/1 summary | complete |
| Remote `tr_lam_ablate_trace` | composite marker plus split outputs | complete |
| Static BC K/R grid | 9/9 original summaries | complete |
| Static BC `K=10,R=2` lambda sweep | 5/5 split summaries in `tail_boost2_static_2026_06_10` | complete |
| Optional tail boost | 9/9 summaries and `all boost jobs exited` | complete |

## Result roots

| Purpose | Remote path |
|---|---|
| DP production shards | `experiments/15_unified_hybrid/results/vast_safe_dp_2026_06_10/` |
| Static original queue | `experiments/15_unified_hybrid/results/offload_chain_2026_06_10/` |
| Remote local-remainder offload | `experiments/15_unified_hybrid/results/research_queue_remote_remainder_2026_06_10/` |
| First optional tail boost | `experiments/15_unified_hybrid/results/tail_boost_2026_06_10/` |
| Second static split race | `experiments/15_unified_hybrid/results/tail_boost2_static_2026_06_10/` |

## Completion conventions

Two long monolithic cells were split during the tail phase to avoid letting one sequential `run_can.py` invocation determine the wall-clock finish.

### Transport lambda ablation

The original monolithic transport ablation was:

```text
tr_lam_ablate_trace --lambda-prior 0 1 inf
```

It was treated as complete by composition:

| Lambda | Source |
|---:|---|
| `0` | original `research_queue_remote_remainder_2026_06_10/sparse_task_extra/unified_demo_1nn_lam-0.0_K-10_R-3_tr_lam_ablate_trace.json` |
| `1` | boosted `tail_boost_2026_06_10/sparse_task_extra/unified_demo_1nn_lam-1.0_K-10_R-3_tr_lam1_trace_boost_nw5.json` |
| `inf` | boosted `tail_boost_2026_06_10/sparse_task_extra/unified_demo_1nn_lam-inf_K-10_R-3_tr_laminf_trace_boost_nw5.json` |

Marker:

```text
experiments/15_unified_hybrid/results/research_queue_remote_remainder_2026_06_10/.markers/tr_lam_ablate_trace.composite_done.json
```

The monolithic original process was stopped after the three lambda outputs existed.

### Static BC lambda sweep

The original monolithic static sweep was:

```text
bc_k10_r2_lam_sweep_no5 --lambda-prior 0 1 2 10 20
```

The canonical completed static sweep outputs are the five independent `n_workers=12` race outputs under:

```text
experiments/15_unified_hybrid/results/tail_boost2_static_2026_06_10/static_offload/bc_lambda_sweep_split_nw12/
```

The earlier `n_workers=6` split outputs under `tail_boost_2026_06_10` also completed and are useful as redundant checks, but the `race2` set finished the final required coverage cleanly.

## Static offload results

### Original K/R grid, `lambda=5`

| K | R | Success | Rate | Wall time |
|---:|---:|---:|---:|---:|
| 5 | 0 | 21/60 | 35.0% | 2223.9 s |
| 5 | 1 | 17/60 | 28.3% | 4421.5 s |
| 5 | 2 | 21/60 | 35.0% | 6656.1 s |
| 10 | 0 | 26/60 | 43.3% | 4263.6 s |
| 10 | 1 | 22/60 | 36.7% | 8972.7 s |
| 10 | 2 | 22/60 | 36.7% | 13283.9 s |
| 20 | 0 | 20/60 | 33.3% | 8830.8 s |
| 20 | 1 | 22/60 | 36.7% | 15945.9 s |
| 20 | 2 | 15/60 | 25.0% | 21757.1 s |

### Split lambda sweep, `K=10,R=2`

Canonical `race2` outputs:

| Lambda | Success | Rate | Wall time |
|---:|---:|---:|---:|
| 0 | 28/60 | 46.7% | 5858.9 s |
| 1 | 18/60 | 30.0% | 5845.3 s |
| 2 | 19/60 | 31.7% | 5856.4 s |
| 10 | 31/60 | 51.7% | 5866.5 s |
| 20 | 18/60 | 30.0% | 5876.8 s |

Redundant first-boost outputs:

| Lambda | Success | Rate | Wall time |
|---:|---:|---:|---:|
| 0 | 25/60 | 41.7% | 10710.7 s |
| 1 | 21/60 | 35.0% | 10761.9 s |
| 2 | 26/60 | 43.3% | 10757.7 s |
| 10 | 28/60 | 46.7% | 10725.5 s |
| 20 | 24/60 | 40.0% | 10757.3 s |

The best static BC cell in the completed set is `K=10,R=2,lambda=10` from `race2` at 31/60 = 51.7%.  Because these were tail-race duplicates, use the output suffix/root when citing exact provenance.

## Remote remainder results

### Adaptation friction lane

| Friction condition | Bank | Success | Rate | Wall time |
|---|---|---:|---:|---:|
| `0p3` | `bank205` | 12/30 | 40.0% | 8562.6 s |
| `0p3` | `bank220` | 12/30 | 40.0% | 8612.2 s |
| `0p3` | canonical | 9/30 | 30.0% | 8574.5 s |
| `0p5` | `bank205` | 7/30 | 23.3% | 8622.7 s |
| `0p5` | `bank220` | 9/30 | 30.0% | 8649.2 s |
| `0p5` | canonical | 6/30 | 20.0% | 8583.8 s |
| `0p7` | `bank205` | 11/30 | 36.7% | 8615.7 s |
| `0p7` | `bank220` | 9/30 | 30.0% | 8285.4 s |
| `0p7` | canonical | 7/30 | 23.3% | 8607.0 s |
| `1p5` | `bank205` | 5/30 | 16.7% | 8266.5 s |
| `1p5` | `bank220` | 8/30 | 26.7% | 8260.9 s |
| `1p5` | canonical | 10/30 | 33.3% | 8241.8 s |
| `2p0` | `bank205` | 6/30 | 20.0% | 8200.6 s |
| `2p0` | `bank220` | 12/30 | 40.0% | 8225.5 s |
| `2p0` | canonical | 11/30 | 36.7% | 8220.4 s |
| `3p0` | `bank205` | 8/30 | 26.7% | 6740.8 s |
| `3p0` | `bank220` | 9/30 | 30.0% | 6738.6 s |
| `3p0` | canonical | 8/30 | 26.7% | 8252.0 s |

### Transport sparse extras

| Cell | Success | Rate | Wall time | Source |
|---|---:|---:|---:|---|
| `tr_demo_phase_trace` | 0/5 | 0.0% | 23882.1 s | original remainder |
| `tr_demo_phase_trace_boost_nw5` | 0/5 | 0.0% | 5653.5 s | optional boost |
| `tr_lam1_trace_boost_nw5` | 0/5 | 0.0% | 5675.4 s | composite lambda-1 source |
| `tr_laminf_trace_boost_nw5` | 0/5 | 0.0% | 5660.2 s | composite lambda-inf source |

The very large rewards observed in `tr_demo_phase_trace` are from the demo-phase dense pseudo-reward.  That reward is scaled and accumulated over the full transport horizon, so values in the hundreds of thousands are not comparable to the normal sparse task reward.  The success indicator remains the reliable task metric here, and all transport sparse-extra rows are 0/5.

## DP production

The validation-gated DP production queue completed 140/140 sharded summaries under:

```text
experiments/15_unified_hybrid/results/vast_safe_dp_2026_06_10/
```

The remote queue log recorded:

```text
[vast-safe] queue_completed_success Thu Jun 11 01:43:05 2026
```

The DP jobs are intentionally sharded by checkpoint, lambda, sigma, and test seed block; analyze them by aggregating the per-shard `unified_summary_*.json` files rather than reading a single monolithic summary.

## Checkpoint calibration lanes (added 2026-06-11)

These lanes ran but were **not** captured in the original closeout tables. Numbers
below are read directly from the on-disk summaries under
`experiments/15_unified_hybrid/results/research_queue_2026_06_10/`.

### Can DP `ep0199` σ_max=0.1 λ-grid (`can_dp_calibration/`)

| Lambda | Success | Rate | Checkpoint |
|---:|---:|---:|---|
| 0.0 | 21/30 | 70.0% | `v_theta_dp_tobs2_h16_v3b_ep0199.pt` |
| 0.5 | 15/30 | 50.0% | `v_theta_dp_tobs2_h16_v3b_ep0199.pt` |
| 1.0 | 20/30 | 66.7% | `v_theta_dp_tobs2_h16_v3b_ep0199.pt` |

Note the non-monotonic dip at λ=0.5 — directly relevant to the calibration-of-guidance
next-direction study. (R=29 / N=30.)

### Square DP `ep0049` λ-grid (`square_dp_checkpoint_calibration/`) — PARTIAL

| Lambda | Success | Rate | Checkpoint |
|---:|---:|---:|---|
| 0.0 | 12/30 | 40.0% | `eta_dp_square_tobs2_h16_ep0199_ep0049.pt` |

Only the λ=0 cell exists on disk (an 18 MB trace, no summary). The λ>0 cells of this
grid were **not** completed/synced.

### Related (already documented elsewhere)

The separate Can DP `ep0199` D1/D2 checkpoint-correction run (2026-06-09) is documented
in `remaining_followups_2026_06_09.md`: D1 (λ=1, σ_max=0.10) → 19/30 = 63.3%;
D2 (λ=10, σ_max=0.25) → 21/30 = 70.0%. Do **not** confuse it with the σ_max=0.1 λ-grid
above.

## Data sync gaps — RESOLVED 2026-06-11 (and a critical invalidation finding)

**Update:** all remote roots below were recovered to local on 2026-06-11 (via a
copy-target instance `40673435` + tar-over-ssh). They are now under
`experiments/15_unified_hybrid/results/`. See
`docs/results_writeups/vast_dp_grid_recovery_analysis_2026_06_11.md` for the full
provenance and per-shard analysis.

| Remote root | What it holds | Status now |
|---|---|---|
| `vast_safe_dp_2026_06_10/` | the 140 DP "production" shards | recovered — **but INVALID, see below** |
| `tail_boost2_static_2026_06_10/` | static λ-sweep JSONs (nw=12) | recovered, valid |
| `tail_boost_2026_06_10/` | BC λ-sweep + transport composites (nw=2) | recovered, valid |
| `offload_chain_2026_06_10/` | static K/R grid JSONs (nw=4) | recovered, valid |
| `research_queue_remote_remainder_2026_06_10/` | adaptation/transport remainder | recovered, valid |

**CRITICAL:** once the per-shard numbers were recovered, the DP production grid
turned out to be an **infrastructure artifact, not a result**. It was launched
with `--n-workers 1` (`queue_vast_safe_remote_2026_06_10.py` L100/L131), which
triggers the chunk-boundary controller-state corruption documented in
`run_can.py:565-577`. Symptoms: Square 0/30 at every checkpoint/λ, Can ≤26.7%,
λ=0 collapsed from the documented 53.3% to 0%, and 6.7× per-rollout slowdown.
**Do not cite any `vast_safe_dp_2026_06_10` number.** The valid DP headline
numbers remain the local `square_dp_prior.md` (Square-DP 66.7% standalone /
70% hybrid) and the Can DP cells, all run with `--n-workers ≥ 2`. The
"140/140 complete" marker only verified completion, not validity.

Process fix: future remote DP/flow-prior offload must pass `--n-workers ≥ 2`,
with a post-run check that flags λ=0 unified success far below the standalone
reference.

## Cleanup status

Final cleanup verified completion first, then stopped redundant remote sessions and leftover stale monolithic work.  The cleanup check returned:

```text
cleanup_verified
---sessions-after---
---active-after---
---counts-after---
dp 140/140
tail_boost 9
race2 5/5
tr_marker yes
```

Thus the remote instance was left with no active tmux sessions and no active experiment processes after cleanup.

## Notes for later analysis

1. Treat `tr_lam_ablate_trace` as a composite result using the marker file, not as a single original monolithic JSON set.
2. Treat the static `K=10,R=2` lambda sweep as split outputs, with `tail_boost2_static_2026_06_10` as the canonical complete set.
3. Do not interpret stale tracebacks in old monolithic static logs as final failure; those logs include killed/redundant attempts after split outputs were already complete.
4. Pull the remote result roots before terminating or recycling the Vast instance if the files are not already synced locally.
