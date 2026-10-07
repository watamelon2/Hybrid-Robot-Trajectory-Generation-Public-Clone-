# K0 diagnostic: --mpc-replans 0 single-shot bug

**Status**: preserved for traceability; **DO NOT use these numbers**.

## What this directory is

The output of the first Phase-C K-sweep launch (PID 30265,
2026-05-29 13:54 PDT, primary Mac CPU).  All 60 rollouts ran the
unified pipeline with `--mpc-replans 0`, which produced
single-shot 16-step rollouts (1 chunk only).  The robot
terminated before reaching the can.

## What went wrong

The original BRIEF specified `--mpc-replans 0` as the
DP-apples-to-apples flag, conflating it with standalone-DP's
"no MPC re-planning between chunks".  But `unified_loop.py:228`
reads

```python
n_chunks = int(n_replans) + 1
```

so `mpc_replans = 0` means **1 chunk total**, not "0 re-plans
beyond the initial plan".  With `--mpc-chunk-steps 8` and
`H = 16`, the single chunk's "last-chunk-takes-up-to-H" rule
(`unified_loop.py:337`) executed 16 env steps and terminated.
`--max-env-steps 400` was a CAP not a target, and 16 < 400 so it
never fired.

## Symptoms

- All 60 rollouts: `final_reward ∈ [0.009, 0.289]`, mean 0.085.
  (Phase-B standalone rewards on the same prior: failures ~1 – 3,
  successes ~25 – 32.)
- Every rollout's `chunks` array has `len == 1` and the chunk's
  `steps_executed == 16`.
- Total K = 0 wall: **68.87 s** for 60 rollouts (~1.1 s / rollout),
  vs the expected ~25 min (the robot never actually moved past
  initial pose).
- Success rate: **0 / 60 = 0.0 %**.

## Fix

Use `--mpc-replans 49` (50 chunks × 8 steps = 400 env steps).
See the parent `BRIEF.md` "Bug discovered + fixed" section and
`experiments/17_dp_hybrid/run_phase_c_ksweep.sh`.

## File inventory

- `unified_flow_lam-0.5_K-0_R-0.json` — per-lambda K = 0 result.
- `unified_summary_flow_K-0_R-0.json` — cross-lambda summary.

Both follow the same JSON schema as the fixed-config run, just
with the buggy `n_chunks=1` / `steps_executed=16` data instead of
the correct `n_chunks=50` / `total_steps≈400`.
