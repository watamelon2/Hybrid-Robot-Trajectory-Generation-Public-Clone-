# Phase-D triage evidence: `--n-workers` flip

These two JSONs are the **smoking gun** for the
`unified_loop.py:466-468` reset+restore bug.  See `../BRIEF.md`
for the full writeup.

## Reproduction

Both runs used **identical** flags except `--n-workers`.  The
runs are short (`--max-env-steps 120` instead of 400) for fast
triage; the full-horizon verification is the next step.

```bash
# Common flags (3 test rollouts at fresh seeds 100000-100002).
COMMON=(
    --task can --prior flow
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt
    --K 0 --H 16
    --mpc-replans 14 --mpc-chunk-steps 8
    --n-demos 0 --n-rollouts 0
    --n-test-rollouts 3 --test-start-seed 100000
    --max-env-steps 120
    --prior-n-steps 100
    --sigma-max 0.25 --sigma-min 0.05
    --lambda-prior 0.5
    --device cpu --seed 0
)

# Run A: n_workers=1 -> _mbd_touches_main_env=True -> reset+restore on.
python -m experiments.15_unified_hybrid.run_can \
    "${COMMON[@]}" --n-workers 1 --output-dir /tmp/n1/
# -> n1_unified_flow_lam-0.5_K-0_R-14.json

# Run B: n_workers=2 -> _mbd_touches_main_env=False -> reset+restore off.
python -m experiments.15_unified_hybrid.run_can \
    "${COMMON[@]}" --n-workers 2 --output-dir /tmp/n2/
# -> n2_unified_flow_lam-0.5_K-0_R-14.json
```

## Result

| seed | n_workers=1 (reset+restore on) | n_workers=2 (reset+restore off) |
|---:|---:|---:|
| 100000 | reward +2.75, FAIL | **reward +27.24, SUCCESS** |
| 100001 | reward +1.89, FAIL | reward +5.78, FAIL |
| 100002 | reward +1.36, FAIL | reward +21.53, FAIL |
| **3-seed succ** | **0/3 = 0.0 %** | **1/3 = 33.3 %** |
| mean reward | 2.00 | 18.18 |
| wall | 55.6 s | 131.5 s |

The wall difference at `n_workers=2` is the subprocess pool's
spawn cost (~30 s for two robosuite workers on macOS); at K = 0
the pool is created but never used (denoise_loop's body never
iterates).  At K > 0 this overhead is amortised across the K
inner iterations.

Even at the truncated 120-step horizon (vs the 400-step Phase-C
protocol) the gap is decisive: 0/3 vs 1/3 success, 9× reward
mean.  At the full 400-step horizon the gap is expected to widen
further (Phase-B standalone hits 80 % over 50 seeds with the
same 400-step max).

## Mechanism (one paragraph)

`unified_loop.py:466-468` runs

```python
if _mbd_touches_main_env:
    env.reset()
    set_sim_state(env, current_state)
```

at every chunk boundary, where `_mbd_touches_main_env =
(n_workers <= 1 or dataset_path is None)`.  This block exists to
restore the main env after MBD's worker pool clobbers it (the
`n_workers <= 1` path runs `rollout_one` on the main env for
each of M perturbed rollouts).  But it fires **even at K = 0**,
where `denoise_loop` never iterates and rollout_fn is never
called — so there's no clobbering to undo, yet the
`env.reset()` still wipes the robosuite controller's goal pose
cache / interpolator state / last-commanded action.
`set_sim_state` only restores Mujoco's `qpos / qvel`, NOT the
Python-side controller state.  Result: every chunk starts with a
controller that thinks "goal pose = current pose", so the
prior's 8-action chunk drives near-zero motion.

The `n_workers=2` path doesn't have this issue because workers
each fork their own env; the main env is never touched, and
`_mbd_touches_main_env` evaluates to False, skipping the
reset+restore entirely.

## Files

- `n1_unified_flow_lam-0.5_K-0_R-14.json`: Run A's per-rollout
  dump.  Each rollout has `chunks: [...15 chunks of 8 steps...]`
  with `chunk_reward` mostly < 0.5, accumulating to `final_reward`
  in [1.36, 2.75].
- `n2_unified_flow_lam-0.5_K-0_R-14.json`: Run B's per-rollout
  dump.  Same chunk structure but `chunk_reward` builds to lift
  thresholds on seed 100000 (success at chunk 14).
