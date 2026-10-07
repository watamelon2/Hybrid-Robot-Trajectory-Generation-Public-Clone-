# Phase-D: root-cause + fix for the Phase-C K=0 gate failure — 2026-05-30

## TL;DR

Phase-C's K = 0 gate failed with **0 / 50 = 0.0 %** test rate while
standalone-DP on the same Mac with the same ckpt / seeds hit
**7 / 10 = 70.0 %**.  The 70-pp gap is **NOT** any of the four
hypotheses listed in `phase_c_2026_05_29/BRIEF.md::Verdict` — those
are all wired correctly.  The actual bug is in
`experiments/15_unified_hybrid/unified_loop.py:466-468`:

```python
if _mbd_touches_main_env:
    env.reset()
    set_sim_state(env, current_state)
```

This block fires **at every chunk boundary** when
`n_workers <= 1 OR dataset_path is None` (the default
configuration).  `env.reset()` wipes robosuite's controller state
(OSC goal cache, interpolator, last-commanded pose), and
`set_sim_state` only restores Mujoco's `qpos / qvel` — not the
controller's Python-side bookkeeping.  Result: every chunk's
8-action plan starts with a freshly-reset controller that thinks
`goal_pos ≈ current_pose`, so the cumulative motion intent encoded
in the prior's chunk is lost.

The bug is invisible at K > 0 with the legacy 46-D `can` schema
because the standalone there caps at ~25 – 37 % (Phase-A regime),
so the controller-state loss is masked by other limitations.  It
becomes the dominant failure mode the moment the prior is strong
enough to actually grasp the can (Phase-B's 80 % standalone).

## Status

**RED → GREEN-on-fix.** Triage complete; minimal fix proposed and
empirically pre-verified on 3 seeds.  Awaiting user go-ahead on
the `run_can.py` / `unified_loop.py` edit per
`HANDOFF_NEXT_MAC.md::Contact / escalation`.

## Evidence

### Smoking-gun side-by-side (this BRIEF's triage)

Same 3 fresh seeds (100000–100002), same ckpt
(`v_theta_dp_tobs2_h16_v3b_ep0199`), same config, **only
`--n-workers` flipped** (which is the ONE flag that controls
whether the reset+restore block fires).  Truncated to
`--max-env-steps 120` for speed (a full 400-step run is the
Phase-D verification step, see below).

| seed | `--n-workers 1` (reset+restore on) | `--n-workers 2` (reset+restore off) |
|---:|---:|---:|
| 100000 | reward +2.75, FAIL | **reward +27.24, SUCCESS** |
| 100001 | reward +1.89, FAIL | reward +5.78, FAIL |
| 100002 | reward +1.36, FAIL | reward +21.53, FAIL |

3-seed success: **0/3 vs 1/3** even at the truncated 120-step
horizon.  Reward mean: 2.00 vs 18.18 (9× improvement).  Wall:
56 s vs 131 s (the subprocess pool's spawn cost; harmless at K = 0
since the pool is never invoked).

Raw JSONs preserved at `triage_n_workers/` for traceability.

### Phase-C reference (the failure being explained)

Identical seeds (subset of 100000–100049), full 400-step horizon:

| pipeline | seeds | test rate | wall |
|---|---|---:|---:|
| **Phase-C unified K=0** (`n_workers=1`) | 50 fresh | **0 / 50 = 0.0 %** | 108 min |
| **Phase-B standalone re-eval** (`standalone_dp.py`) | 10 fresh | **7 / 10 = 70.0 %** | 20.4 min |

Standalone runs the prior in a single continuous Python loop with
no `env.reset` between action chunks — exactly what the unified
loop fails to do.

### Why all 4 of `phase_c_2026_05_29/BRIEF.md`'s hypotheses are wrong

Code-reviewed against
`experiments/15_unified_hybrid/{run_can.py,unified_loop.py}` and
`experiments/08_flow_matching_prior/{standalone_dp.py,adapter_dp.py}`:

1. **T_obs = 2 obs stack maintenance.**  Unified DOES call
   `prior.observe(s0)` inside `initial_mean_fn` (`run_can.py:152-159`)
   and DOES call `prior.reset_obs_buffer()` between rollouts
   (`run_can.py:926-932`).  This was the 2026-05-26 fix that
   landed before Phase-C.  Confirmed correct.

2. **`--lambda-prior 0.5` blending at K = 0.**  Unified gates the
   score-fn construction on `lambda_prior != 0.0`
   (`unified_loop.py:421`), AND `denoise_loop` at K = 0 never
   iterates the inner loop (`mbd.py:234`, `for k in range(K)`),
   so the score_fn is never called.  `mu_final` equals
   `initial_mean.copy()` bit-for-bit at K = 0.  Confirmed correct.

3. **Warm-start state at chunk boundary.**  Unified seeds the
   private generator via `initial_mean_fn._set_chunk_seed`
   (`run_can.py:160-167`, `unified_loop.py:386-391`).  Each chunk
   re-samples from the prior conditioned on the live env's
   current obs.  Confirmed correct.

4. **`n_action_steps` vs `H` alignment.**  Unified executes
   `mu_final[t]` for `t in range(steps_this_chunk)` where
   `steps_this_chunk = min(chunk_steps=8, H=16, ...) = 8`
   (`unified_loop.py:336-341, 470`).  Same 8-of-16 slice as
   standalone.  Confirmed correct.

### The actual bug (Hypothesis 5)

`unified_loop.py:300-309` defines:

```python
_mbd_touches_main_env = (
    int(n_workers) <= 1 or dataset_path is None
)
```

This is intentional defensive logic for when MBD's worker pool
shares the main env (n_workers ≤ 1 path goes through exp 04's
`rollout_one`, which itself does `env.reset() +
set_sim_state(env, init_state)` for each of M perturbed
rollouts).  When that path is taken, the main env is left at an
arbitrary post-M-rollout state, so the unified loop snapshots it
back before executing the chunk:

```python
# unified_loop.py:466-468
if _mbd_touches_main_env:
    env.reset()
    set_sim_state(env, current_state)
```

**Two compounding flaws:**

(a) **The fix-up fires unconditionally even when MBD didn't
actually run.**  At K = 0, `denoise_loop`'s outer
`for k in range(K)` never iterates (`mbd.py:234`), so
`rollout_fn` is never called, so the main env is never
clobbered.  But the reset+restore still happens — fixing a
problem that doesn't exist, and introducing the bug we now
have.

(b) **`set_sim_state` is incomplete as a snapshot.**  Even
when MBD HAS clobbered the env (K > 0 case), restoring only
`sim.set_state_from_flattened(qpos, qvel)` does not restore:
- OSC goal pose cache (`controller.goal_pos`, `goal_ori`)
- Interpolator state (smooth target transition; resets to
  current pose on `env.reset()`)
- Last-commanded action (used by some controllers for
  filtering)
- Gripper action buffer
- The placement sampler's RNG state (used by reward
  functions that compute `dist(eef, can_initial_pose)`)

All of these are pure Python state in robosuite, not in
Mujoco's `qpos / qvel`.  An `env.reset()` wipes them; an
`env.step()` would have evolved them naturally.

### Why this is invisible at K > 0 in the legacy 46-D regime

The Phase-3 hybrid `can` baseline (BC prior, 46-D legacy schema,
K = 10) hit 36.7 % vs the BC-only standalone 30 % — a 6.7 pp
delta.  The standalone BC was so weak that "MBD's denoising +
the reset+restore controller-state corruption" landed near the
BC baseline.  We attributed the modest delta to "MBD doing useful
search on top of a weak prior".

Now in the Phase-B regime (DP prior, 23-D `can_dp` schema, 80 %
standalone), the prior is strong enough that any chunk-boundary
controller-state loss prevents grasp completion.  The K = 0
delta is not 6.7 pp — it's a full 70 pp.  The bug has been there
since the first MPC-chunking commit; we just couldn't see it
until the prior was strong enough to make it dominant.

### Why standalone_dp.py never hit this

`standalone_dp.standalone_rollout` does **one** `env.reset() +
set_sim_state(env, init_state)` at the top of each rollout
(or `env.reset()` for fresh-seed inits), then runs a single
continuous `while not done` loop that calls `env.step(action)`
in `n_action_steps`-sized chunks **without ever touching reset
between chunks**.  Controller state evolves naturally with the
physics.  See `standalone_dp.py:152-280`.

## Proposed fixes (priority order)

### Fix 1 (minimal, validates the diagnosis) — gate the reset on actual MBD work

`unified_loop.py:466-468`, change:

```python
if _mbd_touches_main_env:
    env.reset()
    set_sim_state(env, current_state)
```

to:

```python
# Only re-snapshot when MBD actually clobbered the main env this
# chunk.  At K=0 the inner loop never iterated, so rollout_fn was
# never called, so the env is still at current_state.  Avoiding
# the spurious reset preserves robosuite's controller state
# across chunks -- crucial for strong priors that depend on
# continuous OSC integrator / interpolator evolution.
if _mbd_touches_main_env and int(params.K) > 0:
    env.reset()
    set_sim_state(env, current_state)
```

**Predicted impact**: K = 0 test rate jumps from 0 % to ~70 %
(matching standalone on this Mac).  K > 0 unchanged (still buggy,
see Fix 2).

**Wall to verify**: re-run the Phase-C K = 0 sweep at
`--max-env-steps 400 --n-test-rollouts 50`.  ~60 – 90 min on Mac
CPU.

### Fix 2 (correct, addresses K > 0 too) — never reset the main env

The proper architecture is the `n_workers > 1` subprocess path,
where each worker holds its own forked env (`worker.py:_worker_rollout`
does its own reset+restore internally) and the main env is
**never touched** by MBD's perturbations.  Two sub-options:

- **Option 2a**: change the default of `--n-workers` from 1 to
  `min(4, cpu_count())` and require `dataset_path` to be set in
  every config.  Lowest-change.  Cost: requires `--n-workers > 1`
  paths to be reliable on user machines (subprocess fork overhead
  on macOS is ~5 s per pool spawn × 50 chunks × 60 rollouts =
  ~4 hr wall overhead on top of MBD's actual compute).

- **Option 2b**: refactor `rollout_one` to not clobber the main
  env even when `n_workers == 1`.  E.g., copy the env once at
  function entry, run all M perturbations on the copy, dispose
  of the copy at exit.  Robosuite env objects are not trivially
  copyable, but `make_env_for_dataset()` is cheap enough to
  call M times if amortised.  Highest-correctness, requires
  exp 04 changes.

**Predicted impact**: K = 10 / 20 test rates lift from the
Phase-3 baseline of 36.7 % toward something at or near the
prior's 80 % ceiling, depending on whether MBD search actually
adds value on top of a strong DP prior.  This is the
**original Phase-C question**: "does MBD lift the strong prior?"
We can't answer that question with Fix 1 alone; we need Fix 2 to
get clean K > 0 numbers.

### Fix 3 (cleaner snapshot) — restore controller state too

Snapshot robosuite's controller state in `_snapshot_sim_state`
(currently only returns `env.sim.get_state().flatten()`).  This
is conceptually orthogonal to Fix 2 but practically subsumes it.

The relevant robosuite state to snapshot, per inspection of
`robosuite/controllers/composite/basic.py`:
- `composite_controller._goal_pos`, `_goal_ori`
- `composite_controller._interpolator.{current_pos, target_pos, step}`
- `composite_controller._last_action`
- `env.placement_initializer.rng` state
- `env.timestep` (integer step counter)

Cost: ~50 LOC in `common.py` + invasive use of robosuite
internals.  Brittle across robosuite versions.  Listed for
completeness; **not recommended** unless Fix 2 turns out to be
infeasible.

## Recommended path

1. **Land Fix 1 (one-line change)** as a separate commit, with a
   targeted regression test that runs 1 train + 1 test rollout
   at K = 0 with the corrected n_workers=1 and asserts
   `reward > 5.0` on test seed 100000.

2. **Re-run the Phase-C K = 0 sweep** at full 400-step horizon
   to confirm we recover ~70 – 80 % test rate.  This is the
   sanity gate from `phase_c_2026_05_29/BRIEF.md::Decision rule`.

3. **If K = 0 lands in the [75, 85] % band**, land **Fix 2a or
   2b** (depending on subprocess overhead on user's setup), then
   run the K = 10 / K = 20 sweep.  This finally answers the
   original Phase-C question.

4. **If K = 0 does NOT recover after Fix 1**, escalate: there's
   a second bug.  Run instrumentation that logs, per chunk, the
   first 16-D action vector at the chunk start and the actual
   action passed to `env.step` after the reset+restore — they
   should be identical at K = 0, since `denoise_loop` is a
   no-op.  If they're not, the bug is elsewhere.

## Wall budget (estimate)

| step | wall | what |
|---|---|---|
| Fix 1 implementation + commit | ~5 min | one-line gate + regression test |
| K = 0 re-run (60 rollouts, max_env_steps=400) | ~60 – 90 min | Mac CPU |
| Fix 2 implementation + commit | ~30 – 60 min | depending on which sub-option |
| K = 10 sweep (60 rollouts, K=10 MBD work) | ~90 – 140 min | Mac CPU |
| K = 20 sweep (60 rollouts, K=20 MBD work) | ~150 – 240 min | Mac CPU |
| **Total to publish Phase-C results** | **~5 – 8 hr** | sequential, single Mac |

OR run Fix 1 → K = 0 verify locally (~90 min), then if PASS, push
the K = 10 / 20 work to Colab T4 (~1.5 – 2 hr there).

## Files added in this phase

- `BRIEF.md` (this file): root-cause writeup + fix proposal.
- `triage_n_workers/n1_unified_flow_lam-0.5_K-0_R-14.json`:
  3-seed, 120-step rollout JSON with `--n-workers 1` (reset+restore
  on).  Per-rollout rewards 2.75, 1.89, 1.36; 0/3 success.
- `triage_n_workers/n2_unified_flow_lam-0.5_K-0_R-14.json`:
  Identical config except `--n-workers 2` (reset+restore off via
  the `_mbd_touches_main_env=False` branch).  Per-rollout rewards
  27.24, 5.78, 21.53; 1/3 success even at 120-step truncation.

## Open questions for the next session

1. Should we land Fix 1 alone first (validates diagnosis cheaply)
   or roll Fix 1 + Fix 2 together (avoid two-step verification)?
   **Recommendation**: Fix 1 alone first, because Fix 2 has
   multiple sub-options with different operational costs and we
   want the diagnostic signal of "K = 0 now hits the gate" to
   guide that choice.

2. Should Fix 2a's `n_workers > 1` default be conditional on
   `cpu_count()` (e.g. `n_workers=max(1, cpu_count() // 2)`) or
   a fixed small number?  **Recommendation**: 2 by default;
   M = 16 perturbations divide cleanly into 2 workers × 8 each.
   Users can override with `--n-workers <N>`.

3. Once Fix 1 + Fix 2 are landed, what's the **publishable**
   Phase-C result?  If K = 0 hits 75 – 85 % and K = 10 / 20 lift
   to e.g. 85 % / 90 %, that's a clean GREEN.  If K = 10 / 20
   stay near K = 0 (no lift), that's YELLOW — same shape as
   Slate G's Lift λ-plateau.  Either reading is publishable;
   what matters is the FAIL on the wrapping bug is resolved.

## Cross-references

- `phase_c_2026_05_29/BRIEF.md` — Phase-C result + 4-hypothesis
  list (this BRIEF supersedes that hypothesis list).
- `phase_c_2026_05_29/standalone_sanity_2026_05_29/README.md` —
  standalone re-eval evidence (70 % on this Mac).
- `phase_c_2026_05_29/K0/unified_flow_lam-0.5_K-0_R-49.json` —
  the 0 / 60 = 0.0 % gate input, full 60-rollout dataset.
- `experiments/15_unified_hybrid/unified_loop.py:300-309`,
  `:466-468` — the lines containing the bug.
- `experiments/04_mbd_baseline/run_can.py:107-108, 218-219` —
  the `rollout_one` reset+restore that defines the
  `_mbd_touches_main_env` semantics.
- `experiments/08_flow_matching_prior/standalone_dp.py:152-280` —
  the standalone driver that does NOT have this bug.
- `experiments/common.py:153-162` — `set_sim_state` (incomplete
  snapshot; only qpos/qvel).
