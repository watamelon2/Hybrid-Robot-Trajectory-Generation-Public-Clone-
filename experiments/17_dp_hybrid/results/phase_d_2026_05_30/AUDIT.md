# Phase-D audit: similar `env.reset() + set_sim_state` patterns in the codebase

The Phase-D bug isn't a one-off; it's a footgun in
`common.set_sim_state`'s API: it restores only Mujoco's
`qpos / qvel` and forwards kinematics, **not** robosuite's
Python-side controller state (OSC goal cache, interpolator,
last-commanded pose, gripper buffer, placement sampler RNG,
etc.).  Any code path that does

```python
env.reset()
set_sim_state(env, snapshot)
```

**between** chunks of an active rollout (i.e., where the env was
previously evolving via `env.step()` and we want to "restore" it
to a saved sim state while preserving the *as-if* controller
state) is broken.

Patterns that do this at the **start** of a rollout are OK,
because at that point the controller has no relevant
accumulated state to preserve.

## Audit results

Grepped all `env.reset()` + `set_sim_state()` call pairs across
`experiments/` and classified each by whether it's at a
rollout boundary (safe) or mid-rollout (buggy).

### Safe (rollout-boundary)

These are the canonical "start a fresh rollout from a saved
state" pattern.  No bug.

| file | line | context |
|---|---|---|
| `experiments/03_bc_baseline/eval.py` | 167-168 | per-rollout BC eval init |
| `experiments/04_mbd_baseline/run_can.py` | 107-108 | `rollout_one` per perturbed rollout |
| `experiments/04_mbd_baseline/run_can.py` | 218-219 | `rollout_one_with_obs` per rollout |
| `experiments/04_mbd_baseline/run_can.py` | 378-379 | live-view per-iteration playback |
| `experiments/04_mbd_baseline/run_can.py` | 456-457 | final replay rollout |
| `experiments/04_mbd_baseline/worker.py` | 45-46 | subprocess worker per rollout |
| `experiments/04_mbd_baseline/worker.py` | 89-90 | subprocess worker with obs collection |
| `experiments/04_mbd_baseline/test_cost_rank_parity.py` | 152-153 | per-plan test fixture |
| `experiments/08_flow_matching_prior/standalone_dp.py` | 234-235 | per-rollout init (NOT between chunks) |
| `experiments/16_value_baseline/train.py` | 93-94 | per-BC-rollout init |
| `experiments/common.py` | 196-197 | `run_episode` per-episode init |
| `experiments/17_dp_hybrid/diagnose_chunk0.py` | 73-74 + many | a *diagnostic* that DELIBERATELY recreates both patterns to demonstrate the bug |

### Buggy (mid-rollout / chunk-boundary)

**1 location besides the one we just fixed:**

| file | line | severity |
|---|---|---|
| `experiments/11_oracle_replan/run.py` | 199, 216 | **same bug as Phase-D** |
| ~~`experiments/15_unified_hybrid/unified_loop.py`~~ | ~~466~~ | **FIXED in Phase-D commit `a58af13`** |

#### Detail: `experiments/11_oracle_replan/run.py`

Snippet (annotated):

```python
# Stage 1: plan + execute chunk 0 from init_state.
mu0, _ = denoise_loop(a_bc, ...)
env.reset(); set_sim_state(env, init_state)   # OK: rollout start
for t in range(args.chunk_steps):
    env.step(mu0[t])                          # controller state evolves
snapshot = np.asarray(env.sim.get_state().flatten())

# Stage 2-3: plan chunk 1 from snapshot.
mu1, log1 = denoise_loop(a_bc1, ...)

# Stage 4: execute chunk 1.
env.reset(); set_sim_state(env, snapshot)     # BUG: same Phase-D corruption
for t in range(steps_remaining):
    env.step(mu1[t])                          # controller starts from "goal == current"
```

exp 11's "oracle replan" experiment compares plan-once vs
replan-after-chunk-0.  The replan branch suffers the same
controller-state-loss as Phase-C: at line 216, after the second
`env.reset()`, robosuite's OSC interpolator and goal_pose cache
snap to the snapshot's eef pose, losing whatever the chunk-0
final command was driving toward.

**Impact**: exp 11's published numbers (per `docs/results.md`)
for the replan-on branch are biased downward by however much
controller-state loss costs.  Same mechanism as Phase-D, just
applied once per rollout instead of 50 times.  The bias is
probably much smaller (one corruption event vs 50) but
non-zero.

**Recommendation**: defer.  exp 11 is not on Phase-D's critical
path.  Flagging here for the next codebase-consolidation pass.
A 2-line fix mirroring Phase-D Fix 1 (gate on K > 0) would not
apply directly because exp 11 always replans (K > 0); the
correct fix is Phase-D Fix 2 / 3 (snapshot controller state, or
don't reset the env between chunks).

If the next session re-runs exp 11 with a corrected pipeline,
the replan-on branch may show a meaningful uptick — possibly
overturning whatever the original exp 11 conclusion was.

## Underlying root cause: `set_sim_state` API gap

`experiments/common.py:153-162`:

```python
def set_sim_state(env: Any, state_vec: np.ndarray) -> None:
    """Restore the underlying MuJoCo state of a robosuite env."""
    sim = env.sim
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(state_vec)
    else:
        from robosuite.utils.binding_utils import MjSimState
        sim.set_state(MjSimState.from_flattened(state_vec, sim))
    sim.forward()
```

The docstring "Restore the underlying MuJoCo state" is
**technically** accurate but misleading in practice — most
callers think of this as "restore the env to where it was at
snapshot time", which requires robosuite's Python state too.

The state that's lost on `env.reset() + set_sim_state(snapshot)`:

| state | lives in | restored by `set_sim_state`? |
|---|---|---|
| qpos (body positions) | Mujoco | yes |
| qvel (body velocities) | Mujoco | yes |
| object positions | Mujoco (via body qpos) | yes |
| Mujoco contact forces (next step computes) | derived | yes |
| **OSC `goal_pos`, `goal_ori`** | robosuite (Python) | **NO** |
| **OSC interpolator (current, target, step)** | robosuite (Python) | **NO** |
| **Last commanded action** | robosuite (Python) | **NO** |
| **Gripper action history** | robosuite (Python) | **NO** |
| **Placement sampler RNG state** | robosuite (Python) | **NO** |
| **Internal step counter** | robosuite (Python) | **NO** |

The OSC controller state is the biggest one: a fresh `env.reset()`
re-initializes `goal_pos` to `current_eef_pose`, so the controller
believes "I'm at my target, hold position".  When chunk N+1's
delta-pose actions arrive, they're computed against the wrong
baseline (the reset's "I'm-at-target" pose), so the next 8 steps
diverge from what chunk N+1's plan intended.

## Future work (not blocking Phase-D)

### Option A: rename / re-doc `set_sim_state`

Rename to `set_mujoco_state` to make it obvious it's a partial
snapshot; add a docstring warning that it must only be called
at rollout boundaries.

### Option B: add `snapshot_full_state` + `restore_full_state`

A higher-level API that includes controller state.  Robosuite
exposes the controller via `env.robots[0].composite_controller`;
we'd snapshot the relevant fields and restore them.  Brittle
across robosuite versions but correct.

### Option C: avoid mid-rollout reset+restore entirely

Refactor exp 11 (and any future MPC-style code) to NEVER call
`env.reset()` mid-rollout.  The Phase-D Fix 2a path (subprocess
workers own forked envs, main env is never touched by MBD's M
perturbations) is the right architecture: main env just runs
`env.step()` continuously, MBD's planning happens in side
processes that don't share env with main.

**Recommendation**: do Option A in the next consolidation pass
(cheap), defer Option B / C until they're justified by another
experiment hitting the same trap.

## References

- `BRIEF.md` — Phase-D root cause + fix proposals.
- `smoke_post_fix/README.md` — Phase-D smoke verification.
- Phase-D commit `a58af13` — the Fix 1 + Fix 2a landing.
