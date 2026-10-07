# Unified-driver non-determinism audit (2026-05-26)

While investigating the standalone-vs-unified 3 pp gap on Can/ph
(option 3 of the post-D8 follow-up), the diagnostic uncovered **two
real defects** plus **one residual noise floor**:

## 1. `success_ever` measurement bug (real, fixed)

**Symptom**: unified driver reports `success=False` for rollouts that
robosuite briefly flagged as successful mid-trajectory then "unset"
when the robot continued moving.

**Code**: `experiments/15_unified_hybrid/unified_loop.py:462` only
checked `check_success(env)` at the *final* state.  In contrast,
`standalone_dp.py:171` checks per-step **and early-terminates** on
success.

**Fix**: track `success_ever = success_ever or check_success(env)`
inside the env-step loop; report `success_ever or check_success_now()`.
Patch applied.

**Expected impact**: lifts the unified rate by 1-3 pp on tasks where
the prior briefly succeeds then drifts (DP T_obs=2 is the worst
offender; v6 CFM T_obs=1 was unaffected, which is why the 75.9 %
headline never tripped this).

## 2. `PYTHONHASHSEED` randomization (real, mitigated)

**Symptom**: same code, same args, same checkpoint, same `--seed 0`
gave different demo_0#0 rewards across processes:
`+127.40 (success)` in D7full, `+1.83 / +3.07 / +4.09 (failure)` in
re-runs.

**Code**: `run_can.py:840` uses
`hash(("flow_warmstart", b, _k, _r, chunk_idx))` to derive the
flow prior's chunk seed.  Python's `hash()` on tuples containing
**strings** is salted per-process (PEP 456, since CPython 3.3)
unless `PYTHONHASHSEED` is set.

```
python -c "print(hash(('flow_warmstart', 0, 'demo_0', 0, 0)) & 0xFFFFFFFF)"
4120867754                  # process 1
3415850211                  # process 2
3701148778                  # process 3
2920046184 (always)         # with PYTHONHASHSEED=0
```

Both `warmstart_seed_fn` (line 840) and `rng_lam`'s seed (line 822-825)
are affected.  So *every* prior cell in exps 15-17 that uses the flow
prior has chunk-seeds that vary across Python processes — meaning
each 30-trial result is one realization of a hash-seed lottery.

**Mitigation**: launch with `PYTHONHASHSEED=0` (env var, no code
change required).  All comparison runs from D9 onward use this.

**Permanent fix candidate** (deferred): replace the call sites with a
deterministic stable hash (`hashlib.blake2b` over a canonical bytes
encoding, or simple integer arithmetic over `(b, demo_int, r_idx,
chunk_idx)`).  Would invalidate prior comparison cells but they were
already cross-process non-reproducible.

## 3. Robosuite / MuJoCo numerical noise (residual)

**Symptom**: with `PYTHONHASHSEED=0` and `OMP_NUM_THREADS=1`, the
prior emits a bit-identical action chunk for chunk 0 — yet the
chunk-0 step-0 reward still varies by ±0.04 between processes
(`+3.08` vs `+3.12` on otherwise-identical replays).

**Probable causes**:

- MuJoCo's internal solver state is partially process-dependent
  (warm-start residuals from collision detection).
- Robosuite's `_get_observations(force_update=True)` re-evaluates
  derived quantities that don't round-trip exactly through state
  save/restore.
- Compounding over 240 env steps amplifies sub-step noise enough to
  flip ~10 % of trials between "barely-succeed" and "barely-fail".

**Status**: not fixable without forking MuJoCo.  Quantified noise
floor: ~3 pp on a 30-trial cell.  Treat individual cell results as
samples with implicit ±3 pp realization variance, separate from the
Wilson statistical CI.

## Implications for the D8 probe and forward plan

The D8 K=10 vs D7 K=0 comparison was **across two Python processes**
with different `PYTHONHASHSEED`s.  The "3 recovered, 3 broken, 24
unchanged" pattern conflates:

- MBD's actual effect on the prior chunk
- The hash-seed lottery (different baseline samples)
- MuJoCo numerical noise

The McNemar p=1.000 outcome is **still informative** as "MBD's effect,
if any, is smaller than the combined noise floor at this operating
point", but it can no longer be cleanly attributed to "MBD doesn't
help" alone.

**Re-run plan** (in progress as of 2026-05-26 evening):

| Cell | Status | Args |
|---|---|---|
| `D9 K=0` | running | `PYTHONHASHSEED=0`, success_ever fix, K=0 |
| `D9 K=10` | queued after K=0 | same env vars, K=10 |

Re-running both under the same Python hashseed gives the apples-to-
apples comparison the §2.3 decision rule actually needs.

## Files

- `experiments/15_unified_hybrid/unified_loop.py:217-233, 437-441, 480-486`
  (success_ever patch)
- `experiments/17_dp_hybrid/results/abl_D9_K0_hashseed0/` (in flight)
- `BRIEF_D8_probe_null_2026_05_26.md` — original D8 null analysis;
  numbers there are still valid as point estimates but the paired
  comparison interpretation should be revised once D9 lands.
