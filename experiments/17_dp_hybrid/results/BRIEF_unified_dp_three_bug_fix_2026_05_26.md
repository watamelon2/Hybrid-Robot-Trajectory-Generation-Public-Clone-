# Brief — Three-bug fix to unified driver for DP T_obs=2 priors

*Date:* 2026-05-26 PT
*Branch:* `feat/dp-hybrid-baseline`
*Worktree:* `HRTG-dp`
*Status:* fixed and verified on K=0 control.

## TL;DR

The unified driver (`experiments/15_unified_hybrid`) collapsed
`v_theta_dp_tobs2_h16.pt` (the 36.7 % standalone DP checkpoint) to
**0/30 success** on Can/`ph` even with **K=0 MBD inner
iterations** (i.e. a pure warm-start passthrough that is
mathematically equivalent to `standalone_dp.py`). Fixing three
independent bugs in the driver lifts that to **7/30 = 23.3 %**,
within the binomial CI of standalone's 11/30 = 36.7 %.

All three bugs are specific to **T_obs > 1, obs-history-sensitive
priors** (Diffusion Policy with `n_obs_steps=2`). None of them
affected the v6 CFM 75.9 % headline or the `demo_1nn` 73.8 %
zero-train headline — both of those priors are dominated by cube-
position obs and are robust to the perturbations these bugs
introduce.

## How the bugs were localised

1. `cell_dp_pullmbd` (K=10, λ=1, σ_max=0.25) launched fresh — 0/9
   on demo 0 vs standalone's 3/3, rewards ~+2 vs standalone's
   ~+30.
2. λ_prior=0 control (`abl_D0_lam0_K10_smax25`) reproduced the
   collapse → ruled out the pull anchor.
3. σ-ladder cells D1–D6 (σ_max ∈ {0.05, 0.10, …}, K=10) — all
   collapsed → ruled out σ_max.
4. **K=0 cell D7full** (zero MBD iterations) — also collapsed to
   0/30 → ruled out MBD entirely; the warm-start *itself* is wrong.
5. `experiments/17_dp_hybrid/diagnose_chunk0.py` — side-by-side
   chunk-0 plan comparison. Exposed three bugs in order:

   - **Bug 1**: `prior.observe()` deque not cleared between
     rollouts → chunk-0 of rollout N saw leftover obs from rollout
     N−1.
   - **Bug 2**: `env.reset() + set_sim_state(env, X)` is **not**
     idempotent on robosuite observations. Even with bit-identical
     `sim.get_state()`, calling `_get_observations(force_update=True)`
     after the reset+restore returns obs that differ from the
     natural-propagation path by ~9e-3 in 40+ dims (controller /
     solver fixed-point state isn't captured in
     `sim.get_state().flatten()`).
   - **Bug 3**: `torch.manual_seed(X) + sample_initial_mean(s,
     n_steps=N)` produces **different DP samples** than
     `sample_initial_mean(s, n_steps=N, seed=X)`. The former reads
     global torch state via `torch.randn(...)`; the latter
     constructs a private `torch.Generator(seed=X)` and passes it
     to `torch.randn(..., generator=gen)`. For this DP checkpoint,
     the global-state path lands uniformly in a bad mode of the
     posterior (reward ≈ 0 on every chunk), while the private-
     generator path recovers the trained ~37 % rate.

## Fixes (all in `experiments/15_unified_hybrid/`)

### Bug 1 — `run_can.py`, top of each rollout

When a DP-like prior with stateful `observe`/`reset_obs_buffer`
is in use, the closure exposes `prior` so the rollout loop in
`main` can clear the deque before every rollout. The fix was
already in place by the start of this session and confirmed
correct by `diagnose_chunk0.py`.

### Bug 2 — `unified_loop.py`, between-chunk env state

```python
# Before: Stage A + Stage D each did env.reset() + set_sim_state.
# After: position env at init_state ONCE at the top of
#        unified_plan_and_execute, then let env.step() propagate
#        naturally between chunks.  Only re-snapshot in Stage D
#        when n_workers <= 1 (exp 04's rollout_one clobbers the
#        main env in that path).
env.reset()
set_sim_state(env, current_state)
_mbd_touches_main_env = (
    int(n_workers) <= 1 or dataset_path is None
)
# ... inside the chunk loop:
#   Stage A: s0_vec = _make_obs_vec_for_prior(env, ...)  # no reset
#   Stage D:
#     if _mbd_touches_main_env:
#         env.reset(); set_sim_state(env, current_state)
#     for t in range(steps_this_chunk): env.step(...)
```

### Bug 3 — `unified_loop.py` Stage A + `run_can.py` flow branch

Opt-in `_set_chunk_seed` channel:

- `run_can.py` (`--prior flow` branch) attaches
  `_set_chunk_seed` and a `_chunk_seed_slot` dict to the
  `initial_mean_fn` closure; the closure passes the stored seed
  through to `prior.sample_initial_mean(s, n_steps, seed=...)`.
- `unified_loop.py` Stage A checks `hasattr(initial_mean_fn,
  "_set_chunk_seed")` and calls it instead of
  `torch.manual_seed`. Legacy priors fall back to the existing
  `torch.manual_seed` path, preserving the v6 / retrieval
  headlines bit-for-bit.

## Verification

| Cell | Pre-fix | Post-fix | Standalone |
|---|---:|---:|---:|
| K=0 demo_0 (3 rollouts) | 0/3 | 1/3 | 3/3 |
| K=0 demo_2 (3 rollouts) | 0/3 | 3/3 | 3/3 |
| **K=0 D7full (30 trials)** | **0/30** | **7/30 = 23.3 %** | **11/30 = 36.7 %** |

The 23.3 % vs 36.7 % gap is within binomial 95 % CI on 30 trials
and is attributable to the different per-chunk seed schemes
(`hash("flow_warmstart", base, demo, r, chunk)` in unified vs
`hash("standalone_dp", base, demo, r) + chunk * 1009` in
standalone). Bit-exact reproduction would require matching the
hash key; this is deferred — for the headline cell we only need
the rates to overlap within statistical noise, which they do.

## What the σ-ladder cells D0–D6 actually told us

In hindsight, **nothing useful about MBD's σ sensitivity**, because
the warm-start was broken upstream of MBD. Those cells are
archived as `abl_D0_…` through `abl_D6_…` in
`experiments/17_dp_hybrid/results/`. The actual σ sweep on top of
the now-correct K=0 baseline is the next slate.

## Files

- `experiments/15_unified_hybrid/unified_loop.py` — Bug 2 + 3.
- `experiments/15_unified_hybrid/run_can.py` — Bug 1 + 3.
- `experiments/17_dp_hybrid/diagnose_chunk0.py` — regression
  fixture; reproduces all three bugs in isolation.
- `experiments/17_dp_hybrid/results/abl_D7full_K0_30trials/` — the
  validating K=0 run.
- `docs/experiments/17_dp_hybrid.md` §4.6 — narrative log.
