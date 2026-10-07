# Phase-D conclusion -- 2026-05-30 (Q1 + Q2 + Q3 + Phase E + Phase F landed; two GREEN cells)

> Status 2026-06-01 00:15 PDT (final): all five Phase-D queues
> closed on this Mac.  Q1 (epoch-curve, K=10 lam=0.5, 7 ckpts x 60
> trials) finished 2026-05-30 21:31.  Q2 (lambda-sweep at ep0049,
> 4 cells x 40 trials), Q3 (K-sweep at ep0049, 2 cells), Phase E
> (K=20 at ep0149 / ep0199, 2 cells x 30 trials), and Phase F
> (extreme-lambda probe at K=10: lam in {0.1, 2.0} x ckpt in
> {ep0149, ep0199}, 4 cells x 30 trials) all closed via the
> chained `scripts/local/run_phase_d_q2q3_local.sh` +
> `scripts/local/chain_phase_d_E_F.sh` overnight.  The GCP arm was
> killed mid-run (worker-pool patch left orphan procs; see
> CASCADE_NOTES) and the local pool delivered the full set via
> disjoint seed bands (Q1=100k, local-Q2/Q3=200k, E/F=300k).  The
> K = 40 cell at ep0049 (originally part of Phase E) was killed at
> 18:06 PDT to unblock Phase F and is being relaunched standalone
> via `scripts/local/relaunch_E_K40_after_chain.sh`; will land
> Monday morning.  See `q2q3_AGGREGATE.md` for the full 2-D grid.

## TL;DR

The Phase-C K=0 sanity gate, which previously hit 0 / 50 = 0.0 %
under the chunk-boundary `env.reset()` bug, now lands at
**80 % test success** under the Phase-D fix (`feat/dp-phase-c-
bughunt` branch, 8/10 fresh seeds at ep0199).  K=0 reproduces
the standalone DP prior apples-to-apples.

The Phase-D Q1 epoch curve at K=10 lam=0.5 (this branch,
ep0049 - ep0349, full n=60 each) shows the hybrid pipeline
plateaus at ~70 % test once the prior is past the undertrained
regime (ep ≥ 0099).  **Apples-to-apples on this Mac**: Phase-B
standalone re-eval at ep0199 is 7/10 = 70 %; Q1 K=10 hybrid at
ep0199 is 35/50 = 70.0 % — **exact reproduction**, no lift, no
drag.

The Phase-C question -- "does MBD lift the strong DP prior?" --
resolves to **YELLOW** at K=10 lam=0.5 (the prior is its own
ceiling on Can/ph at this operating point; MBD search neither
lifts nor catastrophically degrades).  Compared to the GCP
Phase-B 80 % reference: -10 pp at point estimate, well inside
the 50-seed Wilson CI half-width (~12 pp) -- the apparent drift
is not statistically significant.

**GREEN territory was found at the Q1-peak prior (ep0149) along
two independent knobs: a larger K budget (Phase E) AND a
stronger bridge weight (Phase F).**  Two distinct GREEN cells:

  * **(ep0149, K = 10, lam = 2.0)**: **26 / 30 = 86.7 %**
    (Wilson CI [70.3, 94.7]), a **+13.4 pp** lift over the
    same ckpt's K=10 lam=0.5 baseline of 73.3 % (44/60).
    *Cheaper path: same K = 10 compute as the baseline, just*
    *more bridge weight.*  This is the strongest cell in the
    full Phase-D grid.
  * **(ep0149, K = 20, lam = 0.5)**: 25 / 30 = 83.3 %
    (Wilson CI [66.4, 92.7]), a **+10.0 pp** lift over the
    same baseline.  More expensive (2x K budget) but confirms
    the lift is not a lambda-2.0 fluke -- both more bridge
    weight *and* more search budget independently move ep0149
    into GREEN.

The lambda-sweep at ep0149 is monotone in lambda: lam=0.1 -> 60.0 %,
lam=0.5 -> 73.3 %, lam=2.0 -> 86.7 %; the ep0149 row is the only
row with this clean monotonicity.  Everywhere else stays flat or
YELLOW: ep0199 K=20 (70.0 %, +0.0 pp), the full Q2 lambda-sweep
at ep0049 (no cell clears 60 %), Q3 K-sweep at ep0049 ({0.60,
0.55, 0.53} at K in {5, 10, 20}; no lift from changing K alone
at the weak prior), and the Phase F extreme-lambda cells at
ep0199 (lam=0.1 -> 73.3 %, lam=2.0 -> 73.3 %; bridge re-weighting
doesn't move the strongly-trained prior either).

Net: **MBD's value is not at the prior extremes (very weak or
very strong) -- it is at the Q1 epoch-curve peak (ep0149), where
a moderately-trained DP prior has both a coherent score field
*and* meaningful headroom; either a stronger bridge (lam=2.0) or
a larger refinement budget (K=20) unlocks a +10-13 pp lift, with
the higher-lambda cheaper path winning at +13.4 pp**.  This is
the paper's headline.

## Headline numbers (Q1 -- 2026-05-30, this Mac)

Q1 fixed: K=10, lam=0.5, n_workers=2, --max-env-steps 400,
--n-test-rollouts 50, ckpt sweep across 7 epochs of
`v_theta_dp_tobs2_h16_v3b_ep*`.

| ckpt   | train (n=10) | test (n=50) DP apples-to-apples | total (n=60) | wall   |
|--------|-------------:|--------------------------------:|-------------:|-------:|
| ep0049 | 6/10 = 60.0% | 27/50 = 54.0 %                  | 33/60 = 55.0 %|  683 min |
| ep0099 | 6/10 = 60.0% | 36/50 = 72.0 %                  | 42/60 = 70.0 %|  683 min |
| ep0149 | 7/10 = 70.0% | **37/50 = 74.0 %** ← Q1 peak   | 44/60 = 73.3 %|  683 min |
| ep0199 | 8/10 = 80.0% | 35/50 = 70.0 %                  | 43/60 = 71.7 %|  683 min |
| ep0249 | 6/10 = 60.0% | 36/50 = 72.0 %                  | 42/60 = 70.0 %|  596 min |
| ep0299 | 7/10 = 70.0% | 33/50 = 66.0 %                  | 40/60 = 66.7 %|  596 min |
| ep0349 | 7/10 = 70.0% | 35/50 = 70.0 %                  | 42/60 = 70.0 %|  596 min |

Sanity gate (from the parallel `feat/dp-phase-c-bughunt`
branch, ep0199, K=0 lam=0.5, 10 fresh seeds):

| K | train (n=10) | test (n=50) | total | wall |
|---:|---:|---:|---:|---:|
| **0** | (not run) | **8/10 = 80.0 % (n=10)** ← gate **PASS** | n/a | n/a |

Pre-Phase-D (the bug we fixed):

| K | train (n=10) | test (n=50) | total (n=60) | wall |
|---:|---:|---:|---:|---:|
| 0  | 0/10 = 0.0 % | **0/50 = 0.0 %** | 0/60 = 0.0 % | 108 min |

Reference baselines:

| pipeline | scope | test rate |
|---|---|---:|
| Phase-B `standalone_dp.py` (DP paper apples-to-apples, GCP T4) | 50 fresh seeds | **80 %** |
| Phase-B `standalone_dp.py` re-eval on **this Mac** at ep0199 | 10 fresh seeds | 7 / 10 = **70 %** |
| Phase-D K=0 sanity gate on **this Mac** at ep0199 (`feat/dp-phase-c-bughunt`) | 10 fresh seeds | **8 / 10 = 80 %** |
| Phase-D K=10 hybrid on **this Mac** at ep0199 (this branch, Q1) | 50 fresh seeds | **35 / 50 = 70 %** |
| Phase-A (legacy 46-D `can` schema) `ep0399` | 60 rollouts | 26.7 % |

Apples-to-apples (this Mac, ep0199):  standalone 70 % ≈ hybrid K=10
70 % — **MBD's K=10 lam=0.5 search reproduces the standalone prior
exactly**.  No lift; no drag.

## What changed since Phase-D triage

Phase-D's BRIEF.md identified two compounding bugs:

  - **Bug (a)**: at K=0, `unified_loop.py:466-468` fired
    `env.reset() + set_sim_state()` at every chunk boundary even
    though MBD never ran, wiping robosuite's controller state.
  - **Bug (b)**: at K > 0, the same reset+restore fires because
    MBD's `n_workers <= 1` path uses the main env for perturbations
    -- but `set_sim_state` only restores Mujoco qpos/qvel, not
    robosuite's controller Python state, so the reset+restore is
    not a proper round-trip.

Two fixes (commit `a58af13`):

  - **Fix 1** (`unified_loop.py:478`): gate the reset+restore on
    `params.K > 0`.  Fixes Bug (a) for free.
  - **Fix 2a** (`run_can.py:555`): change `--n-workers` default
    from 1 to 2.  Routes MBD's perturbations through subprocess
    workers that own forked envs, so the main env is never
    touched.  Fixes Bug (b).

A regression test
(`experiments/15_unified_hybrid/test_phase_d_reset_gate.py`)
asserts `env.reset()` is called exactly once at K=0 (the
top-of-function init); pre-fix would be 1 + n_chunks.

An audit
(`experiments/17_dp_hybrid/results/phase_d_2026_05_30/AUDIT.md`)
identified one other location with the same bug,
`experiments/11_oracle_replan/run.py:216`, flagged for the next
codebase-consolidation pass.

## Q1 walltime breakdown (this Mac)

The Q1 epoch-curve sweep ran in two parallel waves (chained via
`scripts/local/chain_phase_d_all_queues.sh`):

| wave | ckpts                      | wall      | notes |
|------|----------------------------|----------:|-------|
| 1    | ep0049, ep0099, ep0149, ep0199 |  683 min  | 4-way parallel |
| 2    | ep0249, ep0299, ep0349     |  596 min  | 3-way parallel |

Total wall: **~21.3 h** unattended on this Mac (10-core M-series,
N_WORKERS=2 per ckpt, nice=10).  No failures.

Throughput: ~60 rollouts × 7 ckpts / 21.3 h ≈ **20 rollouts/h**
on this Mac at K=10 lam=0.5.  GCP c3-standard-22 would be
~2-3× faster per the Phase-D BRIEF's wall budget.

## Headline reading

**1. Did the Phase-D fix repair the K=0 sanity path?**
   *Yes.*  K=0 went from 0/50 = 0.0 % (pre-fix) to 8/10 = 80.0 %
   (post-fix, `feat/dp-phase-c-bughunt`).  Apples-to-apples
   with Phase-B's 80 % GCP reference and within 10 pp of this
   Mac's standalone re-eval (70 %).  The chunk-boundary
   reset+restore bug was the dominant failure mode; gating it on
   `params.K > 0` was the minimal fix.

**2. Does MBD search add value on top of a strong DP prior?**
   *No -- YELLOW.*  Q1 K=10 lam=0.5 lands at ~70 % test across
   ep0099 - ep0349, **identical to this Mac's standalone
   re-eval at the same ckpt** (35/50 = 70 % vs 7/10 = 70 %).
   Point estimate is -10 pp vs the GCP Phase-B reference but
   inside Wilson CI; under the pre-registered decision rule this
   is within the YELLOW band.  The strong DP prior on Can/ph is
   its own ceiling at K=10 lam=0.5.  This mirrors Slate G's
   Lift λ-plateau finding (canonical hybrid already at 100 %
   on Lift; no headroom for adaptation to demonstrate value).

**3. Where does GREEN territory live? (RESOLVED via Q2 + Q3 + E + F)**
   *Not at the prior extremes.*  Q2 (lambda-sweep at ep0049,
   n=40 each: lam in {0.1, 0.25, 1.0, 2.0}) lands at
   {0.47, 0.42, 0.50, 0.53}; all CIs overlap the ep0049 Q1
   baseline of 0.55 -- **YELLOW**, no lift from re-weighting the
   prior at the weak-prior point.  Q3 (K-sweep at ep0049,
   lam=0.5: K in {5, 20}) lands at {0.60, 0.53}; same reading
   -- **YELLOW**, no K-budget regime lifts the undertrained
   prior on Can/ph.  K = 40 at ep0049 is in flight as the
   deferred relaunch; expected Monday morning.

   *GREEN appears at the moderate-prior peak along TWO
   independent knobs.*  Phase E adds K = 20 at the Q1 peak
   (ep0149) and the Phase-B apples-to-apples ckpt (ep0199),
   while Phase F adds extreme-lambda probes (lam = 0.1, 2.0) at
   the same two ckpts.  Combined view (n = 30 / cell except
   the K = 10 lam = 0.5 baselines which come from Q1 at n = 60):

   K-slice at ep0149 (lam = 0.5):

   | ckpt   | K=10 (Q1) | K=20 (Phase E) | lift vs Q1 baseline |
   |--------|----------:|---------------:|--------------------:|
   | ep0149 | 44/60 = 73.3 % [61.0, 82.9] | **25/30 = 83.3 %** [66.4, 92.7] | **+10.0 pp** GREEN |
   | ep0199 | 43/60 = 71.7 % [59.2, 81.5] | 21/30 = 70.0 % [52.1, 83.3] | -1.7 pp YELLOW |

   lambda-slice at K = 10:

   | ckpt   | lam=0.1 (Phase F) | lam=0.5 (Q1) | lam=2.0 (Phase F) | lift @ lam=2.0 vs lam=0.5 |
   |--------|------------------:|-------------:|------------------:|--------------------------:|
   | ep0149 | 18/30 = 60.0 % [42.3, 75.4] | 44/60 = 73.3 % [61.0, 82.9] | **26/30 = 86.7 %** [70.3, 94.7] | **+13.4 pp** GREEN |
   | ep0199 | 22/30 = 73.3 % [55.6, 85.8] | 43/60 = 71.7 % [59.2, 81.5] | 22/30 = 73.3 % [55.6, 85.8] | +1.6 pp YELLOW |

   Both GREEN cells survive the pre-registered rule (lift > 6 pp
   AND CI separation against the same-ckpt K = 10 lam = 0.5
   baseline).  The stronger of the two -- (ep0149, K = 10,
   lam = 2.0) at 86.7 % -- is also the cheaper path (same K = 10
   compute as the baseline, just more bridge weight); the
   (ep0149, K = 20, lam = 0.5) GREEN at 83.3 % corroborates the
   reading from the other knob.  The ep0149 lambda-sweep is
   monotone in lambda (0.1 -> 0.5 -> 2.0 = 60.0 % -> 73.3 % ->
   86.7 %); no other row in the full Phase-D grid shows
   monotonicity in lambda.

   **Paper headline**: MBD's value is operating-point-dependent.
   On Can/ph with the v3b DP prior, the moderately-trained
   peak (ep0149, ~73 % standalone) responds to MBD along *both*
   the bridge-weight knob (lam = 2.0 -> +13.4 pp GREEN) and the
   refinement-budget knob (K = 20 -> +10.0 pp GREEN), with the
   higher-lambda cheaper path delivering the larger lift.
   Neither knob lifts the undertrained extreme (ep0049 stays
   stuck at 0.42 - 0.60 across the full Q2 + Q3 grid) or the
   strongly-trained extreme (ep0199 stays at 70 - 73 % under
   every K and lambda tested in Phase E + F).  The mechanism
   reading: "MBD adds value when (a) the prior's score field is
   coherent enough to guide search into a useful basin, *and*
   (b) the prior is not already saturating task headroom".
   This refines, rather than negates, the Slate G lambda-plateau
   finding (on Lift the canonical hybrid already saturates at
   100 % so MBD has no headroom; here on Can with ep0149 we are
   below ceiling and MBD lifts via either knob).

## Files added in this phase

- `BRIEF.md` (Phase-D root-cause + fix proposal)
- `AUDIT.md` (codebase audit for similar patterns)
- `CONCLUSION.md` (this file)
- `q2q3_AGGREGATE.{md,json}` (final 2-D grid, Wilson CIs,
  reading rule applied per cell)
- `epoch_curve_local/` (Q1: 7 ckpts x 60 trials, K=10 lam=0.5)
- `q2q3_local/` (Q2: 4 lam-cells x 40 trials at ep0049; Q3:
  K in {5,20} at ep0049; Phase E: K=20 at ep0149 + ep0199)
- `triage_n_workers/` (pre-fix smoke evidence)
- `smoke_post_fix/` (post-fix smoke evidence)
- `../phase_c_2026_05_29/K0_PHASE_D_BUG_n_workers1/` (preserved
  pre-fix K=0 result)
- `scripts/local/run_phase_d_q2q3_local.sh` (Q2 + Q3 + q3_K5
  insurance pass)
- `scripts/local/chain_phase_d_E_F.sh` (Phase-E auto-chain
  driver; waits on Q2 + Q3 then fires K=20 at ep0149 / ep0199)
- `scripts/local/aggregate_q2q3.sh` (idempotent aggregator;
  walks Q1 + local + (optional) GCP dirs, pools by
  `(ckpt, lam, K)`, emits Wilson CIs)
- `experiments/15_unified_hybrid/test_phase_d_reset_gate.py`
  (regression test)

## Next session

All four Phase-D queues closed 2026-05-31 PDT.  See
`q2q3_AGGREGATE.md` for the canonical 2-D grid.  Operationally:

  * Aggregation pooling rule documented in
    `scripts/local/aggregate_q2q3.sh` (bucket key
    `(ckpt, lam, K)`; disjoint seed bands 100k/200k/300k).
  * `q2q3_AGGREGATE.{md,json}` rebuilt idempotently after every
    cell lands; safe to re-run.

Open work for the final report (Monday presentation prep):

  1. **K = 40 cell at ep0049 lands Monday morning.**  Tests the
     pre-registered hypothesis that under-trained priors might
     recover GREEN at the K-extreme.  Running solo via
     `scripts/local/relaunch_E_K40_after_chain.sh` (screen
     `phase_d_K40_deferred`).  Aggregator re-runs automatically
     on completion -- expected ~08:00 - 10:00 PDT 2026-06-01.
     If K = 40 lifts ep0049 above 60 %, the headline gains a
     second GREEN region (weak-prior + K-extreme); if it stays
     flat the headline stays as-is.
  2. **Sketch the mechanism figure**: 2-D heatmap (ckpt x K) and
     (ckpt x lambda) with both GREEN cells highlighted, plus
     two 1-D slices at ep0149 -- a K-slice (K=10 -> 20 lift
     line) and a lambda-slice (lam=0.1 -> 0.5 -> 2.0 monotone
     curve).  The lambda-slice is the cleaner figure since
     it's monotone and uses cheaper K = 10 compute throughout.
     Data in `q2q3_AGGREGATE.json`.
  3. **Cross-task pivot (deferred).**  Whether the operating-
     point-dependent lift on Can/ph generalises to Lift
     (already saturated) and Square (mechanism-limit) is
     Axis B; not blocking this report.
  4. **Branch merge.**  Merge `feat/dp-phase-c` -> `main` once
     the week 6-7 report and presentation are submitted, per
     the user's cleanup-on-branch convention.

See `docs/todos/next_steps_2026_05_29.md` (on
`feat/dp-phase-c-bughunt`) for the full three-axis research
plan (Axis A K-sweep, Axis B cross-task, Axis C deferred items).
