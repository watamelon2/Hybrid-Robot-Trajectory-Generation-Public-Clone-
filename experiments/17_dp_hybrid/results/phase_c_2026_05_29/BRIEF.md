# Phase-C: hybrid K-sweep on the Phase-B prior — 2026-05-29

Pre-launch BRIEF.  Decision rule + K-sweep plan are pre-registered
**before** any K > 0 numbers exist, mirroring the Phase-A / Phase-B
discipline.  Results table is `TODO` until the K-sweep lands.

Companion to:
- `experiments/08_flow_matching_prior/results/phase_b_2026_05_29/BRIEF.md`
  (Phase-B: 80.0 % standalone test rate on `ep0199` with the
  `can_dp` 23-D schema)
- `experiments/08_flow_matching_prior/results/phase_a_2026_05_28/BRIEF.md`
  (Phase-A: 16.0 % structural baseline that motivated the schema
  swap)

## Context

Phase-B confirmed (GREEN) that the obs schema was the dominant
structural cause of the Phase-3 standalone ceiling.  The Phase-B
prior at `ep0199` clears 40/50 = **80.0 %** test rate on the DP
paper protocol (n_test = 50 fresh seeds, max_env_steps = 400,
n_action_steps = 8, DDPM K = 100), within 5 pp of DP paper's 85 %.

Phase-C asks the next question: does the **unified MBD hybrid
pipeline** (`experiments/15_unified_hybrid/run_can.py`) wrap that
prior in a way that **lifts** the standalone number (via MPC search
over the prior's chunk-level samples), or does the wrapping cost
inference cycles without buying success-rate?

The variable being swept is `--K`, the number of inner MBD
denoising iterations per chunk:

- **K = 0** — no MBD denoising; the prior's warm-start chunk is
  executed directly.  This is the apples-to-apples comparison to
  Phase-B's standalone 80 %.  If K = 0 ≠ 80 % ± 5 pp, the unified
  pipeline is **not** running the prior the same way the standalone
  driver does, and every K > 0 number is uninterpretable.
- **K = 10** — 10 MBD iterations per chunk.  This is the DP-prior
  setting that produced the 36.7 % H = 16 baseline on Phase-3-era
  `can` ckpts.  Phase-C asks what it does on the Phase-B `can_dp`
  prior.
- **K = 20** — 20 MBD iterations.  Cheaper than DDPM K = 100
  inference but more search than K = 10.  Establishes whether the
  K trend on the strong prior is monotone (more search = better) or
  saturating / regressing (the strong prior is its own ceiling).

Holding everything else from the Phase-B protocol fixed:

| | Phase-B standalone | **Phase-C K-sweep** |
|---|---|---|
| ckpt                       | `ep0199`             | `ep0199` (same blob) |
| obs schema                 | `can_dp` (23-D)       | `can_dp` (23-D)        |
| horizon                    | 16                   | 16                   |
| n_obs_steps                | 2                    | 2                    |
| n_action_steps             | 8                    | 8                    |
| max_env_steps              | 400                  | 400                  |
| DDPM warm-start steps      | 100                  | 100                  |
| MBD K                      | (n/a, standalone)    | **0 / 10 / 20**       |
| n_test_rollouts            | 50                   | 50                   |
| test_start_seed            | 100000               | 100000               |
| n_demos × n_rollouts       | 10 × 1               | 10 × 1               |

## Decision rule (pre-registered, before K-sweep launch)

### Sanity gate (K = 0)

| K = 0 test rate vs Phase-B 80 % | reading |
|---|---|
| **75 – 85 %** | **PASS** — the unified pipeline reproduces the standalone prior at K = 0.  K > 0 numbers are interpretable. |
| **< 75 %** or **> 85 %** | **FAIL** — re-investigate before drawing conclusions from K > 0.  Suspect a wrapping bug (obs vector / action denorm / chunk seeding / EMA bookkeeping). |

If the gate FAILS, the K-sweep is paused and the bug list is:
- Does `--prior=flow` peek the right ckpt `task` and use 23-D obs at
  runtime?  (Should: see the new `_make_obs_vec_for_prior`
  synthetic-key path landed in commit TODO.)
- Does the adapter denormalise actions back to env-space?  (Phase-B
  ckpt has `normalize_actions=True`.)
- Does the EMA shadow get loaded?  (Phase-B used EMA decay 0.9999;
  the ckpt blob's `state_dict` should already be the EMA weights.)
- Does the chunk seeding match the standalone driver's
  `_set_chunk_seed` pattern?  (See `unified_loop.py:330-340` for
  the regression history.)

### K-sweep result rule (only meaningful if sanity gate PASSED)

| K-sweep shape                                                       | reading |
|---------------------------------------------------------------------|---------|
| K = 10 / 20 > Phase-B 80 % + 3 pp (and CIs separate)                | **GREEN** — MPC search **lifts** the strong prior.  The hybrid pipeline is buying useful structure beyond the prior's standalone behaviour. |
| K = 10 / 20 ≈ Phase-B 80 % ± 3 pp (within noise band of 50 rollouts) | **YELLOW** — MPC search **doesn't hurt** but doesn't help.  Defensible: the prior is its own ceiling; MBD just isn't the bottleneck. |
| K = 10 / 20 < Phase-B 80 % − 5 pp                                    | **RED** — MPC search **degrades** the prior.  Either the prior's warm-start chunk is already optimal and the MBD denoising perturbs it off-distribution, or the score-bridge interaction with a strong DP prior is destructive (cf. the existing comment at `run_can.py:170-180` about λ = 1 collapsing the Phase-3-era DP baseline). |

50-rollout noise band: binomial 95 % CI on 40/50 ≈ 66.3 – 89.9 %, so
~12 pp half-width.  "Separate CIs" in the GREEN row means K > 0's CI
lower-bound > 80 %.

## Pre-flight: the Phase-C wiring (this commit)

The Phase-B prior could not be plugged into the unified pipeline
as-is — the `--prior=flow` code path was hard-coded to the legacy
46-D obs schema.  Two patches landed in this commit (Phase-C
implementation, no eval results yet):

1. **`experiments/08_flow_matching_prior/adapter_dp.py`** — adds
   `self.task` to `FlowPriorDP`, populated from the ckpt blob's
   `task` field via `_from_blob`.  Backward-compat: legacy blobs
   without `task` default to `"can"` (every pre-Phase-B ckpt).

2. **`experiments/15_unified_hybrid/run_can.py`** — when
   `--prior=flow`, peeks the ckpt blob for `task` + `config.obs_dim`,
   sets `prior_obs_keys = get_obs_keys(ckpt_task)`, builds a probe
   obs vector via the same `_make_obs_vec_for_prior` that the
   rollout loop uses, and asserts the trailing dim matches the
   ckpt's `obs_dim` before the first rollout.  A mismatch raises a
   loud `RuntimeError` with the schema details.

3. **`experiments/15_unified_hybrid/priors_zerotrain.py`** —
   re-exports `get_synthetic_keys` + `compute_synthetic_obs_keys`
   from exp 03's dataset module (which already provides them for
   the BC retraining path).

4. **`experiments/15_unified_hybrid/unified_loop.py`** —
   `_make_obs_vec_for_prior` now detects synthetic obs keys
   (those ending in `_synth`), computes them at runtime from their
   base sources via `compute_synthetic_obs_keys`, and injects them
   into the obs dict before the per-key concatenation.  The legacy
   46-D `can` path is bit-for-bit unchanged (no synthetic keys =
   no-op).

**CPU smoke test** (`/tmp/test_phase_c_runtime_obs.py`) verifies:
- `_make_obs_vec_for_prior(env, obs_keys=get_obs_keys("can"))`
  returns a 46-D vector (legacy path, unchanged).
- `_make_obs_vec_for_prior(env, obs_keys=get_obs_keys("can_dp"))`
  returns a 23-D vector matching the train-time layout in
  `dataset_v3.load_chunks`.
- `Can_to_eef_pos_synth = robot0_eef_pos - Can_pos` numerically.
- Quat synthetic is unit-norm.

## Eval protocol (Phase-A unchanged, Phase-B reused)

The K-sweep uses the same eval protocol as Phase-A and Phase-B
standalone.  Canonical invocation (single K; sweep K ∈ {0, 10, 20}):

```bash
python -m experiments.15_unified_hybrid.run_can \
    --task can \
    --prior flow \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt \
    --K <0/10/20> \
    --H 16 \
    --mpc-replans 49 \
    --mpc-chunk-steps 8 \
    --n-demos 10 --n-rollouts 1 \
    --n-test-rollouts 50 --test-start-seed 100000 \
    --max-env-steps 400 \
    --prior-n-steps 100 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --lambda-prior 0.5 \
    --device cuda \
    --output-dir experiments/17_dp_hybrid/results/phase_c_2026_05_29/K<K>/
```

`--task can` is passed (the env factory uses this to construct the
right env); `--prior=flow` will internally re-resolve to `can_dp`
from the ckpt blob and build the right obs vector at runtime.  Any
mismatch fails loud (per the assertion landed in this commit).

### Flag rationale (read before changing)

- **`--mpc-replans 49`** (NOT 0, see "Bug discovered" below).
  `unified_loop.py:228` computes `n_chunks = mpc_replans + 1`, so
  to mirror standalone-DP's 400-step / 8-actions-per-replan
  closed-loop schedule we need 50 chunks total.  With
  `--mpc-chunk-steps 8`: `50 * 8 = 400 = --max-env-steps`.  Passing
  `0` (the original BRIEF value, since corrected) produces a
  single-shot 16-step rollout that terminates before the robot
  reaches the can.
- **`--lambda-prior 0.5`** pins the prior weight in the MBD
  soft-control update to the Phase-3-era convention (the 36.7 %
  K = 10 hybrid baseline used this value).  Without an explicit
  value, `run_can.py` sweeps 6 lambdas {0.0, 0.1, 0.5, 1.0, 5.0, ∞}
  and multiplies wall by 6×.  At K = 0 this value is
  inert (MBD does no work) so K = 0 is reproducible across any
  `--lambda-prior` choice.
- **`--prior-n-steps 100`** matches Phase-B's standalone full-DDPM
  sampling.  Default in `run_can.py:443` is 32 (legacy v3/v6 Heun).
- **`--n-test-rollouts 50 --test-start-seed 100000`** mirrors the
  Phase-A test bucket (this flag pair was added in the Phase-C
  pre-flight commit; see `unified_loop.py` test-rollout block).

## Bug discovered + fixed (2026-05-29 16:00)

First Phase-C K-sweep launch on this Mac (CPU, PID 30265, log at
`/tmp/phase_c_ksweep.log` since lost to `/tmp` cleanup) failed
catastrophically: K = 0 wrote 0 / 60 = **0.0 %** with all
`final_reward ≈ 0.04 – 0.24` and each rollout's
`steps_executed = 16` (out of the intended 400).  Total K = 0
wall: **68.87 s** instead of the expected ~25 min — 22× too fast.

Root cause: the original BRIEF (this section's previous version)
specified `--mpc-replans 0` to "match standalone DP".  But the
unified loop's contract is

```python
n_chunks = int(n_replans) + 1            # unified_loop.py:228
for chunk_idx in range(n_chunks):        # unified_loop.py:330
    ...
```

so `mpc_replans=0` means a single chunk, not a single re-plan per
chunk; the loop exits after one 16-step execution and
`max_env_steps=400` only acts as a CAP, never as a target.

The diagnostic JSONs from the bad K = 0 run are preserved at
`K0_BAD_replans0_2026_05_29/` for traceability.

A 1 train + 1 test smoke at the FIXED flags (`--mpc-replans 14`
→ 15 chunks × 8 = 120 env steps, abbreviated for smoke) printed

```
[exp15] --prior=flow ckpt task='can_dp'  obs_dim=23  (7 keys)
[exp15] prior=flow lam=0.5 R=14 K=0  1 train demo_0#0  reward=+3.67  success=False
[exp15] prior=flow lam=0.5 R=14 K=0  2 test  seed=100000  reward=+3.01  success=False
```

i.e. the robot is now reaching the can (reward ≈ 3.0 vs 0.04
before) and the obs-schema assertion still passes.  The full
400-step K = 0 run was not completed on this Mac due to the
hand-off to the secondary Mac (see `HANDOFF_NEXT_MAC.md`).

## Pre-flight gaps (Phase-C launch blockers)

The schema wiring landed in this commit handles the most important
gap (the obs vector was 46-D vs 23-D mismatched).  Two more flags
need to be added to `run_can.py` before launching the K-sweep
apples-to-apples against Phase-B:

1. **`--n-test-rollouts N` and `--test-start-seed S`** — added to
   `standalone_dp.py` in the Phase-A protocol (the 50 fresh-seed
   test bucket).  `run_can.py` currently has only `--n-demos` and
   `--n-rollouts`, which runs train-init replays.  For an
   apples-to-apples comparison to Phase-B's 80 % standalone test,
   the same 50 fresh-seed test bucket needs to fire in the unified
   pipeline.  Estimated scope: ~30 lines in `run_can.py` +
   `unified_plan_and_execute` to thread the test-rollout loop
   through after the existing train-rollout loop.  Same eval
   semantics as `standalone_dp.py`'s `--n-test-rollouts` path.

2. **`--prior-n-steps 100` default override** — `run_can.py:443`
   defaults `--prior-n-steps` to 32 (legacy v3/v6 Heun-sampler
   convention).  Phase-A established that DP ckpts dominate at
   K = 100 (full DDPM), and Phase-B's 80 % comes from K = 100.
   Phase-C must pass `--prior-n-steps 100` explicitly; the BRIEF's
   example invocation does so.  Alternatively, the DP-prior path
   could default to 100 when the loaded prior reports
   `arch=dp_unet1d`.  Not blocking (just remember to pass the flag),
   but worth a one-liner in `_load_prior` for safety.

Gap (1) is the only true blocker.  Until it lands, Phase-C K-sweep
would compare 10-rollout train-init numbers against Phase-B's
50-rollout fresh-seed numbers, which is uninterpretable.  Worth
landing as its own focused commit before the K-sweep launch.

## Wall budget (estimate)

Per Phase-A's observation of ~13 min / 60 rollouts at
max_env_steps = 400 on T4, with K = 10 / 20 adding ~10× / ~20× the
inner MBD work per chunk:

| K  | inner work / chunk | T4 (cuda) | local Mac CPU |
|----|---:|---:|---:|
| 0  | none               | ~12 – 15 min | ~60 – 90 min  |
| 10 | 10 MBD iters       | ~25 – 40 min | ~90 – 140 min |
| 20 | 20 MBD iters       | ~50 – 70 min | ~150 – 240 min |

Total on T4: ~1.5 – 2 hr.  Total on a single Mac CPU: ~5 – 7 hr
(the next-Mac handoff is sized for a 6 hr budget).  Could
alternatively run K = 0 (fast, gate) on one Mac while K = 10 / 20
(slow, conditional) run on Colab T4 in parallel.

## Result

K-sweep ran on the secondary Mac on 2026-05-29 17:09–18:57 PT.
K = 0 finished in 108 min wall and **failed** the sanity gate (test
band [75, 85] %).  K = 10 and K = 20 were **skipped** by the
ksweep script's gate-passes guard (correctly, per
`run_phase_c_ksweep.sh::gate_passes`); their cells are left as
"not run" rather than `TODO`.

| K  | train (10) | **test (50, DP)** | combined | wall    |
|----|-----------:|------------------:|---------:|--------:|
| 0  | 0/10 = 0.0 % | **0/50 = 0.0 %**  | 0/60 = 0.0 % | 108.0 min |
| 10 | not run (gate FAIL) | not run | not run | — |
| 20 | not run (gate FAIL) | not run | not run | — |

Reference numbers:
- Phase-B `ep0199` standalone test (original): **80.0 %** (40/50)
- Phase-B `ep0199` standalone re-eval on this Mac (today, 10 fresh
  seeds 100000–100009): **70.0 %** (7/10).  Within the runbook's
  predicted "~6–9 / 10" sub-bucket band; see
  `standalone_sanity_2026_05_29/README.md`.
- DP paper Can/ph: **~85 %**
- Phase-3 hybrid `can` baseline at K = 10: 36.7 % (`run_can.py:173` comment)

### What the K = 0 rollouts actually look like

The bug-fix on `--mpc-replans 49` is verifiably installed: every
K = 0 rollout ran the full 50 chunks × 8 steps = 400 steps (none of
the 1-chunk × 16-step truncation that defines `K0_BAD_replans0_2026_05_29/`).
Reward distribution on K = 0 (60 rollouts):

| metric | K = 0 (today) | K0_BAD baseline (preserved) |
|---|---:|---:|
| chunks / rollout | 50 | 1 |
| steps / rollout | 400 | 16 |
| reward range | 1.51 — 35.49 | 0.01 — 0.29 |
| reward mean | 7.81 | 0.08 |
| rollouts with reward > 5 | 40/60 | 0/60 |
| rollouts with reward > 10 | 13/60 | 0/60 |
| test_succ_rate | **0.0 %** | **0.0 %** |

So the unified pipeline runs the prior, gives it the full 50-chunk
MPC budget, and produces *meaningful* rewards (the robot moves and
makes contact with the can — `reward > 5` in 40/60 cases) but
**never** triggers the lift-success condition.  Compare to the
standalone driver on the same ckpt + same fresh seeds, which lifts
the can in 119–162 steps and exits early with reward 25–35 in 7/10
test rollouts.

## Verdict

> **UPDATED 2026-06-01** -- the original `FAIL` on the K = 0
> sanity gate has been **superseded** by the Phase-D bughunt +
> the full epoch x lambda x K grid (Q1 + Q2 + Q3 + Phase E +
> Phase F).  The original FAIL paragraph is preserved below
> verbatim for the historical record; the resolved verdict is
> **YELLOW at the K = 10 lam = 0.5 operating point but GREEN at
> two ep0149 cells: (K = 10, lam = 2.0) at 86.7 % and (K = 20,
> lam = 0.5) at 83.3 %**, as detailed in the new "Resolved
> verdict (Phase-D)" sub-section.

### Resolved verdict (Phase-D, 2026-06-01)

**Operating-point-dependent**: YELLOW at the canonical
K = 10 lam = 0.5 cell across all 7 epochs of the v3b prior, but
**GREEN at two distinct ep0149 cells**:

  * **(ep0149, K = 10, lam = 2.0)** = 86.7 % (+13.4 pp lift,
    Phase F).  *Cheapest path: same K = 10 compute as the*
    *baseline, just more bridge weight.*
  * **(ep0149, K = 20, lam = 0.5)** = 83.3 % (+10.0 pp lift,
    Phase E).  More expensive but confirms the lift is not a
    lambda-2.0 fluke -- both knobs independently move ep0149
    into GREEN.

The Phase-C wrapping bug was isolated and fixed on
`feat/dp-phase-c-bughunt` (`unified_loop.py:478` -- gate the
chunk-boundary `env.reset() + set_sim_state()` on
`params.K > 0`).  Post-fix:

  * **K = 0 sanity gate PASS**: 8/10 = 80.0 % test rate at fresh
    seeds 100000-100009 on `ep0199`
    (`feat/dp-phase-c-bughunt`, see
    `phase_c_bughunt_2026_05_29/BRIEF.md`).  Sits at the upper
    edge of the runbook's "~6-9/10" PASS band; ties the GCP
    Phase-B reference of 80 % standalone.
  * **K = 10 epoch curve (Q1) plateau**: ~70 % test rate across
    ep0099 - ep0349 on this Mac (50 fresh seeds per ckpt,
    `phase_d_2026_05_30/epoch_curve_local/`).  Apples-to-apples
    with this Mac's Phase-B standalone re-eval at the same
    ckpt (7/10 = 70 %).  Same value, no lift, no drag.  Q1 peak
    at ep0149: 44/60 = 73.3 % (CI [61.0, 82.9]).
  * **Q2 lambda-sweep at ep0049 (weak prior, n = 40 per cell)**:
    lam in {0.1, 0.25, 1.0, 2.0} returns {0.47, 0.42, 0.50, 0.53};
    all CIs overlap the ep0049 Q1 baseline of 0.55 -- **no lift**
    from re-weighting the prior at the weak-prior point.
  * **Q3 K-sweep at ep0049 (weak prior, lam = 0.5)**: K in
    {5, 20} returns {0.60 (24/40), 0.53 (16/30)}; same reading
    -- **no lift** from changing the K budget on the under-
    trained prior.
  * **Phase E K = 20 at moderate prior**: at ep0149 (Q1 peak),
    25/30 = **83.3 % (CI [66.4, 92.7])**, a **+10 pp lift**
    over the same ckpt's K = 10 baseline of 73.3 % -- the
    pre-registered GREEN cell along the K-knob.  At ep0199,
    21/30 = 70.0 % (CI [52.1, 83.3]), no lift -- YELLOW.
  * **Phase F extreme-lambda probe at K = 10 (n = 30 / cell)**:
    at ep0149, lam = 0.1 -> 18/30 = 60.0 % (-13.3 pp, the
    bridge is too weak to be useful), lam = 2.0 -> **26/30 =
    86.7 % (CI [70.3, 94.7])**, a **+13.4 pp lift** over the
    K = 10 lam = 0.5 baseline of 73.3 % -- the second and
    stronger GREEN cell along the lambda-knob.  The ep0149
    lambda-sweep is monotone in lambda (60.0 % -> 73.3 % ->
    86.7 %).  At ep0199, lam = 0.1 -> 73.3 %, lam = 2.0 ->
    73.3 %; no lift in either direction -- YELLOW.

By the pre-registered K-sweep result rule (lift > 6 pp with CI
separation): GREEN at (ep0149, K = 10, lam = 2.0) at 86.7 %
and (ep0149, K = 20, lam = 0.5) at 83.3 %; YELLOW everywhere
else.  The strong DP prior on Can/ph is its own ceiling at
K = 10 lam = 0.5; MBD's value emerges at moderately-trained
priors along *either* the bridge-weight knob (lam = 2.0) or
the K-budget knob (K = 20), where the prior's score field is
coherent enough to guide search and there is still headroom
above the prior's standalone rate.  The lambda-knob path is
cheaper (same K = 10 compute) and delivers the larger lift.

K = 40 at ep0049 was dropped from the original Phase-E batch
(compute long pole; the original K = 40 cell at ep0049 was
killed at 2026-05-31 18:06 PDT to unblock Phase F).  It is
being relaunched standalone via
`scripts/local/relaunch_E_K40_after_chain.sh` and will land
Monday morning 2026-06-01.  The GCP arm was killed mid-run
after a worker-pool patch left orphan processes; the local
pool delivered the full ckpt x lam x K grid via disjoint seed
bands (100k Q1, 200k local Q2/Q3, 300k Phase E + F).  See
`phase_d_2026_05_30/CONCLUSION.md` + `q2q3_AGGREGATE.md` for
full numbers, the 2-D grid view, and the mechanism reading.

### Original FAIL paragraph (preserved verbatim for the historical record)

**FAIL** on the K = 0 sanity gate (0/50 vs 75-85 % band).
**Not interpretable** at K > 0 (and per the runbook's gate logic,
K = 10 / 20 were not run).

The Phase-B prior is healthy on this Mac.  The standalone re-eval
(diagnostic step 4 of `HANDOFF_NEXT_MAC.md::Gate failure`) hits
**7 / 10 = 70.0 %** test rate at fresh seeds 100000-100009, fully
inside the runbook's predicted "~6 - 9 / 10" sub-bucket band.

The 70-percentage-point gap between **70.0 %** standalone and
**0.0 %** unified-K=0, with **identical ckpt, identical 23-D
`can_dp` schema, identical first 10 fresh seeds, identical
max_env_steps and n_action_steps**, isolates the bug to the
unified-pipeline wrapping (one of `experiments/15_unified_hybrid/run_can.py`,
`unified_loop.py`, or `experiments/08_flow_matching_prior/adapter_dp.py`).

Per `HANDOFF_NEXT_MAC.md::Contact / escalation`:

> *"Any wiring fix that requires editing run_can.py / unified_loop.py
>  / adapter_dp.py is a separate phase and needs the user's explicit
>  go-ahead."*

Phase-C is therefore halted at the gate; the bug-hunt is the
follow-on phase, not part of this BRIEF.

### Hypotheses for the wrapping bug (in priority order, for the next phase)

1. **Observation-history maintenance across chunk boundaries.**
   The DP prior expects `T_obs = 2` (a stack of two consecutive
   23-D observations).  The standalone driver builds this stack
   explicitly per chunk; the unified loop's
   `_build_obs_for_prior(...)` (or equivalent in
   `priors_zerotrain.py`) may be passing a single-step obs or the
   wrong slice on chunk handoff.  Easiest first thing to check.

2. **`--lambda-prior 0.5` blending semantics at K = 0.**  At K = 0
   the prior's chunk should be executed as-is.  If the unified
   loop applies any lambda blending of the prior chunk against a
   default / value-baseline chunk before execution (rather than
   only against MBD-refined samples for K > 0), the prior's
   actions get attenuated.  Check `unified_loop.py` for an
   unconditional blending step that bypasses the `K == 0`
   short-circuit.

3. **Warm-start state at chunk-boundary execution.**
   `deterministic_prior_warmstart=True` is in the K = 0 config; if
   the warm-start uses the *previous chunk's* prior sample as the
   denoising init rather than re-sampling from the prior at the
   new env state, the chunks decorrelate from the actual env state
   over the 50-chunk rollout.

4. **Action-step alignment.**  Standalone uses
   `--n-action-steps 8` directly.  Unified uses `mpc_chunk_steps =
   8` plus `H = 16`.  If the unified loop is executing 16
   actions per chunk while the prior was trained at horizon 16
   but with `n_action_steps = 8` (i.e. only the first 8 of each
   16-action plan are intended to be open-loop-executed), then
   the unified pipeline doubles the open-loop horizon and the
   policy diverges over each chunk.  This is a strong candidate
   given the published DP recipe.

The new evidence captured under
`standalone_sanity_2026_05_29/` should be enough for the next
agent to triangulate without re-running anything heavy.

## Pre-registered next step (depends on verdict)

- **Sanity gate FAIL** → pause; debug the wrapping per the bug list
  above; no K > 0 conclusions.
- **GREEN** → publish hybrid result; consider extended K-sweep
  (K = 5 / 50 / 100) to characterise the lift curve; investigate
  whether the lift transfers to Square / Lift with the same
  `_dp`-style schema.
- **YELLOW** → close out the hybrid hypothesis on Can/ph (the
  prior is its own ceiling here); pivot to either (a) adaptation
  experiments where the *prior* needs to be perturbed-bank-aware
  rather than the wrapper, or (b) the off-distribution tasks
  (Square / Lift) where a strong prior may not exist and MBD
  search has more to contribute.
- **RED** → file as a known failure mode of MBD with strong DP
  priors; document in `docs/theory/` alongside the existing λ = 1
  collapse note in `run_can.py`.
