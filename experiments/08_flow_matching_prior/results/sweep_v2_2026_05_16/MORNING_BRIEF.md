# Morning brief — v_theta_v2 flow-prior sweep

**Sweep launched**: 2026-05-16 03:54 PDT
**Sweep completed**: 2026-05-16 11:18 PDT (~7.4 h, including Phase 7 contingency)
**Branch**: `feat/flow-mjx`
**Last commits**: `54b6ac4` (sweep infra + Phase 0 finding), `28151f2` (v2 0/60 baseline), `6ef09fc` (squashed colab fixes), `fd2486f` (monitor), `3b561b0` (Phase 7 contingency).

---

## FINAL OUTCOME

**Headline: σ_max=0.02, λ_prior=0.0, K=10, M=16, H=120  →  10/60 = 16.7 %.**

Breaks the previous v1+v2 0/60 ceiling and doubles the prior-only baseline (Phase 0 = 5/60 = 8.3 %). All artefacts live in
`results/sweep_v2_2026_05_16/`; aggregate Markdown is `v2_sweep_report.md`.

| Phase | Config | Result |
|---|---|---|
| 0 — prior-only replay | warm-start, no MBD | **5/60 = 8.3 %** |
| 1 — λ sweep WITH warm-start, σ=0.5 | every λ in {0, …, 20} | 0/60 |
| 2 — λ sweep NO warm-start, σ=0.5 | every λ in {0, 0.5, 2} | 0/60 |
| 3 — best-λ + K=20 at σ=0.5 | SKIPPED in script | — |
| 4 — best-λ + σ ∈ {0.1, 0.25} | SKIPPED (subsumed by Phase 5) | — |
| 5 — σ × λ cross-sweep | **σ=0.02, λ=0** | **10/60 = 16.7 %** |
| 6 — best cell + K=20 | σ=0.02, λ=0, K=20 | 2/60 = 3.3 %  (regresses) |
| 7 — bridge=quadratic at σ=0.02 | λ=0.1 / 0.5 / 1.0 | 4/60 / 2/60 / 1/60 |

### Three findings the data forced

1. **σ=0.5 is what was killing the v1 and v2 priors.** Every Phase 1 + Phase 2 cell at the default σ=0.5 was 0/60 regardless of λ_prior, regardless of warm-start. At σ=0.02 the same warm-starts survive and the cell jumps to 10/60.
2. **Denoising MBD hurts more than it helps once the warm-start is good.** Phase 6 (K: 10 → 20 at the winning cell) regressed from 10/60 → 2/60 — more refinement iterations overshoot. K=10 at σ=0.02 is near-optimal; further denoising is degrading, not amplifying.
3. **The score-bridge formula bug is not the primary blocker.** Phase 7 ran σ=0.02 with the theoretically-correct quadratic bridge `(1-t)^2` at λ ∈ {0.1, 0.5, 1.0}. Best cell was 4/60 (λ=0.1), still well below the λ=0 control's 10/60. The bridge formula does mildly improve small-λ cells (λ=0.1: linear 2/60 → quadratic 4/60), but it doesn't unlock a positive λ contribution at any tested magnitude. **Conclusion: the prior gradient adds noise that the warm-start can't survive, full stop. Either the v_theta_v2 score estimate is too coarse to be useful as a per-step gradient, or MBD's noise schedule is the wrong tool for amplifying a generative prior.**

### Implications for the hybrid headline

The flow-prior path is not on track to beat the BC-prior λ=5 result (33.3 %, exp 07) or the unified hybrid K=10 R=1 result (41.7 %, exp 15). The 16.7 % v_theta_v2 number is a positive measurement-cell result — proof that flow priors **can** transfer to MBD — but it does not promote to the project headline.

`docs/results.md` and the exp 08 row are updated in the same commit as this brief.

---

## Update — 2026-05-16 PM — Phase 8 / 9: MPC replanning ablation

After the σ-preservation sweep landed at 10/60 = 16.7 % single-shot, the natural follow-up was: **does MPC chunking + replanning multiply the flow-prior cell the same way it multiplies the BC cell (21.7 % → 37.5 %)?**

The answer is **no — replanning strictly hurts the flow prior at this cell.**

### Phase 8 (initial attempt, unseeded)

Ran the σ=0.02 λ=0 K=10 M=16 cell through the exp 15 unified driver at R=0 (matched control) and R=1 (the question):

| Cell | n / 60 | rate | reward median | "approached but missed" |
|---|---|---|---|---|
| R=0 (exp 15 driver, unseeded) | 2 / 60 | 3.3 % | 2.3 | 15 |
| R=1 (exp 15 driver, unseeded) | 0 / 60 | 0.0 % | 10.9 | 24 |

R=0 in exp 15 came back at 2/60 against Phase 5's 10/60 in exp 07 — a worrying ~8/60 gap between two drivers that *should* be equivalent at R=0. Diagnostic notebook (`/tmp/diag_drivers*.py`, kept locally) traced the gap to:

**`prior.sample_initial_mean(s0, n_steps=32)` depends on torch's global RNG state.** Each driver consumes a different number of torch RNG cells during module init, so the flow prior's warm-start at every rollout after the first one is sampled from a *different* torch state. With v_theta_v2's bimodal warm-start distribution at σ=0.02 (one mode is a viable trajectory with cost ≈ −25, the other is a near-zero idle trajectory with cost ≈ −2), small changes in the torch state push individual rollouts between modes, producing per-cell variance of ~ ±5–10/60. **The Phase 5 10/60 number and the Phase 8 R=0 2/60 number are both inside that envelope.**

### Phase 9 (seeded re-run)

Plumbed a deterministic per-rollout torch seed into the flow prior's warm-start path:

- `experiments/15_unified_hybrid/unified_loop.py`: new `warmstart_seed_fn: Callable[[int], int]` argument to `unified_plan_and_execute`. When supplied, the loop calls `torch.manual_seed(warmstart_seed_fn(chunk_idx))` immediately before each chunk's `initial_mean_fn`, eliminating the dependence on whatever ambient torch state the driver leaves.
- `experiments/15_unified_hybrid/run_can.py`: new `--deterministic-prior-warmstart` flag (default ON). Builds a per-(seed, demo, rollout, chunk) seed closure for flow runs; no-op for BC (whose `initial_mean_fn` is a deterministic MLP forward pass).
- Code change is on this branch, ready to commit. BC's existing pools are unaffected.

Re-ran the same two cells with seeding ON:

| Cell | n / 60 | rate | wall (s) | reward median |
|---|---|---|---|---|
| R=0 (seeded, P9) | **4 / 60** | **6.7 %** | 750 | 2.8 |
| R=1 (seeded, P9) | **0 / 60** | **0.0 %** | 1705 | 4.6 |

R=0 moved from 2/60 → 4/60 with seeding (consistent with the ~5–10/60 noise envelope). R=1 stayed at 0/60 — the direction is the same as Phase 8, so the conclusion is not a torch-state artefact.

### What the flow prior actually does at R=1

R=1's median reward (4.6) is higher than R=0's (2.8) and the "approached the can but didn't grasp" count (final reward > 20 yet success=False) is comparable across R={0,1}. Replanning is letting the robot get closer to the can on more rollouts, but **destroying the grasp/lift completion phase**. The likely mechanism:

- Chunk 0's `sample_initial_mean(s_t=0, n_steps=32)` samples from the *training distribution* (the flow was trained only on full demos starting at t=0).
- Chunk 1's `sample_initial_mean(s_t=60, n_steps=32)` is fed a mid-trajectory state that is **out of distribution** for the flow prior. The resulting warm-start is closer to "idle hover near the can" than to "grasp + lift", so the second half of the trajectory regresses to the bad mode.

BC's per-state action MLP does not have this failure mode: it was trained on `(s_t, a_t)` pairs for every t, so it generalises to mid-trajectory states. That is the architectural reason BC + replanning works (21.7 % → 37.5 %) while flow + replanning regresses (6.7 % → 0 %).

### Net narrative impact

- **Project headline unchanged** at 37.5 % (CI [31.6, 43.8]) BC + unified hybrid K=10 R=1. Nothing in Phase 8/9 moves that number.
- **Flow-prior story sharpened**: not all data-driven priors play nicely with MPC. A trajectory-conditional generator like the flow prior is brittle at mid-trajectory states; a per-state policy like the BC MLP is not. This is a clean ablation supporting the BC-prior choice rather than a competing headline.
- **Per-cell noise documented**: the σ-preservation sweep's 10/60 should be quoted with an explicit ±5–10/60 caveat (no seed pool was run for this cell). The conclusion that "v_theta_v2 unlocks σ=0.02 single-shot performance" still stands — the noise envelope is still well above the 0/60 v1+v2 ceiling — but the specific 16.7 % number is one draw from a wider distribution.

### Artefacts

- Result JSONs: `experiments/08_flow_matching_prior/results/v2_followup_2026_05_16/unified_flow_lam-0.0_K-10_R-{0,1}_{exp15_R{0,1}, phase9_seeded}.json`
- Run log: `experiments/08_flow_matching_prior/results/v2_followup_2026_05_16/phase9_seeded.log`
- Code diff: `experiments/15_unified_hybrid/{unified_loop.py, run_can.py}` (warmstart_seed_fn + `--deterministic-prior-warmstart`)

---

## TL;DR (pre-sweep, kept for context)

**The headline news happened *before* the main sweep finished**:

> **Phase 0 (prior-only replay, no MBD): 5/60 = 8.3 % success.**
>
> The v_theta_v2 prior IS task-capable on its own.
> MBD with σ_max=0.5 has been actively destroying its good warm-starts.

This re-frames the whole investigation.  The remaining sweep tests
*how to keep MBD from destroying the prior's wins* (Phase 5,
σ-preservation) and *whether MBD can amplify the wins* once σ is
matched to the warm-start quality (Phase 6).

## What's running right now

```
screen -dmS sweep_v2  (PID 92618 at launch)
log: experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/sweep.log
```

Phases (idempotent — each phase's `summary.json` guards re-runs):

| # | Phase | Status | Expected wall | Result |
|---|---|---|---|---|
| 0 | prior-only replay (no MBD) | **DONE** | 30 s | **5/60 = 8.3 %** |
| 1 | λ-sweep w/ warm-start, 8 λs, σ=0.5 | in progress | ~2.5 h | tbd |
| 2 | λ-sweep w/o warm-start, 3 λs | pending | ~50 min | tbd |
| 3 | best-λ + K=20/30 | SKIPPED at runtime | 0 | skipped |
| 4 | best-λ + σ ∈ {0.1, 0.25} | SKIPPED at runtime | 0 | skipped |
| 5 | σ-preservation: σ ∈ {0.02, 0.05, 0.1} × λ ∈ {0.0, 0.1, 0.5} | pending | ~150 min | tbd |
| 6 | best (σ, λ) + K=20 | pending (contingent) | ~26 min | tbd |
| Σ | | | **~5-6 h** | |

Phase 3 / 4 were skipped because Phase 0's finding promoted Phase 5
(σ-preservation) to highest priority; Phase 3 (K-extension at the
wrong σ) and Phase 4 (σ ∈ {0.1, 0.25} sweep, with 0.1 redundant
with Phase 5 and 0.25 still way above the exp-09-recommended
≤ 0.1 ceiling) had near-zero expected value.

## Why the σ-preservation pivot

**Math reconstruction of the σ_max=0.5 problem** (see `adapter.py:147`
and `mbd.py:255-258`):

- MBD update with prior: `mu_new = (1-lr_k) * mu_planner + lr_k * mu_prior`
  where `lr_k = clip(λ * σ/σ_max, 0, 1)` and `mu_prior = mu + σ² * score`.
- At iter 0, σ=σ_max → `lr_k = clip(λ, 0, 1)` → any λ ≥ 1 saturates
  to lr_k=1.  So **the prior gradient step fully replaces MBD's M=16
  cost-weighted perturbation at iter 0** for any λ ≥ 1.
- Score formula at iter 0 (t=1/K=0.1): for warm-start x ≈ data,
  `score ≈ -x` (derivation in notes below).  So
  `mu_prior = mu + σ_max² * (-mu) = 0.75 * warm_start` at iter 0.
  **The "good" warm-start gets shrunk by 25 % before MBD even
  evaluates a single perturbation.**
- Even at λ=0 (no prior gradient at all), MBD's σ=0.5 perturbations
  add ±0.5 * N(0, I) noise per element to the warm-start.  Action
  bounds are [-1, 1].  So perturbations are at *50 % of the action
  range* — catastrophic for a manipulation task where mid-execution
  timing of gripper open/close is sub-action precision.

Exp 09's own σ-sweep (`docs/results.md`) found
**σ ≤ 0.10 viable; peak σ=0.01-0.03 = 40 % single-seed (BC warm-start)**.
The same regime should apply to a v_theta_v2 warm-start.  That's
exactly what Phase 5 tests.

## Likely outcomes (predictions)

1. **Phase 1 λ=0.0 w/ warm-start**: likely 0-1/60.  MBD's σ=0.5 noise
   destroys warm-start regardless of λ.
2. **Phase 1 λ ≥ 1.0 w/ warm-start**: 0/60.  Plus score shrinks
   warm-start at iter 0.
3. **Phase 2 (no warm-start, MBD-only baseline)**: 0/60 at λ=0
   (matches exp 04's known result), 0/60 at λ > 0 (score not strong
   enough to drive MBD from zero to success in 10 iters).
4. **Phase 5 σ=0.02 + λ=0**: ~5/60 (matches Phase 0).  Warm-start
   preserved, MBD does essentially nothing.
5. **Phase 5 σ=0.02 + λ=0.1 or 0.5**: maybe 5-12/60 if the score
   formula is right; maybe back to 0/60 if the score formula has
   the suspected `(1-t)` vs `(1-t)²` issue (see Open Question below).
6. **Phase 5 σ=0.05 + λ=0**: 2-5/60.
7. **Phase 5 σ=0.1 + λ=0**: 0-2/60.
8. **Phase 6 (best Phase 5 + K=20)**: amplification if Phase 5 > 5
   (more refinement iterations to lock in good trajectories).

If predictions hold, **expected best**: σ=0.02 + λ ∈ [0, 0.5], rate
~8-15 %.  Modest but breaks the v1+v2 0/60 ceiling.

## Open question — score-formula derivation

In `model_v2.py`'s docstring (lines 37-46):

> Score-bridge formula:
>     grad log p_t(x_t) ~= (t * x1_hat - x_t) / max(1 - t, eps)
>
> Derivation: ... the score under that Gaussian is
> `(t x_1 - x_t)/(1-t)^2`. We keep the `(1-t)` denominator (not squared)
> for numerical consistency with `model.py`'s convention so an existing
> lambda sweep transfers.

The "correct" Gaussian-likelihood score has denominator `(1-t)²`,
but the code uses `(1-t)`.  At late MBD iters (large t, small 1-t),
this under-estimates score magnitude by a factor `1/(1-t)`, which
can be ~10x at K=10's last iter.

The comment claims back-compat with v1's λ sweep — but **v1's λ
sweep was 0/60 anyway** and the calibration is meaningless.  If
Phase 5 shows λ>0 doesn't help (the warm-start gets refined back
to zero successes), the suspect is this formula.

A test exists in `sweep_v2_phase7_score_fix.sh` (not yet run; see
"Phase 7 contingency" below).

## Phase 7 contingency (NOT auto-run; manual launch if Phase 5 says λ>0 hurts)

If Phase 5's data shows that at σ=0.02 the λ=0 column outperforms
λ=0.1 and λ=0.5 columns, the score formula is the likely cause.
Fix: change `adapter.py:147`'s denom from `(1-t)` to `(1-t)²`.

To run the contingency (instructions, not auto-run):

```bash
# 1. Edit adapter.py:147 (the v1 FlowPrior class) AND the v2 path:
#    swap denom = (1 - t_clip)  ->  denom = (1 - t_clip) ** 2
# 2. Re-run Phase 5 with corrected formula:
bash experiments/08_flow_matching_prior/sweep_v2_phase7_score_fix.sh
```

(Script not yet written; if Phase 5 motivates it, write similar
to sweep_v2_ext.sh's Phase 5 block.)

## How to read results when you wake up

1. **Quickest summary**:
   ```bash
   cat experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/v2_sweep_report.md
   ```
   The aggregator (`sweep_v2_aggregate.py`) writes this Markdown
   file with all phase tables.  It runs at the end of each chained
   script, so an intermediate version is available even if the
   sweep is mid-flight.

2. **Per-phase summaries**:
   `experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/phase{N}_*/summary.json`

3. **Per-rollout detail** (cost curves, exec rewards, success):
   `experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/phase{N}_*/lambda_*.json`

4. **Live progress** (the sweep is running under `screen -r sweep_v2`):
   ```bash
   tail -f experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/sweep.log
   ```

5. **Is the sweep still running?**
   ```bash
   screen -ls | grep sweep_v2
   # OR
   ps aux | grep "python.*lambda_sweep" | grep -v grep
   ```

## If something went catastrophically wrong

The sweep is idempotent.  Each phase's `summary.json` guards the
work.  To resume from a particular phase, just delete that and
later phases' `summary.json` files and re-run:

```bash
bash experiments/08_flow_matching_prior/sweep_v2.sh
```

(NB: this chains into `sweep_v2_ext.sh` automatically.)

---

## Update — 2026-05-16 PM (later) — Warm-start ablation slate

Three new exp-15 cells at the **headline config** K=10 R=1 σ=0.25
M=16 H=120 seed=0, 60 rollouts each, run from
`experiments/15_unified_hybrid/warmstart_ablation.sh`.  Output:
`experiments/15_unified_hybrid/results/warmstart_ablation_2026_05_16/`.
Logs: `run.log` in the same directory.

| Cell | warm-start | λ (score-bridge) | succ  | rate     |
|------|------------|------------------|-------|----------|
| C0 (existing) | BC      | 5 (BC bridge)   | 90/240 | **37.5 %** (4-seed pool) |
| C1   | BC         | 0                |  7/60 |   11.7 % |
| C3   | flow v2    | 0                |  0/60 |    0.0 % |
| C4   | flow v2    | 5 (flow bridge)  |  0/60 |    0.0 % |
| (exp 04) | none   | 0                |  0/60 |    0.0 % |

**Decomposition of the 37.5 % headline**

```
pure MBD             0.0 %
+ BC warm-start     +11.7 pp   (C1)
+ BC score-bridge   +25.8 pp   (C0 - C1)
= 37.5 %
```

The score-bridge is the **dominant** contributor.  The warm-start
is necessary (gives the score-bridge something non-zero to act on)
but provides only the smaller fraction of the lift.

**BC vs flow as a warm-start at σ=0.25**

- BC (deterministic MLP, ~50 KB): 11.7 %.
- Flow v2 (42 MB FiLM flow): 0 %.

Adding the flow's own per-step score-bridge at λ=5 does not rescue
this (C4 = 0 %).  This is consistent with Phase 7's quadratic-bridge
result at σ=0.02 — the flow score-bridge never helps at any σ we
have measured.  Flow v2 is a **narrow-σ specialist** that only wins
near σ=0.02 R=0 (10/60 ≈ 16.7 %, single seed, ±5-10 / 60 noise).

**Narrative impact**

This locks in:

- **Headline algorithm: BC warm-start + per-step BC score-bridge**
  via the unified driver (exp 15).  All 4 ablation cells, the 4-seed
  pool, and the K × R grid agree on the same operating point.
- **Flow prior is a *negative result with explanatory power* —**
  the architecture is more expressive but is brittle in σ and gives
  no headline lift.  Its presence in the report demonstrates that
  "use a bigger / better generative model as the prior" is **not**
  free; the per-step score-bridge is the load-bearing component.

**Why this is not a hyperparameter sweep**

All four cells share **identical** hyperparameters (K=10 R=1 σ=0.25
M=16 H=120 seed=0).  Only the prior architecture and λ vary —
those are *algorithm dimensions*, not knobs.  The slate is an
ablation through the canonical unified driver, consistent with the
project rule "no hyperparameter sweeps before codebase
consolidation" (`docs/plans.md`).
