# Post-consolidation experimental plan: branches A / B / C

**Drafted 2026-05-14, post BC-swap re-run.** Companion to
[`plans.md`](plans.md) (which sets the *project-level* tier
structure) and [`bc_swap_rerun_analysis.md`](bc_swap_rerun_analysis.md)
(which sets the *empirical* problem statement). This doc scopes
**three concurrent feature branches** that together attack the
three things the n=60 re-run revealed:

1. The hybrid does not stat-sig beat plain BC (+11.7 pp, p=0.15).
2. The MBD score-correction adds only ~3 pp on top of pure-BC
   particle-averaging (λ=5: 33.3 % vs λ=∞: 30 %).
3. The two existing hybrid implementations differ by +15 pp at
   the same K-cell because they are not the same algorithm.

Each branch attacks one *layer* of the system (the MBD signal,
the score parameterisation, the coupling) so the three workstreams
can proceed in parallel after a small synchronous prefix.

---

## Dependency graph

```
                        main  (b547357, 2026-05-14)
                          │
                          ▼
         feat/hybrid-coupling  ─── Phase 1 (BLOCKING)
         (Branch C)                consolidates run_can.py
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
   feat/stronger-     feat/principled-  (Phase 2+ on
   mbd-signal         score              feat/hybrid-coupling)
   (Branch A)         (Branch B)
```

**Synchronisation point**: Branch C Phase 1 (the unified driver)
is the eval substrate for A and B. A's training (V-baseline) and
B's training (`v_theta.pt`) can begin immediately in parallel
with C; only the *evaluation* runs need the unified driver to
exist first.

Estimated calendar: C Phase 1 lands within ~2 days of starting,
then A and B headline-evaluation runs unblocked. A and B training
in parallel during C Phase 1.

---

## Branch A: `feat/stronger-mbd-signal`

### Diagnosis being attacked

Exp 07 λ=0 (no BC anchor, pure MBD) is **0/60**. From a random
Gaussian initial trajectory and a 100-step denoising chain,
MBD never reaches the goal. Whatever cost-gradient signal MBD
is using is too noisy or too uninformative to plan with on its
own; everything that "works" in the hybrid is BC pretending to
be MBD. Branch A's hypothesis is that **a stronger / less-noisy
MBD gradient lets the planner contribute information the BC
prior doesn't have**.

### Tasks (concrete)

| # | task                                                                  | scaffolding                              | effort   |
|---|-----------------------------------------------------------------------|------------------------------------------|----------|
| A1 | **σ-schedule sweep** at the unified-driver baseline                  | `experiments/09_sigma_sweep/` (scaffolded) | ~5 h compute, 0.5 d code |
| A2 | **Learned `V_φ(s)` value baseline** trained on BC dataset            | NEW — `experiments/15_value_baseline/`   | ~3 days  |
| A3 | **Particle-count ablation** M ∈ {64, 256, 1024} at λ=5               | one-line flag in unified driver          | ~3 h compute |
| A4 | **Reward shaping** (object-EEF dist, lift-height, contact bonus)     | wrap `04_mbd_baseline/env.py` cost_fn    | ~1 day  |

### Code locations

- `experiments/04_mbd_baseline/mbd.py` -- `cost_fn` is an
  injected callable into `denoise_loop`; A4's reward shaping
  lands here as a new `shaped_cost_fn`.
- `experiments/15_value_baseline/train.py` -- new directory;
  TD-style critic trained on `low_dim_v141` (s, a, r, s') tuples
  from the 20-demo `can/ph` slice. A2's deliverable is a
  `V_phi.pt` checkpoint + an adapter (mirroring
  `08_flow_matching_prior/adapter.py`) that yields a
  `make_value_score_fn(s_0)` for the unified driver.
- `experiments/09_sigma_sweep/run.sh` -- already scaffolded; A1
  just needs the unified driver path and a fresh σ-grid.

### Success criteria

- **A1 σ-sweep**: at least one σ_max value beats the unified-
  driver λ=5 baseline by ≥ 3 pp at n=60 (single-seed
  suggestive). Or: rules out σ as the bottleneck.
- **A2 value baseline**: `V_phi`-augmented MBD reaches ≥ 5/60
  on the λ=0 (no-BC-anchor) configuration. Failing that, the
  hypothesis "MBD is signal-limited" is falsified and we focus
  on the BC side.
- **A3 particles**: M=1024 vs M=64 gap < 2 pp ⇒ MBD is not
  variance-limited; M ≥ 256 is enough. Cheap to settle.
- **A4 reward shaping**: ≥ 5 pp lift on λ=0 (pure-MBD), even if
  it breaks the unshaped-reward fairness of the headline.
  Documented as "best-case MBD" for the paper.

### What lands on `main` from A

`experiments/09_sigma_sweep/` results + JSONs,
`experiments/15_value_baseline/` (full directory),
modifications to `04_mbd_baseline/mbd.py` for shaped-cost hook,
`docs/results.md` rows for σ / V_φ / M / shaped-reward.

---

## Branch B: `feat/principled-score`

### Diagnosis being attacked

The score-bridge term `λ · ∇log π_BC(x_t)` assumes `π_BC` has a
well-defined score function. The current Exp 03 BC is a
deterministic MLP; what we differentiate is the MSE
`‖π_BC(s, x) − x‖²`, treating it heuristically as
`−log p(x | s)`. This is *not* a rigorous score; the empirical
λ=5 sitting 10× higher than the theoretical nominal (§ 2(d) of
`bc_swap_rerun_analysis.md`) hints the calibration is off.
Branch B's hypothesis is that **a rigorously-defined score
lets the bridge term carry information that the deterministic-
MLP gradient cannot**.

### Tasks (concrete)

| # | task                                                                                  | scaffolding                                      | effort   |
|---|---------------------------------------------------------------------------------------|--------------------------------------------------|----------|
| B1 | **Train `v_theta.pt`** on 20-demo `can/ph` via the conditional flow-matching trainer | `experiments/08_flow_matching_prior/train.py` (exists) | ~10-30 h CPU / 3 h GPU |
| B2 | **Plug `FlowPrior` adapter into unified driver**'s `--prior` flag                    | `08_flow_matching_prior/adapter.py` (exists)     | ~0.5 day |
| B3 | **λ-sweep on `v_theta` prior** at n=60 (same 6-cell grid as Exp 07)                  | re-uses Exp 07 runner via unified driver         | ~6 h compute |
| B4 | **Stochastic Gaussian-head BC** as a half-step intermediate (cheaper than diffusion BC) | sibling of `experiments/03_bc_baseline/policy.py` | ~1 day  |

B4 is optional and primarily useful if B1-B3 underperform; a
Gaussian-head BC is a strictly simpler probabilistic policy whose
score is also exact.

### Code locations

- `experiments/08_flow_matching_prior/train.py` -- trainer already
  written; B1 is a long-running compute task on the existing
  scaffolding. Output: `experiments/08_flow_matching_prior/v_theta.pt`.
- `experiments/08_flow_matching_prior/adapter.py` -- `FlowPrior`
  class exposes `sample_initial_mean(s_0)` and
  `make_prior_score_fn(s_0, K)`, both already designed for the
  Exp 07-style runner interface. B2's job: pass `--prior=flow`
  to the unified driver.
- `experiments/03_bc_baseline/policy.py` -- B4 adds a
  `GaussianMLPPolicy` next to the existing `MLPPolicy`; same
  training script with `--head=gaussian`.

### Success criteria

- **B1 training**: `v_theta` reaches a held-out FM loss within
  20 % of the train loss; LQ sanity (`lq_sanity.py`) passes;
  trajectory samples are visually-plausible can-trajectories.
- **B2/B3 evaluation**: at least one λ value with `v_theta` prior
  matches or beats the MLP-BC headline (33.3 %). **Stretch
  goal**: `v_theta` beats MLP-BC at *every* λ (whole-curve
  improvement, not just a single peak).
- **B3 multimodality test**: at λ=∞ (pure prior), sample 100
  trajectories per initial state with `v_theta`; verify
  trajectory diversity > MLP-BC diversity (cluster count or
  pairwise distance metric). The point of a flow prior is
  multimodal samples; if `v_theta` collapses to a single mode,
  the prior is no better than the MLP.
- **B4 (if run)**: Gaussian-BC alone at n=60 should be within
  ±3 pp of the deterministic MLP-BC's 21.7 % (no regression),
  but with exact-score `λ`-tuning behaving better-conditioned.

### What lands on `main` from B

`experiments/08_flow_matching_prior/v_theta.pt` (LFS-tracked),
`experiments/08_flow_matching_prior/results/*.json`,
`docs/results.md` table rows for `v_theta` at each λ,
optionally `experiments/03_bc_baseline/policy.py` Gaussian head.

---

## Branch C: `feat/hybrid-coupling`

### Diagnosis being attacked

Two facts: (i) Exp 05/10's warm-start-only `run_can.py` and Exp
07's score-bridge `run_can.py` differ by +15 pp at K=10, *the
biggest hyperparameter-like gap in the project*; (ii) Exp 07 is
single-shot H=120, Exp 10 has MPC chunking, so the score-bridge
× MPC interaction has *never been measured*. Branch C's
hypothesis is that **the right way to use the score-bridge is
inside an MPC outer loop, and that this combination wins more
than either alone**.

### Phases

#### Phase 1 (BLOCKING for A and B evaluation): consolidation

Unify the two `run_can.py` drivers into one. Concretely:

- Take Exp 07's `score_fn_factory` and `initial_mean_fn`
  machinery (lines 145-205 of
  `experiments/07_lambda_sweep/run_can.py`) as the canonical
  score-bridge path. This gives correct per-step
  `λ ∇log π_prior` semantics.
- Take Exp 05's `mpc_plan_and_execute` (lines 230-440 of
  `experiments/05_hybrid_method/run_can.py`) as the canonical
  MPC chunking path. This gives correct K / R semantics with
  state-resets between chunks.
- Fuse into `experiments/15_unified_hybrid/run_can.py` (new
  directory; name TBD -- maybe `experiments/05b_unified/` to
  keep the existing numbering).
- Flags: `--prior {mlp_bc, flow, gaussian_bc}`,
  `--lambda <float|inf>`, `--K <int>`, `--mpc-replans <int>`,
  `--mpc-chunk-steps <int>`, `--sigma-max <float>`,
  `--num-particles M <int>`, `--deterministic-bc-reset`,
  `--shaped-cost`, `--value-baseline-ckpt <path>`.
- Retire Exp 05/07/10 runners after Phase 2 verification
  passes; keep their result JSONs for archival.

#### Phase 2: λ=5 verification grid (the planned 9-cell run)

9 cells (K ∈ {5, 10, 20}, R ∈ {0, 1, 2}) at λ=5, n=60 each, on
the unified driver. Three outcomes (see `plans.md` § *Framing
for the final paper*):

| outcome                                          | implication                                                         |
|--------------------------------------------------|---------------------------------------------------------------------|
| Best cell > Exp 07 single-shot (33.3 %)          | MPC × score-bridge wins; headline number is the new grid maximum.   |
| Best cell ≈ 33.3 %                                | MPC chunking is neutral on top of score-bridge; score-bridge alone is the algorithm. |
| Best cell < 33.3 %                                | MPC chunking hurts the score-bridge; the warm-start-only formulation had a hidden virtue. |

#### Phase 3: adaptive λ(σ) along the noise schedule

Currently λ is constant across all denoising steps. Try
`λ(σ) = λ_max · σ / σ_max` (high λ early, low λ late) and the
inverse. One-flag addition; ~3 h compute for a 3-point sweep at
the Phase 2 winning cell.

#### Phase 4: truncated denoising chain

Currently K=10 means 10 denoising steps. Try K ∈ {3, 5, 7, 10}
starting from BC-anchor; the hypothesis is that with a strong
BC the planner only needs a few fine-tuning steps. ~3 h compute.

#### Phase 5: replanning trigger (uncertainty-gated MBD)

Only invoke MBD when BC-action variance across an MC-dropout
ensemble exceeds a threshold; otherwise execute BC directly.
Reframes the hybrid as "BC + selective MBD rescue". Code change
~1 day; ~3 h compute.

### Code locations

- `experiments/05_hybrid_method/run_can.py` -- source of the MPC
  chunking machinery (`mpc_plan_and_execute`).
- `experiments/07_lambda_sweep/run_can.py` -- source of the
  score-bridge machinery (`score_fn_factory`).
- `experiments/15_unified_hybrid/run_can.py` -- new driver
  (Phase 1 deliverable).
- `experiments/04_mbd_baseline/mbd.py` -- shared `denoise_loop`;
  may need a small `lambda_schedule_fn` argument for Phase 3.

### Success criteria

- **Phase 1 (consolidation)**: the unified driver must
  *reproduce* both Exp 07 λ=5 (33.3 %) AND Exp 10 K=10 R=2
  (18.3 %) on the same seeds, given the corresponding flag
  combinations. This is the regression-test bar; without it
  we don't know the consolidation is correct.
- **Phase 2 verification**: covered by the three-outcome table
  above. Any of the three is a publishable result; the goal is
  to *decide*, not to win.
- **Phase 3 / 4 / 5**: each is a one-figure ablation; success =
  ≥ 3 pp lift over Phase 2 winner OR a clean negative result
  worth a paragraph.

### What lands on `main` from C

The new `experiments/15_unified_hybrid/` directory + result
JSONs, a deprecation note in Exp 05/07/10 READMEs pointing to
the unified driver, updates to `docs/results.md`, the planned
3-outcome paper-framing decision in `docs/plans.md` § *Framing
for the final paper*.

---

## Cross-branch coordination

- **A's `--value-baseline-ckpt` flag** is added to the unified
  driver in Phase 1 with a no-op default (passes through to a
  zero-baseline). A's evaluation runs only need to point at a
  trained `V_phi.pt`.
- **B's `--prior=flow` flag** is added to the unified driver in
  Phase 1 with a no-op default (falls back to MLP-BC). B's
  evaluation runs only need to point at a trained `v_theta.pt`.
- **A4's shaped-cost hook** is added to the unified driver in
  Phase 1 as `--shaped-cost`. Default off.
- Result-JSON schema is shared: all branches write into
  `experiments/<N>/results/*.json` with the existing key
  layout (`{success, time_s, demo_idx, rollout_idx, seed,
  config}`). The Exp 10 analyzer keeps working unchanged.

## Open questions for the user

1. **Numbering convention** for the unified driver: keep going
   (`experiments/15_unified_hybrid/`) or insert a sibling under
   the existing 05 (`experiments/05_hybrid_method/run_can_v2.py`)?
   I'd default to a new directory because the unified driver is
   a strict superset of both 05 and 07 and the rename keeps the
   archival JSONs valid under their original paths.
2. **Compute prioritisation**: train `v_theta` (B1) on the Mac
   overnight (10-30 h CPU), or wait for the GCE L4 path to
   unblock and train in ~3 h? My recommendation: kick off Mac
   training now in the background, switch to GCE later only if
   the Mac run is unsatisfactory.
3. **Stretch hypothesis to add**: Category D from the brainstorm
   (harder benchmark / OOD initial states) is *not* covered by
   any of A / B / C above. Want it as a fourth branch
   (`feat/harder-benchmark`), or deferred to after A / B / C
   converge?

---

## What's explicitly NOT on these branches

- Hyperparameter tuning *on the current (pre-consolidation)
  codebase*. Forbidden by the discipline added 2026-05-14.
- Seed-pool runs *before* Branch C Phase 1 lands. The "winning
  cell" might not be one of the cells that currently exists.
- README / docs prose updates beyond the result tables. The
  narrative-level rewrites (paper framing, abstract, intro)
  wait until A / B / C deliver a decision.
