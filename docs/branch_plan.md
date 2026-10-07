# Branch plans — per-branch roadmaps

This is the umbrella branch-plan document containing the per-branch
roadmaps for each of the active topic branches. It was produced by
merging `feat/hybrid-coupling` and `feat/stronger-mbd-signal`
into `main`, since each branch had independently added a
`docs/branch_plan.md` for its own roadmap.

**Status as of 2026-05-25**: both branches are now **merged** into
`main` (`595aadf` and `25ca457` respectively). The remaining
"outstanding work" items below have been triaged in
[`plans.md`](plans.md) § 2026-05-25 update — most A and C items have
been **superseded** by the retrieval-prior landing (Slate E, May 19,
73.8 % pooled headline). The umbrella plan below is preserved for
historical reference and for the items that survived the post-Slate-E
triage (chiefly A2 V_φ eval and Phase 3 adaptive `λ(σ)` wiring).

For the live roadmap and headline number, see:

- [`plans.md`](plans.md) — cross-branch overview and current priorities.
- [`results.md`](results.md) — headline numbers (refreshed 2026-05-25).
- [`experiments/README.md`](experiments/README.md) — per-experiment
  lifecycle status and reproduce commands.

Two additional branches merged on the same day, neither carries a
`docs/branch_plan.md`-style spec but both are documented here:

- **`feat/zero-train-priors`** (`2daa166`, 2026-05-25) — owns the
  retrieval-prior family (exp 15 Slates D – J + `priors_zerotrain.py`
  + the dense-reward shim). The Slate timeline and per-slate BRIEFs
  live under `experiments/15_unified_hybrid/results/slate_*/`.
- **`feat/theory`** (`b8de97d`, 2026-05-25, rebased) — owns
  [`docs/theory/`](theory/): soft-control derivation, SNIS bound, and
  Pareto bias-variance dominance theorem.

`feat/flow-v3-karras` and `feat/flow-mjx` remain **off main**; their
roadmaps live on the branches themselves (linked from
[`plans.md`](plans.md)).

Sections:

- [Branch A — `feat/stronger-mbd-signal`](#branch-a--featstronger-mbd-signal)
- [Branch C — `feat/hybrid-coupling`](#branch-c--feathybrid-coupling)

---

## Branch A — `feat/stronger-mbd-signal`


This is the per-branch roadmap for **Branch A**.

- Owns: all of `experiments/09_sigma_sweep/`.
- May-15 closeout shipped on this branch: σ-max sweep (10 cells),
  Stage 5 M-ablation (M ∈ {16, 64, 256}), Job B 4-seed σ=0.03
  pool.
- Cross-branch context (other branches' work + the unified
  status table): [`plans.md`](plans.md) on `main`.
- Origin spec (pre-May-15, kept for reference):
  [`docs/archive/plan_stronger_mbd_signal.md`](archive/plan_stronger_mbd_signal.md).

### Outstanding work

Priority order, applied 2026-05-15 evening after the closeout
triage:

1. **A1.4 — σ-sweep on the unified driver at K=10 R=1 λ=5.**
   The actual May-14 ask (Job A measured K=10 R=0 instead).
   ~10 h compute, no new code. Highest leverage.
2. **A2 — Learned `V_φ(s)` value baseline as MBD control variate.**
   ~3 days code + 5 min training + 2 h eval. Tests whether the
   pure-MBD (λ=0) 0/60 is a variance problem (not a model
   capacity problem). Acceptance bar: pure-MBD with `V_φ`
   subtracted ≥ 5/60.
3. **A3 follow-up (b) — `λ_softmax ∝ log M` scaling correction.**
   Optional, conditional on (1) and (2) landing with budget left.
   Tests whether the May-15 M=256 inverted-U is a fixed-`λ_softmax`
   artifact.
4. **A4 — Reward shaping (`--shaped-cost`).** Optional /
   paper-appendix. Lowest priority.

**Postponed:** A3 follow-up (a) (4-seed CI at M=64 σ=0.05). The
single-seed 50.0 % cross-branch headline is cited with the
standard single-seed caveat; rigour can wait.

**Skipped:** (none in Branch A — all Branch A items are either
done, prioritised, or optional. The skipped items live on
Branch C.)

---

### Spec — A1.4 σ-sweep on unified driver at K=10 R=1

#### Why

Job A (May 15) measured the σ-sweep on the unified driver at
**K=10 R=0** — finding σ=0.05 = 38.3 % as the K=10 R=0 peak. The
May-14 plan asked for the σ-sweep at **K=10 R=1** specifically,
because R=1 is where the unified-driver headline lives
(Job C 4-seed pool = 37.5 % at K=10 R=1 σ=0.25). The two regimes
peak at *different* σ on the exp 05 driver (R=1 wants
σ=0.01-0.03; R=0 wants σ=0.05), so we can't extrapolate Job A
to R=1.

Hypothesis: the unified driver's R=1 peak σ is in the range
[0.01, 0.10], plausibly above the current 37.5 % single-seed
mean. A new peak at σ ∈ {0.01, 0.03, 0.05, 0.10} would shift
the unified-driver headline.

#### What to run

Five σ cells on the unified driver, K=10 R=1, λ=5, M=16, seed=0,
n=60 each:

| σ    | Cell name                                                     |
|------|---------------------------------------------------------------|
| 0.01 | `unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma0.01.json` |
| 0.03 | `unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma0.03.json` |
| 0.05 | `unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma0.05.json` |
| 0.10 | `unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma0.10.json` |
| 0.25 | already exists (Phase 2 cell = 41.7 % seed=0, 37.5 % pool)   |

```bash
# From repo root, hybrid-traj env activated.
for s in 0.01 0.03 0.05 0.10; do
    python experiments/15_unified_hybrid/run_can.py \
        --K 10 --n-replans 1 --lambda-prior 5 \
        --sigma-max $s --M 16 --prior bc \
        --n-demos 20 --n-rollouts 3 --seed 0 \
        --n-workers 2 --no-live-view \
        --output experiments/15_unified_hybrid/results/unified_bc_lam-5.0_K-10_R-1_sigmasweep_lam5_sigma${s}.json &
done
wait
```

Single seed=0 only (matching Job A's protocol). 4-seed CI on
the resulting peak σ can be added later if needed.

#### Compute

~2 h per cell × 5 cells = ~10 h serial. With `--n-workers=2`
inside each cell and 2 cells in parallel = ~5 h wall. M4 Max
local has the budget.

#### Acceptance bar

Two outcomes are both informative:

1. **Peak at σ < 0.25 beats the 41.7 % single-seed / 37.5 % 4-seed
   headline by ≥ 3 pp at seed=0.** Updates the unified-driver
   operating point; warrants a 4-seed rerun at the new σ.
2. **No σ < 0.25 cell exceeds 41.7 % at seed=0.** Confirms that
   σ=0.25 is the unified-driver R=1 operating point; closes the
   σ-tuning question for the paper.

#### Where the result lands

- 5 JSONs in `experiments/15_unified_hybrid/results/`.
- Update the headline in `docs/results.md` § "Sweep tables" with
  a new "**Unified driver σ-sweep at K=10 R=1**" subsection.
- Run-log entry in `docs/run_log.md` summarising the outcome.

#### Risk

- The unified driver's K=10 R=1 cell at the Phase 2 σ=0.25 was
  the only single-seed cell that hit 41.7 %; the 4-seed pool
  brought it to 37.5 %. So the σ=0.25 cell is *already* known
  to be seed-noisy at this configuration. A single-seed σ-sweep
  may pick up the same noise. **Mitigation:** if the peak σ
  cell at seed=0 looks promising, allocate one extra seed
  immediately to triangulate.

---

### Spec — A2 `V_φ(s)` value baseline

#### Why

Pure MBD (λ_prior=0, no BC anchor) is **0/60** on can/ph at every
configuration tested. Two hypotheses:

- **H1 (capacity):** the MBD denoiser cannot represent the policy
  manifold without a BC warm start.
- **H2 (variance):** the per-step cost is too high-variance for
  MBD's softmax-over-cost weighting to make progress without an
  anchor.

If H2 is right, subtracting a learned **state-value baseline**
`V_φ(s_t)` from per-step costs reduces the variance of the
softmax weighting without changing its expectation (the
standard RL control-variate trick from REINFORCE/A2C/PPO). If H2
is wrong, `V_φ` should have no effect on pure-MBD.

This is the only experiment in Branch A that distinguishes the
two hypotheses; everything else (A1.4 σ-sweep, A3 M-ablation,
A4 reward shaping) only tweaks the BC-anchored hybrid.

#### What to build

New experiment dir `experiments/16_value_baseline/`:

```
experiments/16_value_baseline/
  README.md
  __init__.py
  model.py          # V_phi: 2-layer MLP, [obs_dim, 256, 256, 1]
  train.py          # TD(0) regression on BC-demo rollouts
  adapter.py        # plug into MBD denoise_loop as a control variate
  run_can.py        # eval cell: K=10 R=0 λ=0 M=64 σ=0.05 + V_phi
  results/
    .gitkeep
```

Key design choices (codified from
[`archive/plan_stronger_mbd_signal.md`](archive/plan_stronger_mbd_signal.md)
Task A2):

- **Training data:** roll out the canonical `[256, 256]` BC
  policy on `can/ph` for ~500-1000 trajectories. Compute per-step
  cumulative future cost (`R(s_t) = sum_{t'>=t} c(s_{t'}, a_{t'})`).
  Train `V_φ` to regress `R(s_t)` from `s_t`. ~5 min on M4 Max.
- **Plug-in point:** in `experiments/04_mbd_baseline/mbd.py`'s
  `denoise_loop`, the per-trajectory cost `cost_i` becomes
  `cost_i - sum_t V_φ(s_{i,t})`. Same softmax over the new
  signal.
- **MBD config:** K=10 R=0 (single-shot, the regime where pure
  MBD currently scores 0/60), λ_prior=0 (no BC anchor), M=64
  (the May-15 M-ablation peak), σ=0.05 (the exp 05 R=0 peak,
  applies to single-shot), n=60.

```bash
# Train (~5 min)
python experiments/16_value_baseline/train.py \
    --bc-checkpoint experiments/03_bc_baseline/results/policy.pt \
    --n-trajs 500 --epochs 100 \
    --output experiments/16_value_baseline/results/V_phi.pt

# Eval pure-MBD + V_phi (~2 h)
python experiments/16_value_baseline/run_can.py \
    --K 10 --n-replans 0 --lambda-prior 0 \
    --sigma-max 0.05 --M 64 \
    --value-baseline experiments/16_value_baseline/results/V_phi.pt \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --n-workers 2 --no-live-view \
    --output experiments/16_value_baseline/results/pure_mbd_with_vphi.json
```

#### Acceptance bar

- **Success:** pure-MBD with `V_φ` ≥ **5 / 60 = 8.3 %**. This is
  the threshold separating "control variate didn't help" from
  "control variate revealed a recoverable signal." Justification:
  5/60 is comfortably above 0/60 by Wilson 95 % CI (lower bound
  on 5/60 is 3.6 %, upper on 0/60 is 7.5 %; the CIs barely
  overlap).
- **Failure:** pure-MBD with `V_φ` still 0/60 (or below 3/60)
  ⇒ accept H1 (model capacity is the bottleneck, not variance);
  drop A2 from the paper and write it up as a negative result.

#### Where the result lands

- New JSON in `experiments/16_value_baseline/results/`.
- New row in `docs/results.md` per-experiment table (exp 16).
- Run-log entry in `docs/run_log.md` summarising H1 vs H2 outcome.
- If positive, also add a row to the "May 15 headline" table on
  main (since "pure MBD works" would be a paper-grade finding).

#### Risk

- **Open Q (from archive plan):** training data scale. 500
  trajectories × ~120 steps = 60 k samples. Might be too few
  for a 2-layer MLP to converge cleanly. If TD residuals
  plateau at high variance, scale to 2000 trajectories
  (still under 1 h M4 Max compute).
- **Open Q:** the BC-policy rollouts might not cover the
  state-action distribution that MBD explores. If `V_φ` is OOD
  during pure-MBD inference, the control variate adds variance
  rather than reducing it. Mitigation: warm-start the rollouts
  with small action noise to get coverage.

---

### Spec — A3 follow-up (b) `λ_softmax ∝ log M` (optional)

#### Why (conditional)

Skip unless A1.4 + A2 land with compute budget left. The May-15
M-ablation showed inverted-U: M=64 = 50.0 % peak, M=256 = 43.3 %
on the descending side. Hypothesis (one-line):

> With fixed `λ_softmax` and growing M, the softmax over
> per-particle costs gets softer (more even weighting across
> particles), so the planner under-commits and degrades.
> Rescaling `λ_softmax ∝ log M` recovers the per-particle
> weighting strength at the M=16 reference point.

#### What to do

One-line change in `experiments/15_unified_hybrid/unified_loop.py`
(or wherever `λ_softmax` is set):

```python
# Before:
lambda_softmax = LAMBDA_SOFTMAX_DEFAULT  # e.g. 0.5

# After:
lambda_softmax = LAMBDA_SOFTMAX_DEFAULT * math.log(M) / math.log(16)
```

Run two cells: K=10 R=1 λ=5 σ=0.05 with M=256 and (optionally)
M=1024, both with the rescaled `λ_softmax`. Compare to the
existing M=256 = 43.3 % cell.

#### Acceptance bar

- **Hypothesis confirmed:** rescaled M=256 ≥ M=64 = 50 % at
  seed=0 ⇒ inverted-U was a tuning artefact; lock in the new
  scaling for the paper.
- **Hypothesis falsified:** rescaled M=256 stays ≤ 45 % ⇒
  inverted-U is real; M=64 stays the operating point.

Either way, ~2 cells × 2 h = ~4 h on M4 Max.

---

### Spec — A4 reward shaping (optional, lowest priority)

Paper-appendix material. Spec is in
[`archive/plan_stronger_mbd_signal.md`](archive/plan_stronger_mbd_signal.md)
Task A4 and unchanged from May 14.

---

### Cross-branch dependencies

- **Branch C (`feat/hybrid-coupling`)** owns the unified driver
  (`experiments/15_unified_hybrid/run_can.py`). A1.4 + A3
  follow-up (b) modify `experiments/15_unified_hybrid/` files.
  Before merging Branch A back to main, confirm Branch C's
  Phase 3 work hasn't conflicted there. See
  [`branch_plan.md` on `feat/hybrid-coupling`](https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/hybrid-coupling/docs/branch_plan.md).
- **`feat/principled-score` / `feat/flow-mjx`** own the flow
  prior. None of the Branch A items depend on `v_theta`; A2's
  `V_φ` is a different model with a different training target.

---

## Branch C — `feat/hybrid-coupling`


This is the per-branch roadmap for **Branch C**.

- Owns: all of `experiments/15_unified_hybrid/` (the canonical
  hybrid driver — supersedes exp 05 / 07 / 10) and
  `scripts/local/` overnight infrastructure.
- May-15 closeout shipped on this branch: Phase 1 unified driver,
  Phase 2 K × R grid (7/9 cells), Job A (R=0 σ-sweep), Job C
  (4-seed K=10 R=1 = 37.5 % CI [31.6, 43.8]), K=20 R=1 extra,
  umbrella flow-prior narrative.
- Cross-branch context (other branches' work + unified status):
  [`plans.md`](plans.md) on `main`.
- Origin spec (pre-May-15, kept for reference):
  [`docs/archive/plan_hybrid_coupling.md`](archive/plan_hybrid_coupling.md).

### Outstanding work

After the May-15 triage, Branch C has **exactly one** undone
item, two SKIPPED items, and the integration window for Branch A
+ flow-prior follow-ups:

1. **Phase 3 — Adaptive `λ(σ)` schedule** (denoise-chain anneal).
   `UNDONE`. ~1-2 days code + ~5 h compute.

Skipped (with explicit user decision 2026-05-15):

- Phase 2 K=20 R=0 + K=20 R=2 (would not change any K-axis
  conclusion; the remaining 7/9 cells already locked the story).
- Phase 4 truncated denoising chain (`--sigma-floor`). Not
  needed right now; revisit if MBD wall-time becomes a
  bottleneck.
- Phase 5 adaptive replanning trigger. R=2 catastrophe is
  documented; an adaptive trigger is a research extension, not
  a paper requirement.

---

### Spec — Phase 3 adaptive `λ(σ)` schedule

#### Why

The unified driver's `denoise_loop`
(`experiments/04_mbd_baseline/mbd.py`) currently uses a **fixed**
`prior_lambda` across all K denoising iterations. The per-step
update is:

```
mu_new = (1 - lr_k) * sum_m w_m * tau_m + lr_k * (mu + sigma_k**2 * score(mu, k))
with  lr_k = prior_lambda * (sigma_k / sigma_max)
```

So `lr_k` already scales linearly with `σ_k`, but the
*coefficient* `prior_lambda` is constant over k. Two empirical
findings motivate making `λ` itself σ-dependent:

1. **R=0 and R=1 want different σ.** Branch A's exp 05 σ-sweep
   peaks at σ=0.01-0.03 for R=1; Job A's unified-driver σ-sweep
   peaks at σ=0.05 for R=0. Same driver, same prior, different
   regime → different optimal noise level → likely different
   optimal *prior weight* at each noise level.
2. **The early-K iterations** (large `σ_k`) and the **late-K
   iterations** (small `σ_k`) plausibly need different prior
   contributions:
     - Early: large σ ⇒ many particles ⇒ planner has strong
       signal, prior should be weaker (so the planner explores).
     - Late: small σ ⇒ particles cluster ⇒ planner under-explores,
       prior should be stronger (to anchor the mean).

Hypothesis: an annealed `λ(σ)` (high at small σ, low at large σ)
beats fixed `λ=5` at the current operating cell.

#### Proposed schedule

A single-parameter scalar schedule keyed on the noise ratio
`r = σ_k / σ_max ∈ (0, 1]`:

```
λ(σ_k) = λ_max - (λ_max - λ_min) * r
```

- `r = 1` (early iteration, σ_k = σ_max) → λ = λ_min (weak prior).
- `r → 0` (late iteration, σ_k → 0)        → λ → λ_max (strong prior).

With `λ_min = 0` and `λ_max = 10`, this recovers a "pure planner
at high noise, pure prior at low noise" schedule. With
`λ_min = λ_max = 5`, this recovers the current fixed-λ=5 baseline.

#### What to modify

Single change in `experiments/04_mbd_baseline/mbd.py`'s
`denoise_loop`:

```python
# Before (current lr_k formula):
lr_k = prior_lambda_f * (sigma_k / sigma_max)

# After (adaptive lambda):
r = sigma_k / sigma_max
lambda_k = prior_lambda_min + (prior_lambda_max - prior_lambda_min) * (1.0 - r)
lr_k = lambda_k * r
```

Plus two new optional kwargs:

```python
def denoise_loop(
    ...,
    prior_lambda: float = 0.0,         # legacy, kept for back-compat
    prior_lambda_min: float | None = None,
    prior_lambda_max: float | None = None,
) -> tuple[np.ndarray, dict]:
    """If either lambda_min/max is None, fall back to fixed
    prior_lambda (the May-15 default). Otherwise use the
    linearly-annealed schedule."""
```

And in `experiments/15_unified_hybrid/run_can.py`, expose
`--lambda-prior-min` and `--lambda-prior-max` CLI flags. If only
`--lambda-prior` is passed, the legacy fixed-λ path is used.

#### Cells to run

Three cells on the unified driver, K=10 R=1, σ=0.25 M=16, seed=0,
n=60 each, to bracket the hypothesis:

| Cell                                       | λ_min | λ_max | Expected outcome              |
|--------------------------------------------|------:|------:|-------------------------------|
| anneal_lo (mild anneal)                   |   2   |   8   | should match fixed-λ=5 within noise |
| anneal_hi (strong anneal)                 |   0   |  10   | tests the "pure planner early, pure prior late" hypothesis |
| anneal_inv (inverted: strong prior early) |  10   |   0   | controls — should be WORSE than fixed-λ=5 |

```bash
# anneal_lo
python experiments/15_unified_hybrid/run_can.py \
    --K 10 --n-replans 1 \
    --lambda-prior-min 2 --lambda-prior-max 8 \
    --sigma-max 0.25 --M 16 --prior bc \
    --n-demos 20 --n-rollouts 3 --seed 0 \
    --n-workers 2 --no-live-view \
    --output experiments/15_unified_hybrid/results/unified_bc_anneal_lo_K-10_R-1.json

# anneal_hi  (swap lambda values)
# anneal_inv (lambda_min=10, lambda_max=0)
```

Optionally, if `anneal_hi` beats fixed-λ=5 by ≥ 3 pp at seed=0,
extend to a small λ_min × λ_max grid (e.g., 3 × 3 = 9 cells) to
locate the peak.

#### Compute

~2 h per cell × 3 cells = ~6 h serial. With 2 workers parallel:
~3 h wall. Optional follow-up grid: +6 h.

#### Acceptance bar

- **Hypothesis confirmed:** `anneal_hi` (λ_min=0, λ_max=10) at
  seed=0 ≥ 41.7 % + 3 pp = **≥ 45 %**. This would put the
  unified-driver headline above the M=64 single-seed peak
  (50.0 %) at the M=16 default config — a paper-grade finding.
- **No effect:** all three anneal cells within ±3 pp of fixed-λ=5
  ⇒ adaptive schedule does not help at this operating point;
  Phase 3 closes as negative result; document and move on.
- **Inverted (anneal_inv beats anneal_hi):** unexpected; would
  flip the proposed schedule direction. Investigate.

#### Where the result lands

- 3 JSONs in `experiments/15_unified_hybrid/results/`.
- New row in `docs/results.md` § "Sweep tables" — "**Adaptive
  `λ(σ)` ablation (Phase 3)**".
- Run-log entry in `docs/run_log.md` summarising the schedule
  hypothesis test.
- If positive, add to "May 15 headline" table on `main` (and to
  this branch's narrative).

#### Risk

- The proposed schedule is **linear in r**. A more flexible
  parameterisation (e.g., cosine, or exponential decay) might
  fit a real signal better. If the 3-cell bracket is
  inconclusive, escalate to a 2-parameter (`λ_min`, `λ_max`)
  grid; if that's also flat, conclude there is no σ-dependent
  λ regime worth exposing in the paper.
- The `denoise_loop` change touches the algorithm core
  (`experiments/04_mbd_baseline/mbd.py`). The Phase 1
  regression test (`experiments/15_unified_hybrid/regression_test.py`)
  must still pass with the legacy `--lambda-prior` flag → the
  back-compat branch (lambda_min/max = None) is critical and
  the regression test is the gate.

---

### Cross-branch dependencies

- **Branch A (`feat/stronger-mbd-signal`)** plans to add new
  σ-sweep cells (A1.4) and an `λ_softmax ∝ log M` change
  (A3 follow-up b) to the unified driver. Both are read-only
  consumers of `denoise_loop` — they don't change the algorithm
  core, just the CLI flags around it. Phase 3's `denoise_loop`
  modification therefore takes precedence; A1.4 + A3-(b) can
  rebase on top once Phase 3 lands. See
  [`branch_plan.md` on `feat/stronger-mbd-signal`](https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/stronger-mbd-signal/docs/branch_plan.md).
- **`feat/principled-score` / `feat/flow-mjx`** own the flow
  prior. Phase 3's σ-dependent λ schedule is prior-agnostic and
  applies to both BC and flow priors without modification.
