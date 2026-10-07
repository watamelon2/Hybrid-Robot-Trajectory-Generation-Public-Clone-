# Run log

Append-only diary of overnight / multi-VM execution sessions.
Newest at the top.

**See also:** [`results.md`](results.md) for headline numbers,
[`experiments/README.md`](experiments/README.md) for per-experiment
navigational docs (lifecycle, reproduce commands, cross-references).

---

## 2026-06-01 — Phase-D closes (two GREEN cells) + Pareto-proxy branch merge

Two branches landed on `main` overnight via `--no-ff` merges:

- **`feat/dp-phase-c`** (`8e97ca3`, 01:38 PDT) -- closes the
  multi-week DP-prior K-sweep arc.  Lands the Phase-C wrapping-bug
  fix (`unified_loop.py:478` chunk-boundary `env.reset` now gated
  on `params.K > 0`) and the full Phase-D 2-D grid
  (ckpt × λ × K) on Can/`ph` with the v3b flow-matching DP prior.
  See [`experiments/17_dp_hybrid.md`](experiments/17_dp_hybrid.md)
  and the canonical
  [`experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md`](../experiments/17_dp_hybrid/results/phase_d_2026_05_30/CONCLUSION.md).
  Merge resolved a conflict in `unified_loop.py` by combining the
  dp branch's worker-pool reuse pattern (build pool once before the
  chunk loop + `update_init_state` per chunk) with `main`'s
  perturbations-to-MBD-workers thread (commit `016d603`); both
  improvements survive in the merged version.

- **`feat/adapt-pareto-analysis`** (`aaefba2`, 01:39 PDT) -- closes
  the post-T2 analysis arc.  Lands the summary-level (bias², variance)
  proxy script and the asymmetric-robustness theory doc.  Per-step
  action-level measurement is scaffolded but deferred to a journal
  version.  See
  [`docs/results_writeups/pareto_analysis.md`](results_writeups/pareto_analysis.md).

### Phase-D headline numbers (Can/ph, v3b flow-matching DP prior, this Mac)

The full grid is in
[`q2q3_AGGREGATE.md`](../experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.md).
Two GREEN cells, both at the Q1 epoch-curve peak (ep0149):

| cell | n | rate (Wilson 95%) | lift vs Q1 K=10 λ=0.5 baseline |
|---|---:|---:|---:|
| **(ep0149, K=10, λ=2.0)**  | 30 | **86.7 %** [70.3, 92.7] | **+13.4 pp**  (cheapest path) |
| (ep0149, K=20, λ=0.5)      | 30 | 83.3 % [66.4, 92.7]     | +10.0 pp  (K-knob corroboration) |

YELLOW everywhere else (Q2 λ-sweep at the under-trained ep0049
stays at 0.42 - 0.55; Q3 K-sweep at the same ckpt stays at
0.53 - 0.60; Phase E + F at the strongly-trained ep0199 stay
at 0.70 - 0.73 under every (K, λ) combination).  The ep0149
λ-sweep is uniquely monotone in λ (0.60 → 0.73 → 0.87); no other
row in the grid shows λ-monotonicity.

**Paper headline**: MBD's value on Can/`ph` is operating-point
dependent.  At the Q1-peak ckpt (moderately-trained), either a
stronger bridge (λ=2.0) or a larger refinement budget (K=20)
delivers a +10-13 pp lift, with the cheaper λ-knob path winning.
Neither knob lifts the under-trained or strongly-trained
extremes.  Mechanism: MBD adds value only when the prior's score
field is coherent AND there is meaningful headroom above the
prior's standalone rate.

### Compute story

  - Phase-D total wall on this Mac: **~37 h** unattended across
    two sessions (Q1 epoch curve 21.3 h + chained Q2/Q3/E/F 12.7 h
    + Q1 wave 2 catch-up).  No GCP arm needed; the GCP run was
    killed mid-session after a worker-pool patch left orphan
    processes.  Disjoint seed bands (100k Q1, 200k Q2/Q3, 300k
    E/F) made the local + (failed) GCP pooling safe.
  - Chain orchestrator at
    `scripts/local/chain_phase_d_E_F.sh` auto-fired Phase E and F
    after Q2/Q3 closed; the K=40 cell at ep0049 was killed mid-run
    (~12 / 25 rollouts) to unblock Phase F from waiting on the
    slow K=40 bottleneck.

### Deferred to a follow-up branch

K = 40 cell at ep0049 (tests whether a much larger refinement
budget recovers GREEN at the under-trained baseline).  Currently
**paused via SIGSTOP at 2 / 25 rollouts** (PIDs `57230, 57234,
58838, 58839` in state `T`, screen `phase_d_K40_deferred`).
Resume command: `kill -CONT 57230 57234 58838 58839` (or
`pkill -CONT -f "run_can.*K 40.*ep0049"` if PIDs change).  After
K40 completes, the relaunch script
(`scripts/local/relaunch_E_K40_after_chain.sh`) auto-re-runs
`scripts/local/aggregate_q2q3.sh` to add the K=40 row.  Follow-up
work lands on a new branch per the user's
"no direct-to-main after merge" hygiene rule.

---

## 2026-05-25 — Branch consolidation onto main; theory branch; per-experiment docs

Four feature branches merged into `main` today (all `--no-ff`):

- **`feat/stronger-mbd-signal`** (`25ca457`, 15:17 PDT) — brings the
  V_φ state-value control variate scaffolding (exp 16) onto main
  via the unified driver's `--value-baseline-ckpt` flag. Eval cell
  scoped (pure-MBD λ=0 σ=0.05 M=64 + V_φ; acceptance bar ≥ 5/60 to
  support the variance-bottleneck hypothesis H2 over the
  capacity-bottleneck H1); not yet executed at headline protocol.
  See [`experiments/16_value_baseline.md`](experiments/16_value_baseline.md).
- **`feat/hybrid-coupling`** (`595aadf`, 15:13 PDT) — already mostly
  on main from the May-15 closeout; this brings the Phase 3-5 stub
  flags forward (no semantic change).
- **`feat/zero-train-priors`** (`2daa166`, 15:26 PDT) — lands
  **Slates D – J** onto main: Slate D (May 18, λ=1 peak at 75 %),
  Slate E (May 19, **73.8 % pooled headline**), Slate F (May 20,
  mode-selection ablation + Square cross-task fails), Slate G/G-aux
  (May 21, Lift 100 % + Square mech-limit triple-confirmed),
  Slate H (May 21, tool_hang upstream-bug postmortem), Slate I
  (May 22, contrastive obs prior — paused mid-cell-1), and the
  Slate J pre-stage (tool_hang + transport with dense-reward shim
  `env_dense_rewards.py`, commit `18af830`). The canonical 73.8 %
  pooled headline is now strictly on `main`.
- **`feat/theory`** (`b8de97d`, 15:26 PDT, rebased to drop WIP
  commits) — three new theory docs in [`docs/theory/`](theory/):
  - [`01_soft_control.md`](theory/01_soft_control.md) — KL-regularised
    soft-control derivation of the score-bridge.
  - [`02_snis_bound.md`](theory/02_snis_bound.md) — finite-sample
    SNIS variance bound for the MBD score.
  - [`03_pareto_bias_variance.md`](theory/03_pareto_bias_variance.md) —
    Pareto bias-variance dominance theorem (hybrid > each endpoint).

Per-experiment **navigational docs** added under
[`docs/experiments/`](experiments/) on `feat/experiment-docs`
(commit `4f3b4ae`, 17 files, ~2100 lines). One doc per experiment
(01 – 16) plus a master index. Each doc carries the current
lifecycle status, the most recent headline number with source,
the canonical reproduce command, cross-references to predecessor
/ successor / companion experiments, and caveats / WIP notes.

`feat/flow-v3-karras` remains **off** main. Slate K's
parity-ceiling diagnostic (May 25 morning): at
`--score-mode=learned`, MBD's softmax weighting destroys strong
learned priors; the asymmetric `--score-mode=pull` is the fix.
Promotion to main is gated on beating retrieval-prior 73.8 % or
producing a tighter mechanism story than the negative result.

## 2026-05-22 — Slate I paused mid-cell; Slate J queued

**Slate I** (C1 contrastive obs-embedding prior on Square + Can)
launched 03:40 PDT and was paused at 08:22 PDT mid-cell-1 (rollout
6/60 of `I1 demo_1nn_contrastive λ=0`). The earlier 21:51 PDT
auto-fire from the watcher had failed silently (screen registered
Dead with no `run.log`); manual relaunch at 08:17 PDT ran into a
user pulse. Detached `screen` `slate_i`; all 12 processes frozen
(`T+`). Resume with `pkill -CONT -f run_can.py && pkill -CONT -P
<parent>`. ETA on resume ≈ 7.7 h wall.

Slate I goal: test whether a learned obs encoder (trained
contrastively on demo trajectories,
`train_contrastive_encoder.py`) unlocks Square where the five
orthogonal axes tested in Slates F – G all failed. 13 cells, branch
`feat/zero-train-priors` HEAD `18af830` at launch.

**Slate J** pre-staged at Slate H postmortem time (May 21 19:30
PDT): `tool_hang` + `transport` recovery with the dense-reward
shim. 4 cells, ~10 – 12 h. Manual fire only; launcher refuses
if `slate_h` or `slate_i` screens are still alive. Smoke-verified
before stage:

- Tool_hang demo-replay (post-fix): max +6.71, mean +2.58,
  251 / 681 steps above the +5 subgoal bar (pre-fix: identically
  zero across the whole demo).
- 1-demo × 1-rollout integration smokes through the full driver
  at K=4 M=8 H=80: tool_hang +21.60, transport +1.22, can +3.40
  (no wrap; pass-through).

Both `tool_hang` and `transport` ship with sparse-only `reward()`
upstream (`tool_hang.py:217`, `two_arm_transport.py:reward()`); the
dense-reward shim is project-side and active only when the env
name matches.

## 2026-05-21 — Slate G + G-aux + H — Lift hits 100 %, Square mech-limit triple-confirmed, tool_hang exposes upstream bug

Twenty-six cells across Slates G + G-aux + H, all through the
unified driver (exp 15) at the canonical operating point
(K=10 R=1 M=16 H=120, σ=0.25, seed=0, 60 rollouts/cell unless
noted). Combined runtime ≈ 9 h across two parallel screens (G on
n-workers=10, G-aux on n-workers=5, load avg peaked 36 / 16 with
no thrashing).

**G1 — `demo_1nn` λ-sweep on `lift/ph`** (the first non-Can
headline task):

| λ | succ | rate | mean_r |
|---|------|------|--------|
| 0.0 | 33/60 | 55.0 % | +54.29 |
| 0.5 | 44/60 | 73.3 % | +69.80 |
| **1.0** | **60/60** | **100.0 %** | +84.55 |
| 2.0 | 60/60 | 100.0 % | +83.77 |
| 5.0 | 59/60 | 98.3 % | +82.58 |
| 10.0 | 59/60 | 98.3 % | +82.91 |

First non-Can task at the canonical operating point to hit the
strong (≥ 90 %) gate. Same qualitative shape as Can's D2: warm-
start-only already clears the weak gate, bridge lifts to ceiling
at λ=1, over-anchoring at λ=10 still holds. Task scope of the
mechanism is now **pick-place with sub-mm tolerance** (Can, Lift)
but **not contact-rich insertion** (Square).

**G2 — `demo_softnn` β-sweep on Can**: 45.0 % @ β=0 (uniform k-NN),
61.7 % @ β=5; the continuous bias-variance curve does *not* find a
sweet spot above the hard-1-NN peak (75.0 %). Best β=5 sits 13 pp
below hard-1-NN.

**G3 — λ-schedule on Can**: all three schedules (linear,
inverse_linear, cosine) underperform constant λ=1. Inverse_linear
catastrophically 18.3 % (vs constant 75 %). **A2 is filed as
harmful** at the canonical operating point.

**G4 + AUX1 + AUX2 — Square mechanism-limit triple-confirmed**:

- A1 action-history obs (G4): 5.0 / 3.3 / 1.7 % @ N = 1 / 5 / 10
  (longer histories *hurt*).
- σ-tail high-res (AUX1): peak shifted from F2's σ=0.25 (3.3 %) to
  σ=0.30 (5/60 = 8.3 %). Modest 2.5× over F2 but still below the
  F1 weak gate (15 %).
- Soft-NN β-sweep (AUX2): flat 3.3 % across β ∈ {0.5, 1, 2, 5}.
  Mode-mixing recovers hard-1-NN exactly — retrieval identity is
  not the Square bottleneck.

Five orthogonal knobs at the F1 sweet spot now show no signal
above 8 % on Square. **The structural change required is not in any
of {prior identity, bridge λ, σ, MBD compute, obs vocabulary}.**

**H1 — tool_hang at λ=1 σ=0.25 H=300, 20 × 2 = 40 rollouts**:
0 / 40 in 49 min. Falsified the run, traced to upstream:
robosuite-1.5's `tool_hang.py:reward()` is a bare sparse
`1.0 if check_success else 0.0` despite `reward_shaping=True` being
honoured at `__init__`. Same upstream gap on `TwoArmTransport`
(`reward()` prints "No dense reward currently implemented" and
silently overrides `self.reward_shaping = False`). Project-side
dense-reward shim (commit `18af830`,
`experiments/15_unified_hybrid/env_dense_rewards.py`) wraps
`env.step` to add staged distance + subgoal-indicator shaping.

**H3 + H4 — A1 controls on Can and Lift**:

- Can A1: 45 / 41 / 40 (75.0 / 68.3 / 66.7 %) @ N = 1 / 5 / 10.
  Benign at N=1, monotone decay at N ≥ 5.
- Lift A1: 60 / 60 / 59 (100 / 100 / 98.3 %) @ N = 1 / 5 / 10.
  Ceiling-invariant.

Confirms A1 is task-specific; Square's A1 collapse was the
mechanism, not the augmentation. Strengthens the "Square needs
structural change in retrieval/cost, not in obs vocabulary"
reading from Slate G.

## 2026-05-20 — Slate F — Mode-selection ablation; Square cross-task fails (mechanism-limited)

Thirteen cells through the unified driver organised in two tracks,
~5.5 h wall.

**F3 (Can warm-start variants)** at the canonical operating point
(K=10 R=1 M=16 H=120, λ=1, σ=0.25, seed=0):

| prior | succ | rate | Δ vs `demo_1nn` |
|-------|------|------|------------------|
| `demo_1nn` (Slate E seed=0) | 45/60 | 75.0 % | (baseline) |
| `demo_1nn_proprio` | 42/60 | 70.0 % | −5.0 pp |
| `demo_lwr_k5` | 41/60 | 68.3 % | −6.7 pp |
| `demo_1nn_object` | 41/60 | 68.3 % | −6.7 pp |
| `demo_topk_random_k5` | 17/60 | **28.3 %** | **−46.7 pp** |

The 40-pp drop from LWR (68.3 %) → top-k-random (28.3 %)
**isolates mode selection** as the operative property of the
warm-start prior. The 1-NN benefit is not "use a close trajectory"
but rather "lock onto one specific close trajectory and let the
bridge stabilise it". Substantiates the bias-variance framing
predicted in [`theory_post_slate_e.md`](theory_post_slate_e.md) § 4.

**F1 + F2 + budget retry (Square cross-task)**: F1 λ-sweep peak at
λ ∈ {0.5, 1} = 3.3 %, far below the BRIEF weak gate (15 %). Mean
reward rises monotonically (17.6 → ~28 across the λ-axis) and
plateaus — bridge mechanism partially transfers, but no rollout
clears reward 50 across 360 evaluations. Budget retry at 3× MBD
compute (R=3 M=24): same 3.3 %, mean reward actually *drops* to
+20.1, max reward up modestly to +52.0 with one rollout crossing
50. **Mechanism-limited, not budget-limited.**

Story-B's "task-agnostic mechanism" claim narrows from
"task-agnostic" to "task-agnostic within pick-place regimes";
contact-rich insertion needs a different prior or bridge schedule.

Full reading in [`slate_f_final_reading.md`](slate_f_final_reading.md).

## 2026-05-19 — Slate E — 73.8 % pooled headline; bridge tuning is prior-specific

Eleven cells, 4 h 53 min wall. **The canonical headline of the
project.**

**E1 — 4-seed pool at the new λ=1 cell**:

| seed | succ | rate | source |
|------|------|------|--------|
| 0 | 45/60 | 75.0 % | slate_d D2 (existing) |
| 1 | 46/60 | 76.7 % | E1 |
| 2 | 44/60 | 73.3 % | E1 |
| 3 | 42/60 | 70.0 % | E1 |
| **POOL** | **177/240** | **73.8 %** | **CI95 [67.8, 78.9]** |

Per-seed range 70.0 – 76.7 %, std ~2.7 pp. The CI lower bound
(67.8 %) is above the bridge-dominated plateau (~63 %); Story B's
"peak strictly above plateau" prediction passes two-sample z-test
at p = 0.023. Versus BC's 4-seed pool at the same K=10 R=1 σ=0.25
cell (90/240 = 37.5 % CI [31.6, 43.8]):
**+36.3 pp pooled, CIs separated by ~24 pp**.

**E2 — BC λ-curve at σ=0.25 seed=0** (companion to D2):

| λ | BC rate | 1-NN rate | gap |
|---|---------|-----------|------|
| 0 | 11.7 % | 11.7 % | +0.0 pp |
| 0.5 | 21.7 % | 31.7 % | +10.0 |
| **1** | 28.3 % | **75.0 %** | **+46.7** |
| 2 | 36.7 % | 65.0 % | +28.3 |
| **5** | **41.7 %** | 63.3 % | +21.7 |
| 10 | 38.3 % | 65.0 % | +26.7 |
| 20 | 28.3 % | 63.3 % | +35.0 |

Clean inverted-U with BC peak at λ=5 = 41.7 % vs 1-NN peak at
λ=1 = 75.0 %. **Bridge tuning is strongly prior-specific.** Curves
agree exactly at λ=0 (no bridge, just warm-start provenance).

**E3 — σ-tail at λ=1 seed=0**: σ=0.05 65.0 %, σ=0.25 75.0 %,
σ=0.40 53.3 %, σ=1.00 23.3 %. **σ-invariance is λ-specific** — at
λ=5 the cell is σ-flat across 20× range (63 – 68 %); at λ=1 it is
σ-peaked at 0.25 and collapses at σ=1.00. The Slate D σ-invariance
claim must be quoted at λ=5, not λ=1.

Slate E replaces the May-17 64.2 % pool as the headline. Single-
seed peak 75.0 % at the same cell (Slate D D2, May 18) is
substantiated as a 73.8 % pooled headline.

## 2026-05-18 — Slate D — λ-sweep finds new peak; σ-invariance fully substantiated

Eleven cells, 5 h 24 min wall.

**D1 — BC λ=0 4-seed pool**: 36 / 240 = 15.0 % CI [11.0, 20.1]. The
May-17 single-seed digit agreement "BC and 1-NN both 11.7 % at λ=0"
was lucky; the 4-seed picture is BC ≈ 15 % vs 1-NN ≈ 12 %. The
"BC and 1-NN warm-starts are indistinguishable at λ=0" claim
survives at the CI level (95 % CIs overlap by ~5 pp) but with
softening.

**D2 — `demo_1nn` λ-sweep at σ=0.25 seed=0**:

| λ | succ | rate |
|---|------|------|
| 0 | 7/60 | 11.7 % |
| 0.5 | 19/60 | 31.7 % |
| **1** | **45/60** | **75.0 %** |
| 2 | 39/60 | 65.0 % |
| 5 | 38/60 | 63.3 % |
| 10 | 39/60 | 65.0 % |
| 20 | 38/60 | 63.3 % |

**Sharp peak at λ=1 (75.0 %), broad plateau at 63 – 65 % for
λ ∈ [2, 20]**. Beats the May-17 single-seed at λ=5 (63.3 %) by
+11.7 pp at the same seed=0. NEW PROJECT PEAK (subsequently
substantiated as a 4-seed pool in Slate E).

Mechanism: when the per-step pull target is in-distribution, a
*weaker* score-bridge gives the MBD softmax-across-M room to mix
MBD's own cost-aware exploration with the retrieved-demo signal.
At λ ≥ 2 the bridge becomes dominant enough that the cell
collapses to the warm-start trajectory, capping at the plateau.

**D3 — `demo_1nn` σ-tail at λ=5 seed=0**: 8 cells across
σ ∈ [0.05, 1.00] all land in [63.3, 68.3 %], slight upward trend
with σ. **No σ-cliff in the tested range** — σ-invariance across
a 20× range. Contrast with BC σ-sweep: BC drops from 40 % @ σ=0.03
to 3.3 % @ σ=0.25 to 0 % @ σ=0.40. At σ=1.00 the gap is
**68.3 % vs 0 %**.

## 2026-05-17 — Zero-train retrieval priors land; first pooled CI above BC

Two slate-blocks today.

**Morning slate (zero-train priors first probe)**: First
measurement of `--prior=demo_1nn` (1-NN retrieval from the demo
bank, no training). Seed=0 single-seed: 38 / 60 = **63.3 %** at
λ=5 σ=0.25 K=10 R=1 — vs the May-15 BC pool's 37.5 %. NEW PEAK at
the time. `demo_knn k=5` = 16.7 %, `demo_random` = 0 %,
`demo_mean` = 0 % — the prior identity matters; only the in-
distribution 1-NN target works.

**Narrative-followup slate (14 cells, 6 h 16 min wall)**:

- **Slate 1** — `demo_1nn` 4-seed pool at λ ∈ {0, 5}:
  λ=0 pool 28 / 240 = 11.7 % CI [8.2, 16.3] (≈ BC at λ=0);
  **λ=5 pool 154 / 240 = 64.2 % CI [57.9, 70.0]**. CIs disjoint
  from BC pool [31.6, 43.8] — **+26.7 pp gap is statistically
  decisive**.
- **Slate 2** — k-sweep for `demo_knn` at λ=5: monotone collapse
  with sharp knee at k=2 → 3 (k=1 63.3 %, k=2 35.0 %, k=3 16.7 %,
  k=5 16.7 %, k=10 8.3 %, k=20 10.0 %). **Averaging amplifies
  OOD**; each averaging step from the nearest demo costs 17 – 28
  pp until the floor.
- **Slate 3** — σ-sweep for `demo_1nn` at λ=5: **astonishing
  σ-invariance** — flat 63 – 67 % across σ ∈ {0.05, 0.10, 0.15,
  0.25, 0.40}. BC σ-sweep crashes from 40 % @ σ=0.03 to 3.3 %
  @ σ=0.25 to 0 % @ σ=0.40. The 1-NN + bridge method **removes
  σ_max from the hyperparameter tuning surface** at λ=5.
  *Caveat: subsequently shown in Slate E E3 to be λ-specific —
  the σ-invariance holds at λ=5 but not at the new λ=1 peak.*

All three falsifiers from the BRIEF predictions PASSED. Story B
(the score-bridge needs an in-distribution per-step pull target)
holds under every test.

## 2026-05-16 (PM) — Warm-start ablation through the unified driver

Four cells at the May-15 headline config (K=10 R=1 σ_max=0.25
M=16 H=120, seed=0):

| Cell | warm-start | λ | succ | rate |
|------|------------|---|------|------|
| C0 | BC (MLP) | 5 (BC bridge) | 90 / 240 | 37.5 % (4-seed, existing) |
| C1 | BC (MLP) | 0 | 7 / 60 | 11.7 % |
| C3 | flow v2 | 0 | 0 / 60 | 0.0 % |
| C4 | flow v2 | 5 (flow bridge) | 0 / 60 | 0.0 % |

Decomposes the May-15 BC headline into warm-start vs score-bridge
contributions:

1. **The BC score-bridge is the dominant lift, not the warm-start.**
   C0 − C1 = +25.8 pp; the warm-start alone gives only +11.7 pp
   over pure MBD.
2. **Flow v2 is useless at the headline σ.** C3 and C4 both
   collapse to 0/60, indistinguishable from no-warm-start MBD.
   The flow architecture only wins where it was tuned (σ=0.02 R=0,
   4 – 10/60 in the noise envelope).
3. **The flow score-bridge does not rescue the flow warm-start** —
   consistent with Phase 7's quadratic-denominator bridge sweep
   at σ=0.02. The score-bridge contribution lifting BC from
   11.7 % → 37.5 % is specific to BC's deterministic per-step pull
   target.

## 2026-05-16 (overnight) — v_theta v2 σ-preservation + bridge sweeps

Phase 5 σ-preservation sweep: single-seed peak at **σ=0.02 λ=0
K=10 M=16 = 10/60 = 16.7 %**
(`experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/phase5_sigma_preservation/sigma_0.02/`).
Phase 0 prior-only (no MBD): 8.3 %. Phase 7 quadratic-denominator
bridge sweep at σ=0.02: bridge adds no signal beyond what σ=0.02
λ=0 already provides.

Phase 9 seeded replanning ablation (with the new
`--deterministic-prior-warmstart` flag): R=0 = 4/60 = 6.7 %,
R=1 = 0/60. **MPC replanning hurts the flow prior** — the second
chunk's warm-start gets re-sampled on a state the flow's training
distribution doesn't cover well.

Verdict at end of v2 sweep: better than v1 (which was 0/60) but
well below the (May 17 onward) retrieval-prior numbers. Documented
in `experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16/v2_sweep_report.md`.

---

## 2026-05-15 (evening) — Closeout narrative & methodology

This entry collects the long-form discussion of the May 15
closeout. The headline tables live in [`results.md`](results.md);
this entry covers the *why*: what decisions we made, what
mechanisms we hypothesise, and what each cell means in context.

### One-paragraph narrative

The mid-May plan was to (i) replace exp 05's heuristic warm-start
score with the principled exp 07 score-bridge formulation, then
(ii) measure a K × n_replans grid against the new BC. Three feature
branches ran in parallel — Branch A re-tuned σ-max on the exp 05
driver and ran the M-ablation, Branch B trained the flow-matching
prior `v_theta`, Branch C consolidated exp 05 + 07 + 10 into a
single unified driver. Branch C landed first; the unified-driver
headline pushed from 33.3 % (exp 07 single-shot, May 14) to
**41.7 % at K=10 R=1 λ=5 σ=0.25 M=16** (single seed=0; 4-seed
pool tightens to 37.5 %, CI [31.6, 43.8]). The cross-branch peak
of **50.0 % at K=10 R=1 σ=0.05 M=64** sits on the exp 05 driver
(Branch A's M-ablation, single seed). Branch B's `v_theta` debug
showed the "1000× magnitude bug" was caused by OOD `s0=zeros`
conditioning, not a broken model — but the residual 2× off-scale
is still catastrophic in practice (flow-prior cell at n=60 = 0/60).

### K=10 R=0 anomaly in the Phase 2 grid

The K=10 R=0 cell in the Phase 2 grid reads 13/60 = 21.7 %, which
is anomalous against three independent replications at the same
configuration:

- Phase 1 regression (unified, σ=0.25): 19 / 60 = 31.7 %.
- σ-sweep at σ=0.01 (unified): 18 / 60 = 30.0 %.
- σ-sweep at σ=0.03 (unified): 18 / 60 = 30.0 %.
- σ-sweep at σ=0.05 (unified): 23 / 60 = 38.3 %.

So the **honest K=10 R=0 λ=5 number is ~30 %, not 21.7 %.** The
Phase 2 grid cell appears to have caught a bad-seed fluctuation
or a config drift; the per-σ replications converge on 30-38 %.
This does not change the K=10 R=1 = 41.7 % headline, but it does
mean the "+20 pp from MPC chunking" claim should be softened to
"+8 to +12 pp from MPC chunking" using the σ-replicated R=0
baseline.

### R=2 catastrophe — λ-independent OOD hypothesis

Every R=2 cell tested has been 0/60. This is **λ-independent**:
K=10 R=2 was tested at six λ values {0, 1, 2, 5, 10, 20} in
Stage 3 (then skipped to free CPU), and the canonical Phase 2
cell at λ=5 also reads 0/60. Hypothesis: the third planning chunk
hits a state outside the prior's training distribution (the BC
was trained on H=120 horizons; with H/3 ≈ 40 steps per chunk,
the third chunk is consistently at t ≥ 80 which is past most
demos' grasp event). Diagnosis pending — for now R=2 is
empirically off-limits.

### σ=0.25 replication discrepancy (Phase 2 cell vs σ-sweep cell)

In Job A (unified driver σ-sweep at K=10 R=0 λ=5), σ=0.25 appears
twice with different results despite an identical config: 13/60
in the Phase 2 cell vs 19/60 in the σ-sweep cell. Both used
`--n-workers=2`, same seed=0, same M=16. Open question; recorded
here for honesty. The per-σ replications at 0.01-0.10 land in
the 30-38 % range and are the more credible baseline.

### σ-tuning take-aways

Two independent σ-sweeps converged on the same operating *regime*:
the score-bridge sigma_max needs to be **at most 0.10**, with the
optimum near 0.01-0.03 for R=1 and near 0.05 for R=0.

1. **σ ≥ 0.25 collapses the score-bridge.** From σ=0.10 to σ=0.40
   the success rate falls 26.7 → 0 % on the exp 05 driver. The
   strong-reach counts stay high (63-73 %) even at σ ≥ 0.25, so
   the failure mode is **not** "can't reach" — it is "noise
   destroys grasp/lift/place after a successful reach". The
   score-bridge prior cannot overcome that level of additive
   Gaussian noise.
2. **The legacy default σ=0.25 was badly tuned** (-18.4 pp vs BC
   alone on exp 05; 2/60 = 3.3 % vs BC's 13/60 = 21.7 %). This
   is the σ value baked into `experiments/05_hybrid_method/run_can.py`'s
   defaults and used by every published exp 05 / 07 / 10 result.
   The previous "hybrid barely matches BC" narrative is almost
   certainly a tuning artefact.
3. **The two drivers peak at *different* σ.** R=1 wants
   σ=0.01-0.03; R=0 wants σ=0.05. Intuition: with MPC chunking
   the planner gets a "second chance" so it can afford a noisier
   prior signal; in single-shot the prior has to be sharp.
4. σ=0.25 in the Phase 2 grid was a tolerable choice for R=1
   (gave 41.7 %, lost ~2 pp vs σ=0.03) but a bad choice for R=0
   (gave 21.7 % vs 30-38 % at σ ≤ 0.10).

### M-ablation: the inverted-U finding

Holding K=10 R=1 λ=5 σ=0.05 fixed and varying M:

- M=16: 19/60 = 31.7 % (the σ-sweep cell)
- **M=64: 30/60 = 50.0 %** (+18.3 pp; peak)
- M=256: 26/60 = 43.3 % (loses 6.7 pp vs M=64)

**Increasing M from 16 to 64 gives +18.3 pp absolute. Increasing
M further from 64 to 256 LOSES 6.7 pp.** M=64 is the genuine
sweet spot, not a lower bound on a monotone curve. The May 14
plan version of this section framed the M-axis as monotone and
used a lower-bound argument; the M=256 result kills that framing.

Plausible mechanism: at M=16 the softmax over costs is dominated
by 1-2 lucky particles so the update is noisy; at M=64 the cost
surface gets enough samples to resolve the prior's modal
directions; at M=256 the softmax over a much wider range of
perturbations starts to over-explore — with a fixed
`lambda_softmax`, more particles means a softer weighting and the
planner stops committing decisively. Suggests a follow-up
`lambda_softmax`-vs-M sweep at the peak σ=0.05 cell to see
whether a temperature-scaled softmax recovers the M=256
advantage. See the Branch A plan (`docs/branch_plan.md` on
`feat/stronger-mbd-signal`) for the corrected A3 follow-up.

### Seed robustness — Job B and Job C 4-seed pools

Four-seed reruns at two of the strongest cells. **Both pools land
at ~38 % despite different drivers and different σ values — strong
evidence that the +1.7 pp single-seed gap between them was seed
noise, not a real σ or driver effect at R=1.**

Job C — Unified driver, K=10 R=1 λ=5 σ=0.25 M=16:
- Per-seed: 25/60 (0), 25/60 (1), 19/60 (2), 21/60 (3)
- Pooled: 90 / 240 = **37.5 %**, Wilson 95 % CI **[31.6, 43.8]**
- Per-seed mean = 37.5 %, std = 5.0 pp, range = [31.7, 41.7] %

Job B — Exp 05 driver, K=10 R=1 λ=5 σ=0.03 M=16:
- Per-seed: 24/60 (0), 22/60 (1), 20/60 (2), 25/60 (3)
- Pooled: 91 / 240 = **37.9 %**
- Per-seed mean = 37.9 %, std = 3.7 pp, range = [33.3, 41.7] %

**The single-seed 41.7 % headline is +4.2 pp above the Job C
pooled mean and at the upper edge of the 4-seed range; the
honest headline at this cell is 37.5 % ± ~6 pp.** The BC baseline
at 21.7 % is firmly outside the lower CI bound, so the
hybrid-vs-BC delta survives. Against a Wilson CI on the BC
baseline of [12.7 %, 33.0 %] (n=60, single seed), the
hybrid-vs-BC gap at the unified K=10 R=1 cell is **+15.8 pp at
CI lower bounds** — still a defensible improvement, though less
dramatic than the single-seed +20 pp would suggest.

**Implication for the M=64 headline (50.0 %).** That cell has
only been run with seed=0. The 4-seed Wilson CI on the M=16
cells suggests a ~6 pp per-side uncertainty band; under that
assumption the M=64 cell's *pooled* rate would be plausibly in
the range 44-56 %. A 3-seed rerun at M=64 σ=0.05 is the natural
next ablation to lock in the 50 % headline; **postponed**
(2026-05-15 evening) since Job B + Job C already CI-tightened
the M=16 numbers — a third pool at M=64 is rigor we can defer
without the paper headline breaking. See A3 follow-up (a) in the
Branch A plan.

### Flow-prior diagnostic resolution

Today's diagnostic resolved the May 14 `B1_status.md` finding
that `v_theta` samples were "1000× the action bound." The bug was
caused by testing with `s0=zeros` (heavily out-of-distribution
after the obs-normalisation) — not by a broken model.

| Conditioning                       | Sample range       | `‖x‖/step` |
|------------------------------------|--------------------|-----------:|
| `s0 = zeros` (B1_status.md case)   | `[-2957, 3193]`    |     1686   |
| `s0 = obs_mean` (training centre)  | `[-3, 4]`          |      2.7   |
| `s0 = real demo init_obs`          | `[-3, 4]`          |      2.8   |
| Reference: real demo actions       | `[-1, 1]`          |     1.24   |

**Real-state conditioning eliminates the catastrophic case** —
the unified driver always passes a real `s0` at inference, so
this is fixed automatically. **No retrain needed for the magnitude
bug.**

Residual issue: samples are still ~2× the action bound
(`‖x‖/step` = 2.8 vs the demo's 1.24). Robosuite clips executed
actions to `[-1, 1]`, so worst case MBD wastes some rollouts on
clipped trajectories.

Multimodality across 4 different seeds: pairwise per-element L2
≈ 1.3 — the promised flow-matching advantage over deterministic
BC is preserved (BC by construction has L2 = 0 across seeds).

**Final flow-prior eval cell** (K=10 R=1 λ=5 σ=0.25 M=16
`--prior=flow`, n=60): **0/60 = 0.0 %.** The residual 2×
off-scale appears catastrophic in practice. Reward histogram
(rounded):

| reward bucket | count |
|---------------|------:|
| 0 (no reach)  |   ~15 |
| 1             |   ~30 |
| 2             |   ~13 |
| 3             |    ~2 |

Compare to the same cell with the BC prior, where the reward
distribution had multiple cells > 30 and the 25 / 60 successes
had rewards ~40-50. The flow prior is pulling MBD into an action
manifold where the robot never reaches the can.

**Implication for the writeup.** Three scenarios remain open:

1. **Scenario A (preferred):** retrain `v_theta` v2 with FiLM
   s0-conditioning + x_1-parameterization (shipped on
   `feat/principled-score` at `56db351`; awaiting Colab run).
2. **Scenario B (fallback):** keep the BC headline at 41.7 %
   (or 50 % if M=64 holds), and document that the flow-matching
   prior was *attempted* but the trained `v_theta` v1
   underperforms the MLP-BC stand-in at this data scale (20
   demos). Update the paper title to "A Hybrid Score-Bridge
   Approach…" instead of "Flow Matching."
3. **Scenario C (honest middle path):** report both side-by-side.
   Acknowledge BC > flow at 20 demos, motivate that flow becomes
   advantageous at higher demo counts (which we don't have CPU
   budget to verify).

The flow-prior result goes in the writeup either way; the
question is whether it goes into the headline row or into the
"comparisons" appendix.

### What May 15 rewrites in the May 14 summary

May 14 picture:

- BC: 21.7 %
- Hybrid peak (exp 07 single-shot λ-sweep): 33.3 % at λ=5
- Δ(hybrid − BC): +11.7 pp, p=0.15 (not stat-sig at n=60)
- Best K × R cell (warm-start formulation): 8.3 %

May 15 picture (post Job B + Job C 4-seed closeout):

- BC: 21.7 % (unchanged, single seed; CI [12.7, 33.0] at n=60).
- **Hybrid peak (unified driver, K=10 R=1 σ=0.25 M=16):**
    - single seed=0: **41.7 %**
    - **4-seed pool: 37.5 % (CI [31.6, 43.8])** — honest headline.
- **Cross-branch peak with M-ablation tuning** (exp 05 driver,
  σ=0.05 M=64, seed=0): **50.0 %**
    - **M=256 closes at 43.3 % — inverted-U; M=64 is the peak.**
    - 4-seed CI on the M=64 cell postponed.
- Δ(hybrid − BC) at the unified K=10 R=1 cell, *4-seed pooled*:
  **+15.8 pp at CI lower bounds**, down from the single-seed
  +20 pp but still firmly outside the BC CI's upper bound (33.0 %).
- σ-sweep best at K=10 R=1: 40.0 % at σ=0.01 or 0.03 (single
  seed; 4-seed pool at σ=0.03 is 37.9 % — matches the unified
  driver's 37.5 % within noise).
- K-axis confirmed anti-monotone above K=10 (K=20 lands at
  38.3 %); R=2 catastrophe is λ-independent.

The **single-seed headline +28.3 pp at M=64 should be cited with
the caveat that the 4-seed CI on the same cell is not yet in
hand**. The 4-seed M=16 cells have per-seed std of ~5 pp, so the
M=64 pooled rate is plausibly anywhere in 44-56 %. Even at the
low end, the +22 pp absolute gap over BC clears any reasonable
McNemar threshold.

### In-flight cells closeout log

All cells that were in flight at the 16:47 UTC snapshot have now
landed:

| Cell                              | State | Final                                          |
|-----------------------------------|-------|------------------------------------------------|
| Stage 5 M=64                      | DONE  | 30 / 60 = **50.0 %**                            |
| Stage 5 M=256                     | DONE  | 26 / 60 = 43.3 % (inverted-U; M=64 is peak)     |
| K=20 R=1 BC extra                 | DONE  | 23 / 60 = 38.3 % (K=10 confirmed as peak)        |
| K=10 R=1 FLOW extra (Branch B)    | DONE  | 0 / 60 = 0.0 % ⚠                                |
| Job A σ=0.10 (unified K=10 R=0)   | DONE  | 22 / 60 = 36.7 %                                |
| Job B (4-seed σ=0.03 exp 05)      | DONE  | pooled 91 / 240 = 37.9 %                        |
| Job C (4-seed K=10 R=1 unified)   | DONE  | pooled 90 / 240 = **37.5 % CI [31.6, 43.8]**    |

### Docs cleanup (this commit)

`results.md` was trimmed from 834 → ~200 lines (headlines only).
Narrative, methodology choices, mechanism hypotheses, and
"implication for the writeup" prose moved here. The `Open items`
table and `Headline summary` triple-finding narrative from
`results.md` have been removed — superseded by `plans.md` and
the May 14 entries below in this file.

---

## 2026-05-14 (overnight + evening)  BC-swap re-run lands; codebase consolidation discipline added

### Outcome

All three n=60 re-runs with the new `[256, 256]` BC finished and
are committed to `main` (merge `1a1cd52`, then a single follow-up
commit covering the plans rewrite, the analysis-doc dedup, and
this run-log entry). Headline numbers, all `can/ph`, seed=0,
n=60:

| Cell                                            | Success            |
|-------------------------------------------------|-------------------:|
| Exp 03 BC closed-loop, **new `[256, 256]`**     | **13/60 = 21.7 %** |
| Exp 03 BC closed-loop, old `[512, 512]`         |   4/60 = 6.7 %     |
| Exp 07 λ-sweep, **λ=5.0 (new peak)**             | **20/60 = 33.3 %** |
| Exp 07 λ-sweep, λ=∞ (pure-BC plan via MBD avg)    |  18/60 = 30.0 %    |
| Exp 07 λ-sweep, λ=0.5 (old peak; now flank)       |  11/60 = 18.3 %    |
| Exp 10 K × R, best cell K=10 R=2 (λ_softmax=0.5) |  11/60 = 18.3 %    |
| Exp 10 K × R, K=10 R=0 / K=20 R=0 (open-loop)    |   0/60 = 0.0 %     |

Statistical claims (Wilson 95 %, two-sided exact / z-test):

- **BC new vs old**: +15 pp, **p=0.018** (z=2.36). Stat-sig.
- **Hybrid peak new vs old (λ=5 vs λ=0.5)**: +23 pp, **p=0.002**
  (z=3.10). Stat-sig; inverted-U peak shifted by 10×.
- **Hybrid peak vs new BC**: +11.7 pp, **p=0.15** (z=1.43). NOT
  stat-sig at n=60; CIs overlap (BC [13.1, 33.6], λ=5 [22.7, 45.9]).

### What changed on disk

Branch flow:

```
main 6adadeb (Charles-test merge, morning)
  |
  feat/bc-swap-rerun:
    50133b7  gitignore _archive_old_bc/
    82986a8  remove old-BC tracked JSONs/PNGs
    1f3059f  Exp 03 BC eval at n=60 w/ new ckpt
    10de782  scripts/local/run_pareto_full.sh
    f3fde77  scripts/local/run_lambda_bc_full.sh + chain_bc_swap_rerun.sh
    bf50604  scripts/local/monitor_bc_swap_rerun.sh (watch-compatible)
    edf3234  re-run JSONs (33 files) + bc_swap_rerun_analysis.md + results.md update
    2f5b578  README finalize
  |
main 1a1cd52  Merge --no-ff feat/bc-swap-rerun
     HEAD     plans.md rewrite (consolidation discipline)
              + analysis-doc dedup + this run-log entry
              (single squashed commit)
```

Cherry-picks: none. Everything in this branch was originated on
`feat/bc-swap-rerun` post the morning `Charles-test` merge.

### Why exp 07 λ-curve peak moved from 0.5 to 5.0

Mechanistic: λ controls how strongly the per-step denoising
update is pulled toward `∇log π_BC(x)`. Under the old weak BC,
small λ tolerates BC errors before they dominate the planner's
progress; under the new strong BC, BC is already most of the
answer and you want λ large enough that MBD only nudges, never
overwrites. The fact that λ=5 (33.3 %) and λ=∞ (30 %) are
within 3 pp quantifies how little extra value MBD adds on top of
a strong BC. Plot:
`experiments/07_lambda_sweep/results/lambda_curve.png`.

### Most consequential finding: the algorithmic confound

Exp 05/10 and Exp 07 are **not the same algorithm**:

- Exp 05/10's `run_can.py` uses the BC prior **only as a
  warm-start** (initial value for the denoising chain). No
  per-step `∇log π_BC` term.
- Exp 07's `run_can.py` wires a `prior_score_fn` hook into
  `denoise_loop`, adding `λ · ∇log π_BC(x_t)` at every
  denoising step. This is the score-bridge formulation in eq.1
  of the Week-4 report.

At the same K=10 cell, warm-start-only (Exp 10 K=10 R=2) gets
**18.3 %** and score-bridge (Exp 07 λ=5) gets **33.3 %**. The
+15-pp gap is **larger than any hyperparameter effect this
project has ever measured**.

A McNemar exact paired test inside Exp 10's 540 rollouts (BC
subroutine vs hybrid, same chunk start states) gives
**p ≈ 9 × 10⁻²⁹** that BC > Hybrid under the warm-start-only
formulation. Discordant pairs 167 (BC wins) vs 22 (hybrid wins).
Full table in `docs/bc_swap_rerun_analysis.md` § 3.

### New discipline: no hyperparameter tuning before consolidation

Adopted 2026-05-14 in response to the +15-pp algorithmic gap.
Rule: **no λ-sweeps, K × R grids, σ-schedules, seed pools, or
M-particle ablations until the two hybrid implementations are
unified into a single canonical algorithm**. Recorded in:

- `docs/plans.md` § *Top priority: codebase consolidation
  (BLOCKING)* (the new top section) and § *Explicit
  de-prioritisations* (leading entry).
- `docs/bc_swap_rerun_analysis.md` § 6.
- Persistent memory (Cascade).

### Bugs caught tonight

- `scripts/local/chain_bc_swap_rerun.sh`: the three analyzer
  subprocess calls all failed silently with `OSError: [Errno 9]
  Bad file descriptor`. Root cause: chain script uses
  `exec > >(tee -a ...) 2>&1`, which interacts badly with the
  python subprocess's stdio inheritance when nohup'd. The
  analyzers work fine when invoked manually; I re-ran them by
  hand and committed the regenerated `summary.{md,png}` +
  `lambda_curve.png`. **Not fixed**; the chain hack is a
  one-shot tool and not worth re-engineering.
- An earlier paired-stats attempt used `r.get("hybrid_success")`
  instead of `r.get("success")` for the hybrid outcome key,
  giving all-False results (i.e. "hybrid never wins"). Fixed by
  checking the actual JSON key (`success`) and re-running the
  McNemar tabulation.

### Docs updated tonight

- `docs/results.md`: BC-swap RESOLVED banner; headline table
  swapped for new BC numbers; Exp 03 section rewritten with
  21.7 %.
- `docs/plans.md`: rewrite per discipline above; three paper
  framings rewritten for post-rerun reality.
- `docs/bc_swap_rerun_analysis.md`: new long-form analysis
  (~250 lines after the docs-dedup pass). Headline + Wilson
  CIs + McNemar paired test + the warm-start-vs-score-bridge
  discussion + recommended follow-ups (deferred behind the
  consolidation).
- `README.md`: status table now reads "done (new BC)" for
  Exp 03 / 05+ / 07; headline paragraph names the score-bridge
  peak at 33.3 %.
- `docs/run_log.md`: this entry.

### Outstanding for next session

1. **Codebase consolidation** (top priority per `plans.md`).
   Unify Exp 05/10's `run_can.py` with Exp 07's `prior_score_fn`
   path; add `--mpc-replans` / `--mpc-chunk-steps` to the
   score-bridge driver; retire the duplicate. ~1 day refactor.
2. **Verification run**: 9-cell K × R grid at **λ=5** on the
   unified algorithm, n=60 each, ~5 h locally. Decides whether
   MPC chunking adds value on top of the score-bridge.
3. After (2) lands: 3-seed pool on the winning cell + Exp 03
   BC. ~12 h overnight. Settles the +11.7 pp claim or shows it
   doesn't survive.
4. Failure taxonomy on the new JSONs (cheap; independent).

---

## 2026-05-14 (late)  Incorporate `Charles-test` -- new BC checkpoint changes everything

### What changed on disk

Switched to `feat/further-experiments`, cherry-picked the morning's
results work (analyzers, runners, exp 07 / 10 JSONs, results.md,
run_log fixes, exp 11 numpy fix -- 7 commits total), then
incorporated 4 commits from `origin/Charles-test`:

| commit    | what it brings                                                                              |
|-----------|---------------------------------------------------------------------------------------------|
| `eee088d` | (originally `836daae`) **new `[256, 256]` BC checkpoint** -- 80k params, loss 0.0095        |
| `475fd03` | (originally `686824f`) 3 exp-05 tuning JSONs + PNGs (`can_ph_tuning_v{1,2,3}`, n=5 each)    |
| `cb4be6a` | (originally `034e401`) `docs/archive/charles_tuning_analysis.md` -- Robin's writeup of those 3 runs |
| `a3e87df` | (originally `378a107`) gitignore the Windows `hybrid-traj/` venv                            |
| `d13bb79` | (originally `0a30884`) `env/requirements.txt` packaging pin (same fix as our `d203d06`)     |

Skipped: `d0d254d` ("Testing", added 8849 Windows venv files) and
`22a310f` (the deletion of same). We never had the venv on this
branch.

The local OLD `[512, 512]` `policy.pt` (md5 `d6d0b895`, untracked,
1.16 MB) was moved aside to
`experiments/03_bc_baseline/results/_archive/policy_v1_512x512_2026-05-10.pt`
and gitignored, so the May-13 exp 07 / 10 results remain
reproducible from disk on this machine but the canonical tracked
checkpoint is now Charles's.

### Why his result is better -- and what to learn

Charles's analysis doc has the clean breakdown (read it in full).
The TL;DR:

1. The original Exp-03 BC used `hidden=[512, 512]` (290k params),
   trained for 500 epochs on the 20-demo `can/ph` headline slice.
   Final training loss was **0.0060**, but task success was only
   **6.7 % (4/60)**. The model memorised the 23k training
   transitions and didn't generalise.
2. Charles re-trained at `hidden=[256, 256]` (80k params), same
   data, same epochs, same seed=0. Final training loss was
   **0.0095** (*higher*), but BC-alone task success is suggestively
   ~60 % (3/5 in the v2 and v3 columns).
3. This is a textbook capacity/generalisation trade-off. The
   smaller MLP cannot fit the training transitions to <0.006 loss,
   so it has to learn the coarser "approach / grasp / lift / place"
   structure, which transfers to fresh `env.reset()` init states.
   The bigger MLP can fit the transitions exactly but the exact
   transitions don't recur at test time.

The lesson is not specific to this project: **when a BC policy is
trained on a small demo set, the architecture's interpolation
behaviour matters more than its expressive capacity.** Future BC
checkpoints in this repo should start at `[256, 256]` and only
grow if the *test-time* success rate (not the training loss)
demands it.

### Implications for the May-13 results

The 8.3 % pareto best and 10 % λ-sweep peak are **valid numbers
for the OLD BC** that warm-started them. But:

- The comparison vs BC in `summary.md` and `taxonomy.md` (where
  per-cell hybrid is compared against per-cell BC = 1.7 % - 11.7 %)
  is now apples-to-oranges -- those BC numbers used the OLD
  checkpoint baked into the JSONs at run time. Re-running with the
  new BC will both raise the BC column and (probably) raise the
  hybrid column, but it's not obvious which moves more.
- The λ-sweep inverted-U at lam=0.5 was specifically the BC-prior
  pull strength that struck the best balance against the OLD BC.
  With the new BC -- which is better on its own -- the optimal
  λ may shift or the inverted-U may flatten entirely (Charles's v3
  with `λ=1.0, K=20, sigma_max=0.10` shows hybrid "preserves" BC
  but doesn't convert any of BC's failures, so the peak may
  collapse to 0 differential improvement at his configuration).

### Why exp 08 is interesting *again*

The May-13 / 14 run-log entries argued exp 08 (v_theta) was
optional because the BC-prior λ-peak was the headline. With the
new BC at ~60 %, that argument inverts:

- **BC alone is now the strong baseline (~60 % suggestive).** The
  hybrid framework needs to beat that, not the old 6.7 %.
- **BC is deterministic and unimodal.** If the task admits multiple
  valid solutions (e.g. different grasp orientations), BC will
  collapse to one mode. A generative prior like `v_theta` can in
  principle represent the multimodality.
- **The published method uses `v_theta`, not BC.** Our BC-prior
  experiment was a stand-in for the trained `v_theta`. To make a
  faithful "MBD + flow-matching prior" claim, we need `v_theta`
  trained properly.
- Cost is still ~3 h on GPU / ~10-30 h on Mac CPU.

So exp 08 moves from "medium priority, only if needed" to
"medium-high, needed to test the strong-form hypothesis that the
hybrid framework beats well-tuned BC on Can."

### Branch hygiene note

All this work landed on `feat/further-experiments`. The 7 result
commits also exist on `feat/mjx-rollouts` from earlier in the day
(same content, different shas: `ad70aa6`, `bc2c6a7`, `18d780c`,
`4c99f54`, `b7a6180`, `5af775b`, `1a1d811`). The duplication is
intentional and benign -- `feat/mjx-rollouts` was published
publicly, so force-pushing to remove the result commits would
break anyone who pulled. Future merges into `main` will dedupe
naturally.

### Outstanding (rewritten priority -- supersedes the 2026-05-14 morning list)

1. **`python experiments/03_bc_baseline/eval.py --task can`**
   on the new `policy.pt` to get the proper BC-alone n=60 number.
   ~5 minutes; lowest-cost / highest-info action available.
2. **Re-run exp 10 K×R pareto + exp 07 λ-BC sweep** with the new
   BC warm-start. Same hyperparameters as the May-13 runs so the
   comparison is clean. ~5 h each locally; ~10 h total.
3. **After 1-3 settle**, exp 14 3-seed pool on whichever cell wins.
4. **Then** decide whether exp 08 (v_theta train) and exp 13
   (Square) are worth the time given the post-swap headline.

---

## 2026-05-14  Local finish of Exp 10 pareto + Exp 07 lambda-BC sweep

### Outcome

The two big overnight sweeps are now both at full coverage:

- **Exp 10 (K x n_replans grid)**: 9 / 9 cells, n=60 each. Best
  cells: `K5_R1`, `K10_R2`, `K20_R2`, all at **8.3 %** success.
  Worst: `K10_R0` (single-shot) at **0.0 %**, with BC baseline on
  the same demo set at 11.7 % -- so the open-loop hybrid is
  strictly *worse* than just executing BC here.
- **Exp 07 (BC-prior lambda sweep)**: 6 / 6 cells, n=60 each.
  Clean inverted-U: lambda=0 and lambda=inf both bottom out, peak
  at **lambda=0.5 = 10.0 %** -- 6 / 60 successes, vs 0 % at the
  pure-planner end and 1.7 % at the pure-prior end.

Both sweeps' artefacts:

```
experiments/10_kxreplans_grid/results/summary.{json,md,png}
experiments/10_kxreplans_grid/results/taxonomy.{json,md}
experiments/07_lambda_sweep/results/summary.md
experiments/07_lambda_sweep/results/summary_bc.json
experiments/07_lambda_sweep/results/summary_flow.json
experiments/07_lambda_sweep/results/lambda_curve.png
```

### How the night finished

GCE finished 6 of the 9 pareto cells and 2 of the 6 BC-lambda
cells before the IP-cap + n2-stockout situation made further
overnight dispatch impractical. The remaining **3 pareto cells
(`K10_R2`, `K20_R1`, `K20_R2`)** and **4 BC-lambda cells
(lam in {0.1, 0.5, 1.0, 5.0})** ran locally on the Mac (M4 Max,
12 perf + 4 eff cores) in parallel via two new runner scripts:

- `scripts/local/run_remaining_pareto.sh`
- `scripts/local/run_remaining_lambda_bc.sh`

with monitors (watch-compatible):

- `scripts/local/monitor_pareto.sh`
- `scripts/local/monitor_lambda_bc.sh`

The full 7-cell oversubscription stayed inside the M4 Max budget
(aggregate ~1050-1100 % out of 1600 % theoretical). Measured
per-plan time during the 7-cell phase was **1.4 - 1.9x slower**
than once the 4 lambda cells finished and only the 3 pareto cells
remained -- K10_R2 went from 226 s/plan to 118 s/plan, K20_R1
from 285 s to 162 s, K20_R2 from 313 s to 229 s. End-to-end wall
time for the local half of the night: **~5 h** to finish ~17 h
of GCE-equivalent work. Net cloud-cost avoidance: ~$2.

### Bugs / friction fixed

- `experiments/07_lambda_sweep/run_can.py`: filename now
  disambiguates prior (`lambda_bc_{lam}.json` for BC,
  `lambda_{lam}.json` for flow). Before, the two priors collided
  at the same path and the local "skip if exists" check matched
  flow-prior runs by mistake.
- `experiments/05_hybrid_method/run_can.py`: numpy 2.x
  `OverflowError` on `np.uint32(hash(...))` -- masked the hash
  with `0xFFFFFFFF` so the seed survives both numpy 1.x (GCE) and
  numpy 2.x (local).
- Stale smoke-test files (`lambda_0.0.json`, `lambda_1.0.json`,
  n=2 each) moved to `_stale_smoke_backup/` and added to
  `.gitignore` so they don't pollute future analyses.
- `.gitignore` now also excludes `logs/` (local-run stdout/stderr)
  and the personal `scripts/local/charles_pre_pull_message.txt`.

### Honest read of the results

**The hybrid is not yet a strict improvement over BC on Can.**
Across the 9 pareto cells, the mean delta (hybrid - BC) is
**-1.7 pp**, and hybrid wins on only **4 / 9** cells. The wins
are all on configurations with R >= 1 (multi-shot MPC); R=0
(single-shot, open-loop) is consistently worse than BC. This
matches Exp 04's open-loop story: a single MBD plan is brittle
to compounding error, even with a good warm-start.

The lambda sweep is more cleanly positive: the inverted-U is
exactly the shape we wanted to see, and the combination
(lam=0.5) is **6x better than either extreme**. That's the
right kind of ablation evidence for the hybrid story even when
the absolute number is small.

Flow-prior cells (4 of 6 lambdas covered) are all 0 / 60. As
the 2026-05-13 entry noted, `v_theta` is undertrained
(88 s + 54 s on the two runs we have); a real retraining pass
is the prerequisite for the flow-vs-BC comparison being meaningful.

### Outstanding for future sessions

A. Retrain `v_theta` (Exp 08) so the flow-prior side of the
   lambda sweep is non-trivial.
B. Try the 3 winning pareto configs (K5_R1, K10_R2, K20_R2) on
   the Square task (Exp 13) to check that the R>=1 advantage
   generalises.
C. Seed sweep (Exp 14) on at least `K10_R2` so the 8.3 % number
   has an error bar attached.
D. MJX Gate-4 (the GPU speedup measurement) is still queued on
   Colab; not blocking but would unlock a much larger M (number
   of plan samples) at the same wall budget.

---

## 2026-05-13  MJX migration + Exp 06-14 audit + Exp 12 runner

### Outcome

MJX migration shipped on `feat/mjx-rollouts` (8 commits, pushed):

| Commit  | Gate / topic                                                |
|---------|-------------------------------------------------------------|
| 53bbc2f | Gate-1: simplified PickPlaceCan MJCF + smoke                |
| b52046e | Gate-2a: JAX OSC controller + MJX batched rollout           |
| fedaa76 | Gate-2b: OSC math parity vs robosuite -- **8.03e-12 Nm**    |
| fd53270 | Gate-2c: cost rank parity vs robosuite -- **rho = +0.485**  |
| 8d253e8 | Gate-3: `--mjx` flag in `run_can.py` (Mac CPU smoke 57 s)   |
| 015332d | Gate-4 prep: GCE L4 infra + `test_mjx_gpu_smoke.py`         |
| 99ea3b1 | Colab notebook (`colab_mjx_gate4.ipynb`)                    |
| 22a357b | PAT prompt for the private-repo clone                       |

Gate-4 (the real GPU speedup measurement) has two paths: Colab T4/L4
(unblocked, the notebook works), and GCE L4 (blocked on quota).

### Audit of `feat/further-experiments` Exp 06-14

Re-reading `docs/topic_introduction/topic_introduction_and_literature_review.md`
and `docs/plans.md` revealed that the MJX migration is only Tier 4
item (h) out of 14 planned experiments. The honest status of the
others:

| #  | Implementation | Verified locally this session                 |
|----|----------------|-----------------------------------------------|
| 06 | full           | reproduces published 3.1 % / 14.8 % SR->succ  |
| 07 | full           | `--smoke --prior bc` passes end-to-end (2 min)|
| 08 | full           | `lq_sanity.py` passes; `train.py --smoke` OK  |
| 09 | bash wrapper   | unchanged; ready for overnight                |
| 10 | bash wrapper   | unchanged; ready for overnight                |
| 11 | full           | not run locally (~80 min)                     |
| 12 | basis.py only  | **run_can.py written this session** (`1a1d811`)    |
| 13 | bash wrapper   | unchanged; ready for overnight                |
| 14 | full           | unchanged; ready for overnight                |

### Verified copy-pasteable commands (Mac CPU, hybrid-traj env)

```bash
# 06 -- failure taxonomy (~1 s, no env)
conda run -n hybrid-traj python experiments/06_failure_taxonomy/run.py \
    --inputs experiments/04_mbd_baseline/results/can_ph_seed*.json \
             experiments/05_hybrid_method/results/can_ph.json \
             experiments/05_hybrid_method/results/can_ph_mpc.json

# 08 -- flow-matching prior, 2-D ring sanity (~10 s)
conda run -n hybrid-traj python experiments/08_flow_matching_prior/lq_sanity.py

# 08 -- trainer smoke (~few seconds)
conda run -n hybrid-traj python experiments/08_flow_matching_prior/train.py \
    --smoke

# 07 -- lambda-sweep smoke with BC prior (~2 min)
conda run -n hybrid-traj python experiments/07_lambda_sweep/run_can.py \
    --smoke --prior bc --no-live-view

# 12 -- DCT-basis MBD smoke (~70 s)
conda run -n hybrid-traj python experiments/12_action_basis_mbd/run_can.py \
    --smoke --no-live-view

# Gate-4 GPU smoke (Mac -> exit 20; L4 -> exit 0)
conda run -n hybrid-traj python experiments/04_mbd_baseline/test_mjx_gpu_smoke.py

# Gate-3 end-to-end MJX smoke through run_can.py (Mac CPU, ~60 s)
conda run -n hybrid-traj python experiments/04_mbd_baseline/run_can.py --mjx \
    --H 20 --M 4 --K 2 --n-demos 1 --n-rollouts 1 --seed 0 \
    --no-live-view --video-dir "" --output /tmp/can_mjx_smoke.json
```

All seven of these were run end-to-end this session and exited
cleanly. Numbers reproduce the published table (Exp 04 0/60, Exp 05
1/60 + 53 % strong-reach, Exp 05b 4/60 + 14.8 % SR->succ).

### Exp 12 runner -- what was written

`experiments/12_action_basis_mbd/run_can.py` (389 LOC). Mirrors
exp 05's single-shot configuration but substitutes
`basis.denoise_loop_basis` for `mbd.denoise_loop`. New CLI:

  * `--n-modes` -- DCT basis size. 4, 8, 16, 32, 120 are the
    suggested sweep points (120 == H == "no projection").
  * Inherits `--no-warm-start`, `--deterministic-bc-reset`,
    `--n-demos`, `--n-rollouts`, `--seed`, `--output`,
    `--no-live-view`, etc.

Smoke verified locally:

```
[exp12] n_modes=4  H=120  M=16  K=4  warm_start=True  sigma=(0.05, 0.25)
[exp12]   1/2  demo_0  init=-3.06  final=-11.09  exec=+6.06  success=False  (31.9s plan)
[exp12]   2/2  demo_1  init=-2.43  final= -7.96  exec=+6.93  success=False  (33.3s plan)
[exp12] success rate: 0/2 = 0.0%   wall=67.5s
```

Cost descends in both rollouts. The headline 5-point sweep over
`n_modes in {4, 8, 16, 32, 120}` at `--n-demos 20 --n-rollouts 3`
is the next overnight item once exp 08 produces a non-toy
`v_theta.pt`.

### Priority list going into next overnight

Per `docs/plans.md` Tier 1, the highest-leverage gaps are:

1. **Exp 08 full train**: `train.py --task can --n-demos 20
   --horizon 120 --epochs ~10000` on GPU (Colab T4 is fine). The
   1.5-day previous overnight produced an undertrained v_theta
   (final loss ~88, 54 s wall). A real run needs ~3 h on T4.
2. **Exp 07 full sweep with `--prior flow`**: once the trained
   v_theta lands, run all 6 lambdas (~80 min each) on the
   GCE fleet.
3. **Exp 14 3-seed sweep**: Exp 04 seeds 0/1/2 already done; need
   Exp 05 + 05b seeds 1/2. ~19 h overnight.
4. **Exp 12 n_modes sweep**: ~5 * 80 min = ~7 h overnight.

The Tier-3 generalisation rows (Exp 13 Square, Exp 11 oracle
replan) are gated on the headline number landing first.

---

## 2026-05-12  Overnight 1 -- seed-sweep + pareto + lambda-sweep-bc

### Outcome (as of 2026-05-13 ~02:00 PDT)

Launcher PID 26612, RUN_ID `r46394`, state dir
`$TMPDIR/overnight-46394/`, still in flight. Expected completion
around 10:30 AM PDT given the IN_USE_ADDRESSES=8 throttle.

| Cluster   | Slug             | Result                                  |
|-----------|------------------|-----------------------------------------|
| r45286    | 04-s0            | done -- `can_ph_seed0.json`, 0/60       |
| r45286    | 04-s1            | done -- `can_ph_seed1.json`, 0/60       |
| r45286    | 04-s2            | done -- `can_ph_seed2.json`, 0/60       |
| r46394    | 04-s0            | running, rollout 59/60 at 04:54 UTC    |
| r46394    | 04-s1            | running, rollout 59/60 at 04:54 UTC    |
| r46394    | 04-s2 (DUP)      | KILLED at 04:54 UTC (12 min in apt)    |
| r46394    | 05-s0/s1, 05b-s0 | running, fixed-tarball                  |
| r46394    | (queue, 17 items)| 05-s2, 05b-s1/s2, 9 pareto, 6 lam-bc    |

Headline finding: **MBD-only (exp 04) gives 0/60 reproducibly across
seeds 0, 1, 2.** Variance is zero; the Wilson 95 % CI on 0/180 is
`[0 %, 2.0 %]`. The 0 % success rate of exp 04 is not a fluke.

The bug-fixed exp 05 / 05b / pareto results are still pending. The 6
lambda-prior-BC items are last in queue; they exercise the per-step
composite-score plumbing with the well-trained BC policy from exp 03
as the prior (substitute for the undertrained v_theta).

### Mid-session correction (02:15 PDT)

The `lambda-sweep-bc` pattern I added crashed all 4 in-flight BC-lambda
VMs with `FileNotFoundError: experiments/03_bc_baseline/results/policy.pt`.
The pattern was passing `--bc-checkpoint <local-path>` but `policy.pt`
is gitignored, so it was not in the repo tarball uploaded by
`upload_repo.sh`. Fixed by adding a `BC_GCS` -> `bc-gcs` metadata path
mirroring the existing `VTHETA_GCS` -> `vtheta-gcs` wiring (commit
`0c8f008`):

  * `launch_overnight.sh` defaults `BC_GCS=$BUCKET/bc/policy.pt`
  * `launch.sh` adds `bc-gcs=$BC_GCS` to instance metadata
  * `startup.sh` `gsutil cp`'s policy.pt into the canonical location
    before the experiment runs
  * Uploaded `policy.pt` to `gs://hybrid-traj-robin-2026-results/bc/`

The 4 lost slugs (`lam-bc-0`, `lam-bc-0-1`, `lam-bc-5-0`, `lam-bc-inf`)
were manually re-appended to the live launcher's queue.txt after
re-upload of the patched repo tarball.

### Bugs caught and fixed tonight  (all committed to feat/further-experiments)

1. **`scripts/gce/launch_overnight.sh` RUN_ID sanitization** (`af75b38`)
   GCE VM names must match `[a-z][-a-z0-9]{0,61}`. An operator-set
   RUN_ID with uppercase or symbols was producing invalid names and
   every gcloud-create failed. RUN_ID is now lowercased and stripped
   to `[a-z0-9]`, falling back to the timestamp digits if empty.

2. **`scripts/gce/launch_overnight.sh` stdin redirect** (`af75b38`)
   `nohup ... &` on macOS does NOT redirect stdin like GNU nohup
   does. The inherited stdin from the parent shell becomes invalid
   once the terminal session ends and every subsequent gcloud
   subprocess (Python) crashes with `OSError: [Errno 9] Bad file
   descriptor` during `init_sys_streams`. Forcing `exec </dev/null`
   at the top of the launcher fixes it for all descendant
   subprocesses.

3. **`experiments/05_hybrid_method/run_can.py` duplicate `--seed`** (`901b383`)
   Argparse declared `--seed` twice. Python rejected it at import
   time with `ArgumentError: conflicting option string`, so every
   exp 05 / exp 05b fan-out VM crashed before its first rollout.
   Five sweep VMs in the previous launcher run silently produced
   no data because of this -- caught only after the `watch.sh` fix
   below exposed their crash logs in the fleet table.

4. **`scripts/gce/watch.sh` set -e + pipefail truncation** (`901b383`)
   `last_log=$(... | grep -E ... | tail | cut ...)` -- when no line
   matched (typical for freshly-booted VMs), grep exited 1, pipefail
   propagated, set -e killed the script mid-loop. Output table
   silently truncated to the first 2 VMs. `|| true` on the
   assignment lets the empty match fall through to the default
   `(no tag yet)` placeholder. Fixing this is what surfaced bug #3.

5. **`scripts/gce/launch_overnight.sh` `lambda-sweep-bc` pattern** (`901b383`)
   New built-in pattern that runs the 6-lambda sweep with the BC
   policy from exp 03 as the prior instead of the (currently
   undertrained) flow-matching v_theta. Lets us collect lambda-sweep
   data this overnight without waiting for v_theta to retrain.

### Wasted compute

Approximate (n2-standard-2 at ~$0.10/h):

| Cause                                                 | Cost     |
|-------------------------------------------------------|----------|
| First launcher's 4 sweep VMs killed when RUN_ID bug   | ~$0.10   |
| Second launcher's 6 sweep VMs killed after stdin bug  | ~$0.13   |
| Five exp 05/05b crashes from the `--seed` bug         | ~$0.50   |
| Undertrained v_theta v2 (54 s) -- not useful          | ~$0.01   |
| 3 r46394 04-* duplicates of finished r45286 work      | ~$1.20   |
| **Total wasted**                                      | **~$1.94** |

Cheaper than expected given the number of incidents. The duplicate
issue (last row) is the biggest single waste and is preventable.

### Known limitations carried into tomorrow

A. **IN_USE_ADDRESSES regional cap = 8.** Launcher only gates on
   `CPUS_ALL_REGIONS` (limit 24 -> 12 concurrent VMs). The IP cap
   binds first, so steady-state concurrency is 8. Launcher
   self-corrects via re-queue + retry, but burns ~75 s per failed
   dispatch (175 of them so far tonight). To lift: enable Private
   Google Access on the subnet, add Cloud NAT (apt / pip still need
   public egress), and pass `--no-address` to launch.sh's gcloud
   create. Not worth doing at 02:00 -- queue for tomorrow.

B. **Duplicate-dispatch on launcher restart.** The launcher emits
   the full pattern list every start and has no way to skip slugs
   already in flight under a previous RUN_ID. Tonight that cost us
   ~$1.20 in duplicate 04-s0/s1 (the third, 04-s2, was killed
   manually at 04:54 UTC after 12 min, saving ~$0.40 of the ~$0.60
   it would have wasted). Add a `SKIP_SLUGS` env var or a
   `--resume-with` flag that takes precedence over emitted slugs.

C. **`v_theta` is undertrained** (88 s on the original, 54 s on
   v2). All flow-prior lambda data so far is `0/60` across the
   board. Need a real retraining run -- e.g. `--epochs 200000` on
   n2-standard-4 for ~3 h -- before re-running the
   `--prior flow` sweep. Tonight's `lambda-sweep-bc` is the BC-prior
   stand-in.

D. **`watch.sh` false-positive matches.** The serial-output filter
   regex catches `Error` (capital E) but case-sensitivity on the
   `[a-z]error` strings from `otelopscol` shows lines we don't want
   in the LAST_LOG column. Cosmetic, not blocking.

### Local data fetched as of 02:00 PDT

```
experiments/04_mbd_baseline/results/can_ph_seed{0,1,2}.json   # 3 x 141 KB
experiments/07_lambda_sweep/results/lambda_{0.0,0.1,0.5,1.0,5.0,inf}.json
```

All exp 04 seeds 0/60; lambda-sweep results all 0/60 (expected --
flow prior was undertrained, 6-VM r46394 sweep). The BC-prior
sweep currently in the launcher queue is the meaningful version.

### Morning checklist

1. `bash scripts/gce/watch.sh` -- confirm fleet has drained.
2. `bash scripts/gce/launch_overnight.sh` -- check the launcher
   process exited cleanly (`all items done.  exiting.` in
   `launcher.log`).
3. Fetch remaining results:
   ```
   gsutil -m cp "gs://hybrid-traj-robin-2026-results/runs/ht-r46394-*/*/results/*.json" \
       /tmp/r46394_results/
   ```
4. Dispatch results to their experiment dirs:
   - `r46394-04-s*` -> `experiments/04_mbd_baseline/results/can_ph_seed*.json`
   - `r46394-05-s*` -> `experiments/05_hybrid_method/results/can_ph_seed*.json`
   - `r46394-05b-s*` -> `experiments/05_hybrid_method/results/can_ph_mpc_seed*.json`
   - `r46394-K*-R*` -> `experiments/10_kxreplans_grid/results/K*_R*.json`
   - `r46394-lam-bc-*` -> `experiments/07_lambda_sweep/results/lambda_bc_*.json`
5. Delete remaining TERMINATED VMs by hand:
   `gcloud compute instances list --filter="status=TERMINATED" ...`
   (the launcher only reaps its own RUN_ID; once it exits, nothing
   is left to clean up the final batch automatically).

### Closure note (added 2026-05-14)

Follow-up status of the morning checklist above:

| step | bucket                                 | status                                                                                                  |
|------|----------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1-3  | drain fleet / launcher exit / `gsutil` | done (no live VMs; partial JSON fetches dispatched piecewise as cells finished)                         |
| 4a   | `04-s*` -> exp 04 seed files           | done -- `experiments/04_mbd_baseline/results/can_ph_seed{0,1,2}.json` committed                         |
| 4b   | `05-s*` -> exp 05 single-shot seeds    | **not done** -- the seed-1/2 single-shot runs were on the queue but never landed locally                |
| 4c   | `05b-s*` -> exp 05b MPC seeds          | **not done** -- same: the seed-1/2 MPC runs never landed locally                                        |
| 4d   | `K*-R*` -> exp 10 pareto cells         | done (9 / 9 cells committed; 6 from GCE, 3 finished locally on 2026-05-14 -- see 2026-05-14 entry)      |
| 4e   | `lam-bc-*` -> exp 07 BC lambda cells   | done (6 / 6 cells committed; 2 from GCE, 4 finished locally on 2026-05-14)                              |
| 5    | delete TERMINATED VMs                  | done manually                                                                                           |

The missing 4b / 4c (3-seed pools for exp 05 and 05b) are now the
open half of Tier 2 (c) in `plans.md`; tracked as item C in the
2026-05-14 "Outstanding for future sessions" list.
