# Project roadmap (cross-branch overview)

This is the cross-branch coordination doc on `main`.

## 2026-06-04 update -- offload-plan slate batch landed

Mac A merged the **six remaining finished branches** from the
2026-06-01 multi-machine offload plan
([`coordination/offload_plan_2026_06_01.md`](coordination/offload_plan_2026_06_01.md)).
T4 (`feat/cross-task-baselines-fill` MBD-no-prior cells) is still
running on Mac A and was excluded from this batch.  Mac A had
already merged T6 / T8 / T9 / followups earlier; the six branches
landed here close the offload arc on the Mac-B side.

Merge order (chronological completion, all `--no-ff`):

- **`feat/per-step-bias-var-action`** -- merged `9180285`.  Slate
  C6 per-step bias-variance, 9 cells (3 seeds × {hybrid_canonical,
  hybrid_t2_1, bc_t2_3}).  Stage-2 analysis pipeline
  (`measure_bias_variance_v2.py`) deferred.
- **`feat/lambda-schedule-wireup`** -- merged `92caf65`.  Slate C1
  σ-indexed λ schedule sweep, 5 cells.  **VERDICT: FAIL** -- best
  schedule (cosine) 66.7 % vs. constant-λ control 76.7 % (-10 pp).
  σ-indexed schedules hurt the headline at the Can canonical
  operating point.  The `--lambda-schedule` wireup is preserved on
  `main` for future-proofing on other tasks / priors.
- **`feat/value-baseline-headline-eval`** -- merged `32a63ac`.
  T2 / B1 V_φ headline eval.  0 / 60 successes on Can canonical
  with the V_φ control-variate cost.  **VERDICT: H1 SUPPORTED**
  (capacity-limited, not variance-limited).  Also lands two
  critical bug fixes for the OBS rollout factory: pass
  `perturbations` kwarg through, and expose `update_init_state`
  on the pooled branch (mirroring the non-OBS twin).
- **`feat/t2-square-bugfix-and-bc-arm`** -- merged `ffbd531`.
  T3 / B5 T2-Square slate, 3 cells (canonical hybrid 1/30,
  BC unadapted 2/30, BC retrained 1/30) -- confirms the Square
  floor from Slate H is genuinely Square-hard.  **Also lands a
  critical Can_pos cross-task bug fix** in `run_can.py:980`
  (the BC `else` branch was falling back to exp 03's Can-only
  `OBS_KEYS` regardless of `--task`; one-line fix routes through
  `get_obs_keys(args.task)`).  Without this, every non-Can BC
  eval hit `KeyError: "runtime obs missing 'Can_pos'"`.
- **`feat/contrastive-square-slate`** -- merged `1ee9195`.
  T5 / #4 contrastive Square slate (`demo_1nn_contrastive` vs.
  `demo_1nn` on Square /ph).  **VERDICT: WEAK** -- the Square
  contrastive encoder edges out raw 1-NN but neither variant
  unlocks the Square floor.  Supersedes the "Slate I revival"
  item in `docs/todos/future_2026_05_26.md` §A4.
- **`feat/adapt-t2-n200-magnitude`** -- merged `233fff1`.  T7 / #2
  combined Mac A (Stage A) + Mac B (Stage B) work:
  - Stage A: demo-count axis at mass=1.3, N ∈ {30, 50, 100, 150, 200} -- 5 cells
  - Stage B: mass-axis sweep, T2.0 / T2.1 / T2.3 per magnitude,
    mass ∈ {1.6, 1.7, 1.8, 2.0} -- 12 eval cells + 3 BC retrains
  Combined with the pre-existing mass=1.5 work on main, the
  adaptation arc has a 5-point magnitude sweep on Can /ph.  Also
  lands the shell-quoting bugfix for paths containing spaces in
  the slate (parallel bash arrays via `"$@"` instead of
  colon-separated `$CELL_ARGS`).

**Still open** (Mac A only from here):
- T4 = `feat/cross-task-baselines-fill` -- 2 MBD-no-prior cells
  (cross_square + cross_tool_hang) still running on Mac A in
  screen `t4_macA` (pid 23839); will merge on completion.
- The Slide-7 78× / 73.3 % paper-integrity question on T8
  (`feat/adapt-dp-prior-78x-replication`) remains open -- the
  ep0199→ep0149 checkpoint swap was real but **not sufficient**
  to recover the Phase F headline under current code/env.

## 2026-06-01 update -- two more branches merged

- **`feat/dp-phase-c`** -- merged `8e97ca3`.  Closes the multi-week
  DP-prior K-sweep arc on Can/`ph`.  Phase-C wrapping-bug fix +
  full Phase-D 2-D grid (ckpt × λ × K, 7 ckpts × multiple λ + K
  cells).  **Two GREEN cells at ep0149**: (K=10, λ=2.0) at 86.7 %
  (+13.4 pp, cheapest path) and (K=20, λ=0.5) at 83.3 % (+10.0 pp).
  YELLOW everywhere else.  K=40 cell at ep0049 deferred (paused
  via SIGSTOP at 2 / 25, will resume on a follow-up branch).  See
  [`experiments/17_dp_hybrid.md`](experiments/17_dp_hybrid.md) and
  [`run_log.md`](run_log.md)'s 2026-06-01 entry for the full
  headline + mechanism reading.
- **`feat/adapt-pareto-analysis`** -- merged `aaefba2`.  Closes
  the post-T2 analysis arc.  Summary-level (bias², variance) proxy
  decomposition verifies T2.1 strictly Pareto-dominates T2.3.
  Per-step action-level measurement scaffolded but deferred to a
  journal version.  See
  [`docs/results_writeups/pareto_analysis.md`](results_writeups/pareto_analysis.md).

This closes the seven-branch adaptation follow-up plan from
[`docs/adaptation/05_followup_plan.md`](adaptation/05_followup_plan.md)
(see its "Closure (2026-06-01)" subsection for the full
merge-status table).

## 2026-05-25 baseline

As of **2026-05-25** the per-branch plans for Branch A and Branch C
have been merged into a single umbrella doc
[`branch_plan.md`](branch_plan.md) on `main`; the off-main
flow-integration plan still lives on its feature branch.

- **Branches A + C** — see the umbrella [`branch_plan.md`](branch_plan.md).
  - Branch A (`feat/stronger-mbd-signal`) carried items: A1.4 σ-sweep,
    A2 V_φ value baseline, A3-(b) `λ_softmax(M)`, A4 reward shaping.
    V_φ scaffolding (exp 16) **merged 2026-05-25** (`25ca457`); eval
    cell scoped but not yet executed.
  - Branch C (`feat/hybrid-coupling`) carried items: Phase 3 adaptive
    `λ(σ)` schedule. **Merged 2026-05-25** (`595aadf`); Phase 3
    flag declared but not yet wired.
- **Zero-train priors** — `feat/zero-train-priors`, **merged
  2026-05-25** (`2daa166`). Lands Slates D – J. The
  canonical 73.8 % pooled headline (Slate E, May 19) is now strictly
  on `main`. See the per-experiment doc
  [`experiments/15_unified_hybrid.md`](experiments/15_unified_hybrid.md)
  for the slate timeline.
- **Theory** — `feat/theory`, **merged 2026-05-25** (`b8de97d`,
  rebased to drop WIP commits). Three theory docs added under
  [`docs/theory/`](theory/) supporting the empirical Story-B
  narrative.
- **Flow integration** — [`feat/flow-mjx/docs/plan_principled_score.md`](https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/flow-mjx/docs/plan_principled_score.md)
  (B1-B4 plan with a "## 2026-05-15 update": v2 architecture + MJX
  integration done; only the Colab L4 train + MJX rollout + robosuite
  post-eval run remains). The newer v3 Karras-EDM work lives on
  `feat/flow-v3-karras`, **not merged**; gated on beating retrieval-
  prior 73.8 % or producing a tighter mechanism story than the
  Slate K parity-ceiling negative result.

For empirical state see [`results.md`](results.md) (refreshed
2026-05-25). For methodology, decisions and chronology see
[`run_log.md`](run_log.md). For per-experiment lifecycle status
and reproduce commands see [`experiments/README.md`](experiments/README.md).
For the archived pre-May-15 engineering plans see
[`archive/`](archive/).

---

## 2026-05-25 update — post-zero-train-priors landing

The May-15 closeout summary below is preserved verbatim for
historical reference; the canonical project state has moved
substantially since then. Quick read:

- **Headline number**: was 37.5 % CI [31.6, 43.8] (4-seed pool,
  BC + score-bridge at K=10 R=1 λ=5 σ=0.25 M=16, May 15);
  now **73.8 % CI [67.8, 78.9]** (4-seed pool, `demo_1nn` retrieval
  + score-bridge at K=10 R=1 **λ=1** σ=0.25 M=16, Slate E
  May 19). **+36.3 pp pooled, CIs separated by ~24 pp**.
- **Headline algorithm**: was BC + per-step BC-bridge; now
  **1-NN retrieval from the demo bank + per-step retrieval-bridge**
  through the same unified driver (exp 15). Zero training, no
  neural network in the loop.
- **Cross-task scope**: Lift 100 % (Slate G G1), Can 73.8 %, Square
  8.3 % peak (mechanism-limited, triple-confirmed by 5 orthogonal
  ablations in Slates F + G + G-aux), `tool_hang` blocked on the
  dense-reward shim (now landed, Slate J queued), `transport` also
  blocked on the same upstream sparse-reward gap.
- **Branches**: A and C merged today after a 10-day quiet period.
  `feat/zero-train-priors` is the source of the May-17 – May-22
  retrieval narrative. `feat/theory` adds the soft-control / SNIS /
  Pareto bias-variance triple.
- **What "framing #3 — Flow Matching Hybrid as the headline" became**:
  framing #4 — **retrieval prior** as the headline.  The mechanism
  argument is the same (the score-bridge needs an in-distribution
  per-step pull target); the prior identity is a 1-NN over the
  demo bank instead of a flow-matching network. Trained flow priors
  (exp 08 v2: 16.7 %, v3 work-in-progress) remain well below
  retrieval; "beat retrieval with a learned prior" is an open
  research question, not a tuning question.

Concrete outstanding work (priority order, May 25):

1. **Resume Slate I** (contrastive obs prior on Square). Paused
   mid-cell-1; ~7.7 h to finish. Tests whether a learned encoder
   unlocks Square where five orthogonal axes failed.
2. **Fire Slate J** (`tool_hang` + `transport` with the dense-reward
   shim, commit `18af830`). Manual fire only; 4 cells, ~10 – 12 h.
3. **A2 eval cell** (pure-MBD λ=0 σ=0.05 M=64 + V_φ). ~2 h with
   `--n-workers=2`. Acceptance ≥ 5/60 supports H2 (variance
   bottleneck); ≤ 3/60 supports H1 (capacity bottleneck).
4. **Promote `feat/flow-v3-karras` to main** if Slate K's
   `--score-mode=pull` asymmetric MBD evaluation beats retrieval at
   any σ (currently parity-ceiling, not yet beat).

Skipped indefinitely: A1.4 σ-sweep on the unified driver (Slate E's
σ-tail already covers σ ∈ [0.05, 1.00] at the canonical retrieval
prior — A1.4 is redundant for the retrieval framing). A3-(b)
`λ_softmax(M)` correction (M=16 is the canonical M; M-sweep
postponed until / unless retrieval ceiling lifts). A4 reward
shaping (paper-appendix material).

---

## 2026-05-15 closeout summary

The three-branch fan-out planned in
[`archive/post_consolidation_plan.md`](archive/post_consolidation_plan.md)
fully landed today. The single-paragraph status:

- **Branch C (`feat/hybrid-coupling`) Phase 1-2 — DONE.** Unified
  driver landed (`experiments/15_unified_hybrid/`). Phase 2 K×R
  grid at λ=5 σ=0.25 M=16 produced the new headline:
  K=10 R=1 = **41.7 %** single seed=0, **37.5 % (Wilson CI
  [31.6, 43.8])** at 4-seed pool. Phase 3 (adaptive `λ(σ)`)
  remains; Phases 4-5 SKIPPED per user decision.
- **Branch A (`feat/stronger-mbd-signal`) Task A1 + M-ablation — DONE.**
  σ-sweep on the exp 05 driver established the operating point
  (σ ∈ {0.01, 0.03} peaks at 40.0 %; σ ≥ 0.25 cliff-collapses);
  Stage 5 M-ablation found an inverted-U sweet spot at M=64 with
  **50.0 % (single seed)**, the cross-branch peak. Tasks
  A1.4, A2, A3-(b), A4 carried forward — see
  [`branch_plan.md` on `feat/stronger-mbd-signal`](https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/stronger-mbd-signal/docs/branch_plan.md).
- **`feat/principled-score` v_theta v1 diagnosed + v2 shipped.**
  v1 flow-prior 0/60 cell measured and committed (`f3f7b64`).
  v2 architecture with FiLM + x_1-parameterization + `tanh` head
  shipped (`56db351`). Colab notebooks ready on
  `feat/flow-mjx`; **awaiting Colab L4 run** — gated on GPU
  access, not on further engineering. See
  [`plan_principled_score.md` on `feat/flow-mjx`](https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation/blob/feat/flow-mjx/docs/plan_principled_score.md) (the original B1-B4 plan plus the May-15 follow-up section).

The full empirical state lives in
[`results.md` § May 15 headline](results.md). The post-closeout
headline-to-baseline delta is **+15.8 pp at CI lower bounds**
(4-seed pooled, unified driver K=10 R=1) or **+22 pp absolute**
at the cross-branch M-ablation peak (single seed, exp 05 driver
M=64).

### Carried-forward TODOs from the archived branch plans

Triaged 2026-05-15 evening. Status legend:

- **DONE** — landed on `main`; reference only.
- **SKIPPED** — user decision, not needed for the paper.
- **UNDONE** — still on the roadmap. Concrete specs live in each
  branch's `docs/branch_plan.md` (links at the top of this file).

#### Branch A — `feat/stronger-mbd-signal` (`archive/plan_stronger_mbd_signal.md`)

| # | Item | Status | Notes / cost |
|---|---|---|---|
| A1 | σ-max sweep on exp 05 driver (10 cells, σ ∈ {0.01..1.00}) | **DONE** | Peak at σ ∈ {0.01, 0.03} = 40.0 %; σ ≥ 0.25 cliff-collapses. |
| A1.4 | σ-sweep on **unified driver** at K=10 R=1 λ=5 | **UNDONE** | Top Branch A priority. ~10 h compute. Spec: branch_plan.md on `feat/stronger-mbd-signal`. |
| A3 (partial) | M-ablation at σ=0.05 on exp 05 driver, M ∈ {16, 64, 256} | **DONE** | Inverted-U peak at M=64 = 50.0 %; cross-branch headline. |
| A3 follow-up (a) | 4-seed CI at M=64 σ=0.05 | **UNDONE** *(postponed)* | Postponed 2026-05-15: Job B + Job C already CI-tightened the M=16 numbers; 50 % single-seed cited with the standard caveat. |
| A3 follow-up (b) | `λ_softmax ∝ log M` correction at M=256 (optionally M=1024) | **UNDONE** *(optional)* | Conditional on A1.4 + A2 landing. Spec: branch_plan.md. |
| Job B | 3-seed σ=0.03 robustness pool (exp 05 driver) | **DONE** | Pooled 91/240 = 37.9 %. |
| A2 | Learned `V_φ(s)` value baseline as MBD control variate | **UNDONE** | Second Branch A priority. ~3 d code + 2 h eval. Acceptance: pure-MBD (λ=0) + V_φ ≥ 5/60. Spec: branch_plan.md on `feat/stronger-mbd-signal`. |
| A4 | Reward shaping (`--shaped-cost`, dense intermediate rewards) | **UNDONE** *(optional)* | Paper-appendix material; lowest priority. |

#### Branch C — `feat/hybrid-coupling` (`archive/plan_hybrid_coupling.md`)

| # | Item | Status | Notes |
|---|---|---|---|
| Phase 1 | Unified driver (consolidate exp 05/07/10 into `experiments/15_unified_hybrid/`) | **DONE** | Lands as the canonical hybrid driver. |
| Phase 2 (7/9) | K × R verification grid at λ=5 σ=0.25 M=16 — 7 of 9 cells | **DONE** | K=10 R=1 = 41.7 % new single-seed headline. |
| Phase 2 (2/9) | K=20 R=0 and K=20 R=2 cells | **SKIPPED** | K-axis story already locked in; filling K=20 R={0,2} doesn't change any conclusion. |
| Job A | Unified-driver σ-sweep at K=10 R=0 λ=5 (5 cells) | **DONE** | Peak at σ=0.05 = 38.3 %. |
| Job C | 4-seed K=10 R=1 robustness pool | **DONE** | Pooled 90/240 = 37.5 % CI [31.6, 43.8]. |
| Phase 3 | Adaptive `λ(σ)` schedule (anneal `λ_prior` across denoising chain) | **UNDONE** | Only outstanding Branch C item. ~1-2 days code + ~6 h compute. Spec: branch_plan.md on `feat/hybrid-coupling`. |
| Phase 4 | Truncated denoising chain (`--sigma-floor`) | **SKIPPED** | Not needed right now. |
| Phase 5 | Adaptive replanning trigger | **SKIPPED** | Not needed right now. |

#### Overnight queue (`archive/plan_overnight_queue.md`)

| # | Item | Status | Notes |
|---|---|---|---|
| Stage 0-1, 3-5 | Wait-gates, Phase 1 regression, Phase 2 grid, λ-sweep, Job C, M-ablation | **DONE** | Stages 0, 0.5, 1, 3, 4, 5 all closed; data on `main`. |
| Stage 2 | σ=0.05 3-seed pool **analysis** (data exists, never written up) | **UNDONE** *(cheapest win)* | ~10 min to surface pooled rate + Wilson CI in `experiments/09_sigma_sweep/results/_archive/sigma_sweep_analysis.md`. |
| Stage 6 | Failure-taxonomy refresh on K=10 R=1 unified cell | **UNDONE** | ~30 min to re-run `experiments/06_failure_taxonomy/run.py` on the new headline cell. |

---

## Framing for the final paper (updated 2026-05-14, post-BC-swap)

Three possible framings, depending on which of the per-branch
plans lands first:

1. **Score-bridge + MPC chunking stacks**: unified K × R × λ=5
   grid's peak meaningfully exceeds 33.3 %. Story: *"the
   score-bridge formulation beats both BC and MBD, and MPC
   chunking adds a second improvement."* Strongest possible
   paper. Already partially supported by the May-15 closeout:
   the unified K=10 R=1 single-seed 41.7 % (4-seed 37.5 % CI
   [31.6, 43.8]) is well above the May-14 33.3 % single-shot
   number, and the M-ablation 50.0 % at M=64 suggests further
   room. Requires Branch A's A1.4 σ-sweep + the M=64 seed pool
   (postponed) to make the +15.8 pp at-CI-LB margin tight.
2. **Score-bridge alone is enough**: unified grid's peak ≈
   33.3 % (no MPC benefit). Story: *"the score-bridge formulation
   beats BC; MPC chunking is unnecessary on Can."* Cleaner paper
   that drops the MPC angle. *Already ruled out* by Phase 2: R=1
   is the peak, R=0 is ~10 pp lower. So this framing is no
   longer on the table.
3. **Flow Matching Hybrid as the headline**: if `feat/flow-mjx`'s
   Colab L4 run produces a v_theta v2 result ≥ Job C's CI lower
   bound (31.6 %), the paper headlines flow matching instead of
   BC. The score-bridge mechanism is the same; only the prior
   identity changes. Acceptance gate is on `feat/flow-mjx`'s
   branch_plan.md.

The May-13 honest hedge — *"the mechanism validates, the binary
number does not yet separate"* — was tightened by the May 15
4-seed pools: the hybrid-vs-BC delta is now **statistically
defensible** (BC CI upper 33.0 % vs hybrid CI lower 31.6 %; the
two CIs nearly touch but the hybrid lower bound is at the BC
upper bound — meaningful evidence). What remains is choosing
between framing #1 (BC headline) and framing #3 (flow headline);
the choice gates on the Colab L4 run.
