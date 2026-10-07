# Next-steps research note — 2026-05-25 18:00 PT

*Companion to `plan_2026_05_25.md` in the main worktree. That
file is the rolling status; this one is the **researched** view
of what comes after the demo-scaling slate lands.*

> **Closure note (2026-06-01)**: the demo-scaling slate landed
> and the subsequent T2 4-knob robustness arc completed.  All 7
> follow-up branches merged into `main`; see
> [`05_followup_plan.md`](05_followup_plan.md) "Closure
> (2026-06-01)" for the merge-status table and
> [`docs/results_writeups/pareto_analysis.md`](../results_writeups/pareto_analysis.md)
> for the post-hoc Pareto-dominance verdict.  The research note
> below is preserved as a historical record of the
> live-from-2026-05-25 view of "what comes next".

## 0. TL;DR

The slate currently running on Can will, by ~21:00 PT tonight,
produce the **first quantitative scaling law in this project**.
Once it lands, the highest-leverage next moves are, in order:

1. **Aggregate + figure** for the 6-cell scaling curve (~30 min).
2. **Cost-time worker-propagation fix** (~30 min of focused
   editing across 4 files; the smoke is already on disk from the
   previous attempt).
3. **Cost-time slate** (4 cells × 3 methods × 50 min ≈ 10 h
   wall), assuming the fix passes smoke. This is the first
   experiment that directly substantiates limitation §8.1 #5
   of the presentation brief.
4. **Brief updates** — strike §8.1 #2 (no scaling argument)
   immediately; partial strike of §8.1 #5 after cost-time lands.

Dynamics, incremental-prior, and SNIS-bound writeup follow.

## 1. Slate-completion analysis plan

The demo-scaling slate writes per-cell JSONs to a single dir:

```
experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25/
  unified_demo_1nn_lam-1.0_K-10_R-1_Nb{5,10,20,50,100,200}.json
```

Each contains the per-rollout reward+success array + the params
block (with `prior_bank_size`). The aggregation pipeline is
straightforward:

### 1.1 Aggregation script (to write in HRTG-slate)

`experiments/15_unified_hybrid/aggregate_demoscale.py`:

- For each Nb in {5, 10, 20, 50, 100, 200}, load the JSON,
  compute success-rate + Wilson 95% CI.
- Decompose each Nb's eval into **in-bank** (`demo_i` with
  `i < Nb`) and **out-of-bank** rollouts; report per-population
  rates. (At Nb=20, all eval states are in-bank, so the
  decomposition is degenerate but still tabled.)
- Emit a CSV + a matplotlib figure
  (`scaling_law.png` + `scaling_law.pdf`).

### 1.2 Figure design

A single **two-panel figure**:

- **Left panel.** Log-x scaling curve. x = `log2(N_b)`,
  y = success rate, Wilson 95% CI bars. Annotate the canonical
  73.8% baseline as a horizontal dashed line. **Annotate the
  knee** at the predicted N_b ≈ 20 (where bank coverage = eval
  set size).
- **Right panel.** Mode-coverage decomposition. For each Nb,
  two bars side-by-side: in-bank-rate (blue) and OOB-rate
  (orange). At Nb ≥ 20 the OOB bar disappears (full coverage).
  This panel makes the **mechanism** of the scaling law visible
  — it's not just "more data = better", it's "more modes
  covered = more states with low-variance anchored proposals".

### 1.3 Theoretical interpretation

The shape we expect:

| N_b | Pred. rate (overall) | Mechanism reading |
|---:|---:|---|
| 5 | ~20-25% | mode coverage 25%; IB ≈ 80%, OOB ≈ 0% |
| 10 | ~35-50% | mode coverage 50%; IB ≈ 60% (aliasing kicks in), OOB ≈ 10% |
| 20 | ~70-75% | full mode coverage; IB ≈ 73% (canonical) |
| 50 | ~70-75% | saturated; possible overshoot to 80% if extra demos sharpen the 1-NN |
| 100 | ~70-75% | saturated |
| 200 | 73.8% | canonical baseline reproduced |

**If the curve actually saturates flat from N_b=20 onward**,
Theorem 1' Corollary 1 is confirmed: $\rho(\bar\tau)$ is
**bank-coverage-limited**, not bank-size-limited. This is a
publishable finding.

**If the curve continues to climb past N_b=20** (e.g.,
N_b=200 reaches 85%), Corollary 1's mode-anchoring story is
incomplete: extra demos provide some benefit beyond mode coverage.
This would motivate a refinement to the corollary admitting
within-mode density effects.

### 1.4 Already-observed surprise (live cell)

Nb50 mid-cell is showing **85% success rate** in the first 13
rollouts (more than the canonical 73.8%). If this rate holds to
the cell end (~30 min from now), the curve is **non-monotone** —
peaks at Nb=50 and then comes back down to 73.8% at Nb=200. This
is a real finding worth its own paragraph in the analysis:
extra demos eventually start to **hurt** because the 1-NN starts
picking similar-but-suboptimal matches when the bank is dense.
This is the **density-vs-coverage tradeoff** predicted as a
follow-up corollary to `02_snis_bound.md`.

## 2. Cost-time worker-propagation fix (concrete patch)

The blocking item is the only unchecked box in
`docs/adaptation/01_cost_time_spec.md` §3:

> Cost-function pass-through. Verify
> `experiments/04_mbd_baseline/run_can.cost_fn` reads the goal
> from the live env state ... REMAINING.

The cost_fn itself is goal-agnostic (`run_can.py:178` just returns
`-rewards`), so the issue is upstream: the workers' envs.

### 2.1 Root cause

`experiments/04_mbd_baseline/worker.py:28` defines
`_worker_init(dataset_path_str)`. Each worker creates a fresh env
via `make_env_for_dataset`, with the canonical (unperturbed)
goal. The main process applies `apply_goal_perturbation`, but
the workers never call it. So:

- Main env's `bin2_pos`: perturbed (after my 05edaf6 fix).
- Workers' envs' `bin2_pos`: unperturbed (canonical).
- The rewards computed by workers (= cost_fn input for MBD) are
  computed against the **canonical** goal, not the perturbed one.

This silently disables the perturbation for the MBD planning loop
— the very mechanism we want to test.

### 2.2 The patch (4 files, ~50 lines)

**File 1: `experiments/04_mbd_baseline/worker.py`**

```python
def _worker_init(
    dataset_path_str: str,
    task: str | None = None,
    goal_xyz: tuple[float, float, float] | None = None,
    mass_scale: float | None = None,
    friction_scale: float | None = None,
) -> None:
    global _worker_env
    from common import make_env_for_dataset

    _worker_env, _ = make_env_for_dataset(Path(dataset_path_str))
    if hasattr(_worker_env, "reward_shaping"):
        _worker_env.reward_shaping = True

    # Propagate runtime env perturbations (cost-time / dynamics)
    # from the main process to this worker.
    if task is not None and goal_xyz is not None and goal_xyz != (0.0, 0.0, 0.0):
        # Defer the import so workers that don't perturb don't pay
        # for the exp15 import.  The exp15 dir must be on sys.path,
        # which the module-level _p loop above arranges.
        sys.path.insert(0, str(_EXPERIMENTS_DIR / "15_unified_hybrid"))
        from env_perturbations import apply_goal_perturbation
        apply_goal_perturbation(_worker_env, task, goal_xyz)

    if mass_scale not in (None, 1.0) or friction_scale not in (None, 1.0):
        sys.path.insert(0, str(_EXPERIMENTS_DIR / "15_unified_hybrid"))
        from env_perturbations import apply_dynamics_perturbation
        apply_dynamics_perturbation(
            _worker_env,
            mass_scale=mass_scale,
            friction_scale=friction_scale,
        )
```

**File 2: `experiments/04_mbd_baseline/run_can.py`** — `make_rollout_fn` signature:

```python
def make_rollout_fn(
    n_workers: int,
    dataset_path: Path,
    perturbations: dict | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...
    init_args = (
        str(dataset_path),
        perturbations.get("task") if perturbations else None,
        perturbations.get("goal_xyz") if perturbations else None,
        perturbations.get("mass_scale") if perturbations else None,
        perturbations.get("friction_scale") if perturbations else None,
    )
    pool = ctx.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=init_args,
    )
```

**File 3: `experiments/15_unified_hybrid/unified_loop.py`** —
`unified_plan_and_execute` and `make_rollout_fn` callers need to
forward `perturbations`. ~5-line addition.

**File 4: `experiments/15_unified_hybrid/run_can.py`** — build the
`perturbations` dict from CLI args and pass through:

```python
perturbations = {
    "task": args.task,
    "goal_xyz": tuple(args.goal_perturbation_xyz),
    "mass_scale": getattr(args, "mass_scale", None),
    "friction_scale": getattr(args, "friction_scale", None),
}
```

### 2.3 Smoke test

A single-rollout smoke at the perturbed cell verifies the worker's
env receives the perturbation:

```bash
python experiments/15_unified_hybrid/run_can.py \
    --task can --prior demo_1nn --lambda-prior 1 \
    --K 4 --M 16 --sigma-max 0.25 --mpc-replans 1 \
    --n-workers 2 --n-demos 1 --n-rollouts 1 \
    --goal-perturbation-xyz 0.20 0.00 0.00 \
    --smoke --output-name-suffix=_worker_pert_smoke \
    --output-dir /tmp/worker_pert_smoke
```

Expected: rollout reward drops noticeably vs canonical, because
the worker's reward fn now sees the moved bin2. Diagnostic: a
single `print(_worker_env.bin2_pos)` in `_worker_init` after the
perturbation step (remove before commit).

## 3. Cost-time slate plan (after fix smoke passes)

Per `01_cost_time_spec.md` §2:

```
4 cells × 3 methods × 60 rollouts each = 720 evaluations total.
Wall-clock: ~10 h (or ~5 h if BC + MBD reuse warm envs).
```

| Cell | Δx (m) | Predicted hybrid | Predicted BC | Predicted MBD-no-prior |
|---|---|---:|---:|---:|
| C1.1 | (0.00, 0, 0) | 75% | 22% | 0% |
| C1.2 | (0.05, 0, 0) | 70% | 10% | 0% |
| C1.3 | (0.10, 0, 0) | 50% | 0% | 0% |
| C1.4 | (0.20, 0, 0) | 30% | 0% | 0% |

The key signal: **hybrid-vs-BC gap widens monotonically with
Δx** while hybrid's per-seed standard error stays in the
canonical ~2 pp envelope. This is the empirical companion to
`01_cost_time_spec.md` §5.1 ("Theorem 1' specialised to
cost-time"): the variance term is constant in Δx; only the bias
term grows.

## 4. Dynamics experiment design (independently of cost-time)

Per `02_dynamics_spec.md` (`feat/adapt-dynamics`, not yet read in
this session). Concretely on Can:

```
--mass-scale ∈ {0.7, 1.0, 1.3, 1.5}
--friction-scale ∈ {0.5, 1.0, 1.5, 2.0}     (orthogonal sweep)
```

The worker-propagation fix from §2.2 also handles dynamics
perturbations (the `mass_scale` / `friction_scale` kwargs are in
the patch). So **once cost-time's worker fix smokes clean, dynamics
is fire-and-forget**.

The two adaptation axes test different parts of $p^{\mathrm{model}}$:

| Experiment | Changes which factor of $J$ | What it tests |
|---|---|---|
| Cost-time | the running cost $\ell(x, u, k)$ via goal | "can the hybrid auto-retarget the planner?" |
| Dynamics | the constraint $x_{k+1} = f(x_k, u_k)$ | "can the hybrid handle perturbed physics?" |

Cost-time is the **cleanest pitch** (no retraining, no
representation question), dynamics is the **more impressive
demonstration** (the env is genuinely OOD from the demos).
Running both makes §8.1 #5 of the brief a strike-through.

## 5. Brief updates after each milestone

Mapping milestones to specific paragraphs in
`presentation_brief_2026_05_22.md`:

| Milestone | Paragraph(s) updated |
|---|---|
| Demo-scaling lands tonight | §8.1 #2 (no scaling argument): **strike**, replace with "scaling law shown: saturates at N_b ~= eval-coverage threshold; see Figure X". §8.2 #6: **strike** (moved from future-work to done). §9: add **claim #6** ("the retrieval prior obeys a bank-coverage scaling law"). §12 one-slide summary: append a bullet on the scaling figure. |
| Cost-time lands +24h | §8.1 #5 (sim-only / no perturbation): **partial strike** — leave the "no real-robot" half. §9: add **claim #7** (cost-time adaptation: hybrid-vs-BC gap widens with goal displacement). §12: append a bullet. |
| Dynamics lands +48h | §8.1 #5: **full strike** — both perturbation axes covered. §9: add **claim #8**. §12: append. |
| SNIS bound writeup | §8.3 already has the bound stated; the writeup is the 2-3 page theory section it points to. **Strike §8.1 #7** (theory is empirical-only) once the writeup lands. |

After all four milestones, the brief's limitations list drops
from 13 items to 9, and §9's publishable claims grow from 5 to
8.

## 6. Literature positioning (for the paper, not the brief)

Three literatures the scaling-law result connects to:

### 6.1 Retrieval-augmented imitation / planning

- **VINN** (Pari et al., NeurIPS 2022): 1-NN over visual features
  for imitation. No scaling argument given.
- **R3M** (Nair et al., CoRL 2022) and **VINNIE** (Cui et al.,
  RA-L 2023): contrastive embeddings for retrieval, again no
  scaling argument.
- **Retrieval-augmented diffusion planning** (Hua et al., RA-L
  2024): closest in spirit; uses retrieved demos to condition
  the diffusion prior. Their bank scaling is implicit; doesn't
  produce a curve.

**Our positioning.** First retrieval-prior + online-control method
to produce an explicit **scaling law** with theoretical prediction
(Corollary 1 of `02_snis_bound.md`) and empirical confirmation.

### 6.2 Neural scaling laws

- **Kaplan et al. 2020**, **Hoffmann et al. 2022 (Chinchilla)**:
  loss vs. params, data, compute. All study trained models.
- **Henighan et al. 2020 (autoregressive scaling)**: extension to
  generation tasks; still trained.

**Our positioning.** A scaling law for a **parameter-free** prior
(retrieval over the demo bank). The relevant axis is not
parameters or compute, it's **bank coverage** of the deployment
state distribution. Qualitatively different — saturation comes
from coverage, not capacity.

### 6.3 SNIS variance and self-normalised estimators

- **Owen 2013, Ch. 9**: classical SNIS theory.
- **Whiteley & Lee 2021** (Stat Sci review): SNIS in SMC samplers.
- **Pan et al. 2024 (MBD)** explicitly cites SNIS variance as a
  limitation.

**Our positioning.** Theorem 1 + Corollary 1 of `02_snis_bound.md`
give the first **explicit variance-reduction bound** for an
anchored SNIS proposal (anchored at the retrieval-prior's
nearest demo). The scaling law in this slate is the **empirical
companion** that confirms the bound's quantitative shape.

## 7. Risk register

Things that could go sideways in the next 48 h:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Nb50 / Nb100 / Nb200 don't replicate canonical 73.8% | low | medium | seed sensitivity at single-seed is real; reproduce at canonical N_b=200 with seed=1 if it lands < 65% |
| Cost-time worker fix breaks demo-scaling worktree | low | low | isolated worktrees solve this; the fix lives on `feat/adapt-cost-time`, slate runs from `HRTG-slate` on `feat/adapt-demo-scaling` |
| `apply_dynamics_perturbation` is stub on cost-time branch | high | low | only matters for dynamics; merge dynamics impl in before firing that slate |
| 4-cell cost-time slate exceeds 12 h wall | medium | low | drop BC + MBD-no-prior columns; rerun later |
| Theorem 1' Corollary 1 falsified by Nb=200 climbing past 78% | low | high | revise corollary to admit within-mode density effects; spec a "density vs coverage" follow-up |

## 8. What I'm NOT planning (and why)

Following the standing instruction (no hyperparameter sweeps
until codebase is consolidated):

- **No λ-fine-grain sweep** (originally a Slate-E follow-up).
  Deferred — would need single canonical algorithm settled first.
- **No 2-D (λ, σ) grid** for the brief's drawback #11. Same
  reason.
- **No Square contrastive embedding retry**. Slate I was the
  attempt and showed the limit is on the cost-function side, not
  retrieval. Pursuing further on Square is low ROI until the
  bridge formulation is revisited.
- **No SOTA reproduction** (Diffusion Policy, Diffuser). The
  brief's §7.2 framing handles this in prose; running their
  numbers would be 1-3 days for a citation.

These are recorded so future-me / collaborators can see they're
deferred by **deliberate scope discipline**, not oversight.

---

*End of next-steps note. Next action when demo-scaling lands:
write `aggregate_demoscale.py` per §1.1, then the scaling-law
figure per §1.2.*
