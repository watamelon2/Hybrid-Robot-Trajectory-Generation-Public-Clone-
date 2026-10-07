# Branch plan: `feat/hybrid-coupling` (Branch C)

Companion to [`post_consolidation_plan.md`](post_consolidation_plan.md)
on `main`. This doc specifies the engineering steps for *this*
branch only.

**Branch goal**: unify Exp 05/10's warm-start-only `run_can.py`
with Exp 07's score-bridge `run_can.py` into a single canonical
driver, run the verification grid, and execute the three
coupling ablations (adaptive λ, truncated denoising, replanning
trigger). This branch is **BLOCKING** for branches A and B's
headline-evaluation runs.

---

## Phase 1: codebase consolidation (BLOCKING)

### Deliverable

`experiments/15_unified_hybrid/run_can.py` — a single driver
that is a strict superset of both Exp 05's and Exp 07's CLIs,
and that **reproduces** both runners' headline numbers bit-for-
bit when given the right flags.

### Why a new directory, not a v2 inside 05 or 07

The old Exp 05 / 07 / 10 result JSONs reference their original
runner paths in the metadata; renaming or mutating those runners
breaks reproducibility of the JSONs already on `main`. A new dir
is the least-invasive change. The old runners get a one-line
deprecation banner pointing here.

### Module structure

```
experiments/15_unified_hybrid/
├── __init__.py
├── README.md
├── run_can.py            # CLI driver
├── unified_loop.py       # the actual hybrid loop (score-bridge + MPC)
├── regression_test.py    # asserts exp 07 lambda=5 and exp 10 K=10 R=2 match
└── results/              # JSON outputs go here
```

Splitting `run_can.py` (argparse / IO) from `unified_loop.py`
(the actual algorithm) keeps the test in `regression_test.py`
small — it imports `unified_loop` and calls it with stubbed args.

### CLI surface (union of 05 + 07 + new flags)

| flag                             | type    | default                                       | source        |
|----------------------------------|---------|-----------------------------------------------|---------------|
| `--task`                         | str     | `"can"`                                       | both          |
| `--dataset`                      | Path    | auto                                          | both          |
| `--prior`                        | choices | `bc` (was `flow` in exp 07)                   | 07 (extended) |
| `--prior-checkpoint`             | Path    | `experiments/03_bc_baseline/results/policy.pt` | 07           |
| `--bc-checkpoint`                | Path    | same                                          | 07 alias      |
| `--lambda-prior`                 | float*  | `5.0`  (was nargs="+" sweep in exp 07)        | 07            |
| `--lambda_`                      | float   | `0.5`                                         | both          |
| `--warm-start / --no-warm-start` | bool    | True                                          | 07            |
| `--n-demos`                      | int     | `20`                                          | both          |
| `--n-rollouts`                   | int     | `3`                                           | both          |
| `--H`                            | int     | `120`                                         | both          |
| `--M`                            | int     | `16`                                          | both          |
| `--K`                            | int     | `10`                                          | both          |
| `--sigma-max`                    | float   | `0.25`                                        | both          |
| `--sigma-min`                    | float   | `0.05`                                        | both          |
| `--seed`                         | int     | `0`                                           | both          |
| `--n-workers`                    | int     | `1`                                           | both          |
| `--device`                       | str     | `"cpu"`                                       | both          |
| `--mpc-replans`                  | int     | `0`                                           | 05            |
| `--mpc-chunk-steps`              | int     | `0` (auto = H//(R+1))                         | 05            |
| `--deterministic-bc-reset`       | bool    | `True` (flip of exp 05's default; matches exp 07's de-facto behaviour) | 05 |
| `--output`                       | Path    | `experiments/15_unified_hybrid/results/<auto>.json` | 05      |
| `--smoke`                        | flag    | False                                         | 07            |
| `--live-view / --no-live-view`   | bool    | False                                         | both          |
| **`--prior` choices extension**  |         |                                               | NEW           |
| ··· `bc`                         |         | exp 03 MLP-BC, deterministic head             | exp 07 alias  |
| ··· `flow`                       |         | exp 08 flow-matching prior                    | exp 07 alias  |
| ··· `gaussian_bc`                |         | (Branch B B4) Gaussian-head BC                | NEW (no-op until B4) |
| **`--value-baseline-ckpt`**      | Path    | None (no-op)                                  | NEW (Branch A) |
| **`--shaped-cost`**              | flag    | False                                         | NEW (Branch A) |
| **`--lambda-schedule`**          | choices | `constant` (others: `linear`, `inverse_linear`) | NEW (Phase 3) |
| **`--K-truncate`**               | int     | `K` (no truncation)                           | NEW (Phase 4) |
| **`--replan-trigger`**           | choices | `always` (other: `uncertainty`)               | NEW (Phase 5) |

The "no-op" entries for `--value-baseline-ckpt`, `--shaped-cost`,
`--prior gaussian_bc` exist in Phase 1 so branches A and B can
target the unified driver immediately; their *implementations*
land on their respective branches.

### Algorithm (pseudocode)

```python
def unified_hybrid(args):
    prior = load_prior(args.prior, args.prior_checkpoint)
    value_baseline = load_value(args.value_baseline_ckpt)  # may be None
    cost_fn = make_cost_fn(env, shaped=args.shaped_cost,
                           value_baseline=value_baseline)

    n_chunks = args.mpc_replans + 1
    chunk_steps = args.mpc_chunk_steps or (args.H // n_chunks)

    for demo_idx, init_state in enumerate(initial_states):
        env.reset_to(init_state)
        if args.deterministic_bc_reset:
            env.seed(args.seed * 10_000 + demo_idx)

        for chunk in range(n_chunks):
            s_0 = env.get_obs()
            initial_mean = prior.sample_initial_mean(s_0) if args.warm_start else zeros
            score_fn = prior.make_score_fn(s_0, K=args.K)
            lambda_schedule = make_lambda_schedule(
                args.lambda_schedule, args.lambda_prior, K=args.K)

            mu = denoise_loop(
                initial_mean,
                cost_fn=cost_fn,
                prior_score_fn=score_fn,
                lambda_schedule_fn=lambda_schedule,
                M=args.M, K=args.K_truncate or args.K,
                sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                lambda_softmax=args.lambda_,
            )

            execute_actions(env, mu[:chunk_steps])
            if env.success():
                record_success(demo_idx, chunk_idx=chunk)
                break
```

Concretely, the unified loop combines:

- Exp 07's `score_fn_factory(s_0, K)` returning `score_fn(mu, k)`
  with the `λ * grad_log_prior` semantics at every denoising
  iteration `k`.
- Exp 07's `initial_mean_fn(env, init_state, s_0)` returning
  a BC-warmstarted trajectory.
- Exp 05's MPC chunking: chunk_steps execution per outer-loop
  iteration with state snapshot/restore.
- Exp 04's `denoise_loop` extended to accept
  `lambda_schedule_fn: Callable[[int], float]` so Phase 3 can
  vary λ with k.

### Phase 1 implementation steps

| step | task                                                                  | files touched                                            |
|------|-----------------------------------------------------------------------|----------------------------------------------------------|
| 1.1  | Create `experiments/15_unified_hybrid/` dir + skeleton                | new                                                      |
| 1.2  | Copy `07/run_can.py` as the starting body (has both score_fn and warm-start) | new `15/run_can.py`, new `15/unified_loop.py`     |
| 1.3  | Port `mpc_plan_and_execute` from `05/run_can.py` into `unified_loop.py` | `15/unified_loop.py`                                  |
| 1.4  | Extend `04_mbd_baseline/mbd.py` `denoise_loop` to accept `lambda_schedule_fn` (default = constant) | `04_mbd_baseline/mbd.py` (additive, no behaviour change) |
| 1.5  | Add no-op flags for Branches A and B (`--value-baseline-ckpt`, `--shaped-cost`, `gaussian_bc` choice) | `15/run_can.py`                                  |
| 1.6  | Write `15/regression_test.py` asserting two reproducibility targets    | new                                                      |
| 1.7  | Add deprecation banners (`05/README.md`, `07/README.md`, `10/README.md`) pointing to `15/` | three READMEs                                |
| 1.8  | `15/README.md` documenting the unified driver + migration guide        | new                                                      |

### Regression test (the Phase 1 acceptance bar)

`15/regression_test.py` runs the unified driver with two flag
combinations and asserts both headline success rates match the
existing JSONs to within ±1/60 (i.e. exact, allowing a single
flip from RNG-implementation differences):

| flag combination                                                   | reproduces                                          | target            |
|---------------------------------------------------------------------|-----------------------------------------------------|-------------------|
| `--prior=bc --lambda-prior=5 --mpc-replans=0 --K=10 --M=16 --sigma-max=0.25` | exp 07 λ=5 single-shot                              | 20/60 = 33.3 %    |
| `--prior=bc --lambda-prior=0.5 --mpc-replans=2 --K=10 --M=16 --sigma-max=0.25` | exp 10 K=10 R=2 (warm-start-only λ-softmax peak)    | 11/60 = 18.3 %    |

If either differs by > 1/60, Phase 1 is **not complete** —
something in the consolidation is semantically off and must be
fixed before merging to main.

### Phase 1 effort estimate

| step | wall time |
|------|-----------|
| 1.1-1.5 (coding) | ~6-8 h |
| 1.6 regression test compute (2 × n=60 runs) | ~3 h |
| 1.7-1.8 (docs) | ~1 h |
| **Total** | **~10-12 h** (one focused day) |

### Phase 1 deliverable commit message preview

```
feat(15_unified_hybrid): consolidate score-bridge + MPC into one driver

Single canonical run_can.py replacing the divergent exp 05 and
exp 07 implementations. Reproduces both headlines:
  - exp 07 λ=5 single-shot: 20/60 = 33.3 %
  - exp 10 K=10 R=2:         11/60 = 18.3 %

Old drivers retained with deprecation banner; their JSONs stay
valid for archival.
```

---

## Phase 2: λ=5 verification grid

### Deliverable

9 cells (K ∈ {5, 10, 20}, R ∈ {0, 1, 2}) at λ-prior=5 on the
unified driver, n=60 each, written to
`experiments/15_unified_hybrid/results/verify_K{K}_R{R}.json`.

### Commands

```bash
# uses the existing parallel-runner pattern from
# scripts/local/run_pareto_full.sh, retargeted at exp 15:
PYTHONPATH=. python -m experiments.15_unified_hybrid.run_can \
    --prior=bc --lambda-prior=5 \
    --K=${K} --mpc-replans=${R} --mpc-chunk-steps=$((120/(R+1))) \
    --n-demos=20 --n-rollouts=3 --seed=0 \
    --output=experiments/15_unified_hybrid/results/verify_K${K}_R${R}.json
```

A small wrapper script `scripts/local/run_unified_verify.sh`
forks 9 cells in parallel (mirrors `run_pareto_full.sh`).

### Acceptance criteria

Three pre-registered outcomes, copying the table from the master
plan:

| outcome                                          | implication                                                         |
|--------------------------------------------------|---------------------------------------------------------------------|
| Best cell > 33.3 %                                | MPC × score-bridge wins; headline number is the new grid max.       |
| Best cell ≈ 33.3 % (within ±3 pp)                 | MPC chunking is neutral; score-bridge alone is the algorithm.       |
| Best cell < 33.3 %                                | MPC chunking *hurts* the score-bridge; document and investigate.    |

### Phase 2 effort estimate

- ~5 h compute (parallel) for the 9 × n=60 grid.
- ~30 min analysis (re-use `experiments/10_kxreplans_grid/analyze.py`
  pointed at the new dir).

---

## Phase 3: adaptive λ(σ) schedule

### Hypothesis

Constant λ across all K denoising steps is sub-optimal. At high
σ (early steps, trajectory is noisy) the BC anchor should
dominate (high λ); at low σ (late steps, trajectory is near
final) MBD's local search should dominate (low λ).

### Tasks

- 3.1: Implement `make_lambda_schedule` in `unified_loop.py`
  - `constant`:  λ(k) = λ_prior
  - `linear`:    λ(k) = λ_prior · (K − k) / K       (high early, zero at end)
  - `inverse_linear`: λ(k) = λ_prior · k / K         (low early, high at end)
- 3.2: Sweep all three at the Phase 2 winning cell, n=60.
- 3.3: One-figure result `experiments/15_unified_hybrid/results/lambda_schedule.png`.

### Acceptance criteria

≥ 3 pp lift over Phase 2 winner OR a clean negative result
(all three within ±2 pp, schedule shape doesn't matter).

### Phase 3 effort estimate

~2 h coding, ~3 h compute.

---

## Phase 4: truncated denoising chain

### Hypothesis

With a strong BC anchor (the new [256, 256] BC reaches 21.7 %
alone), the planner doesn't need 10 denoising iterations; it
needs maybe 3-5 fine-tuning steps. Longer chains waste compute
and may erode the BC structure.

### Tasks

- 4.1: K ∈ {3, 5, 7, 10} at the Phase 2 + Phase 3 winning
  configuration, n=60 each.
- 4.2: Plot `success vs K`.

### Acceptance criteria

If K=3 or K=5 matches K=10 within ±2 pp, denoising can be
shortened by 2-3× with no quality loss — a strong compute-
efficiency claim for the paper.

### Phase 4 effort estimate

~1 h coding (just a CLI sweep), ~3 h compute.

---

## Phase 5: replanning trigger

### Hypothesis

Currently MBD runs at every chunk boundary unconditionally. If
we only invoke MBD when a BC-uncertainty estimator says BC is
unsure, we avoid wasting compute on "easy" chunks and may
improve overall success by letting BC handle what it handles
well.

### Tasks

- 5.1: Train an MC-dropout BC ensemble (`experiments/03_bc_baseline/`
  re-trained with dropout=0.1) OR use ensemble-of-5 BC seeds.
- 5.2: Add `--replan-trigger=uncertainty --trigger-threshold=<float>`
  to the unified driver.
- 5.3: Sweep threshold ∈ {0.1, 0.3, 0.5, 1.0 (= always)} at
  Phase 2 + 3 + 4 winning configuration, n=60 each.

### Acceptance criteria

≥ 3 pp lift over Phase 4 winner OR a clean negative result.
Stretch goal: compute reduction (fewer MBD invocations) without
success-rate loss, which is itself paper-worthy.

### Phase 5 effort estimate

~1 day coding (uncertainty estimator + threshold logic),
~3 h compute.

---

## Open questions for the user (Branch C specifically)

1. **Numbering**: I picked `experiments/15_unified_hybrid/`.
   Acceptable? Alternative: `experiments/05_hybrid_method/`'s
   driver becomes the canonical one, with the old machinery
   archived as `run_can_v1.py`. Pro: keeps "exp 05" as *the*
   hybrid experiment. Con: invalidates the existing 05 JSONs
   in their original paths.
2. **Should the unified driver be CPU-only or also wire in the
   `feat/mjx-rollouts` work**? Right now Phase 1 keeps it CPU
   (matches existing 05/07). Wiring mjx is essentially a
   separate scope (Tier 4(h) of `plans.md`).
3. **Phase ordering**: Phases 2-5 are pre-registered as listed.
   But Phase 5 (replanning trigger) is the most speculative.
   Run Phases 1-4 first, then decide whether to do 5? Or
   commit to the full 5-phase rollout up front?

---

## Rollback plan

If Phase 1 fails the regression test (unified driver doesn't
reproduce exp 07 + exp 10 headlines), **do not merge**. Branch
stays open; we investigate. The old `05/07/10` runners remain
the source of truth on `main` until consolidation is verified.
