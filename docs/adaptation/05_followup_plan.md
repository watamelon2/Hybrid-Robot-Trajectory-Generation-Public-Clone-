# Adaptation follow-up plan (post-2026-05-28)

After the `feat/adapt-incremental-prior` branch closed (merge
commit `01a743a`), 7 follow-up branches were checked out as
worktrees off `main @ b752d88` to extend the incremental-
adaptation story.  This doc records the plan, the branch
structure, and the data-prereq convention.

## Branch table

| Branch | Worktree | Wall | Step ref |
|---|---|---:|---|
| `feat/adapt-t2-multiseed` | `HRTG-multiseed` | ~7 h | (1) re-run T2.0-T2.4 at seeds 1, 2 |
| `feat/adapt-t2-democount` | `HRTG-democount` | ~3 h | (2) Can mass=1.3 × demos {1, 2, 3, 10} |
| `feat/adapt-t2-magnitude` | `HRTG-magnitude` | ~6 h | (3) mass {1.1, 1.2, 1.4, 1.5} × {T2.0, T2.1, T2.3} |
| `feat/adapt-t2-square` | `HRTG-square` | ~3-6 h | (4) Square mass=1.3 ladder; BC arm waits on (5) |
| `feat/adapt-bc-multitask-obs` | `HRTG-bcmulti` | ~4 h | (5) BC dataset.py refactor + Lift/Square retrain |
| `feat/adapt-t2-lambda` | `HRTG-lambda` | ~3 h | (6) T2.1 at λ ∈ {0.5, 2, 5} |
| `feat/adapt-pareto-analysis` | `HRTG-analysis` | ~1 day | (7+8) (bias, var) measurement + theory writing |

All 7 branches share the base commit `b752d88` so they can land
on main in any order (modulo the (5) before (4)-BC constraint).

## Dependency graph

```
(1) multiseed       ───┐
(2) demo-count      ───┤
(3) magnitude       ───┤   all parallel, no shared code edits
(6) lambda          ───┤
(7+8) analysis      ───┘

(5) bc-multitask-obs  ─────>  unblocks Square BC retraining
                              │
(4a) Square hybrid     ───────┘   (4b) Square BC after (5) lands
```

Only (5) touches existing code paths (`experiments/03_bc_baseline/
dataset.py`, `train.py`); every other branch adds new launchers
and result dirs only.

## Data-prereq convention (NEW after 2026-05-28)

Issue: `experiments/15_unified_hybrid/data/` is gitignored, and
the perturbed-demo banks (`can_mass1p3_205demos.hdf5` etc.) lived
in the `HRTG-incremental` worktree -- now removed.  Re-collecting
per worktree is wasteful (10 min × 7 worktrees = 70 min).

**Convention:** perturbed banks live in
`/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/`
(outside every worktree).  Each slate references this path via the
`SHARED_DATA` env var:

```bash
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
```

Files expected to live there (build once, reuse across branches):

| File | Producer | Used by |
|---|---|---|
| `can_mass1p3_5demos.hdf5` | `slate_incremental_collect_*` | -- |
| `can_mass1p3_205demos.hdf5` | derived from above | (1), (2), (6) |
| `can_mass1p3_220demos.hdf5` | derived from above | (1) |
| `can_mass1p1_205demos.hdf5` | new collect at mass=1.1 | (3) |
| `can_mass1p2_205demos.hdf5` | new collect at mass=1.2 | (3) |
| `can_mass1p4_205demos.hdf5` | new collect at mass=1.4 | (3) |
| `can_mass1p5_205demos.hdf5` | new collect at mass=1.5 | (3) |
| `can_mass1p3_201..204demos.hdf5` | derive from existing | (2) |
| `can_mass1p3_210demos.hdf5` | derive from existing | (2) |
| `square_mass1p3_5demos.hdf5` | new collect for Square | (4) |
| `square_mass1p3_205demos.hdf5` | derived from above | (4) |

## Merge feasibility -- summary

Yes, all 7 branches should merge cleanly to main with one care:
- (5) must merge before (4)-BC cells run.
- `docs/results.md` updates land via section-stub files
  (`docs/results_writeups/<branch_slug>.md`) per branch, then
  one final integration commit on `main` weaves them in.
  This avoids merge conflicts on the central writeup.

## Execution recommendation

Priority order (highest credibility return first):

1. `feat/adapt-t2-multiseed` -- the single most important
   credibility upgrade.  Without this, every T2 number rests on
   `seed=0`.

2. `feat/adapt-t2-magnitude` -- strongest claim-strengthener.
   Maps the perturbation-magnitude-conditional bite zone
   constructively, turning the headline from a single-point
   demonstration into a characterised regime.

3. `feat/adapt-bc-multitask-obs` -- prerequisite for (4)-BC and
   any future cross-task BC comparison.  Pure code work;
   parallelisable with (1) and (2).

4. `feat/adapt-t2-square` (hybrid first, BC after 5) --
   cross-task confirmation that the T2 framing transfers.

5. `feat/adapt-t2-democount` + `feat/adapt-t2-lambda` --
   ablations that round out the operating-point characterisation.

6. `feat/adapt-pareto-analysis` -- after all data is in, close
   the loop between the bias-variance Pareto theorem and the
   measured action-space (bias, variance) of T2.1 vs T2.3.

The compute budget is ~22 h serial.  With careful queueing on the
existing 5-worker infra, this fits in a long weekend.

## Closure (2026-06-01)

All 7 branches have landed.  Merge status:

| # | branch | status | merge commit | landing date |
|---|---|---|---|---|
| 1 | `feat/adapt-t2-multiseed`    | merged | `d093044` | 2026-05-28 |
| 2 | `feat/adapt-t2-magnitude`    | merged | (multiseed batch) | 2026-05-28 |
| 3 | `feat/adapt-bc-multitask-obs`| merged | (multiseed batch) | 2026-05-28 |
| 4 | `feat/adapt-t2-square`       | merged | `8082c84` | 2026-05-28 |
| 5a| `feat/adapt-t2-democount`    | merged | `544c771` | 2026-05-28 |
| 5b| `feat/adapt-t2-lambda`       | merged | `4636f90` | 2026-05-28 |
| 6 | `feat/adapt-pareto-analysis` | **merged** | `aaefba2` | **2026-06-01** |

Ships the cell-level (bias², variance) proxy decomposition (T2.1
strictly Pareto-dominates T2.3) plus the asymmetric-robustness
theory doc; per-step action-level measurement is deferred to a
journal version (scaffolding in
`experiments/15_unified_hybrid/measure_bias_variance.py`).

See:

  - [`docs/results_writeups/pareto_analysis.md`](../results_writeups/pareto_analysis.md)
  - [`docs/theory/08_asymmetric_robustness_model.md`](../theory/08_asymmetric_robustness_model.md)
  - [`experiments/15_unified_hybrid/results/proxy_bias_variance/bias_variance_proxy.md`](../../experiments/15_unified_hybrid/results/proxy_bias_variance/bias_variance_proxy.md)
