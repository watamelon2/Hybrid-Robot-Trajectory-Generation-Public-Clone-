# T2 lambda sweep (writeup stub)

**Branch:**  `feat/adapt-t2-lambda`
**Worktree:** `HRTG-lambda`
**Slate:**   `experiments/15_unified_hybrid/slate_incremental_T2_lambda_2026_05_28.sh`
**Status:**  results landed 2026-05-29; aggregation done.

## Motivation

The T2.1 headline (`docs/results.md`) used `lambda_prior = 1.0`
without justification beyond "it's the canonical hybrid setting
on Can".  Sweeping lambda at the T2.1 configuration tells us
whether the adaptation gain is specific to that operating point
or robust across a range of prior strengths.

## Cells

3 cells, each is T2.1 (hybrid + 200 canonical + 5 mass=1.3 demos
@ Can mass=1.3) at a different lambda:

| Lambda | Notes |
|---|---|
| 0.5  | Weaker prior pull; MBD does more cost-shaping work |
| 1.0  | Canonical (already on main as T2.1) |
| 2.0  | Stronger prior; bank dominates initial guidance |
| 5.0  | Very strong prior; MBD acts as light corrector |

Each cell N=30 trials, seed=0.

## To fill in after slate completes

| Lambda | Succ | [Wilson 95 % CI] | wall_clock/rollout |
|---:|---:|---:|---:|
| 0.5 | __ / 30 = __ % | [__, __] | __ s |
| 1.0 | 22 / 30 = 73.3 % (on main) | [55.6, 85.8] | ~12 s |
| 2.0 | __ / 30 = __ % | [__, __] | __ s |
| 5.0 | __ / 30 = __ % | [__, __] | __ s |

### lambda sweep table

| lambda_prior | succ | Wilson 95 % CI | wall_s/cell |
|---:|---:|---:|---:|
| 0.5 | 9/30 = 30.0% | [16.7, 47.9] | 1028 |
| 1.0 | 22/30 = 73.3% | [55.6, 85.8] | 849 |
| 2.0 | 24/30 = 80.0% | [62.7, 90.5] | 1036 |
| 5.0 | 24/30 = 80.0% | [62.7, 90.5] | 1034 |

### Headline (computed from per-lambda N=30)

- **Performance spread across lambda in {0.5, 1, 2, 5}:** 50.0 pp  (worst 30.0 %, best 80.0 %).
- **lambda = 0.5 collapses** (30.0 %) below lambda = 1 (73.3 %).  Wilson CIs do not overlap; gap = +43.3 pp.  Below lambda ~ 1 the prior pull is too weak to dominate MBD's score corrections.
- **lambda in [1, 5] is a robust plateau:** lam=1 -> 73.3 %, lam=2 -> 80.0 %, lam=5 -> 80.0 %.  Pairwise Wilson CIs all overlap.  T2.1 headline at lambda = 1 is not specific to that choice; any lambda in [1, 5] would have produced the same headline.

### Predicted outcomes

- lambda=0.5: -10 to -20 pp vs lambda=1.  MBD needs prior pull
  to escape local minima around the canonical-bank-dominated mode.
- lambda=2.0: +/- 5 pp vs lambda=1.  Robust regime.
- lambda=5.0: could drop noticeably because the bank's 200
  canonical demos overwhelm the 5 perturbed ones in score
  averaging, washing out the adaptation signal.

### Failure mode

If lambda=5 actually IMPROVES over lambda=1: the adaptation signal
isn't "more demos = better prior" but rather "the 200 canonical
demos plus the cost are doing the work, the 5 demos are noise".
That would weaken the T2 narrative.  Unlikely given the T2.0 result
(13 % with just the 200 canonical demos) but worth checking.

## Integration into docs/results.md

The T2.1 paragraph gains a sentence: "Robustness check -- T2.1
holds within Wilson-CI overlap across lambda in [A, B], confirming
the headline is not lambda=1 specific."  A small table goes into
the appendix.
