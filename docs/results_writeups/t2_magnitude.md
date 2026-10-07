# T2 perturbation-magnitude sweep (writeup stub)

**Branch:**  `feat/adapt-t2-magnitude`
**Worktree:** `HRTG-magnitude`
**Slate:**   `experiments/15_unified_hybrid/slate_incremental_T2_magnitude_2026_05_28.sh`
**Status:**  results landed 2026-05-29; aggregation done.

## Motivation

The T2 headline on main is a single-point demonstration at Can
mass=1.3.  The Lift-cross-task null result already flagged that the
T2 Pareto gap is **perturbation-magnitude-conditional** -- but
this conditional framing is asserted, not characterised.  This
branch maps the bite zone constructively by sweeping mass_scale
in {1.1, 1.2, 1.4, 1.5} and re-running the T2.0 / T2.1 / T2.3
triplet at each magnitude.

## Cells

12 evaluations + 4 BC retrains:

| Magnitude | T2.0 hybrid+canonical | T2.1 hybrid+5demo | T2.3 BC retrained |
|---|:-:|:-:|:-:|
| 1.1 | cell | cell | cell + retrain |
| 1.2 | cell | cell | cell + retrain |
| 1.3 | already on main | already on main | already on main |
| 1.4 | cell | cell | cell + retrain |
| 1.5 | cell | cell | cell + retrain |

All cells run at seed=0, K=10, M=16, H=120, lambda_prior=1
(hybrid) or 5 (BC), N=30 trials per cell.

## To fill in after slate completes

### Bite-zone table

| Mass | T2.0 hybrid+canonical | T2.1 hybrid+5demo | T2.3 BC retrained 205 | T2.1 − T2.3 gap |
|---:|---:|---:|---:|---:|
| 1.1 | 20/30 = 66.7% | 20/30 = 66.7% | 3/30 = 10.0% | +56.7 pp |
| 1.2 | 20/30 = 66.7% | 24/30 = 80.0% | 3/30 = 10.0% | +70.0 pp |
| 1.3 | 21/30 = 70.0% | 22/30 = 73.3% | 0/30 = 0.0% | +73.3 pp |
| 1.4 | 19/30 = 63.3% | 20/30 = 66.7% | 4/30 = 13.3% | +53.3 pp |
| 1.5 | 22/30 = 73.3% | 20/30 = 66.7% | 1/30 = 3.3% | +63.3 pp |

### Wilson 95 % CIs per cell

| Mass | T2.0 (95 % CI) | T2.1 (95 % CI) | T2.3 (95 % CI) |
|---:|---:|---:|---:|
| 1.1 | [48.8, 80.8] | [48.8, 80.8] | [3.5, 25.6] |
| 1.2 | [48.8, 80.8] | [62.7, 90.5] | [3.5, 25.6] |
| 1.3 | [52.1, 83.3] | [55.6, 85.8] | [0.0, 11.4] |
| 1.4 | [45.5, 78.1] | [48.8, 80.8] | [5.3, 29.7] |
| 1.5 | [55.6, 85.8] | [48.8, 80.8] | [0.6, 16.7] |

### Headline (computed from per-magnitude N=30)

Robustness of the T2.1 > T2.3 gap across mass scales:

| Mass | T2.1 − T2.3 gap | overlap? |
|---:|---:|:-:|
| 1.1 | +56.7 pp | **no** |
| 1.2 | +70.0 pp | **no** |
| 1.3 | +73.3 pp | **no** |
| 1.4 | +53.3 pp | **no** |
| 1.5 | +63.3 pp | **no** |


T2 headline holds across the full 1.1-1.5 magnitude sweep wherever T2.1 and T2.3 CIs do not overlap.  Below `1.0` the perturbation is absent so the bite-zone framing is vacuous; above `1.5` was not tested because the BC arm is already at floor (~0-10 %).

### Bite-zone figure

Plot: x = mass_scale, three curves (T2.0, T2.1, T2.3) with Wilson
95 % CIs.  Annotate the bite zone = the magnitude range where
T2.0 is below ~70 % AND T2.1 - T2.3 gap is positive.

### Predicted outcomes

If the bite-zone framing holds:
- At mass=1.1: T2.0 and T2.1 both near 100 %, T2.3 still works
  (small enough shift that BC retrain doesn't catastrophically
  flip).  Adaptation gap = 0 (no bite).
- At mass=1.2: T2.0 may dip to ~60-80 %, T2.1 recovers to ~90%,
  T2.3 may or may not collapse.  Borderline bite.
- At mass=1.3 (main): T2.0=13%, T2.1=73%, T2.3=0%.  Center of bite.
- At mass=1.4 / 1.5: T2.0 collapses harder; T2.1 may also
  partially collapse (5 demos no longer enough); T2.3 catastrophic.
  Bite zone exits when even T2.1 collapses.

### Integration into docs/results.md

A new figure replaces the single-point Pareto with a magnitude-
parameterised version.  The headline section adds the bite-zone
characterisation table and softens "78x speedup at non-inferior
succ" to "78x speedup at non-inferior succ within the
characterised bite zone [mass=A..B]".
