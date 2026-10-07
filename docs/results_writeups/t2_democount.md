# T2 demo-count saturation sweep (writeup stub)

**Branch:**  `feat/adapt-t2-democount`
**Worktree:** `HRTG-democount`
**Slate:**   `experiments/15_unified_hybrid/slate_incremental_T2_democount_2026_05_28.sh`
**Status:**  results landed 2026-05-29; aggregation done.

## Motivation

T2.1 used 5 mass=1.3 demos and T2.4 used 20.  The current writeup
claims "5 demos are enough", but the 5-vs-20 comparison only shows
that more demos do not help past 5 -- it does NOT show that 5 is
the saturation point.  3 or even 2 demos may already suffice.

This branch sweeps n_perturbed in {1, 2, 3, 10} (filling in the
gaps in the existing {0, 5, 20} ladder) so the saturation knee can
be located.

## Cells

Each cell is T2.1-style (hybrid + canonical 200 + N mass=1.3 demos
@ mass=1.3 eval, lambda=1, K=10, M=16, seed=0, N=30 trials).

| Bank | Existing on main? | This branch | n_perturbed |
|---|:-:|:-:|---:|
| can_mass1p3_200demos (= T2.0) | yes | | 0 |
| can_mass1p3_201demos          |     | new | 1 |
| can_mass1p3_202demos          |     | new | 2 |
| can_mass1p3_203demos          |     | new | 3 |
| can_mass1p3_205demos (= T2.1) | yes | | 5 |
| can_mass1p3_210demos          |     | new | 10 |
| can_mass1p3_220demos (= T2.4) | yes | | 20 |

4 new cells: n_perturbed in {1, 2, 3, 10}.

## To fill in after slate completes

### Demo-count saturation table

| n_perturbed | label | succ | Wilson 95 % CI |
|---:|---|---:|---:|
| 0 | T2.0 (canonical only) | 21/30 = 70.0% | [52.1, 83.3] |
| 1 | 1 perturbed demo | 20/30 = 66.7% | [48.8, 80.8] |
| 2 | 2 perturbed demos | 20/30 = 66.7% | [48.8, 80.8] |
| 3 | 3 perturbed demos | 21/30 = 70.0% | [52.1, 83.3] |
| 5 | T2.1 (5 perturbed) | 22/30 = 73.3% | [55.6, 85.8] |
| 10 | 10 perturbed demos | 18/30 = 60.0% | [42.3, 75.4] |
| 20 | T2.4 (20 perturbed) | 21/30 = 70.0% | [52.1, 83.3] |

### Saturation knee (computed from N=30 per cell)

- **T2.1 anchor (n=5):** 22/30 = 73.3%  [55.6, 85.8]

- **Per-N overlap with the T2.1 (n=5) CI:**
  - n= 1: 20/30 = 66.7%  [48.8, 80.8]  -- yes (saturated)
  - n= 2: 20/30 = 66.7%  [48.8, 80.8]  -- yes (saturated)
  - n= 3: 21/30 = 70.0%  [52.1, 83.3]  -- yes (saturated)
  - n=10: 18/30 = 60.0%  [42.3, 75.4]  -- yes (saturated)

- **Saturation knee:** as small as **n=1** demos already lies within the T2.1 (n=5) Wilson 95 % CI.  More demos do not reliably add lift; the T2 adaptation effect is essentially demo-count-saturated at the very first perturbed demo.

### Predicted outcomes

If 5 is genuinely the saturation knee:
- {1, 2} should be below T2.1 (e.g. 30-50 %).
- {3} could be on the rising flank (~60-70 %).
- {10} should be at or near T2.4 (~75-80 %).

If 3 already saturates:
- {3} sits at or above T2.1 within CI overlap.
- Headline tightens to "3 demos suffice".

### Integration into docs/results.md

A new figure `experiments/15_unified_hybrid/results/.../democount.png`
plots succ as a function of n_perturbed (log-x), with Wilson CIs.
The headline section gains one paragraph: "5-demo saturation
point verified -- 1 / 2 / 3 demos yield {x, y, z}% succ, 10
saturates within CI overlap of 5."

## Caveats

- All 30 trials per cell at seed=0.  Pooling with multi-seed
  results (if the multi-seed branch lands first) would give
  90 trials per cell for tighter CIs.
- Random draw of which perturbed demos enter the bank: the
  derive script always takes the first N of the perturbed-20.
  An ablation over which N out of 20 to use is out of scope but
  worth a footnote.
