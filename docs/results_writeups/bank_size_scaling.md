# Bank-size scaling for zero-training retrieval

**Slate:** `experiments/15_unified_hybrid/slate_bank_size_scaling_2026_06_07.sh`
**Status:** completed 2026-06-07.
**Primary question:** is the `demo_1nn` retrieval prior already saturated by a small demo bank, or is the 73.8% headline bank-size-limited?

**Primary answer:** the retrieval prior is strongly bank-size sensitive at tiny `N_b`, but it saturates quickly. Success rises from 23.3% at `N_b=5` to 65.0% at `N_b=20`, then stays statistically flat through `N_b=50,100,200`. The 200-demo bank is therefore not uniquely responsible for the zero-training headline; the main effect is finite mode coverage, with a knee around 20-50 demos.

## Motivation

The current zero-training headline uses the full Robomimic Can bank and can be criticized as memorization. A bank-size curve directly tests the finite-sample / mode-coverage story: if retrieval success rises steeply then saturates, the method is behaving like nonparametric nearest-neighbor mode coverage rather than a hidden trained policy.

## Cells

Retrieval-only first pass, all Can/ph, score-bridge, canonical operating point:

| Bank size `N_b` | Prior | λ | K | M | H | Trials |
|---:|---|---:|---:|---:|---:|---:|
| 5 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |
| 10 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |
| 20 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |
| 50 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |
| 100 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |
| 200 | `demo_1nn` | 1 | 10 | 16 | 120 | 60 |

The evaluation init-state protocol remains fixed at 20 demos × 3 rollouts; only `--prior-bank-size` changes.

## Decision gates

- **Strong support for mode-coverage theory:** monotone, concave-saturating curve with a knee by `N_b <= 20` or `N_b <= 50`, and the 200-demo cell within Wilson-CI overlap of the knee.
- **Bank-size-limited retrieval:** monotone rise with no visible plateau by `N_b = 200`; next step is larger-bank or augmentation scaling, not metric tuning.
- **Metric problem:** non-monotone curve or severe drop at intermediate sizes; next step is task-aware distance / phase-conditioned retrieval.

**Outcome:** the slate passes the strong mode-coverage gate. The curve has a clear knee by `N_b=20` and the `N_b=20` Wilson interval `[52.4%, 75.8%]` overlaps the full-bank `N_b=200` interval `[55.8%, 78.7%]`. The small decline from `N_b=100` to `N_b=200` is within seed noise and should not be interpreted as a real negative scaling effect.

Generated figure: `docs/results_writeups/bank_size_saturation.pdf`.

## Analysis

- **Tiny banks are not enough.** `N_b=5` and `N_b=10` score 23.3% and 35.0%, so nearest-neighbor retrieval needs enough examples to cover the relevant Can modes.
- **Most of the gain arrives by 20 demos.** The largest jump is `N_b=10 -> 20` (+30.0 pp), while `N_b=20 -> 200` adds only +3.3 pp.
- **The 50-200 range is a plateau.** `N_b=50`, `100`, and `200` all land between 68.3% and 70.0%, with overlapping Wilson intervals.
- **Compute is not the differentiator here.** Wall time is approximately 23 minutes per 60-rollout cell across bank sizes, so the scaling curve is about prior coverage, not runtime.
- **Paper implication.** The zero-training result should be described as nonparametric mode coverage plus MBD score-bridging, not as a hidden trained policy and not as a full-bank memorization artifact.

## Follow-up decision

Stage-2 matched BC scaling is optional rather than urgent. If included, it should be framed as a comparison against the retrieval curve, not as a prerequisite for interpreting this slate. For near-term figures, use the full curve and highlight the `N_b=20` knee plus the `N_b=50-200` plateau.

## Results table

| Bank size `N_b` | Success | Rate | Wilson 95% CI | Mean reward | Notes |
|---:|---:|---:|---:|---:|---|
| 5 | 14/60 | 23.3% | [14.4%, 35.4%] | 10.619 | under-covered |
| 10 | 21/60 | 35.0% | [24.2%, 47.6%] | 18.384 | still under-covered |
| 20 | 39/60 | 65.0% | [52.4%, 75.8%] | 36.139 | knee / most gain recovered |
| 50 | 41/60 | 68.3% | [55.8%, 78.7%] | 36.298 | plateau |
| 100 | 42/60 | 70.0% | [57.5%, 80.1%] | 34.711 | best observed, not significant vs 50/200 |
| 200 | 41/60 | 68.3% | [55.8%, 78.7%] | 35.270 | full-bank plateau |
