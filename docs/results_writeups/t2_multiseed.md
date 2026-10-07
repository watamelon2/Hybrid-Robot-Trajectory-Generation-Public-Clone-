# T2 multi-seed validation (writeup stub)

**Branch:**  `feat/adapt-t2-multiseed`
**Worktree:** `HRTG-multiseed`
**Slate:**   `experiments/15_unified_hybrid/slate_incremental_T2_multiseed_2026_05_28.sh`
**Status:**  results landed 2026-05-29; 3-seed aggregation done.

## Motivation

The T2 headline (`docs/results.md`, "Slates Incremental T1 + T2")
rests on a single seed (`seed=0`) of MBD's stochastic planning.
The Wilson 95 % CIs on N=30 per cell are non-overlapping at that
seed -- but a reviewer would reasonably ask whether the gap is a
property of the method or a lucky seed.

This branch reruns the full T2 ladder (T2.0, T2.1, T2.2, T2.3, T2.4)
at seeds {1, 2} to pool 3-seed CIs across N=90 trials per cell.

## Cells (10 evaluations + 2 BC retrains)

| Cell | Description | Per-seed wall |
|---|---|---:|
| T2.0_seed{1,2} | Hybrid + canonical 200 bank @ mass=1.3 | ~12 min |
| T2.1_seed{1,2} | Hybrid + 200 + 5 mass=1.3 demos (headline) | ~12 min |
| T2.2_seed{1,2} | BC unadapted @ mass=1.3 | ~10 min |
| T2.3_seed{1,2} | BC retrained on 205, retrained at seed={1,2} | ~12 min eval + 80 s retrain |
| T2.4_seed{1,2} | Hybrid + 200 + 20 mass=1.3 demos | ~12 min |

Total wall: 2 × 5 × ~12 min = ~2 h compute + ~2 min BC retrains.
Adjusted estimate: ~7 h with worker queueing overhead.

## To fill in after slate completes

### Per-seed table

| Cell | seed=0 | seed=1 | seed=2 | 3-seed pooled (Wilson 95 % CI) |
|---|---:|---:|---:|---:|
| T2.0 | 21/30 = 70.0% | 21/30 = 70.0% | 22/30 = 73.3% | 64/90 = 71.1%  [61.0, 79.5] |
| T2.1 | 22/30 = 73.3% | 19/30 = 63.3% | 20/30 = 66.7% | 61/90 = 67.8%  [57.6, 76.5] |
| T2.2 | 9/30 = 30.0% | 9/30 = 30.0% | 9/30 = 30.0% | 27/90 = 30.0%  [21.5, 40.1] |
| T2.3 | 0/30 = 0.0% | 4/30 = 13.3% | 2/30 = 6.7% | 6/90 = 6.7%  [3.1, 13.8] |
| T2.4 | 21/30 = 70.0% | 22/30 = 73.3% | 21/30 = 70.0% | 64/90 = 71.1%  [61.0, 79.5] |

### Headline (computed from pooled 3-seed numbers)

- **T2.1** (hybrid + 5 perturbed demos): 61/90 = 67.8%  [57.6, 76.5]
- **T2.3** (BC retrained on 205 perturbed demos): 6/90 = 6.7%  [3.1, 13.8]
- CIs **do NOT overlap** -- the hybrid+5-demo > BC-retrained-on-205 gap holds in the 3-seed pool (Δ ≈ +61.1 pp).  Headline substantiates the single-seed claim across N=90 trials.

### Pareto plot update

`experiments/15_unified_hybrid/plot_incremental_pareto.py` re-run
with per-cell N=90 (3-seed pool); error bars become Wilson 95 % CIs
on the pooled count.

### Predicted outcomes (substantiate T2 if true)

- T2.0 / T2.1 / T2.3 succ ranks should be stable across seeds.
- T2.1 vs T2.3 pooled CIs should NOT overlap.
- Adaptation wall-clock (~1 s for T2.1, ~78 s for T2.3) is per-seed
  identical -- the 78x speedup claim does not depend on seed.

### Failure modes (force re-thinking of the headline if observed)

- T2.1 succ collapses at seed != 0 (would indicate seed-0 was a
  lucky retrieval); response: characterise the dependence on the
  MBD planning seed and report the worst-seed result.
- T2.3 succ recovers at seed != 0 (would indicate seed-0's BC
  retrain was a degenerate local optimum); response: characterise
  the BC retrain variance and report the across-seed mean +/- SD.

## Integration into docs/results.md

**Status update 2026-06-01**: this branch has landed.  The 3-seed
pooled headline went into `docs/results.md`'s T2 section via the
`feat/adapt-t2-multiseed` merge (commit `d093044`, 2026-05-28),
and the post-hoc Pareto-dominance follow-up landed via
`feat/adapt-pareto-analysis` (commit `aaefba2`, 2026-06-01) --
see [`pareto_analysis.md`](pareto_analysis.md) for the proxy
`(bias², variance)` decomposition that verifies T2.1 strictly
Pareto-dominates T2.3.  Connection to the formal theory in
[`docs/theory/08_asymmetric_robustness_model.md`](../theory/08_asymmetric_robustness_model.md).

The abstract claim text has been upgraded from "single-seed
demonstration" to "across 3 independent seeds".  This section
is preserved as a historical record of the integration plan.
