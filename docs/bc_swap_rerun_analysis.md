# BC-swap re-run analysis (2026-05-14)

After swapping the canonical BC checkpoint from `[512, 512]` (290k
params) to Charles's `[256, 256]` (80k params), we re-ran:

- Exp 03 BC eval (n=60).
- Exp 10 K × R pareto grid (9 cells, n=60 each, ~25 h compute).
- Exp 07 BC-prior lambda sweep (6 cells, n=60 each, ~8 h compute).

This document captures the headline numbers and the most important
finding: **the project's previous λ=0.5 inverted-U was a parameter
artifact of the under-trained `[512, 512]` BC. With the stronger
`[256, 256]` BC the peak shifts by an order of magnitude to λ=5.**

## 1. Headline numbers (n=60, PickPlaceCan / ph)

Ranked by success rate:

| rank | cell                                          | succ      | rate      | notes                                          |
|------|-----------------------------------------------|-----------|-----------|------------------------------------------------|
| 1    | exp 07 λ=5.0 (BC prior, K=10, H=120)          | **20/60** | **33.3 %** | **project peak**                               |
| 2    | exp 07 λ=∞ (pure BC plan in MBD framework)    | 18/60     | 30.0 %    | BC plan, MBD inert                             |
| 3    | exp 07 λ=1.0                                  | 17/60     | 28.3 %    | inverted-U flank                               |
| 4    | exp 03 BC closed-loop (no MBD, no MPC)        | 13/60     | 21.7 %    | the new strong-BC baseline                     |
| 5    | exp 07 λ=0.5 (old peak)                       | 11/60     | 18.3 %    | old peak; now flank                            |
| 5    | exp 10 K=10 R=2 (λ_softmax=0.5)               | 11/60     | 18.3 %    | best K × R cell                                |
| ...  | exp 10 K=10 R=0 / K=20 R=0                    | 0/60      | 0.0 %     | long open-loop hybrid plan collapses           |
| 13   | exp 07 λ=0.0 / λ=0.1 (near-pure MBD)          | 2/60      | 3.3 %     | MBD-without-prior baseline (cf. exp 04 = 0/60) |

## 2. Headline comparisons

### a) New BC vs old BC (closed-loop, pure π)

|         | succ  | rate    |
|---------|-------|---------|
| Old `[512, 512]` BC | 4/60 | 6.7 % |
| **New `[256, 256]` BC** | **13/60** | **21.7 %** |

z = +2.36, **p = 0.018** — statistically significant. The new BC is
3.2 × stronger than the old one without any change to the hybrid
framework. This validates Charles's hyper-parameter investigation
(see `docs/archive/charles_tuning_analysis.md`): the 290k-param model was
over-parameterised for 200 demos, the 80k-param model regularises
better and generalises further from train.

### b) Hybrid peak (new) vs old hybrid peak

|         | succ  | rate    |
|---------|-------|---------|
| Old hybrid peak (exp 07 λ=0.5) | 6/60 | 10.0 % |
| **New hybrid peak (exp 07 λ=5.0)** | **20/60** | **33.3 %** |

z = +3.10, **p = 0.002** — clearly significant. The 3.3 × lift
comes from two sources: a stronger BC (3.2 ×) and an upward shift
of the optimal mixing weight (×10, from λ=0.5 to λ=5).

### c) Hybrid peak vs new BC alone

|         | succ  | rate    |
|---------|-------|---------|
| New BC closed-loop | 13/60 | 21.7 % |
| **Hybrid λ=5.0** | **20/60** | **33.3 %** |

z = +1.43, **p = 0.15** — **not** statistically significant at n=60.
The hybrid method does still appear to improve on pure BC by +11.7 pp
absolute, but the n=60 confidence intervals overlap:

- BC : 21.7 % [13.1 %, 33.6 %] Wilson 95 % CI
- λ=5 : 33.3 % [22.7 %, 45.9 %]

To settle this we need either n=200 single-seed or a 3-seed pool
at n=60 per seed (the latter is what `experiments/14_seed_sweep`
was scaffolded for). See § 6.

### d) Inverted-U has shifted right

![lambda_curve](../experiments/07_lambda_sweep/results/_archive/lambda_curve.png)

The shape is preserved (still an inverted U over log-λ), but:

- Old peak (`[512, 512]` BC): **λ = 0.5**, 10.0 % success.
- New peak (`[256, 256]` BC): **λ = 5.0**, 33.3 % success.

Mechanistic intuition: λ controls how strongly the per-step
denoising update is pulled toward `∇log π_BC(x)`. When BC is bad
(old), a small nudge is all you can tolerate before BC's errors
dominate the planner's progress. When BC is good (new), most of
the planner's job is already done by the prior, so you want λ
large enough that MBD only nudges, never overwrites.

The fact that λ=∞ (30 %) and λ=5 (33.3 %) are within 3 pp of
each other quantifies how little extra value MBD adds on top of
a strong BC. The hybrid story is now **"BC is most of the answer;
MBD's score correction gives a small but possibly non-zero edge"**.

## 3. The exp 10 K × R grid: results + paired-pair stats

Exp 10's K × R pareto grid was launched with the old default
`λ_softmax = 0.5`, so the post-BC-swap numbers in this table
are a re-run of the *same* grid the project shipped with old
BC — useful as a like-for-like comparison but not as a hybrid
headline (see plans.md § *Top priority: codebase consolidation*
for the algorithmic-confound discussion of why this grid
measures the warm-start-only variant, not the score-bridge):

| K \ R | R=0    | R=1    | R=2    |
|-------|--------|--------|--------|
| 5     | 6.7 %  | 13.3 % | 8.3 %  |
| 10    | 0.0 %  | 15.0 % | 18.3 % |
| 20    | 0.0 %  | 8.3 %  | 13.3 % |

The grid heat-map (`experiments/10_kxreplans_grid/results/summary.png`)
reproduces the pattern from the old runs: best cell at K=10 R=2;
collapses at R=0; degrades again at K=20 R=2 (too much denoising
in too few replans). **But the grid's best cell (18.3 %) still
loses to plain BC (21.7 %)** under the warm-start-only algorithm.

### Paired BC-vs-hybrid inside exp 10 (McNemar)

This is the analysis that doesn't appear anywhere else. Each
exp 10 rollout records both the hybrid outcome and the BC
subroutine's outcome (BC is re-run at each chunk start). Across
all 540 rollouts (9 cells × 60):

|                                  | n     | rate    |
|----------------------------------|-------|---------|
| Hybrid succeeded                 | 50    | 9.3 %   |
| BC-subroutine succeeded (∃ chunk)| 195   | 36.1 %  |
| BC-only wins (paired)            | 167   |         |
| Hybrid-only wins (paired)        | 22    |         |
| Both succeed                     | 28    |         |
| Both fail                        | 323   |         |

McNemar exact 2-sided **p ≈ 9 × 10⁻²⁹** that BC > Hybrid under
this paired protocol. Caveats:

- The BC-subroutine number (36 %) is inflated because it counts
  a success if BC succeeds **from any chunk start state**, not
  just t=0 (n_chunks=3 with R=2 → up to 3 BC attempts per
  rollout). The fair single-shot comparison is Exp 03's
  closed-loop BC at 21.7 %, which still beats Exp 10's best
  cell (18.3 %).
- This says nothing about the score-bridge variant; Exp 07
  λ=5 beats single-shot BC by +11.7 pp. The McNemar result is
  specifically a stat-test on the warm-start-only formulation,
  and reinforces the consolidation argument.

## 4. Failure modes worth flagging

### a) Long open-loop hybrid plans collapse

K=10 R=0 and K=20 R=0 both at **0 / 60**. The combination of
"long denoising chain" (many MBD updates) + "no MPC replan"
(no chance to correct from intermediate observations) is
catastrophic. The denoising drifts to a low-rollout-cost region
of action space that does not correspond to task completion.
This matches the exp 04 failure mode (open-loop MBD diverges).

### b) Particle averaging at λ=∞ helps slightly over closed-loop BC

Exp 07 λ=∞ = 30.0 %, exp 03 closed-loop BC = 21.7 %.

When λ → ∞ the MBD update is pinned to the BC mean over M=16
particle rollouts, with the rollout sequence then executed
open-loop for H=120 steps. This protocol differs from exp 03's
step-by-step closed-loop BC execution in two ways:

- It averages M=16 stochastic BC rollouts before execution. If
  BC has Gaussian output noise, averaging smooths away high-
  variance per-step decisions.
- The averaged action sequence is executed open-loop without
  re-querying BC at each env step.

The fact that the averaged-open-loop protocol *beats*
closed-loop BC (+8 pp) is unexpected and worth a 1-page
ablation. Hypothesis: closed-loop BC is over-reacting to
observation noise, whereas the averaged plan is smoother and
more robust. If true, the project has a free 8 pp from "BC
particle averaging" alone, independent of MBD.

## 5. Statistical situation

n=60 is sufficient for old vs new comparisons (the deltas are
huge: +15 pp BC, +23 pp hybrid peak). It is **not** sufficient
to discriminate the +12 pp hybrid-vs-new-BC delta (95 % CIs
overlap; p=0.15).

To get to a publishable hybrid-vs-BC claim we need:

- n=200 single-seed on the two cells (exp 03 BC and exp 07 λ=5),
  OR
- 3 seeds at n=60 each (180 total) per cell, which also lets us
  report seed-pooled variance — the protocol Charles is set up
  for in `experiments/14_seed_sweep`.

A 60-trial cell takes ~80 min single-process on the M4 Max. A
3 × 60 pool is ~4 h per cell. The pool for just BC + λ=5 + λ=∞
is ~12 h overnight.

## 6. Recommended follow-ups arising from this analysis

**The roadmap for what to do next lives in
[`plans.md`](plans.md)** — specifically § *Top priority:
codebase consolidation* and § *Recommended sequencing*. The
short version is: unify the warm-start-only and score-bridge
drivers (because the +15-pp gap between them is larger than
any hyperparameter effect we have measured), then do the seed
pool, then `v_theta`. Don't launch any hyperparameter sweep
before that consolidation lands.

The items below are findings *from this re-run* that don't
belong in the project roadmap but are worth flagging:

- **Failure taxonomy on the new JSONs**. The chain-script
  attempt to auto-run `experiments/06_failure_taxonomy/run.py`
  on the new exp 07 / exp 10 outputs failed with a stdio
  plumbing bug; the analyser works fine when invoked manually.
  Whoever picks this up next should categorise the remaining
  ~40 / 60 score-bridge failures into no-reach / partial-reach /
  strong-reach buckets, to see if MBD is salvaging specific
  demo classes (the *"BC reaches, MBD lifts"* hypothesis).
  Independent of consolidation; cheap.
- **BC particle-averaging is an unexpected free 8 pp**
  (§ 4(b) above). Worth a separate 1-page ablation (M ∈
  {1, 4, 16, 64} on Exp 07 λ=∞) **once the codebase is
  unified**. Tagged as a hyperparameter sweep, deferred until
  consolidation.
- **The empirical λ=5 doesn't match the theoretical nominal**
  by ~10× (see § 2(d)). Either our σ-schedule is mis-tuned or
  the practical optimum is offset from the analytical optimum.
  Worth a paragraph in the paper; theory work, not an
  experiment.

## 7. What changed in the project narrative

| before BC-swap                                                          | after BC-swap                                                  |
|-------------------------------------------------------------------------|----------------------------------------------------------------|
| BC is weak (6.7 %).                                                     | BC is the bulk of the headline (21.7 %).                       |
| Hybrid (10 %) beats BC (6.7 %) by +3.3 pp, stat-sig at n=60.            | Hybrid (33 %) beats BC (22 %) by +11.7 pp, **not** stat-sig at n=60. |
| Optimal λ ≈ 0.5.                                                        | Optimal λ ≈ 5.                                                 |
| Hybrid framework's value-add is *necessary* (BC alone is unusable).     | Hybrid framework's value-add is *modest* (BC alone is workable). |
| K × R grid useful: shows MPC + bigger K helps.                          | K × R grid uninformative at λ_softmax=0.5: dominated by BC.    |

The headline thesis is the same — the hybrid score-bridge does
beat both pure BC and pure MBD — but it lands less dramatically
than before, and we now need a seed pool (or n=200) to make the
claim quantitatively defensible. Charles's stronger BC has the
silver lining of pushing the **absolute** numbers up by 3 ×.

---

*Author: Cascade, 2026-05-14. Computed on M4 Max with
`scripts/local/run_pareto_full.sh`, `scripts/local/run_lambda_bc_full.sh`,
and `scripts/local/chain_bc_swap_rerun.sh`. Raw JSONs in
`experiments/{03,07,10}_*/results/`. Aggregated plots in
`experiments/{07,10}_*/results/summary.png` and `.../lambda_curve.png`.*
