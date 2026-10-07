# Slate F final reading — mode-selection confirmed, Square mechanism-limited

*Written 2026-05-20 10:45 PT.  Slate F finished at 05:48 PT,
Square budget-retry finished at 07:19 PT.  Supersedes
`docs/slate_f_interim_reading.md` (which read only F1 alone).*

---

## TL;DR

Two clean headlines, in tension by task:

### Track B (Can warm-start variants) — **mode-selection thesis confirmed**

Four new zero-train priors run on Can at the canonical
operating point (K=10 R=1 M=16 H=120, λ=1, σ=0.25, seed=0,
n_rollouts=60), against the Slate-E `demo_1nn` baseline of
75.0 %:

| prior                | succ  | rate    | mean_r  | max_r |
|----------------------|-------|---------|---------|-------|
| `demo_1nn` (Slate E) | 45/60 | 75.0 %  | (n/a)   | (n/a) |
| `demo_1nn_proprio`   | 42/60 | **70.0 %** | +35.6 | +57.0 |
| `demo_lwr_k5`        | 41/60 | **68.3 %** | +34.8 | +53.8 |
| `demo_1nn_object`    | 41/60 | **68.3 %** | +33.5 | +57.9 |
| `demo_topk_random_k5`| 17/60 | **28.3 %** | +15.5 | +53.9 |

Reading:

1. **Three of four variants land in [68.3 %, 70.0 %]** —
   within ~5-7 pp of the baseline 75.0 %.  Both single-obs-subset
   1-NN variants and the locally-weighted regression variant
   retain almost all of the warm-start benefit.

2. **`demo_topk_random_k5` collapses to 28.3 %.**  The
   construction is: take the same 5 nearest neighbors LWR uses,
   but pick *one at random* instead of weighting them.  The drop
   from 68.3 % (LWR) to 28.3 % (random-pick) isolates **mode
   selection** as the operative property of 1-NN.  LWR works
   because it puts ~all weight on the closest demo (1/√d
   weighting; closest demo dominates); top-k-random destroys this
   by sampling uniformly among modes.

3. **Bias-variance framing is corroborated.**  The theory doc
   predicted that variance-inducing warm-starts hurt because the
   bridge has to recover from trajectory-level multi-modality
   each replan.  Top-k-random adds variance by construction;
   LWR / 1NN-subset preserve mode selection and behave like
   slightly noisier 1-NN.

4. **Obs-subset slightly favours proprio (70.0 %) over object
   (68.3 %)**, but the gap is well within seed noise (this was
   one seed each, no CI).  Both subsets nearly reproduce the
   full-obs 1-NN baseline, suggesting the retrieval signal is
   distributed across obs dims with no single critical
   sub-vector.  A small proprio edge is consistent with the
   action-relevant signal living in eef / gripper state.

### Track A (Square cross-task) — **mechanism-limited, not budget-limited**

Hypothesis test from the interim doc: was F1's 0-3.3 % success
on Square due to (a) insufficient MBD budget, or (b) the
mechanism not transferring at all?

The budget-retry cell tripled MBD work (R=3 M=24 vs R=1 M=16) at
the F1 sweet spot (λ=1, σ=0.25, seed=0):

| cell                | R | M  | succ  | rate   | mean_r | max_r | frac>50 | wall    |
|---------------------|---|----|-------|--------|--------|-------|---------|---------|
| F1 λ=1 (baseline)   | 1 | 16 | 2/60  | 3.3 %  | +28.6  | +45.6 | 0 / 60  | 26.5 m  |
| Budget retry        | 3 | 24 | 2/60  | 3.3 %  | +20.1  | +52.0 | 1 / 60  | 90.1 m  |

Verdict: **mechanism-limited.**  The success rate is identical
(2/60 = 3.3 %).  Mean reward actually *dropped* by 8.5 points
(+28.6 → +20.1) at 3× compute -- consistent with more aggressive
exploration falling apart on Square's contact geometry.  One
rollout *did* exceed reward 50 (max_r 52.0 vs F1's 45.6), so a
sliver of additional progress is visible, but the
"approach + grasp" barrier still isn't cleared.

Story-B's "task-agnostic mechanism" claim is **weakened** for
Square.  The Can-tuned bridge + Can-retrieved-then-flat-shipped
warm-start does not generalise to Square's contact-rich
insertion regime, and no amount of MBD compute at the same
prior + obs vocabulary fixes it.

### Track A.2 (Square σ-tail at λ=1) — small, expected

| σ_max | succ  | rate   | mean_r | max_r |
|-------|-------|--------|--------|-------|
| 0.05  | 0/60  | 0.0 %  | +29.7  | +43.0 |
| 0.25  | 2/60  | 3.3 %  | +28.6  | +45.6 |
| 0.40  | 1/60  | 1.7 %  | +26.7  | +44.7 |
| 1.00  | 0/60  | 0.0 %  | +20.3  | +38.9 |

Mean reward and max reward both peak near σ=0.25.  σ=1.00 is
clearly worse (the bridge target is too noisy to track).  But
nothing here clears the success threshold either, so this is
aux evidence at best.

---

## Full F1 + F2 + F3 table

| track | task   | prior                  | λ    | σ    | R | M  | succ  | rate    |
|-------|--------|------------------------|------|------|---|----|-------|---------|
| F1    | square | demo_1nn               | 0.0  | 0.25 | 1 | 16 | 0/60  | 0.0 %   |
| F1    | square | demo_1nn               | 0.5  | 0.25 | 1 | 16 | 2/60  | 3.3 %   |
| F1    | square | demo_1nn               | 1.0  | 0.25 | 1 | 16 | 2/60  | 3.3 %   |
| F1    | square | demo_1nn               | 2.0  | 0.25 | 1 | 16 | 0/60  | 0.0 %   |
| F1    | square | demo_1nn               | 5.0  | 0.25 | 1 | 16 | 0/60  | 0.0 %   |
| F1    | square | demo_1nn               | 10.0 | 0.25 | 1 | 16 | 0/60  | 0.0 %   |
| F2    | square | demo_1nn               | 1.0  | 0.05 | 1 | 16 | 0/60  | 0.0 %   |
| F2    | square | demo_1nn               | 1.0  | 0.40 | 1 | 16 | 1/60  | 1.7 %   |
| F2    | square | demo_1nn               | 1.0  | 1.00 | 1 | 16 | 0/60  | 0.0 %   |
| F3    | can    | demo_lwr_k5            | 1.0  | 0.25 | 1 | 16 | 41/60 | 68.3 %  |
| F3    | can    | demo_topk_random_k5    | 1.0  | 0.25 | 1 | 16 | 17/60 | 28.3 %  |
| F3    | can    | demo_1nn_proprio       | 1.0  | 0.25 | 1 | 16 | 42/60 | 70.0 %  |
| F3    | can    | demo_1nn_object        | 1.0  | 0.25 | 1 | 16 | 41/60 | 68.3 %  |
| retry | square | demo_1nn               | 1.0  | 0.25 | 3 | 24 | 2/60  | 3.3 %   |

Wall-time totals: Slate F = 5h 42m (13 cells × ~26 min each, one
invocation per cell except F1 which packed 6 cells).  Retry +
1h 30m.  Total = 7h 12m.

---

## Acceptance gate evaluation

The Slate F BRIEF defined separate gates for the two tracks:

**Track A (Square cross-task headline).**

| gate     | requirement                          | verdict   |
|----------|--------------------------------------|-----------|
| strong   | peak ≥ 30 %, peak > both endpoints   | **fail**  |
| weak     | peak in [15 %, 30 %]                 | **fail**  |
| falsify  | peak ≤ max(λ=0, λ=10), or flat       | **fires** |

After the budget-retry, the falsification holds at 3× compute as
well; mechanism does not transfer at this prior + operating
point.  Mean-reward partial transfer is interesting but does not
satisfy the BRIEF's success-rate gates.

**Track B (Can warm-start variant headline).**

| gate     | requirement                                                  | verdict   |
|----------|--------------------------------------------------------------|-----------|
| strong   | one variant ≥ baseline 75 % (within seed noise of ±5 pp)     | **pass**  |
| variance | top-k-random meaningfully worse than top-k-weighted          | **pass**  |
| separation | mode-selecting vs mode-mixing visible at single seed          | **pass**  |

The bias-variance prediction is confirmed.  Variance-inducing
warm-starts (top-k-random: 28.3 %) underperform mode-selecting
warm-starts (1NN / LWR / 1NN-subset: 68-75 %) by ~40 pp at the
same operating point.

---

## What goes into the headline tables

The Slate F update to `docs/results.md`:

- Append a Track B subsection under Slate F: "warm-start variant
  ablation on Can substantiates mode-selection".  Promote
  `demo_lwr_k5` and `demo_1nn_proprio` as near-baseline alternatives;
  flag `demo_topk_random_k5` as the negative control that confirms
  the bias-variance framing.

- Append a Track A subsection under Slate F: "Square cross-task
  transfer fails at the canonical operating point even at 3×
  compute; mean reward shows partial mechanism transfer but does
  not clear the insertion threshold".

- Update `docs/theory_post_slate_e.md` § 4 with the F3 ablation
  -- this is the cleanest empirical support for the bias-variance
  framing so far.

---

## Open follow-ups carried forward

1. **Multi-seed F3 variants.**  Track B is one seed.  4-seed pool
   for `demo_lwr_k5`, `demo_1nn_proprio`, `demo_1nn_object`,
   `demo_topk_random_k5` at λ=1 σ=0.25 to get CIs and confirm the
   variant ordering.  ~8 h wall on the same operating point.

2. **Why does Square fail?**  Three sub-hypotheses to test
   (in increasing cost order):
   - (i) obs-key vocabulary too narrow for the contact geometry;
     try retrieval on obs + recent action history.
   - (ii) horizon too short; bump H to 200 (Square's typical
     trajectory length in PH demos).
   - (iii) MBD step distribution too biased toward Can-style
     pick-place; explore a per-task diffusion schedule.

3. **Cross-task on `lift` or `tool_hang`.**  Square may be a
   pathological case (peg-in-hole is exceptionally
   contact-rich).  Cross-task on `lift` (similar to Can but
   simpler) could establish whether the mechanism transfers
   at all to non-Can tasks.

---

## Source

- AGGREGATE.json:
  `experiments/15_unified_hybrid/results/slate_f_2026_05_20/AGGREGATE.json`
- Slate F BRIEF (gates predefined):
  `experiments/15_unified_hybrid/results/slate_f_2026_05_20/BRIEF.md`
- Interim reading (F1 only):
  `docs/slate_f_interim_reading.md`
- Budget-retry result:
  `experiments/15_unified_hybrid/results/slate_f_square_retry_2026_05_20/unified_demo_1nn_lam-1.0_K-10_R-3_square_retry_R3_M24_lam1.json`
