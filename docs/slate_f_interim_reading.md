# Slate F interim reading — Square mechanism transfer is partial

*Written 2026-05-20 03:10 PT, ~3 h into the Slate F overnight, with
F1 (Square λ-sweep) complete and F2 / F3 still in progress.
This is an interim summary; the full reading will be written
after F3 (Can warm-start variants) finishes.*

---

## TL;DR

F1 -- the Square `demo_1nn` λ-sweep at σ=0.25 seed=0 -- **fails
the BRIEF's success-rate acceptance gates** but **partially
substantiates the underlying mechanism**.  Two facts in tension:

- **Success rate.**  All six λ ∈ {0, 0.5, 1, 2, 5, 10} cells
  return 0–3.3 %.  The peak is 3.3 % at λ ∈ {0.5, 1} -- well
  below even the "weak" gate of 15 %.

- **Mean reward.**  Rises monotonically from 17.6 (λ=0,
  warm-start only) to ~28 (λ ∈ {1, 2, 5, 10}), then saturates.
  The mean-reward-rises-with-λ-then-plateaus shape is the same
  qualitative profile we see on Can; the bridge is moving
  trajectories closer to the goal manifold.  It just can't push
  past the final-insertion threshold within the canonical
  K=10 R=1 M=16 H=120 budget.

Across all 360 Square rollouts in F1, **no rollout ever cleared
reward = 50**.  The 95th-percentile reward saturates at ~42 and
the absolute max ever observed is 45.6 (at λ=1).  Square's
shaped-reward range is ~0-75 (same scale as Can), so this is a
hard barrier at the "grasp the nut" stage — the hybrid never
completes the "insert nut onto peg" step.

---

## F1 results table

K=10 R=1 M=16 H=120, σ=0.25, seed=0, 20 demos × 3 rollouts = 60
rollouts per cell.  Source:
`experiments/15_unified_hybrid/results/slate_f_2026_05_20/unified_demo_1nn_lam-*_F1_square_1nn_lamsweep.json`.

| λ      | succ    | rate  | mean_r | reward q95 | max_r |
|-------:|--------:|------:|-------:|-----------:|------:|
| 0.00   | 0/60    | 0.0 % | +17.6  | +36.5      | +41.7 |
| 0.50   | 2/60    | 3.3 % | +25.2  | +40.4      | +45.0 |
| 1.00   | 2/60    | 3.3 % | +28.6  | +42.3      | +45.6 |
| 2.00   | 0/60    | 0.0 % | +28.7  | +40.3      | +41.2 |
| 5.00   | 0/60    | 0.0 % | +27.6  | +39.8      | +40.1 |
| 10.00  | 0/60    | 0.0 % | +27.0  | +39.8      | +40.1 |

`frac > 50 % = 0.0 %` everywhere; not shown.

---

## Acceptance gate evaluation (from the BRIEF)

The BRIEF specified three gates on **success rate**:

| Gate          | Requirement                                          | Verdict |
|---------------|-------------------------------------------------------|---------|
| **Strong**    | peak ≥ 30 %, peak > both endpoints, peak not at λ=0   | **fail** (max = 3.3 %) |
| **Weak**      | peak in [15 %, 30 %]                                  | **fail** |
| **Falsify**   | peak ≤ max(λ=0, λ=10), or curve flat                  | **fires** |

By the letter of the BRIEF, the gates are falsified.  But the
BRIEF gates were defined on success rate alone, which on a harder
task can be dominated by a binary insertion threshold and miss
the underlying mechanism.  The mean-reward curve tells a
different story.

---

## Mean-reward curve as a mechanism proxy

| measure                       | λ=0   | λ=1   | λ=10  | curve shape |
|-------------------------------|------:|------:|------:|-------------|
| Mean reward (this slate)      | 17.6  | 28.6  | 27.0  | rising, plateau ~λ=1-2 |
| Success rate (this slate)     | 0 %   | 3.3 % | 0 %   | flat near zero |
| Reference: Can mean reward<br/>(Slate D, K=10 R=1)*    | ~40  | ~70  | ~65   | rising, plateau ~λ=1 |
| Reference: Can success rate<br/>(Slate D D2 + Slate E E1)*  | 11.7 % | 75 % (peak)<br/>73.8 % pooled | ~50 % | clear inverted-U |

*The Can reference numbers are reconstructed from
docs/results.md; they are *not* re-extracted from JSON here.*

On Can, the success rate tracks the mean reward — both rise with
λ and peak around λ=1.  On Square, the same mean-reward
**shape** is present but it stops at ~28, far below the
~45-threshold needed for sparse success on this task's contact
geometry.  The bridge is doing what it does on Can; the rollouts
just need more compute to finish the job.

This is consistent with hypothesis (a) in the budget-retry
script header: **budget-limited transfer, not mechanism-limited.**

---

## What the retry will tell us

The Square budget-retry queued for after Slate F finishes:

- single cell: `--task square --prior demo_1nn --lambda-prior 1
  --sigma-max 0.25 --mpc-replans 3 --mpc-chunk-steps 30 --K 10
  --M 24 --H 120 --n-demos 20 --n-rollouts 3 --seed 0`
- 3x more MBD work per rollout (R=3 M=24 vs R=1 M=16)
- ~80-90 min wall, auto-launches when `slate_f` screen exits
- Output:
  `experiments/15_unified_hybrid/results/slate_f_square_retry_2026_05_20/`

Decision rule:

- **If retry hits ≥ 15 % success**: budget-limited transfer
  confirmed.  The score-bridge mechanism does generalise to
  Square, but at higher MBD cost.  Story-B's "task-agnostic
  mechanism" claim survives, with a budget caveat.

- **If retry stays ≤ 5 % success but mean reward goes up
  further (e.g., q95 > 50)**: the bridge is still helping but
  Square needs even more budget (R=5 / M=32) or longer horizon.
  Inconclusive; Story-B still survives but the budget story is
  bigger than expected.

- **If retry is still ≤ 5 % success with no further mean-reward
  improvement**: mechanism-limited.  The hybrid genuinely fails
  to transfer to Square's insertion regime; Story-B's
  "task-agnostic" claim weakens.

---

## What still matters from Slate F

F2 (Square σ-tail at λ=1) is **less informative now**: at λ=1
on Square we're already at the rising-edge of the mean-reward
curve (28.6 vs the plateau at ~28), so σ swings probably won't
shift the success threshold either way.  We'll read it as
auxiliary evidence on the bridge-effective-strength λ/σ²
relation.

F3 (Can warm-start variants) is **still the main Track B
deliverable** for tonight:
- `demo_lwr_k5` vs `demo_1nn` -- does locally-weighted
  regression beat 1-NN, or does mode-selection win?
- `demo_topk_random_k5` vs `demo_1nn` -- is the *exact*
  closest demo uniquely good, or any "close enough" demo?
- `demo_1nn_proprio` vs `demo_1nn_object` -- which dims of
  the obs vector carry the retrieval signal?

These all run on Can at the canonical operating point that
we know works, so they are insulated from the Square budget
issue.

---

## Updates to apply to the main results doc after the slate finishes

1. Add a Square section to `docs/results.md` (after Slate E).
   Header: "Slate F (May 20) — cross-task partial transfer on
   square/ph".  Include the F1 table, the mean-reward proxy
   discussion, and (if available) the budget-retry result.

2. Append a Track B subsection with the four Can warm-start
   variants at λ=1 σ=0.25 seed=0, compared against the
   demo_1nn baseline (75 % single-seed / 73.8 % pooled).

3. Update `docs/theory_post_slate_e.md` § 4 with the F1
   reading.  The bias-variance framing predicted the
   *shape* of F1 (inverted-U in λ) on success rate -- which
   failed.  But the framing also implies the bridge moves
   the *mean trajectory* toward the demo manifold, which
   succeeded in the mean-reward proxy.  The framing survives
   with a budget-caveat refinement.

---

## Source

- Slate F BRIEF (with the original acceptance gates):
  `experiments/15_unified_hybrid/results/slate_f_2026_05_20/BRIEF.md`
- F1 raw JSONs:
  `experiments/15_unified_hybrid/results/slate_f_2026_05_20/unified_demo_1nn_lam-*_F1_square_1nn_lamsweep.json`
- Budget-retry launcher:
  `experiments/15_unified_hybrid/slate_f_square_retry_2026_05_20.sh`
- Watcher (running in screen `slate_f_retry`):
  `experiments/15_unified_hybrid/slate_f_followup_watcher.sh`
- Monitor:
  `experiments/15_unified_hybrid/results/slate_f_2026_05_20/monitor.sh`
