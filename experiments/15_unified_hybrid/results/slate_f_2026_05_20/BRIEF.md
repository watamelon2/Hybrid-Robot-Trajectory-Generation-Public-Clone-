# Slate F -- cross-task generalisation + warm-start variants (May 20)

The post-Slate-E follow-up.  Slate E (May 19) substantiated the
73.8 % pooled headline (Wilson CI [67.8, 78.9]) at the canonical
operating point on **Can** with `demo_1nn` retrieval + lambda=1
score-bridge + sigma=0.25.  Two open questions:

1. **Does the headline algorithm transfer to a different task?**
   The score-bridge + retrieval claim should be task-agnostic --
   nothing in the algorithm references "can-ness".  If the
   inverted-U-in-lambda shape replicates on Square (`square/ph`,
   the standard 2nd-tier robomimic task), Story B's mechanism
   claim survives the "you tuned to Can" criticism.

2. **Is the simplest retrieval rule (1-NN) actually the best?**
   Slate E confirmed BC is uniformly below 1-NN at every lambda.
   But within the retrieval-prior family, we've only swept k=1
   (1-NN), k=5 with uniform mean (demo_knn -- worse than 1-NN
   per May 17 narrative_followup), uniform random demo
   (demo_random -- much worse), and dataset-mean trajectory
   (demo_mean -- worst).  Four unexplored variants this Slate
   fills in.

## Two tracks, 13 cells, ~5.5 h

All cells: K=10 R=1 M=16 H=120, 20 demos x 3 rollouts = 60
rollouts per cell, n-workers=10.  Operating point as on May 19.

### Track A -- Square cross-task headline (~3.75 h, 9 cells)

**F1** (~2.5 h, 6 cells): `demo_1nn` lambda-sweep at sigma=0.25
seed=0 on `square/ph`, lambda in {0, 0.5, 1, 2, 5, 10}.  Mirrors
Slate D's D2 lambda-sweep on the new task.

**F2** (~1.25 h, 3 cells): `demo_1nn` sigma-tail on `square/ph`
at lambda=1 seed=0, sigma in {0.05, 0.40, 1.00}.  Mirrors Slate
E's E3 on the new task.  Fixes lambda=1 by assumption (matching
Can); if F1 shows the Square peak is elsewhere, we re-run F2 at
the right lambda in a follow-up.

### Track B -- Can warm-start variants (~1.7 h, 4 cells)

**F3**: 4 new zero-train priors at lambda=1 sigma=0.25 seed=0
on `can/ph`, each compared against the May-19 `demo_1nn` cell
(45/60 single-seed, 177/240 pooled = 73.8 %).

- `demo_lwr_k5` -- locally-weighted regression: k-NN with
  inverse-distance weights.  Theoretical sit between `demo_1nn`
  (k=1) and `demo_knn` (uniform mean over k=5 -- which collapsed
  to ~50 % in narrative_followup).
- `demo_topk_random_k5` -- pick one of the top-5 demos uniformly
  at random (per-chunk deterministic via seeded RNG).  Tests
  whether the nearest demo is uniquely good or whether "close
  enough" suffices.
- `demo_1nn_proprio` -- 1-NN on robot-proprio dims only (drops
  object pos/quat from the retrieval distance).  Tests whether
  the matching is really about object pose or just robot state.
- `demo_1nn_object` -- 1-NN on object dims only.  Complementary
  ablation.

## Predictions / acceptance gates

### Track A -- Square headline

If the Story-B mechanism is real, F1 should produce a clear
inverted-U in lambda on Square, peaked strictly above both
endpoints.  Acceptance gates:

- **Strong (mechanism transfers)**: peak of the F1 curve >= 30 %,
  peak > both lambda=0 and lambda=10, with the peak location
  not at lambda=0.  The shape may be shifted (square is a
  harder task; peak rate will likely be lower in absolute
  terms).
- **Weak (signal too noisy)**: peak in [15 %, 30 %].  Suggestive
  but not conclusive.
- **Falsification**: peak <= the higher of {lambda=0, lambda=10},
  OR no clear peak (flat curve).  Either falsifies the
  "score-bridge sweet spot" claim on this task.

For F2, the prediction follows Slate E's E3 finding -- at
lambda=1 we expect a sigma-peaked curve (sweet-spot regime),
NOT flat (which would indicate the bridge-dominated regime).
If F2 shows a flat sigma curve at lambda=1 on Square, Square's
optimum is at higher lambda than Can's.

### Track B -- Warm-start variants

Baseline to beat at lambda=1 sigma=0.25 seed=0 on Can:
`demo_1nn` = 45/60 = 75.0 %.

Per-variant predictions:

- **`demo_lwr_k5`**: expect a few pp below 1-NN
  (in [60 %, 75 %]).  LWR de-noises the action target by
  blending the top neighbours, which loses some of the
  mode-selection benefit but gains stability.  If LWR is
  *above* 1-NN, that's a new headline candidate.
- **`demo_topk_random_k5`**: expect within ~5 pp of 1-NN
  (in [65 %, 75 %]).  The top-5 demos at a given obs are
  typically very similar action-wise, so any of them should
  work.  A big drop here would suggest the 1-NN's exact
  pick matters.
- **`demo_1nn_proprio`**: ambiguous.  If proprio carries most
  of the retrieval signal, expect ~75 % (matching 1-NN).  If
  object matters, expect a drop into the 40-60 % range.
- **`demo_1nn_object`**: complementary to proprio.  Expect a
  significant drop (probably 20-50 %) because at the start
  of a rollout, many demos are at very similar object poses
  but very different robot configurations; matching on
  object-only would pick a demo with a far-away robot, whose
  initial action would push the robot in the wrong direction.

The single most informative variant pair is
`demo_1nn_proprio` vs `demo_1nn_object`.  Together they
decompose the retrieval signal into "what does the robot need
to do" (proprio) vs "what's the object's situation" (object).

## What lands in `docs/results.md` after Slate F

If Track A passes acceptance: a new section
"Slate F (May 20) -- cross-task replication on `square/ph`"
with the F1+F2 readout and the Square row added to the
headline table.

If Track B finds a variant above 75 %: that variant becomes
the new Can headline.  Otherwise: `demo_1nn` stays the
headline, and the variants are filed as ablations showing
the simplicity is robust.

## Determinism / reproducibility

All cells use a single seed (0 for F1/F2/F3) and the existing
warm-start determinism contract.  Stochastic variants
(`demo_topk_random`) use a per-chunk RNG seeded from
`(rng_seed, counter)`, identical pattern to `demo_random` --
re-running the same cell with the same seed yields
bit-identical results.

`n-workers=10` matches the Slate E run pattern.

## Failure modes / fallbacks

- If a cell hangs > 1 h, `screen -X -S slate_f kill` and
  inspect `run.log`.  Each cell has its own python invocation,
  so a hang in one does not propagate.
- If `--task square` produces a key error in the obs vector,
  the engineering prerequisite (task-aware OBS_KEYS) was
  not applied -- check that `experiments/03_bc_baseline/
  dataset.py:get_obs_keys` exists and that
  `experiments/15_unified_hybrid/priors_zerotrain.py:74`
  imports it.  Smoke-tested 2026-05-19 at engineering time.

## Source

- Launcher: `experiments/15_unified_hybrid/slate_f_2026_05_20.sh`
- Output dir: `experiments/15_unified_hybrid/results/slate_f_2026_05_20/`
- Branch: `feat/zero-train-priors`, HEAD `b543fec` at engineering;
  preamble commits (task-aware obs + 4 new variants) push HEAD
  forward before launch.
