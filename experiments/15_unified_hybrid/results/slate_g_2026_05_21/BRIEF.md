# Slate G -- algorithm-knob sweep + Lift cross-task + Square action-history (May 21)

Post-Slate-F follow-up.  Slate F (May 20) closed two questions and
opened three.

**Slate F closures**

- Can warm-start variants (`demo_lwr_k5`, `demo_topk_random_k5`,
  `demo_1nn_proprio`, `demo_1nn_object`) all sit within a few pp of
  the `demo_1nn` baseline.  Mode-selection is the operative
  property; 1-NN is not magic.
- Square (`square/ph`) is *mechanism*-limited, not budget-limited:
  3.3 % peak at lambda=1, identical at R=3 M=24, mean reward
  saturates around 28.6 (Square reward range 0-75).  The hybrid
  grasps but never inserts.

**Slate F open follow-ups (Slate G addresses three of three)**

1. Cross-task on Lift -- distinguish "Can-only" from "pick-place
   class" mechanism scope.  G1.
2. Square retry with a richer obs vocabulary.  G4 tests the
   simplest such vocabulary expansion (action history).
3. A continuous knob over the bias-variance tradeoff that supersedes
   the discrete 1-NN / k-NN / lwr family.  G2.

One extra question, surfaced by the new infrastructure during
Slate G engineering:

4. The historical lr_k schedule already includes a natural
   sigma-decay (`lr_k = lambda * sigma_k / sigma_max`).  Does an
   *additional* lambda schedule (linear / cosine anneal-out vs
   inverse anneal-in) help on Can?  G3.

## Four tracks, 18 cells, ~8 h

All cells: K=10 R=1 M=16 H=120, 20 demos x 3 rollouts = 60
rollouts per cell, n-workers=10, seed=0.

### Track A -- G1 (Lift lambda-sweep, ~2.5 h, 6 cells)

`demo_1nn` lambda-sweep on `lift/ph` at sigma=0.25 seed=0, lambda
in {0, 0.5, 1, 2, 5, 10}.  Single sweep invocation (run_can.py
loops internally).

**Acceptance gates**

- **Strong (mechanism transfers to all pick-place)**: peak >= 60 %
  with peak above both lambda=0 (warm-start only) and lambda=10
  (bridge-dominated tail).  Lift is the easiest robomimic task;
  a 60 % bar leaves headroom for headline-quality replication.
- **Weak (signal positive, transfer partial)**: peak in [30, 60 %]
  with the inverted-U shape preserved.  Still substantiates
  Story B's pick-place-class claim.
- **Falsification**: peak <= max(lambda=0, lambda=10), OR flat
  curve.  Would suggest the mechanism is Can-specific.

### Track B -- G2 (Soft 1-NN beta-sweep on Can, ~3 h, 6 cells)

`demo_softnn` on `can/ph` at lambda=1 sigma=0.25 seed=0, k=5,
beta in {0, 0.1, 0.5, 1, 2, 5}.  Loop in shell since
`--retrieval-beta` is single-valued.

The soft-1NN prior (A3, added 2026-05-20) interpolates continuously
between the two endpoints of the bias-variance tradeoff that the
Slate F variants approximate discretely:

- beta = 0 -> uniform mean over top-k = `demo_knn` k=5.  High bias,
  low variance.  Per narrative_followup (May 17), `demo_knn` k=5
  hit ~50 % at the old operating point.
- beta -> infinity -> hard argmax = `demo_1nn`.  Low bias, high
  variance.  75 % at the May-19 canonical point.

**Acceptance gates**

- **Strong (continuous knob beats both endpoints)**: some beta in
  (0, 5) >= 78 % (clearing the 75 % hard-1NN baseline by at least
  3 pp).  Would make `demo_softnn` the new Can headline.
- **Weak (curve is monotone, hard 1-NN is the optimum)**: smooth
  rise from ~50 % at beta=0 to ~75 % at beta=5, no interior peak.
  Confirms mode-selection is the operative property and motivates
  treating beta as a regulariser rather than a free hyperparameter.
- **Falsification**: non-monotone or flat curve below 50 %.  Would
  indicate the soft-mean is fundamentally worse than the closest
  variants in Slate F.

### Track C -- G3 (Lambda schedule on Can, ~1.3 h, 3 cells)

`demo_1nn` on `can/ph` at lambda=1 sigma=0.25 seed=0, with
`--lambda-schedule` in {linear, inverse_linear, cosine}.
Constant=1 (the historical baseline) hit 75.0 % single-seed at
this operating point.

The historical lr_k formula is

    lr_k = prior_lambda * (sigma_k / sigma_max)

which already anneals lr_k linearly in noise level.  The new
`--lambda-schedule` flag multiplies this by an additional
per-iteration factor:

- linear: 1.0 at k=0 -> 0.0 at k=K-1 (bridge strong early when
  noise is high; faded at low-sigma tail so MBD's local refinement
  dominates).
- inverse_linear: 0.0 at k=0 -> 1.0 at k=K-1 (bridge weak early,
  strong late).  Counterintuitive but tests whether MBD-led
  exploration + bridge-led stabilisation is better than the reverse.
- cosine: 0.5 * (1 + cos(pi * k / (K-1))); smooth anneal-out from
  1.0 to 0.0.

**Acceptance gates**

- **Strong (annealing helps)**: at least one schedule >= 80 %
  (5 pp above the constant baseline).
- **Weak (schedules within noise of constant)**: all schedules in
  [70, 80 %].  Result: filed as a robustness check; constant
  stays the default.
- **Falsification of the "anneal-out" intuition**: linear and
  cosine both significantly below constant.  Would suggest the
  late-iteration bridge contribution is important.

### Track D -- G4 (Square action-history, ~1.3 h, 3 cells)

`demo_1nn` on `square/ph` at lambda=1 sigma=0.25 seed=0 with
`--action-history-len` in {1, 5, 10}.  The bank is rebuilt once
per action-history length (cache key includes it).

Slate F established a 3.3 % Square ceiling at this operating point.
Hypothesis: Square's contact-rich insertion phase produces obs
vectors that are ambiguous between "approach", "grasp", "lift",
and "insert"; the same instantaneous state can correspond to any
of these depending on what the agent has been doing.  Action
history makes the bank's retrieval distance phase-aware without
training a phase encoder.

**Acceptance gates**

- **Strong (action history unlocks Square)**: any
  `--action-history-len` cell >= 15 % success (>= 4x the F1 peak).
  Would substantiate the contact-disambiguation hypothesis and
  re-open the Square cross-task headline question.
- **Weak (mean-reward improvement only)**: success rate <= 6 %
  but mean reward > 35 (vs F1's 28.6).  Suggests the bridge
  pushes further along the reward landscape but still misses the
  insertion endpoint.
- **Falsification**: all three cells <= the F1 baseline of
  3.3 % AND mean reward unchanged.  Action history is not the
  binding constraint on Square; bridge schedule, horizon, or
  cost shaping are more promising directions.

## What lands in `docs/results.md` after Slate G

- A new "Slate G (May 21)" section with G1-G4 readouts.
- If G1 passes the weak gate: Lift row added to the cross-task
  headline table (alongside Can and Square).
- If G2 finds beta*>0 above 78 %: soft-1NN becomes the new Can
  headline candidate (pending Slate H multi-seed pool).
- If G3 finds a schedule above 80 %: that schedule becomes the
  Can default for subsequent slates.
- If G4 passes the strong gate: A1 (action history) becomes a
  default-on flag for Square; Slate H sweeps `--action-history-len`
  in combination with lambda.

## Engineering prerequisites (all landed 2026-05-21)

- **Tasks** (`experiments/03_bc_baseline/dataset.py`): `lift`,
  `transport`, `tool_hang` are now first-class.  G1 uses `lift`.
- **D2** (`priors_zerotrain.py`): on-disk bank cache keyed by
  (dataset_path, mtime, n_demos, obs_keys, action_history_len).
  Saves ~30-60 s/cell after the first invocation.
- **A1** (`priors_zerotrain.py`, `unified_loop.py`, `run_can.py`):
  per-timestep action history augmentation, parameterised by
  `--action-history-len N`.  Bank obs and runtime s0_vec stay
  synchronised; default N=0 preserves bit-for-bit compat.
- **A2** (`experiments/04_mbd_baseline/mbd.py`, `unified_loop.py`,
  `run_can.py`): `--lambda-schedule` accepts
  {constant, linear, inverse_linear, cosine}.  `constant` (default)
  is bit-for-bit reproducing the historical lr_k.
- **A3** (`priors_zerotrain.py`, `run_can.py`): `demo_softnn`
  prior with `--retrieval-beta` temperature.  Endpoints verified
  to match `demo_1nn` (beta=100) and `demo_knn` (beta=0) exactly.

## Determinism / reproducibility

All cells single-seed (seed=0).  The bank's npz cache is content-
addressed by hash of (dataset_path, mtime, n_demos, obs_keys,
action_history_len); rerunning a cell hits the cache.

Stochastic priors (none used in Slate G; `demo_random` and
`demo_topk_random` not on this slate) use a per-chunk RNG seeded
from `(rng_seed, counter)`.

## Failure modes / fallbacks

- If a cell hangs > 1 h, `screen -X -S slate_g kill` and inspect
  `run.log`.
- If G4 produces a key error on Square -> action-history bank
  build path is wrong.  Inspect `_build_action_history` and the
  obs-aug branch in `load_demo_bank`.  Smoke-tested 2026-05-21
  on Lift with H=120 K=10 R=1 N=5 succ=3/3.
- If G1 (Lift) shows the same flat-curve pathology as Square
  F1, the mechanism is genuinely Can-specific.  Slate H would
  then pivot to "why does Can work" rather than "how to make
  Square work".

## Source

- Launcher: `experiments/15_unified_hybrid/slate_g_2026_05_21.sh`
- Output dir: `experiments/15_unified_hybrid/results/slate_g_2026_05_21/`
- Branch: `feat/zero-train-priors`.  Engineering commit lands
  before launch.
