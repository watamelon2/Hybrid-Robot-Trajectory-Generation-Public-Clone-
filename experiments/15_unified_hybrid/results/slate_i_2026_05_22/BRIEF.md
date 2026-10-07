# Slate I -- C1 contrastive obs embedding for Square (May 22)

## Why this slate

Slate G + G-aux (May 21) confirmed the Square mechanism-limit
**triple**:

| axis (Slate F/G test)              | best succ | gate cleared? |
|------------------------------------|----------:|---------------|
| prior identity (F3 variants)       |   3.3 %   | no            |
| bridge lambda (F1 lam-sweep)       |   3.3 %   | no            |
| sigma (F2 + AUX1 high-res)         |   8.3 %   | no (close)    |
| MBD compute (F1 budget retry K,M,R)|   3.3 %   | no            |
| obs vocabulary (G4 A1 hist)        |   5.0 %   | no            |

Combined: five orthogonal knobs at the F1 sweet spot give a
Square peak of 8.3 %, well below the weak gate of 15 % and far
below the strong gate of 30 %.  **The structural change must
not lie in any of {prior identity, lambda, sigma, MBD compute,
raw obs vocabulary}.**

The one remaining hypothesis is that **the raw obs vector does
not separate Square's contact phases.**  Concretely:

- approach phase: gripper above peg, square nut held;
- contact phase: gripper at peg level, nut threading;
- recovery phase: gripper above peg, nut threaded onto peg.

In raw obs, "approach" and "recovery" look nearly identical
(same gripper height, same nut pose modulo a few mm of slop),
but they require **opposite** actions.  1-NN over raw obs
cannot reliably tell these apart, so the retrieved action
window is phase-correct only ~50 % of the time on contact-
critical states.

C1 = train a small contrastive encoder on demo trajectories
so that **temporally-adjacent states inside the same demo
embed close together**.  This forces the encoder to factor out
the spatial ambiguity and represent phase identity instead.
Retrieve via 1-NN in embedding space; the actions follow the
phase, not the raw spatial state.

## Encoders trained on May 21 PDT

All three trained with `train_contrastive_encoder.py` defaults
(60 epochs, batch 512, embed_dim 32, hidden 128, k_max 4,
tau 0.1, seed 0):

| task   | demos | steps  | final loss | pos_sim | neg_sim | gap   | train wall |
|--------|------:|-------:|-----------:|--------:|--------:|------:|-----------:|
| square |   200 | 30 154 |     0.294  |  +0.960 |  +0.004 | 0.956 |     90 s   |
| can    |   200 | 26 939 |     0.309  |  +0.964 |  +0.004 | 0.959 |     68 s   |
| lift   |   200 | 11 188 |     0.484  |  +0.956 |  +0.004 | 0.952 |     27 s   |

Lift has 2.5x fewer pairs (~50 steps/demo) so its loss is
higher per epoch but the same gap is reached.  All checkpoints
live in `artifacts/contrastive_encoder_{task}_2026_05_21.pt`
(plus a sidecar `.meta.json` for at-a-glance inspection).

C1 smoke (1 demo x 1 rollout on Square at K=4 M=8 H=80, lambda=1,
sigma=0.25): ran clean, reward=+12.33, no crash.  Wiring through
`unified_loop._make_obs_vec_for_prior` -> bank -> encoder
-> retrieval is verified.

## Tracks

All cells: canonical operating point (K=10 R=1 M=16 H=120,
seed=0, 20 demos x 3 rollouts = 60 rollouts/cell), n-workers=10.

### I1 -- C1 lambda-sweep on Square (6 cells, ~3.7 h)

Direct mirror of Slate F's F1 lambda-sweep with `demo_1nn`
replaced by `demo_1nn_contrastive`.  lambda in {0, 0.5, 1, 2,
5, 10} at sigma=0.25 seed=0.

Acceptance gates:

- **STRONG** (Square unlocked): peak succ >= 25 %.  Reopens
  Square; C2/C3 become further structural improvements rather
  than replacements.  This is the headline result that proves
  retrieval *was* the bottleneck after all.
- **WEAK** (retrieval helps): peak succ in [15, 25] %.
  Beats the F1 / AUX1 8.3 % peak by >= 2x; C1 is the right
  axis but C2 phase-conditioning is needed for full unlock.
- **FALSIFICATION** (mech limit holds): peak succ <= 10 %.
  Retrieval identity in *any* representation is not the
  bottleneck; the structural change must be in the cost
  function or sampling distribution.  Square is filed as
  "needs new bridge", not "needs new prior".

### I2 -- C1 sigma-tail on Square (4 cells, ~2.5 h)

sigma in {0.20, 0.30, 0.40, 1.00} at lambda=1 seed=0.
Mirror of AUX1's sigma-tail but in embedding space.  Tests
whether C1 reshapes the sigma-curve (e.g., peak moves to
higher sigma because embedded retrieval is robust to bigger
perturbations).

Acceptance: same gates as I1.  If I1 is strong but I2's
curve looks identical to AUX1's (peak still at sigma~0.30
at the same succ), the C1 win is purely retrieval; sigma
behaviour is unchanged.  If I2 finds a different peak, C1
is reshaping the bias-variance tradeoff and we should
re-evaluate sigma-scheduling.

### I3 -- C1 control on Can (3 cells, ~1.6 h)

lambda in {0.5, 1, 2} at sigma=0.25 seed=0, with the **Can-
specific** encoder.  Verifies C1 does not break a working
task.

Acceptance: each cell within +/- 10 pp of Can baseline
(`demo_1nn` lambda=1 = 75 %).  A regression below 65 %
indicates the encoder is task-specifically overfitting and
the contrastive trainer needs per-task tuning (smaller
hidden dim?  bigger embed dim?  longer training?).  A
preservation above 70 % at lambda=1 means C1 is a clean
drop-in for the retrieval prior on the working tasks.

## Engineering posture

- 13 cells, ~36 min/cell at n-workers=10 (Square wall ~25 min
  baseline, +5 - 10 min for encoder load + bank projection
  per cell).  Total wall ~7.8 h.
- Encoders pre-baked; bank cache pre-warmed by Slate F/G.
  Per-cell startup is ~2 - 3 s once everything is on disk.
- Watcher: `slate_i_watcher.sh` polls for the `slate_h` screen
  to disappear, then auto-fires Slate I in a fresh `slate_i`
  screen.  Launch it any time:
  `screen -dmS slate_i_watcher bash experiments/15_unified_hybrid/slate_i_watcher.sh`
  Cancel by killing the watcher screen.
- All three Slate H gates are independent of C1, so the order
  is H first (4.6 h), I second (7.8 h).  Expected combined
  finish: ~9 AM Friday (if started 6 PM Thursday).

## Source

- Launcher:    `experiments/15_unified_hybrid/slate_i_2026_05_22.sh`
- Watcher:     `experiments/15_unified_hybrid/slate_i_watcher.sh`
- Encoders:    `artifacts/contrastive_encoder_{can,lift,square}_2026_05_21.pt`
- Trainer:     `experiments/15_unified_hybrid/train_contrastive_encoder.py`
- Prior code:  `experiments/15_unified_hybrid/priors_zerotrain.py` (kind="demo_1nn_contrastive")
- Driver CLI:  `--prior demo_1nn_contrastive --contrastive-encoder-path <pt>`
- Branch:      `feat/zero-train-priors`.

## What happens after Slate I

If **STRONG**: Slate J = sigma-schedule + soft-NN on C1 (re-run
AUX2 in embedding space; the soft-NN curve under C1 may differ
qualitatively from the raw soft-NN curve).  Plus seed pool to
get CIs on the new peak.

If **WEAK**: Slate J = C2 phase-conditional bridge (HMM over
demo phases, feed phase one-hot into the score function).
~1 week impl.  C1 retrieval stays as the prior.

If **FALSIFIED**: defer C2/C3 for now; switch to data side --
fine-tune the bridge on Square mistake-trajectories (active
data collection).  Filed as a longer-horizon ticket; no
overnight slate.
