#!/usr/bin/env bash
# Slate G -- algorithm-knob sweep + Lift cross-task + Square action-history
# (May 21 overnight).
#
# Post-Slate-F follow-up.  Slate F (May 20) found:
#   - Can warm-start variants (demo_lwr/topk_random/proprio/object) are
#     all within +/- a few pp of demo_1nn (mode-selection is operative).
#   - Square (`square/ph`) is mechanism-limited at the canonical
#     operating point: 3.3 % peak at lambda=1, no improvement at
#     R=3 M=24, mean reward saturates around 28.6 (Square reward
#     range 0-75).  The hybrid can grasp but never insert.
#
# Open follow-ups Slate F surfaced:
#   1. Cross-task on Lift (simpler than Square) -- distinguish
#      "pick-place only" from "Can only" mechanism scope.
#   2. Square retry with a richer obs vocabulary (action history)
#      to disambiguate contact phases.
#   3. The bias-variance curve under the new soft-1NN prior
#      (continuous knob, supersedes 1-NN vs k-NN at k=5).
#   4. Time-conditional lambda schedule on Can -- can we beat
#      75 % at the canonical operating point by annealing the
#      bridge as MBD denoises?
#
# Four tracks through the unified driver, all at K=10 R=1 M=16
# H=120, 20 demos x 3 rollouts = 60 rollouts/cell, n-workers=10:
#
#   Track A (G1, ~2.5 h, 6 cells)
#     demo_1nn lambda-sweep on `lift/ph`, lambda in {0, 0.5, 1,
#     2, 5, 10} at sigma=0.25 seed=0.  Mirrors slate_f F1 on the
#     simpler task.  Acceptance: inverted-U with peak above
#     lambda=0; distinguishes Can-only from pick-place-class
#     mechanism scope.
#
#   Track B (G2, ~3 h, 6 cells)
#     demo_softnn beta-sweep on `can/ph` at lambda=1 sigma=0.25
#     seed=0, k=5, beta in {0, 0.1, 0.5, 1, 2, 5}.  Draws the
#     bias-variance curve directly: beta=0 recovers demo_knn k=5
#     (uniform mean -- variance-dominated, ~50 % per
#     narrative_followup), beta>>1 recovers demo_1nn (mode-
#     selection, 75 % at this operating point).  Acceptance:
#     monotone-ish curve from variance regime to mode regime;
#     the "sweet spot" beta location informs whether soft-1NN
#     beats hard 1-NN with finite smoothing.
#
#   Track C (G3, ~1.3 h, 3 cells)
#     demo_1nn on `can/ph` at lambda=1 sigma=0.25 seed=0,
#     --lambda-schedule in {linear, inverse_linear, cosine}.
#     Tests whether annealing the score-bridge across MBD
#     iterations beats the historical constant=1 schedule
#     (which already includes the natural sigma-decay on lr_k).
#     Acceptance: at least one schedule >= 75 % (matches
#     constant baseline) with at most one significantly
#     worse.  A peak above 80 % would be a new Can headline.
#
#   Track D (G4, ~1.3 h, 3 cells)
#     demo_1nn on `square/ph` at lambda=1 sigma=0.25 seed=0
#     with --action-history-len in {1, 5, 10}.  Square hypothesis:
#     action history disambiguates contact phases (approach /
#     grasp / lift / insert) when the instantaneous state alone
#     is ambiguous.  Acceptance: any --action-history-len cell
#     beats Slate F's 3.3 % Square peak at lambda=1.  A >= 15 %
#     success rate substantiates the action-history hypothesis.
#
# Total: 18 cells, ~8 h wall.  Run in screen "slate_g".

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_g_2026_05_21"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate G: Lift lambda-sweep (G1) + softnn beta-sweep (G2) + lambda-schedule (G3) + Square action-history (G4)" | tee -a "$LOG"
echo "Total ~8 h wall, 18 cells." | tee -a "$LOG"

# ----- G1: demo_1nn lambda-sweep on Lift at sigma=0.25 seed=0 --------
# Single sweep invocation; run_can.py loops internally and writes one
# JSON per lambda.  Lift demos average ~50 timesteps; H=120 gives
# generous tail.  lambda=0 anchors the warm-start-only point.
echo "##### G1: demo_1nn lambda-sweep on Lift #####" | tee -a "$LOG"
TAG="G1_lift_1nn_lamsweep"
echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task lift \
  --mpc-replans 1 --mpc-chunk-steps 60 \
  --K 10 --M 16 --H 120 \
  --sigma-max 0.25 --sigma-min 0.01 \
  --n-demos 20 --n-rollouts 3 \
  --n-workers 10 \
  --output-dir "$OUTDIR" \
  --output-name-suffix "_$TAG" \
  --prior demo_1nn --lambda-prior 0 0.5 1 2 5 10 --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

# ----- G2: demo_softnn beta-sweep on Can at lambda=1 sigma=0.25 ------
# --retrieval-beta is single-valued, so loop in shell.  At beta=0 the
# soft-1NN collapses to demo_knn k=5 (uniform mean over top-5);
# beta>>1 recovers demo_1nn.  Six points span the variance->mode
# transition with a log-ish stride.
echo "##### G2: demo_softnn beta-sweep on Can #####" | tee -a "$LOG"
for BETA in 0 0.1 0.5 1 2 5; do
  # File-safe beta tag (matches priors_zerotrain.make_zero_train_prior).
  BETA_TAG="${BETA/./p}"
  TAG="G2_can_softnn_k5_b${BETA_TAG}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_softnn --knn-k 5 --retrieval-beta "$BETA" \
    --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- G3: lambda-schedule comparison on Can at lambda=1 sigma=0.25 --
# Three non-constant schedules at the May-19 canonical operating point.
# The 'constant' schedule's outcome is already known (75.0 % single
# seed); these cells answer whether annealing on top of the natural
# sigma-decay helps.
echo "##### G3: lambda-schedule on Can #####" | tee -a "$LOG"
for SCHED in linear inverse_linear cosine; do
  TAG="G3_can_1nn_lam1_sched-${SCHED}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 \
    --lambda-schedule "$SCHED" --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- G4: action-history on Square at lambda=1 sigma=0.25 -----------
# Square retry with the A1 augmentation.  Slate F's F1 hit a 3.3 %
# ceiling on Square at this operating point; action history is the
# narrowest engineering knob targeting the contact-disambiguation
# hypothesis.  Bank is rebuilt once per action-history-len (cache key
# includes it), so each cell pays a ~30-60 s bank-build cost on top
# of the eval wall.
echo "##### G4: action-history on Square #####" | tee -a "$LOG"
for HIST in 1 5 10; do
  TAG="G4_square_1nn_lam1_hist${HIST}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 \
    --action-history-len "$HIST" --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
