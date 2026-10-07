#!/usr/bin/env bash
# Slate I -- C1 contrastive obs embedding for Square (May 22 overnight).
#
# Post-Slate-G/G-aux follow-up.  Slate G + G-aux (May 21) confirmed
# the Square mechanism-limit triple:
#   - A1 action-history obs (G4):     <=5% across hist in {1, 5, 10}.
#   - sigma-tail high-res (AUX1):     peak at sigma=0.30 -> 8.3 %.
#   - soft-NN beta-sweep (AUX2):      flat 3.3 % across beta in {0.5, 1, 2, 5}.
# Combined with Slate F's MBD-budget retry, five orthogonal axes
# (prior identity, lambda, sigma, MBD compute, raw obs vocabulary)
# leave Square below 9 % at the F1 sweet spot.
#
# The remaining structural hypothesis is that the raw obs vector
# doesn't separate Square's contact phases (a fingertip-on-peg
# state and a fingertip-just-released state look nearly identical
# in 46-d raw obs but require very different actions).  C1 tests
# this directly: replace the obs->bank-index map with a *learned
# contrastive embedding* trained on temporal proximity inside
# demos.  Phase-equivalent states embed close together; argmin in
# embedding space recovers a phase-correct neighbour where raw
# 1-NN was ambiguous.
#
# Three tracks through the unified driver at the canonical
# operating point (K=10 R=1 M=16 H=120, sigma=0.25 by default,
# seed=0, n-demos=20, n-rollouts=3 = 60 rollouts/cell,
# n-workers=10):
#
#   Track A (I1, ~3.7 h, 6 cells)
#     demo_1nn_contrastive lambda-sweep on `square/ph`.
#     lambda in {0, 0.5, 1, 2, 5, 10} at sigma=0.25 seed=0.
#     Encoder: artifacts/contrastive_encoder_square_2026_05_21.pt
#     (60 epochs, embed_dim=32, gap=0.956 on train pairs).
#
#     Acceptance:
#       - STRONG (Square unlocked):   peak succ >= 25%.  Reopens
#                                      Square and motivates C2/C3
#                                      as further structural
#                                      improvements rather than
#                                      replacements.
#       - WEAK   (retrieval helps):   peak succ in [15%, 25%].
#                                      Beats the F1/AUX1 8.3% peak;
#                                      C1 is the right axis but
#                                      C2 phase-conditioning is
#                                      still needed.
#       - FALSIFICATION (mech limit): peak succ <= 10%.  Retrieval
#                                      identity is not the
#                                      bottleneck; the structural
#                                      change must be in the
#                                      cost function (C2) or
#                                      sampling (C3).  Square is
#                                      filed as "needs new
#                                      bridge", not "needs new
#                                      prior".
#
#   Track B (I2, ~2.5 h, 4 cells)
#     demo_1nn_contrastive sigma-tail on `square/ph` at the
#     I1 best lambda (assumed lambda=1 ahead of I1; if I1 finds
#     a peak elsewhere, re-queue I2 around it).
#     sigma in {0.20, 0.30, 0.40, 1.00} at lambda=1 seed=0.
#     Mirrors AUX1's sigma-tail study but in embedding space;
#     answers whether the C1 prior reshapes the sigma-curve
#     (e.g., peak moves to higher sigma because embedded
#     retrieval is robust to bigger perturbations).
#
#     Acceptance: same gates as I1.  If I1 is strong but I2's
#     curve looks identical to AUX1's, the C1 win is purely
#     retrieval (sigma is unchanged).  If I2 finds a different
#     peak, C1 is reshaping the bias-variance tradeoff.
#
#   Track C (I3, ~1.6 h, 3 cells)
#     demo_1nn_contrastive control on `can/ph`.
#     lambda in {0.5, 1, 2} at sigma=0.25 seed=0.
#     Encoder: artifacts/contrastive_encoder_can_2026_05_21.pt
#     (separate encoder per task -- different obs vocabulary).
#
#     Acceptance: each cell within +/- 10 pp of Can baseline
#     (demo_1nn lambda=1 -> 75 %).  Verifies C1 doesn't BREAK
#     a working task -- i.e., the learned embedding preserves
#     the Can mode-selection benefit.  A regression below 65 %
#     would indicate C1 is task-specific (over-fit to Square)
#     and motivate a per-task hyperparameter sweep on the
#     contrastive trainer.
#
# Total: 13 cells, ~7.8 h wall.  Run in screen "slate_i".
#
# Encoders were trained on 2026-05-21 by
# experiments/15_unified_hybrid/train_contrastive_encoder.py with
# default hparams (60 epochs, batch 512, embed_dim 32, k_max 4,
# tau 0.1, seed 0).  All three have final gap ~0.95 on train pairs.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_i_2026_05_22"
LOG="$OUTDIR/run.log"
ART="$REPO/artifacts"
ENC_SQUARE="$ART/contrastive_encoder_square_2026_05_21.pt"
ENC_CAN="$ART/contrastive_encoder_can_2026_05_21.pt"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate I: C1 contrastive prior on Square (I1 lambda, I2 sigma) + Can control (I3)" | tee -a "$LOG"
echo "Total ~7.8 h wall, 13 cells." | tee -a "$LOG"

# Sanity check: encoder artifacts present.
for ENC in "$ENC_SQUARE" "$ENC_CAN"; do
  if [[ ! -f "$ENC" ]]; then
    echo "FATAL: missing encoder $ENC" | tee -a "$LOG"
    echo "Train it with experiments/15_unified_hybrid/train_contrastive_encoder.py first." | tee -a "$LOG"
    exit 2
  fi
done
echo "Encoders OK: $(basename "$ENC_SQUARE"), $(basename "$ENC_CAN")" | tee -a "$LOG"

# ----- I1: C1 lambda-sweep on Square at sigma=0.25 seed=0 ------------
echo "##### I1: demo_1nn_contrastive lambda-sweep on Square #####" | tee -a "$LOG"
TAG="I1_square_c1_lamsweep"
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
  --prior demo_1nn_contrastive \
  --contrastive-encoder-path "$ENC_SQUARE" \
  --lambda-prior 0 0.5 1 2 5 10 \
  --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

# ----- I2: C1 sigma-tail on Square at lambda=1 seed=0 ---------------
# Single sigma per invocation (--sigma-max is single-valued), so we
# loop in shell.  Each cell launches with --lambda-prior 1; the
# encoder + bank are reloaded per cell, but the bank cache hits the
# Square npz so the per-cell startup is ~1 s.
echo "##### I2: demo_1nn_contrastive sigma-tail on Square #####" | tee -a "$LOG"
for SIG in 0.20 0.30 0.40 1.00; do
  SIG_TAG="${SIG/./p}"
  TAG="I2_square_c1_lam1_sigma${SIG_TAG}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max "$SIG" --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn_contrastive \
    --contrastive-encoder-path "$ENC_SQUARE" \
    --lambda-prior 1 \
    --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- I3: C1 control on Can at sigma=0.25 seed=0 -------------------
# 3 lambdas in one invocation so the encoder + bank are loaded once.
echo "##### I3: demo_1nn_contrastive control on Can #####" | tee -a "$LOG"
TAG="I3_can_c1_lamsweep_control"
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
  --prior demo_1nn_contrastive \
  --contrastive-encoder-path "$ENC_CAN" \
  --lambda-prior 0.5 1 2 \
  --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
