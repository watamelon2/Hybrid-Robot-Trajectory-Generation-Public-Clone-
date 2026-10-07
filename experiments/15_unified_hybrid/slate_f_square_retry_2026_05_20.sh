#!/usr/bin/env bash
# Slate F follow-up -- Square budget-retry (May 20).
#
# F1's six-lambda sweep on `square/ph` showed mean reward rising
# from 17.6 (lambda=0) to ~28 (lambda=1-10) but ZERO rollouts (out
# of 360) cleared the success threshold.  Max reward ever achieved
# was 45.6 -- the hybrid completes "approach + grasp the nut" but
# never the final "insert nut onto peg" stage.
#
# Two competing hypotheses:
#
#   (a) Budget-limited:  K=10 R=1 M=16 H=120 was tuned to Can.
#       Square's contact-rich insertion needs more rollout samples
#       and more replans.  The mean-reward-rises-with-lambda
#       monotone-then-flat shape transfers, but the threshold
#       doesn't clear within budget.
#
#   (b) Mechanism-limited:  the algorithm does not extend to Square
#       at all.  Increasing budget would still produce near-zero
#       success.
#
# This single-cell retry distinguishes them by holding lambda fixed
# at the F1 sweet spot (lambda=1) and tripling the per-rollout MBD
# compute via R=3 (4 chunks) and M=24 (50% more samples per iter).
#
# Per-rollout MBD work: R=3 M=24 = 4 chunks * 24 samples * 10 iters
#                                = 960 sample-evaluations
#                     vs F1's     R=1 M=16 = 2 * 16 * 10
#                                = 320  (=> ~3x more compute).
#
# Budget: ~80-90 min wall at n-workers=10.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_f_square_retry_2026_05_20"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate F retry: Square at R=3 M=24 lambda=1 sigma=0.25 seed=0" | tee -a "$LOG"
echo "Compare against F1 lambda=1 cell (2/60 = 3.3%, mean_r=+28.6, max_r=+45.6)." | tee -a "$LOG"

TAG="square_retry_R3_M24_lam1"
echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task square \
  --mpc-replans 3 --mpc-chunk-steps 30 \
  --K 10 --M 24 --H 120 \
  --sigma-max 0.25 --sigma-min 0.01 \
  --n-demos 20 --n-rollouts 3 \
  --n-workers 10 \
  --output-dir "$OUTDIR" \
  --output-name-suffix "_$TAG" \
  --prior demo_1nn --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
