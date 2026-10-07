#!/usr/bin/env bash
# Slate F -- cross-task generalisation + warm-start algorithm
# variants (May 20 overnight).
#
# Two tracks through the unified driver at the canonical operating
# point (K=10 R=1 M=16 H=120, 20 demos x 3 rollouts = 60 rollouts
# per cell, n-workers=10):
#
#   Track A (Square cross-task headline replication, ~3.75 h):
#
#     F1: demo_1nn lambda-sweep at sigma=0.25 seed=0 on `square/ph`,
#         lambda in {0, 0.5, 1, 2, 5, 10} (6 cells, one sweep
#         invocation). Mirrors slate_d D2 on a new task. Acceptance:
#         clear inverted-U with a peak above lambda=0 (warm-start
#         only) and lambda=10 (bridge-dominated tail).
#
#     F2: demo_1nn sigma-tail on `square/ph` at lambda=1 seed=0,
#         sigma in {0.05, 0.40, 1.00} (3 cells, loop in shell).
#         Mirrors slate_e E3 on a new task. Acceptance: a sigma curve
#         that is either flat (matching the lambda=5 Can regime) or
#         peaked at moderate sigma (matching the lambda=1 Can regime).
#         Either outcome substantiates the mechanism; the question is
#         which regime Square sits in.
#
#   Track B (Can warm-start algorithm variants, ~1.7 h):
#
#     F3: 4 new zero-train priors at lambda=1 sigma=0.25 seed=0 on
#         `can/ph` (4 cells, loop in shell):
#         - demo_lwr_k5      : k-NN with inverse-distance weights.
#                              Hypothesis: recovers most of 1-NN's
#                              mode-selection while reducing variance.
#         - demo_topk_random_k5 : random pick from top-k. Hypothesis:
#                              any "close enough" demo works (since
#                              k=1 already plateaus around 75%).
#         - demo_1nn_proprio : 1-NN on robot-only obs dims (drops
#                              object pos/quat). Info ablation.
#         - demo_1nn_object  : 1-NN on object-only obs dims. The
#                              complementary info ablation.
#
# Total: 13 cells, ~5.5 h wall.  Started 2026-05-20 ~02:30 PT in
# screen "slate_f".  ETA finish ~08:00 PT.
#
# Branch: feat/zero-train-priors.  Engineering prerequisite:
# task-aware OBS_KEYS (Slate F preamble commits) -- without those,
# `--task square` raises a key error in the obs concatenation.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_f_2026_05_20"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate F: Square cross-task (F1 lambda-sweep + F2 sigma-tail) + Can warm-start variants (F3)" | tee -a "$LOG"
echo "Total ~5.5 h wall, 13 cells." | tee -a "$LOG"

# ----- F1: demo_1nn lambda-sweep on Square at sigma=0.25 seed=0 ------
# Single sweep invocation; run_can.py loops internally and writes one
# JSON per lambda.  lambda=0 included to anchor the warm-start-only
# point (no bridge), lambda=10 to bracket the bridge-dominated tail.
echo "##### F1: demo_1nn lambda-sweep on Square #####" | tee -a "$LOG"
TAG="F1_square_1nn_lamsweep"
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
  --prior demo_1nn --lambda-prior 0 0.5 1 2 5 10 --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

# ----- F2: demo_1nn sigma-tail on Square at lambda=1 seed=0 ----------
# sigma_max is single-valued in run_can.py, so loop in shell.
echo "##### F2: demo_1nn sigma-tail on Square at lambda=1 #####" | tee -a "$LOG"
for SIGMA in 0.05 0.40 1.00; do
  TAG="F2_square_1nn_lam1_sigma${SIGMA}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max "$SIGMA" --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- F3: warm-start algorithm variants on Can at lambda=1 sigma=0.25 -
# Four new zero-train priors at the May-19 headline operating point.
# Compares against the 73.8% pooled / 75% single-seed demo_1nn cell.
echo "##### F3: warm-start variants on Can #####" | tee -a "$LOG"
for VARIANT in demo_lwr demo_topk_random demo_1nn_proprio demo_1nn_object; do
  TAG="F3_can_${VARIANT}_lam1"
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
    --prior "$VARIANT" --knn-k 5 \
    --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
