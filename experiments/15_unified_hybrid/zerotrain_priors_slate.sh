#!/usr/bin/env bash
# Zero-training warm-start priors survey, overnight slate.
#
# Four priors x two lambdas = eight cells, all through the unified
# driver (exp 15) at the **headline config** so cells compose
# directly with the existing 2x2 warm-start ablation table.
#
#   prior         lambda    label             purpose
#   demo_1nn      0         Z1a               1-NN retrieval warm-start
#   demo_1nn      5         Z1b               1-NN + retrieved-target bridge
#   demo_knn_k5   0         Z2a               5-NN-mean warm-start
#   demo_knn_k5   5         Z2b               5-NN + retrieved-target bridge
#   demo_random   0         Z3a               random demo (state-cond ablation)
#   demo_random   5         Z3b               random demo + retrieved bridge
#   demo_mean     0         Z4a               dataset-mean trajectory
#   demo_mean     5         Z4b               dataset-mean + bridge
#
# Each cell:  20 demos x 3 rollouts = 60 rollouts;  K=10 R=1 sigma=0.25
#             M=16 H=120 seed=0; 10 parallel rollout workers.
#
# Estimated wall: ~25-30 min per (prior, lambda).  Running both
# lambdas in one process per prior to save spawn / env-load overhead:
# 4 invocations x ~55 min = ~3.7 h.  Comfortable for overnight.
#
# Launched 2026-05-17 from feat/zero-train-priors branch.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/zerotrain_priors_2026_05_17"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

COMMON=(
  --task can
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.01
  --n-demos 20 --n-rollouts 3 --seed 0
  --n-workers 10
  --output-dir "$OUTDIR"
)

run_prior () {
  local PRIOR="$1"; shift
  local TAG="$1"; shift
  echo "=== $(date '+%H:%M:%S')  $TAG  prior=$PRIOR  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --prior "$PRIOR" \
    --lambda-prior 0 5 \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  prior=$PRIOR  done ===" | tee -a "$LOG"
}

date | tee "$LOG"
echo "Zero-train priors survey: 4 priors x 2 lambdas = 8 cells, ~3.7 h total" | tee -a "$LOG"

run_prior demo_1nn    Z1
run_prior demo_knn    Z2 --knn-k 5
run_prior demo_random Z3
run_prior demo_mean   Z4

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
