#!/usr/bin/env bash
# Narrative-strengthening follow-up to the zero-train priors survey.
#
# Three slates, all through the unified driver at the canonical
# operating point (K=10 R=1 sigma_min=0.01 M=16 H=120, 20 demos x
# 3 rollouts = 60 rollouts per cell).  Each slate closes one
# specific gap between the current data and "Story B"
# (the score-bridge needs an in-distribution per-step pull target):
#
#   Slate 1: 4-seed pool of demo_1nn at lambda in {0, 5}
#            -- honest CI on the 63.3 % single-seed headline
#            -- 6 cells, ~2.5 h
#
#   Slate 2: k-sweep for demo_knn at lambda=5, k in {2, 3, 10, 20}
#            -- monotonicity test for the "averaging amplifies OOD"
#               mechanism; k=1 is demo_1nn, k=5 is already done
#            -- 4 cells, ~1.7 h
#
#   Slate 3: sigma-sweep for demo_1nn at lambda=5,
#            sigma_max in {0.05, 0.10, 0.15, 0.40}
#            -- robustness of the 1-NN win across noise levels
#            -- 4 cells, ~1.7 h
#
# Total: 14 cells, ~5.9 h.  Launched 2026-05-17 in screen "narrative".

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/narrative_followup_2026_05_17"
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
  --n-demos 20 --n-rollouts 3
  --n-workers 10
  --output-dir "$OUTDIR"
)

run () {
  local TAG="$1"; shift
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

date | tee "$LOG"
echo "Narrative follow-up: Slate 1 (seed pool) + 2 (k-sweep) + 3 (sigma-sweep)" | tee -a "$LOG"
echo "Total ~5.9 h wall" | tee -a "$LOG"

# ----- Slate 1: 4-seed pool of demo_1nn (seeds 1, 2, 3 to combine with seed 0) -----
echo "##### Slate 1: 4-seed pool of demo_1nn #####" | tee -a "$LOG"
for SEED in 1 2 3; do
  run "S1_1nn_seed${SEED}" \
    --prior demo_1nn --lambda-prior 0 5 --seed "$SEED"
done

# ----- Slate 2: k-sweep for demo_knn at lambda=5, seed=0 -----
echo "##### Slate 2: k-sweep for demo_knn at lambda=5 #####" | tee -a "$LOG"
for K in 2 3 10 20; do
  run "S2_knn_k${K}" \
    --prior demo_knn --knn-k "$K" --lambda-prior 5 --seed 0
done

# ----- Slate 3: sigma-sweep for demo_1nn at lambda=5, seed=0 -----
echo "##### Slate 3: sigma-sweep for demo_1nn at lambda=5 #####" | tee -a "$LOG"
for SIGMA in 0.05 0.10 0.15 0.40; do
  # Re-define COMMON for this cell (override sigma-max).  We inline
  # the call rather than mutating COMMON to keep the loop readable.
  TAG="S3_1nn_sigma${SIGMA}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max "$SIGMA" --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 5 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
