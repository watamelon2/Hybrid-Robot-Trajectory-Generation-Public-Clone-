#!/usr/bin/env bash
# Quick probe: which Lift perturbation degrades the canonical hybrid?
# 5 demos x 2 rollouts = 10 trials per cell, 3 perturbation candidates.
# Wall: ~5 min per cell x 3 = ~15 min.
set -euo pipefail
REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/probe_lift_perturb_2026_05_28"
mkdir -p "$OUTDIR"
cd "$REPO"
source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
LOG="$OUTDIR/run.log"
date | tee "$LOG"

COMMON=(
  --task lift
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 5 --n-rollouts 2
  --n-workers 5
  --seed 0
  --prior demo_1nn --lambda-prior 1
  --output-dir "$OUTDIR"
)

run () {
  local TAG="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

run "probe_mass2p0"   --mass-scale 2.0     --friction-scale 1.0
run "probe_fric0p5"   --mass-scale 1.0     --friction-scale 0.5
run "probe_fric0p3"   --mass-scale 1.0     --friction-scale 0.3

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
