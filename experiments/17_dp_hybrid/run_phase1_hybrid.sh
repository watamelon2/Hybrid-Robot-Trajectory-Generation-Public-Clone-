#!/usr/bin/env bash
# Phase 1 §2.3 — DP + pull-mode MBD headline cell on Can.
# Same 30-trial config as standalone (10 demos × 3 rollouts, seed 0).
# Expected wall: ~3 h with n_workers=4.
# Falsifiable claim: succ >= 50% (lifts under-trained DP by >= 13 pp).
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

OUT=experiments/17_dp_hybrid/results/cell_dp_pullmbd
mkdir -p "$OUT"
LOG="$OUT/run.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] starting DP + pull-MBD headline cell"
echo "  output: $OUT"
echo "  log:    $LOG"

python -u experiments/15_unified_hybrid/run_can.py \
    --task can \
    --prior flow --score-mode pull \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 \
    --K 10 --M 16 --lambda-prior 1.0 \
    --sigma-max 0.25 --sigma-min 0.05 --prior-n-steps 100 \
    --deterministic-prior-warmstart \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 4 --seed 0 --device cpu \
    --output-dir "$OUT" \
    --output-name-suffix _dp_pullmbd_seed0

echo "[$(date)] DP + pull-MBD headline cell finished"
touch "$OUT/done"
