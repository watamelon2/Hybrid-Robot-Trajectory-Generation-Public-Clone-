#!/usr/bin/env bash
# Phase 1 §2.2 — reproduce DP standalone 36.7 % on Can.
# 10 demos × 3 rollouts = 30 trials, single seed.
# Expected wall: ~15 min. Expected succ: 36.7 % (11/30, ±5 pp).
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

OUT=experiments/17_dp_hybrid/results/baseline_standalone_dp
mkdir -p "$OUT"
LOG="$OUT/run.log"

# Send all subsequent output to both terminal and log via exec redirection.
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] starting standalone DP reproduction"
echo "  output: $OUT"
echo "  log:    $LOG"

python -u experiments/08_flow_matching_prior/standalone_dp.py \
    --task can \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --n-demos 10 --n-rollouts 3 \
    --n-action-steps 8 --max-env-steps 240 --n-steps 100 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --seed 0 --device cpu \
    --output-dir "$OUT" \
    --output-name standalone_dp_repro.json

echo "[$(date)] standalone DP repro finished"
echo "[$(date)] writing done marker"
touch "$OUT/done"
