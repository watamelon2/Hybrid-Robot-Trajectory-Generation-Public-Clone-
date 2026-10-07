#!/usr/bin/env bash
set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUT="$REPO/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
CKPT="$OUT/v_theta_dp_square_tobs2_h16_ep0199.pt"
LOG="$OUT/restart_remaining_12_workers.log"

echo "=== square-dp remaining hybrid restart start $(date '+%F %T %Z') ===" | tee -a "$LOG"
echo "checkpoint=$CKPT" | tee -a "$LOG"
echo "lambdas=0.5,1,2 n_workers=12" | tee -a "$LOG"

"$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task square \
  --prior flow --score-mode pull \
  --prior-checkpoint "$CKPT" \
  --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 \
  --K 10 --M 16 --lambda-prior 0.5 1 2 \
  --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 \
  --n-demos 0 --n-rollouts 0 \
  --n-test-rollouts 30 --test-start-seed 100000 \
  --n-workers 12 --seed 0 --device cpu \
  --output-dir "$OUT" --output-name-suffix "_square_dp_pull_test30" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== square-dp remaining hybrid restart done rc=$rc $(date '+%F %T %Z') ===" | tee -a "$LOG"
exit "$rc"
