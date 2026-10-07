#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUT="$REPO/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
CKPT="$OUT/v_theta_dp_square_tobs2_h16_ep0199.pt"
LOG="$OUT/run.log"
MODE="${MODE:-smoke}"
DEVICE="${DEVICE:-cpu}"
N_WORKERS="${N_WORKERS:-4}"
mkdir -p "$OUT"

run_train_smoke () {
  "$PYTHON" "$REPO/experiments/08_flow_matching_prior/train_dp.py" \
    --smoke \
    --task square_dp \
    --horizon 16 \
    --n-obs-steps 2 \
    --normalize-actions \
    --normalize-obs \
    --device "$DEVICE" \
    --output "$OUT/v_theta_dp_square_smoke.pt" \
    --log "$OUT/train_log_square_smoke.json"
}

run_train_full () {
  "$PYTHON" "$REPO/experiments/08_flow_matching_prior/train_dp.py" \
    --task square_dp \
    --n-demos 200 \
    --horizon 16 \
    --n-obs-steps 2 \
    --epochs 200 \
    --save-every 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --adam-beta1 0.95 \
    --normalize-actions \
    --normalize-obs \
    --device "$DEVICE" \
    --output "$CKPT" \
    --log "$OUT/train_log_square_dp.json"
}

run_standalone_eval () {
  local ckpt="${1:-$CKPT}"
  "$PYTHON" "$REPO/experiments/08_flow_matching_prior/standalone_dp.py" \
    --task square \
    --prior-checkpoint "$ckpt" \
    --n-demos 0 --n-rollouts 0 \
    --n-test-rollouts 30 --test-start-seed 100000 \
    --n-action-steps 8 --max-env-steps 400 \
    --n-steps 100 \
    --device cpu \
    --output-dir "$OUT" \
    --output-name "standalone_square_dp_test30.json"
}

run_hybrid_eval () {
  local ckpt="${1:-$CKPT}"
  "$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --prior flow --score-mode pull \
    --prior-checkpoint "$ckpt" \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 \
    --K 10 --M 16 --lambda-prior 0 0.5 1 2 \
    --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 \
    --n-demos 0 --n-rollouts 0 \
    --n-test-rollouts 30 --test-start-seed 100000 \
    --n-workers "$N_WORKERS" --seed 0 --device cpu \
    --output-dir "$OUT" --output-name-suffix "_square_dp_pull_test30"
}

echo "=== square DP prior slate start $(date '+%F %T %Z') mode=$MODE ===" | tee -a "$LOG"
case "$MODE" in
  smoke)
    run_train_smoke 2>&1 | tee -a "$LOG"
    ;;
  train)
    run_train_full 2>&1 | tee -a "$LOG"
    ;;
  eval)
    run_standalone_eval "$CKPT" 2>&1 | tee -a "$LOG"
    ;;
  hybrid)
    run_hybrid_eval "$CKPT" 2>&1 | tee -a "$LOG"
    ;;
  all)
    run_train_full 2>&1 | tee -a "$LOG"
    run_standalone_eval "$CKPT" 2>&1 | tee -a "$LOG"
    run_hybrid_eval "$CKPT" 2>&1 | tee -a "$LOG"
    ;;
  *)
    echo "unknown MODE=$MODE; expected smoke|train|eval|hybrid|all" | tee -a "$LOG"
    exit 2
    ;;
esac
rc=$?
echo "=== square DP prior slate done $(date '+%F %T %Z') rc=$rc ===" | tee -a "$LOG"
exit "$rc"
