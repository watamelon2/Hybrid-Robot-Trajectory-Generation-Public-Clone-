#!/usr/bin/env bash
set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUT="$REPO/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
CKPT="$OUT/v_theta_dp_square_tobs2_h16_ep0199.pt"
LOG="$OUT/handoff_to_12_workers.log"
OLD_PID="${OLD_PID:-23208}"
LAM05="$OUT/unified_flow_pull_lam-0.5_K-10_R-49_square_dp_pull_test30.json"
LAM10="$OUT/unified_flow_pull_lam-1.0_K-10_R-49_square_dp_pull_test30.json"
LAM20="$OUT/unified_flow_pull_lam-2.0_K-10_R-49_square_dp_pull_test30.json"

echo "=== handoff-to-12 start $(date '+%F %T %Z') old_pid=$OLD_PID ===" | tee -a "$LOG"
echo "waiting for lambda=0.5 output: $LAM05" | tee -a "$LOG"

while [[ ! -s "$LAM05" ]]; do
  if ! kill -0 "$OLD_PID" 2>/dev/null; then
    echo "old run pid $OLD_PID is not active before lambda=0.5 completed" | tee -a "$LOG"
    break
  fi
  sleep 60
done

if [[ -s "$LAM05" ]]; then
  echo "lambda=0.5 output detected at $(date '+%F %T %Z')" | tee -a "$LOG"
else
  echo "WARNING: lambda=0.5 output not detected; proceeding only because old pid stopped" | tee -a "$LOG"
fi

if kill -0 "$OLD_PID" 2>/dev/null; then
  echo "terminating old 4-worker run pid=$OLD_PID" | tee -a "$LOG"
  for child in $(pgrep -P "$OLD_PID" 2>/dev/null || true); do
    kill -TERM "$child" 2>/dev/null || true
  done
  kill -TERM "$OLD_PID" 2>/dev/null || true
  sleep 10
  for child in $(pgrep -P "$OLD_PID" 2>/dev/null || true); do
    kill -KILL "$child" 2>/dev/null || true
  done
  kill -KILL "$OLD_PID" 2>/dev/null || true
else
  echo "old run pid=$OLD_PID already stopped" | tee -a "$LOG"
fi

if [[ -s "$LAM10" && -s "$LAM20" ]]; then
  echo "lambda=1 and lambda=2 outputs already exist; nothing to run" | tee -a "$LOG"
  exit 0
fi

echo "=== launching lambda=1,2 with 12 workers $(date '+%F %T %Z') ===" | tee -a "$LOG"
"$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task square \
  --prior flow --score-mode pull \
  --prior-checkpoint "$CKPT" \
  --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 \
  --K 10 --M 16 --lambda-prior 1 2 \
  --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 \
  --n-demos 0 --n-rollouts 0 \
  --n-test-rollouts 30 --test-start-seed 100000 \
  --n-workers 12 --seed 0 --device cpu \
  --output-dir "$OUT" --output-name-suffix "_square_dp_pull_test30" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== lambda=1,2 12-worker run done rc=$rc $(date '+%F %T %Z') ===" | tee -a "$LOG"
exit "$rc"
