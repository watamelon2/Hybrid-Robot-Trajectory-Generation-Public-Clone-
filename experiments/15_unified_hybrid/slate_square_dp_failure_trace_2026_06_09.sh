#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
SRC_OUT="$REPO/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/square_dp_failure_trace_2026_06_09"
CKPT="$SRC_OUT/v_theta_dp_square_tobs2_h16_ep0199.pt"
N_WORKERS="${N_WORKERS:-10}"
RUN="${RUN:-0}"
if [ "$RUN" = "1" ]; then
  mkdir -p "$OUTDIR"
fi

cmd=(
  "$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py"
  --task square
  --prior flow --score-mode pull
  --prior-checkpoint "$CKPT"
  --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400
  --K 10 --M 16 --lambda-prior 1
  --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100
  --n-demos 0 --n-rollouts 0
  --n-test-rollouts 30 --test-start-seed 100000
  --n-workers "$N_WORKERS" --seed 0 --device cpu
  --save-step-trace
  --output-dir "$OUTDIR" --output-name-suffix "_square_dp_lam1_trace_test30"
)

echo "Manual-only Square-DP failure-trace scaffold."
echo "Default RUN=0 prints the command only. Set RUN=1 to execute."
echo "${cmd[*]}"

if [ "$RUN" = "1" ]; then
  "${cmd[@]}"
fi
