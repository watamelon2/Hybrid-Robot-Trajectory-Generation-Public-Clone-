#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUTDIR="$REPO/experiments/17_dp_hybrid/results/ep0199_checkpoint_correction_2026_06_09"
CKPT="$REPO/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt"
N_WORKERS="${N_WORKERS:-10}"
RUN="${RUN:-0}"
CELL="${1:-plan}"
if [ "$RUN" = "1" ]; then
  mkdir -p "$OUTDIR"
fi

run_cell () {
  local cell="$1"
  local lam="$2"
  local smax="$3"
  local k="$4"
  local tag="$5"
  local cmd=(
    "$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py"
    --task can
    --prior flow --score-mode pull
    --prior-checkpoint "$CKPT"
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 --max-env-steps 240
    --K "$k" --M 16 --lambda-prior "$lam"
    --sigma-max "$smax" --sigma-min 0.05 --prior-n-steps 100
    --deterministic-prior-warmstart
    --n-demos 0 --n-rollouts 0
    --n-test-rollouts 30 --test-start-seed 100000
    --n-workers "$N_WORKERS" --seed 0 --device cpu
    --output-dir "$OUTDIR" --output-name-suffix "_${tag}"
  )
  echo
  echo "[$cell] ${cmd[*]}"
  if [ "$RUN" = "1" ]; then
    "${cmd[@]}"
  fi
}

case "$CELL" in
  plan)
    echo "Manual-only DP checkpoint correction scaffold."
    echo "Default RUN=0 prints commands only. Set RUN=1 to execute."
    echo "Cells: D1_ep0199, D2_ep0199, all"
    ;;
  D1_ep0199)
    run_cell D1_ep0199 1.0 0.10 10 D1_ep0199_lam1_K10_smax10
    ;;
  D2_ep0199)
    run_cell D2_ep0199 10.0 0.25 10 D2_ep0199_lam10_K10_smax25
    ;;
  all)
    run_cell D1_ep0199 1.0 0.10 10 D1_ep0199_lam1_K10_smax10
    run_cell D2_ep0199 10.0 0.25 10 D2_ep0199_lam10_K10_smax25
    ;;
  *)
    echo "unknown cell '$CELL'; expected plan|D1_ep0199|D2_ep0199|all" >&2
    exit 2
    ;;
esac
