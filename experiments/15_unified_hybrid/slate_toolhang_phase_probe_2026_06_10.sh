#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/toolhang_phase_probe_2026_06_10"
N_WORKERS="${N_WORKERS:-6}"
RUN="${RUN:-0}"
CELL="${1:-plan}"
if [ "$RUN" = "1" ]; then
  mkdir -p "$OUTDIR"
fi

run_cell () {
  local cell="$1"
  local task="$2"
  local horizon="$3"
  local replans="$4"
  local chunk="$5"
  local demos="$6"
  local rollouts="$7"
  local tag="$8"
  local cmd=(
    "$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py"
    --task "$task"
    --prior demo_1nn --lambda-prior 1
    --combine-mode score
    --H "$horizon" --mpc-replans "$replans" --mpc-chunk-steps "$chunk" --max-env-steps "$horizon"
    --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05
    --n-demos "$demos" --n-rollouts "$rollouts"
    --n-workers "$N_WORKERS" --seed 0
    --save-step-trace --save-executed-actions --log-gate-diagnostic
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
    echo "Manual-only ToolHang/Transport phase-probe slate."
    echo "Default RUN=0 prints commands only. Set RUN=1 to execute."
    echo "Cells: TH_key_short, TH_phase_long, TR_key_short, all"
    ;;
  TH_key_short)
    run_cell TH_key_short tool_hang 180 2 60 3 1 TH_key_short_trace
    ;;
  TH_phase_long)
    run_cell TH_phase_long tool_hang 300 3 75 5 1 TH_phase_long_trace
    ;;
  TR_key_short)
    run_cell TR_key_short transport 180 2 60 2 1 TR_key_short_trace
    ;;
  all)
    run_cell TH_key_short tool_hang 180 2 60 3 1 TH_key_short_trace
    run_cell TH_phase_long tool_hang 300 3 75 5 1 TH_phase_long_trace
    run_cell TR_key_short transport 180 2 60 2 1 TR_key_short_trace
    ;;
  *)
    echo "unknown cell '$CELL'; expected plan|TH_key_short|TH_phase_long|TR_key_short|all" >&2
    exit 2
    ;;
esac
