#!/usr/bin/env bash
# Slate HARD -- get Square and tool_hang off 0% so the cross-task and
# sparse comparisons become meaningful.
#
# WHY: Section G Square = 0% even for the score-ref, and Section S
# tool_hang = 0% with gate_frac=0.  Both are config failures, not
# combination-rule failures: H=120 is too short for these long-horizon
# tasks, and sigma_demo=1 is far below tool_hang's balanced point.  This
# slate re-runs the strong score path (and product) at longer horizons,
# and gives tool_hang its own sigma_demo calibration.
#
# SLOW (long-horizon tasks).  Override budgets via env.  ~9 cells.
#   screen -dmS mbd_hard bash experiments/15_unified_hybrid/slate_mbd_hardtask_config_2026_06_06.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_hardtask_config_2026_06_06"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate HARD  Square + tool_hang config rescue" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
BASE=(
  --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05
  --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[hard] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${BASE[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# --- Square: longer horizon, score path (+ one product) ---
run_cell "H_square_score_h200" --task square --combine-mode score --lambda-prior 1 \
  --H 200 --mpc-replans 3 --mpc-chunk-steps 66 --n-demos 10 --n-rollouts 3
run_cell "H_square_score_h300" --task square --combine-mode score --lambda-prior 1 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 --n-demos 10 --n-rollouts 3
run_cell "H_square_prod_h300"  --task square --combine-mode tempered --temper-p 0 \
  --lambda-prior 1 --sigma-demo 16 --H 300 --mpc-replans 4 --mpc-chunk-steps 75 \
  --n-demos 10 --n-rollouts 3

# --- tool_hang: own sigma_demo calibration (max-gate) @ long horizon ---
for S in 16 64 256; do
  run_cell "H_th_Dg${S}" --task tool_hang --combine-mode tempered --temper-p inf \
    --lambda-prior 1 --sigma-demo "$S" \
    --H 300 --mpc-replans 4 --mpc-chunk-steps 75 --n-demos 5 --n-rollouts 2
done
# tool_hang score + product at the long horizon
run_cell "H_th_score_h300" --task tool_hang --combine-mode score --lambda-prior 1 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 --n-demos 5 --n-rollouts 2
run_cell "H_th_prod_s64_h300" --task tool_hang --combine-mode tempered --temper-p 0 \
  --lambda-prior 1 --sigma-demo 64 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 --n-demos 5 --n-rollouts 2

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate HARD  ALL DONE ===" | tee -a "$LOG"
