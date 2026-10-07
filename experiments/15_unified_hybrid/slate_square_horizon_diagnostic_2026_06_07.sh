#!/usr/bin/env bash
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_square_horizon_diagnostic_2026_06_07"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO" || exit 1

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate Square horizon diagnostic: score-bridge at H>=200" | tee -a "$LOG"
echo "N_WORKERS = $N_WORKERS" | tee -a "$LOG"
echo "OUTDIR    = $OUTDIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
BASE=(
  --task square
  --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05
  --n-workers "$N_WORKERS" --seed 0
  --prior demo_1nn --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if compgen -G "$OUTDIR/*${TAG}*.json" > /dev/null; then
    echo "=== $(date '+%H:%M:%S')  $TAG  SKIP existing JSON ===" | tee -a "$LOG"
    return 0
  fi

  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${BASE[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "=== $(date '+%H:%M:%S')  $TAG  FAILED status=$status ===" | tee -a "$LOG"
    return "$status"
  fi

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

FAIL=0

run_cell "square_warmstart_h300" \
  --lambda-prior 0 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 \
  --n-demos 10 --n-rollouts 3 || FAIL=1

run_cell "square_score_h200_lam1" \
  --lambda-prior 1 \
  --H 200 --mpc-replans 3 --mpc-chunk-steps 66 \
  --n-demos 10 --n-rollouts 3 || FAIL=1

run_cell "square_score_h300_lam1" \
  --lambda-prior 1 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 \
  --n-demos 10 --n-rollouts 3 || FAIL=1

run_cell "square_score_h300_lam2" \
  --lambda-prior 2 \
  --H 300 --mpc-replans 4 --mpc-chunk-steps 75 \
  --n-demos 10 --n-rollouts 3 || FAIL=1

echo "" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then
  echo "=== $(date '+%H:%M:%S')  ALL SQUARE DIAGNOSTIC CELLS DONE ===" | tee -a "$LOG"
else
  echo "=== $(date '+%H:%M:%S')  SQUARE DIAGNOSTIC FINISHED WITH FAILURES ===" | tee -a "$LOG"
fi
exit "$FAIL"
