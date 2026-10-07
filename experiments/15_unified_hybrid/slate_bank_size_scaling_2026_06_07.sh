#!/usr/bin/env bash
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_bank_size_scaling_2026_06_07"
N_WORKERS="${N_WORKERS:-10}"
BANK_SIZES="${BANK_SIZES:-5 10 20 50 100 200}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO" || exit 1

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate bank-size scaling: retrieval-only demo_1nn + score-bridge" | tee -a "$LOG"
echo "BANK_SIZES = $BANK_SIZES" | tee -a "$LOG"
echo "N_WORKERS  = $N_WORKERS" | tee -a "$LOG"
echo "OUTDIR     = $OUTDIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_bank_size () {
  local BANK_SIZE="$1"
  local TAG="bank_size_${BANK_SIZE}"

  if compgen -G "$OUTDIR/*${TAG}*.json" > /dev/null; then
    echo "=== $(date '+%H:%M:%S')  N_b=$BANK_SIZE  SKIP existing JSON ===" | tee -a "$LOG"
    return 0
  fi

  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  N_b=$BANK_SIZE  starting ===" | tee -a "$LOG"

  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers "$N_WORKERS" \
    --seed 0 \
    --prior demo_1nn --lambda-prior 1 \
    --prior-bank-size "$BANK_SIZE" \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "=== $(date '+%H:%M:%S')  N_b=$BANK_SIZE  FAILED status=$status ===" | tee -a "$LOG"
    return "$status"
  fi

  echo "=== $(date '+%H:%M:%S')  N_b=$BANK_SIZE  done ===" | tee -a "$LOG"
}

FAIL=0
for BANK_SIZE in $BANK_SIZES; do
  run_bank_size "$BANK_SIZE" || FAIL=1
done

echo "" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then
  echo "=== $(date '+%H:%M:%S')  ALL BANK-SIZE CELLS DONE ===" | tee -a "$LOG"
else
  echo "=== $(date '+%H:%M:%S')  BANK-SIZE SLATE FINISHED WITH FAILURES ===" | tee -a "$LOG"
fi
exit "$FAIL"
