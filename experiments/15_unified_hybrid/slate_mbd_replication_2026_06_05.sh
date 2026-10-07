#!/usr/bin/env bash
# Slate T2 -- put a confidence interval on "finite-p beats both endpoints".
#
# WHY: the headline tempered result (A2_cal_p1 20% vs product 6.7% vs
# max-gate 3.3%) is at n=30, where the 95% CI is ~+/-15pp -- not enough
# to call p=1 a real winner.  This re-runs the four key cells at n=90
# (n-demos 30 x n-rollouts 3) so the ranking is statistically meaningful.
#
# Canonical Can, calibrated sigma_demo (auto).  4 cells x 3x rollouts.
#
#   screen -dmS mbd_repl bash experiments/15_unified_hybrid/slate_mbd_replication_2026_06_05.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_replication_2026_06_05"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
if [ -z "${SIGMA_CAL:-}" ]; then
  SIGMA_CAL="$(python "$REPO/experiments/15_unified_hybrid/pick_calibrated_sigma.py" \
    --results-dir "$GATED_DIR" --fallback 1 2>>"$LOG")"
fi
echo "Slate T2  replication n=90  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 30 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --combine-mode tempered --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
  --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[repl] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# A2_ prefix so the analyzer's Section A picks them up (own dir = own table)
for P in 0 1 2 inf; do
  PT=$(echo "$P" | tr '.' 'p')
  run_cell "A2_n90_p${PT}" --temper-p "$P"
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2  ALL DONE ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/analyze_tempered_slate.py" \
  --results-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || true
