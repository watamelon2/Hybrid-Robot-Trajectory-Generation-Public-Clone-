#!/usr/bin/env bash
# Slate T1 -- does the tempered family close the gap to the score-bridge
# as the MBD sample budget M grows?
#
# WHY: at M=16 the tempered/gated reweighting is 3-20% while the additive
# score-bridge is 66.7% (SLATE_FINDINGS_2026_06_05 §2).  Hypothesis: the
# generalised-mean reweighting of only 16 importance samples is
# high-variance, so it needs many more samples to rank candidates as well
# as the per-sample score gradient does.  If tempered p=1 climbs toward
# the score-bridge as M -> 256, that variance story is confirmed; if it
# plateaus low, the weakness is structural (the reweighting throws away
# the gradient information the score path keeps).
#
# Canonical Can, calibrated sigma_demo (auto).  Larger M is ~linearly
# slower per iter.  8 cells.
#
#   screen -dmS mbd_mscale bash experiments/15_unified_hybrid/slate_mbd_tempered_Mscale_2026_06_05.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_tempered_Mscale_2026_06_05"
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
echo "Slate T1  M-scaling  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[mscale] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# tempered p=1 (the best M=16 cell) across the sample budget
for M in 16 64 256; do
  run_cell "T_temp_p1_M${M}" --M "$M" --combine-mode tempered --temper-p 1 \
    --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
done
# endpoints at the largest budget (do they also benefit?)
run_cell "T_temp_p0_M256"   --M 256 --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "T_temp_pinf_M256" --M 256 --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
# score-bridge reference at matched budgets (the path that already wins)
for M in 16 64 256; do
  run_cell "T_score_M${M}" --M "$M" --combine-mode score --lambda-prior 1
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T1  ALL DONE ===" | tee -a "$LOG"
