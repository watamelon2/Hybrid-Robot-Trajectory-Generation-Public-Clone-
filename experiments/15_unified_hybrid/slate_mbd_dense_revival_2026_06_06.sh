#!/usr/bin/env bash
# Slate DR (follow-up #1) -- the §7.1 "continuous revival" test on a
# SOLVABLE task where the gate actually engages.
#
# WHY: the sparse slate (Section S) tried this on tool_hang, but there
# the goal is unreachable in H (0% everywhere) and the gate never fired
# (gate_frac=0), so the product-vs-gate contrast could not be observed.
# Can and Lift ARE solvable AND the gate engages at calibrated sigma
# (gate_frac ~0.35-0.4).  Re-running raw-sparse vs dense x {product,
# max-gate, score} there finally tests the doc 09 §7.1 prediction:
#   raw (flat J)  -> product ~ gate ~ score ~ BC          (no signal)
#   dense shaping -> product/score climb CONTINUOUSLY while the max-gate
#                    lags behind a likelihood threshold.
# Compare BOTH success and mean reward (the continuous signal).
#
# Two solvable tasks, own output dir each so the Section S analyzer works
# per-dir.  6 cells/task x 2 = 12.  No new code (uses --raw-sparse-reward).
#
#   screen -dmS mbd_dr bash experiments/15_unified_hybrid/slate_mbd_dense_revival_2026_06_06.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
BASEDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_dense_revival_2026_06_06"
N_WORKERS="${N_WORKERS:-10}"
TASKS="${TASKS:-can lift}"
mkdir -p "$BASEDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$BASEDIR/run.log"
date | tee "$LOG"
if [ -z "${SIGMA_CAL:-}" ]; then
  SIGMA_CAL="$(python "$REPO/experiments/15_unified_hybrid/pick_calibrated_sigma.py" \
    --results-dir "$GATED_DIR" --fallback 1 2>>"$LOG")"
fi
echo "Slate DR  dense-revival  TASKS='$TASKS'  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"

run_cell () {
  local OUTDIR="$1"; local TAG="$2"; shift 2
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[dr] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0 \
    --warm-start --prior demo_1nn --log-gate-diagnostic \
    --output-dir "$OUTDIR" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for T in $TASKS; do
  OUTDIR="$BASEDIR/$T"
  mkdir -p "$OUTDIR"
  # raw sparse cost (flat J): expect product ~ gate ~ score ~ BC
  run_cell "$OUTDIR" "S_raw_product" --task "$T" --raw-sparse-reward --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
  run_cell "$OUTDIR" "S_raw_maxgate" --task "$T" --raw-sparse-reward --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
  run_cell "$OUTDIR" "S_raw_score"   --task "$T" --raw-sparse-reward --combine-mode score    --lambda-prior 1
  # dense-shaped cost (signal present): expect product/score climb > gate
  run_cell "$OUTDIR" "S_dense_product" --task "$T" --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
  run_cell "$OUTDIR" "S_dense_maxgate" --task "$T" --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
  run_cell "$OUTDIR" "S_dense_score"   --task "$T" --combine-mode score    --lambda-prior 1
  echo "--- $T summary ---" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/analyze_tempered_slate.py" \
    --results-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || true
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate DR  ALL DONE ===" | tee -a "$LOG"
