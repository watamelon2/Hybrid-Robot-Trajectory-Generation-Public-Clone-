#!/usr/bin/env bash
# Slate DP3 (follow-up #3) -- demo-phase pseudo-cost on the unsolved
# sparse task (doc 09 §7.1 route 3).
#
# WHY: tool_hang stays 0% at every horizon/sigma with the hand-crafted
# dense shaping (HARD slate) AND the gate never engages.  Route 3 swaps
# that shaping for a ZERO-TRAIN demo-phase progress reward (reward =
# scale * fraction along the nearest demo).  Question: does a
# progress-shaped cost get tool_hang off 0% where hand-shaping did not,
# and -- on a SOLVABLE task (Can, where the gate engages) -- does the
# product revive faster than the max-gate as this signal turns on (the
# §7.1 prediction the sparse slate could not test)?
#
# Long-horizon (tool_hang) cells are SLOW.  Override via env.  9 cells.
#   screen -dmS mbd_dp3 bash experiments/15_unified_hybrid/slate_mbd_demo_phase_2026_06_06.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_demo_phase_2026_06_06"
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
echo "Slate DP3  demo-phase reward  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[dp3] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" \
    --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 \
    --n-workers "$N_WORKERS" --seed 0 \
    --warm-start --prior demo_1nn --demo-phase-reward --log-gate-diagnostic \
    --output-dir "$OUTDIR" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# --- Can (solvable, gate engages): the §7.1 continuous-revival test ---
# demo-phase reward ON for all; compare product vs max-gate vs score.
CAN_BUDGET=(--task can --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --n-demos 10 --n-rollouts 3)
run_cell "DP_can_product" "${CAN_BUDGET[@]}" --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "DP_can_maxgate" "${CAN_BUDGET[@]}" --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "DP_can_score"   "${CAN_BUDGET[@]}" --combine-mode score    --lambda-prior 1

# --- tool_hang (unsolved): does demo-phase progress beat hand-shaping? ---
TH_BUDGET=(--task tool_hang --H 300 --mpc-replans 4 --mpc-chunk-steps 75 --n-demos 5 --n-rollouts 2)
run_cell "DP_th_score"   "${TH_BUDGET[@]}" --combine-mode score    --lambda-prior 1
run_cell "DP_th_product" "${TH_BUDGET[@]}" --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "DP_th_maxgate" "${TH_BUDGET[@]}" --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
# higher demo-phase scale (stronger progress signal) + bigger sigma_demo
run_cell "DP_th_score_s30"   "${TH_BUDGET[@]}" --combine-mode score --lambda-prior 1 --demo-phase-scale 30
run_cell "DP_th_product_sd16" "${TH_BUDGET[@]}" --combine-mode tempered --temper-p 0 --lambda-prior 1 --sigma-demo 16
run_cell "DP_th_maxgate_sd16" "${TH_BUDGET[@]}" --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo 16

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate DP3  ALL DONE ===" | tee -a "$LOG"
