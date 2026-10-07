#!/usr/bin/env bash
# Slate S -- sparse-reward regime: does the product revive continuously
# while the max-gate stays threshold-gated as the cost densifies? (§7.1)
#
# WHY: on sparse-terminal tasks (tool_hang/transport) J is flat over every
# non-goal rollout -> p_J uniform -> ESS=M -> no cost gradient -> BOTH
# product and max-gate fall back to BC.  This is NOT a combination-rule
# problem; it's signal density.  PREDICTION (doc 09 §7.1): once dense
# shaping is added, the product's target shifts CONTINUOUSLY with
# lambda*grad(shaping), while the max-gate only responds once the model
# branch crosses the demo's fixed likelihood threshold.  So:
#   raw-sparse:  product ~ max-gate ~ score ~ BC   (all flat)
#   dense-shaped: product/score improve > max-gate  (gate lags)
# Compare MEAN REWARD (not just success) -- sparse success can stay ~0
# while reward/gate_frac already reveal the divergence.
#
# Uses the new --raw-sparse-reward flag (skips wrap_env_with_dense_reward).
# EXPLORATORY + slow (sparse tasks are long-horizon); chained LAST.
# Override TASK / budget via env.  6 cells.
#
#   screen -dmS mbd_sparse bash experiments/15_unified_hybrid/slate_mbd_sparse_reward_2026_06_06.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_sparse_reward_2026_06_06"
N_WORKERS="${N_WORKERS:-10}"
TASK="${TASK:-tool_hang}"
N_DEMOS="${N_DEMOS:-5}"
N_ROLLOUTS="${N_ROLLOUTS:-2}"
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
echo "Slate S  sparse-reward  TASK=$TASK  SIGMA_CAL=$SIGMA_CAL  n=${N_DEMOS}x${N_ROLLOUTS}" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task "$TASK" --mpc-replans 3 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[sparse] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# --- raw sparse reward (flat J): expect product ~ max-gate ~ score ~ BC ---
run_cell "S_raw_product"  --raw-sparse-reward --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "S_raw_maxgate"  --raw-sparse-reward --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "S_raw_score"    --raw-sparse-reward --combine-mode score    --lambda-prior 1

# --- dense-shaped (signal returns): expect product/score > max-gate ---
run_cell "S_dense_product" --combine-mode tempered --temper-p 0   --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "S_dense_maxgate" --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SIGMA_CAL"
run_cell "S_dense_score"   --combine-mode score    --lambda-prior 1

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate S  ALL DONE ===" | tee -a "$LOG"
