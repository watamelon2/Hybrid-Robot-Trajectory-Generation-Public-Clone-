#!/usr/bin/env bash
# Slate E -- lambda x p 2-D operating surface (canonical Can, calibrated sigma).
#
# WHY: Section A only sliced the temper-p axis at mix_demo (lambda)=1.
# The product<->max-gate family actually has TWO knobs: p (how the two
# experts fuse) and lambda=mix_demo (how strongly the demo factor enters).
# This maps the full surface at the calibrated sigma_demo so we can see
# whether the best cell is an interior (finite p, moderate lambda) point
# -- the strongest possible "a tuned product beats both endpoints" claim.
#
# sigma_demo auto-derived from Section D (override SIGMA_CAL=...).
# lambda=1 column already covered by slate_mbd_calibrated_AC; here we add
# lambda in {0.5, 2} x p in {0, 1, 5, inf} = 8 cells.  ~12 min/cell -> ~1.6 h.
#
#   screen -dmS mbd_grid bash experiments/15_unified_hybrid/slate_mbd_lambda_p_grid_2026_06_05.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_lambda_p_grid_2026_06_05"
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
echo "Slate E  lambda x p grid  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --combine-mode tempered --sigma-demo "$SIGMA_CAL"
  --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[grid] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for LAM in 0.5 2; do
  LT=$(echo "$LAM" | tr '.' 'p')
  for P in 0 1 5 inf; do
    PT=$(echo "$P" | tr '.' 'p')
    run_cell "E_lam${LT}_p${PT}" --lambda-prior "$LAM" --temper-p "$P"
  done
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate E  ALL DONE ===" | tee -a "$LOG"
