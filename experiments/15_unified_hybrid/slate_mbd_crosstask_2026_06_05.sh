#!/usr/bin/env bash
# Slate G -- cross-task generality of the product<->max-gate finding.
#
# WHY: every tempered/gated result so far is Can-only.  Pan's MBD-Demo
# and our product make TASK-INDEPENDENT claims, so the headline
# comparison must transfer.  This runs the comparison on Lift and Square
# (both robomimic ph datasets are installed).  Each task gets its OWN
# mini sigma_demo calibration (max-gate at sigma in {1,4,16} with the
# passive diagnostic) so we can read post-hoc where ldem-lmod~0 for that
# task, plus product (p=0) at sigma in {1,4} and the legacy score-product
# reference.  6 cells/task x 2 tasks = 12.  ~12 min/cell -> ~2.4 h.
#
#   screen -dmS mbd_xtask bash experiments/15_unified_hybrid/slate_mbd_crosstask_2026_06_05.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_crosstask_2026_06_05"
N_WORKERS="${N_WORKERS:-10}"
TASKS="${TASKS:-lift square}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate G  cross-task  TASKS='$TASKS'" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[xtask] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for T in $TASKS; do
  # per-task sigma_demo mini-calibration (max-gate)
  for S in 1 4 16; do
    run_cell "G_${T}_Dg${S}" --task "$T" --combine-mode tempered --temper-p inf \
      --lambda-prior 1 --sigma-demo "$S"
  done
  # product at the balanced + gate-dominant points
  run_cell "G_${T}_prod_s1" --task "$T" --combine-mode tempered --temper-p 0 \
    --lambda-prior 1 --sigma-demo 1
  run_cell "G_${T}_prod_s4" --task "$T" --combine-mode tempered --temper-p 0 \
    --lambda-prior 1 --sigma-demo 4
  # legacy score-product reference (the operating mode behind the Can headline)
  run_cell "G_${T}_score_ref" --task "$T" --combine-mode score --lambda-prior 1
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate G  ALL DONE ===" | tee -a "$LOG"

python "$REPO/experiments/15_unified_hybrid/analyze_tempered_slate.py" \
  --results-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || true
