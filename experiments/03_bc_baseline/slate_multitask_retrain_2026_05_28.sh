#!/usr/bin/env bash
# Slate BC multi-task retrain.
#
# Trains 4 BC policies on the now-multitask-aware experiments/03_bc_baseline
# pipeline:
#
#   - Lift canonical    (200 demos)
#   - Lift + 5 mass=1.3 (205 demos)
#   - Square canonical  (200 demos)
#   - Square + 5 mass=1.3 (205 demos)
#
# These checkpoints feed the BC arm of feat/adapt-t2-square (and any
# future cross-task BC ablation).  Each train runs 500 epochs at batch
# 512 on CPU -- matches the canonical Can BC budget (commit e3474c3).
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (5)
# Branch:      feat/adapt-bc-multitask-obs
# Worktree:    HRTG-bcmulti
# Wall-clock:  ~4 trainings x ~80 s = ~6 min CPU.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/03_bc_baseline/results"
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/multitask_retrain.log"
date | tee "$LOG"
echo "Slate BC multi-task retrain (Lift, Square; canonical + mass=1.3)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

retrain () {
  local TASK="$1"          # lift | square
  local DATASET="$2"       # absolute path
  local TAG="$3"           # canonical | mass1p3_205
  local OUT="$OUTDIR/policy_${TASK}_${TAG}.pt"
  local TRAIN_LOG="$OUTDIR/training_log_${TASK}_${TAG}.json"
  if [ -f "$OUT" ]; then
    echo "[$TASK $TAG] skip (exists): $OUT" | tee -a "$LOG"
    return
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TASK / $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/03_bc_baseline/train.py" \
    --dataset "$DATASET" --task "$TASK" \
    --output "$OUT" --log "$TRAIN_LOG" \
    --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
    --seed 0 --device cpu 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TASK / $TAG  done ===" | tee -a "$LOG"
}

# --- Lift -----------------------------------------------------------
LIFT_CANON="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/lift/ph/low_dim_v141.hdf5"
retrain "lift" "$LIFT_CANON" "canonical_200"

LIFT_205="$SHARED_DATA/lift_mass1p3_205demos.hdf5"
if [ -f "$LIFT_205" ]; then
  retrain "lift" "$LIFT_205" "mass1p3_205"
else
  echo "[lift mass1p3_205] skip: $LIFT_205 not in SHARED_DATA" | tee -a "$LOG"
  echo "  Build it first via collect+derive on the Lift cross-task slate." | tee -a "$LOG"
fi

# --- Square ---------------------------------------------------------
SQ_CANON="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/square/ph/low_dim_v141.hdf5"
retrain "square" "$SQ_CANON" "canonical_200"

SQ_205="$SHARED_DATA/square_mass1p3_205demos.hdf5"
if [ -f "$SQ_205" ]; then
  retrain "square" "$SQ_205" "mass1p3_205"
else
  echo "[square mass1p3_205] skip: $SQ_205 not in SHARED_DATA" | tee -a "$LOG"
  echo "  feat/adapt-t2-square's slate auto-builds this on first run." | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL RETRAINS DONE ===" | tee -a "$LOG"
ls -lh "$OUTDIR"/policy_{lift,square}_*.pt 2>/dev/null | tee -a "$LOG"
