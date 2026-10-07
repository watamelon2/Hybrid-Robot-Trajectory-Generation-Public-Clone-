#!/usr/bin/env bash
# Slate Incremental T2 -- perturbation-magnitude sweep.
#
# Maps the bite zone: where does T2.1 stop dominating T2.3?
# Sweeps mass_scale in {1.1, 1.2, 1.4, 1.5} (the canonical mass=1.3
# is already on main).  For each magnitude:
#   - T2.0: hybrid + canonical 200 bank @ that mass
#   - T2.1: hybrid + 200 + 5 mass-X demos @ that mass  (headline)
#   - T2.3: BC retrained on 205 mass-X demos @ that mass
#
# Predicted outcome: bite zone is bounded -- at mass close to 1.0
# the hybrid never falls into the bite zone (T2.0 ~ T2.1 ~ T2.3
# all near canonical) so no adaptation gain; at very high mass
# even T2.1 collapses.  Headline is conditional on the bite zone
# being non-trivial.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (3)
# Branch:      feat/adapt-t2-magnitude
# Worktree:    HRTG-magnitude
# Wall-clock:  ~6 h (4 magnitudes x 3 hybrid cells x ~12 min) +
#              4 BC retrains x ~80 s + 4 collections x ~10 min.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_magnitude_2026_05_28"
N_WORKERS="${N_WORKERS:-10}"        # override e.g. to 8 on a 10-core Mac
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate Incremental T2 perturbation-magnitude sweep" | tee -a "$LOG"
echo "SHARED_DATA = $SHARED_DATA" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Per-magnitude: collect + derive + 3 eval cells + 1 BC retrain.

run_magnitude () {
  local MASS="$1"                  # e.g. 1.1
  local TAG="${MASS//./p}"         # e.g. 1p1
  local P20="$SHARED_DATA/can_mass${TAG}_20demos.hdf5"
  local B205="$SHARED_DATA/can_mass${TAG}_205demos.hdf5"
  local B220="$SHARED_DATA/can_mass${TAG}_220demos.hdf5"

  echo "" | tee -a "$LOG"
  echo "============== MAGNITUDE mass=$MASS ==============" | tee -a "$LOG"

  # Stage 0: collect if missing.
  if [ ! -f "$B220" ] || [ ! -f "$B205" ]; then
    echo "[mag $MASS] collecting perturbed demos..." | tee -a "$LOG"
    LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
    mkdir -p "$LOCAL_DATA"
    python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
      --task can --mass-scale "$MASS" --friction-scale 1.0 \
      --n-demos-target 20 \
      --output "$LOCAL_DATA/can_mass${TAG}_20demos.hdf5" \
      --combine-output "$LOCAL_DATA/can_mass${TAG}_220demos.hdf5" \
      --verbose 2>&1 | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
      --task can \
      --perturbed-20 "$LOCAL_DATA/can_mass${TAG}_20demos.hdf5" \
      --out-5 "$LOCAL_DATA/can_mass${TAG}_5demos.hdf5" \
      --out-205 "$LOCAL_DATA/can_mass${TAG}_205demos.hdf5" 2>&1 | tee -a "$LOG"
    cp "$LOCAL_DATA"/can_mass${TAG}_{20,205,220,5}demos.hdf5 "$SHARED_DATA/"
  fi

  # Stage 1: BC retrain on 205 mass-X demos (CPU, ~80 s).
  local CKPT="$OUTDIR/policy_mass${TAG}_205.pt"
  local TRAIN_LOG="$OUTDIR/training_log_mass${TAG}_205.json"
  if [ ! -f "$CKPT" ]; then
    echo "[mag $MASS] BC retrain on 205 mass=$MASS demos..." | tee -a "$LOG"
    local TRAIN_T0=$(date +%s)
    python "$REPO/experiments/03_bc_baseline/train.py" \
      --dataset "$B205" --task can \
      --output "$CKPT" --log "$TRAIN_LOG" \
      --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
      --seed 0 --device cpu 2>&1 | tee -a "$LOG"
    echo "[mag $MASS] BC retrain wall: $(($(date +%s) - TRAIN_T0))s" | tee -a "$LOG"
  fi

  # Stage 2: the 3 eval cells.
  local COMMON=(--task can --mpc-replans 1 --mpc-chunk-steps 60
                --K 10 --M 16 --H 120
                --sigma-max 0.25 --sigma-min 0.05
                --n-demos 10 --n-rollouts 3
                --n-workers "$N_WORKERS" --seed 0
                --mass-scale "$MASS"
                --output-dir "$OUTDIR")

  for CELL_SPEC in \
    "T2_0_hybrid_canonical:--prior demo_1nn --lambda-prior 1" \
    "T2_1_hybrid_5demo_adapted:--prior demo_1nn --lambda-prior 1 --prior-bank-dataset $B205 --prior-bank-task can" \
    "T2_3_bc_retrained_205demos:--prior bc --lambda-prior 5 --bc-checkpoint $CKPT" \
  ; do
    local CELL_TAG="${CELL_SPEC%%:*}"
    local CELL_ARGS="${CELL_SPEC#*:}"
    local TAG_FULL="${CELL_TAG}_mass${TAG}"
    echo "" | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  starting ===" | tee -a "$LOG"
    # shellcheck disable=SC2086
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --output-name-suffix "_$TAG_FULL" \
      $CELL_ARGS 2>&1 | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  done ===" | tee -a "$LOG"
  done
}

run_magnitude 1.1
run_magnitude 1.2
run_magnitude 1.4
run_magnitude 1.5

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL MAGNITUDES DONE ===" | tee -a "$LOG"
echo "Combine with mass=1.3 cells already on main to make the bite-zone plot." | tee -a "$LOG"
