#!/usr/bin/env bash
# Slate Incremental T2 -- Square cross-task ladder (hybrid arm).
#
# Tests whether the T2 framing transfers from Can to Square.
# Per Slate G, Square is mechanism-limited (tight nut-on-peg
# clearance) and demo-sensitive -- a far better cross-task probe
# than Lift (which saturated under every available perturbation).
#
# Hybrid cells only (T2.0, T2.1, T2.4).  BC cells (T2.2, T2.3) are
# blocked on feat/adapt-bc-multitask-obs landing first -- the BC
# baseline currently hard-codes Can's obs schema in dataset.py.
# A companion slate slate_incremental_T2_square_bc_*.sh will be
# added on this branch once step 5 merges.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (4a)
# Branch:      feat/adapt-t2-square
# Worktree:    HRTG-square
# Wall-clock:  ~3 h (3 hybrid cells x ~50 min) + ~10 min collect.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_square_hybrid_2026_05_28"
N_WORKERS="${N_WORKERS:-10}"        # override e.g. to 8 on a 10-core Mac
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate Incremental T2 Square cross-task (hybrid arm)" | tee -a "$LOG"
echo "SHARED_DATA = $SHARED_DATA" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Stage 0: collect perturbed Square demos at mass=1.3 if missing.
P20="$SHARED_DATA/square_mass1p3_20demos.hdf5"
B205="$SHARED_DATA/square_mass1p3_205demos.hdf5"
B220="$SHARED_DATA/square_mass1p3_220demos.hdf5"
B5="$SHARED_DATA/square_mass1p3_5demos.hdf5"
if [ ! -f "$B220" ] || [ ! -f "$B205" ]; then
  echo "[square] collecting Square mass=1.3 perturbed demos..." | tee -a "$LOG"
  LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
  mkdir -p "$LOCAL_DATA"
  python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
    --task square --mass-scale 1.3 --friction-scale 1.0 \
    --n-demos-target 20 \
    --output "$LOCAL_DATA/square_mass1p3_20demos.hdf5" \
    --combine-output "$LOCAL_DATA/square_mass1p3_220demos.hdf5" \
    --verbose 2>&1 | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
    --task square \
    --perturbed-20 "$LOCAL_DATA/square_mass1p3_20demos.hdf5" \
    --out-5 "$LOCAL_DATA/square_mass1p3_5demos.hdf5" \
    --out-205 "$LOCAL_DATA/square_mass1p3_205demos.hdf5" 2>&1 | tee -a "$LOG"
  cp "$LOCAL_DATA"/square_mass1p3_{20,205,220,5}demos.hdf5 "$SHARED_DATA/"
fi
ls -lh "$SHARED_DATA"/square_mass1p3*.hdf5 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Hybrid cells (Square at mass=1.3) -------------------------------
# Note: --task square is honoured throughout the hybrid pipeline
# (priors_zerotrain.get_obs_keys is task-aware; mass perturbation
# registry _ROBOT_LINK_BODY_PREFIXES has the Square entry).
COMMON=(--task square --mpc-replans 1 --mpc-chunk-steps 60
        --K 10 --M 16 --H 120
        --sigma-max 0.25 --sigma-min 0.05
        --n-demos 10 --n-rollouts 3
        --n-workers "$N_WORKERS" --seed 0
        --mass-scale 1.3
        --output-dir "$OUTDIR")

run () {
  local TAG="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# T2.0_square: hybrid + canonical 200 Square bank @ mass=1.3.
run "T2_0_square_hybrid_canonical_bank_mass1p3" \
  --prior demo_1nn --lambda-prior 1

# T2.1_square: hybrid + 200 + 5 mass=1.3 demos @ mass=1.3.
run "T2_1_square_hybrid_5demo_adapted_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$B205" \
  --prior-bank-task square

# T2.4_square: saturation probe (200 + 20).
run "T2_4_square_hybrid_20demo_saturation_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$B220" \
  --prior-bank-task square

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  HYBRID ARM DONE ===" | tee -a "$LOG"
echo "BC arm (T2.2, T2.3) blocked on feat/adapt-bc-multitask-obs." | tee -a "$LOG"
