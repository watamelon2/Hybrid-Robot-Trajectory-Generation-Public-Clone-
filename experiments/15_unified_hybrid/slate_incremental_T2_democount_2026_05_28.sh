#!/usr/bin/env bash
# Slate Incremental T2 -- demo-count saturation sweep.
#
# Sweeps the number of mass=1.3 demos appended to the canonical 200
# Can bank: n_perturbed in {1, 2, 3, 5, 10, 20}.  Tests whether
# T2.1 (5 demos) actually sits on the saturation knee or whether 2-3
# demos already suffice.  5 and 20 are pre-existing (T2.1 and T2.4 on
# main).  This slate fills in {1, 2, 3, 10}.
#
# Predicted outcome: saturation by ~3-5 demos.  If observed, the
# "minimum effective demos" claim can be tightened from 5 to 3.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (2)
# Branch:      feat/adapt-t2-democount
# Worktree:    HRTG-democount
# Wall-clock:  ~3 h (4 hybrid cells x ~40 min each).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_democount_2026_05_28"
N_WORKERS="${N_WORKERS:-10}"        # override e.g. to 8 on a 10-core Mac
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate Incremental T2 demo-count sweep" | tee -a "$LOG"
echo "SHARED_DATA = $SHARED_DATA" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Prereq: need the 20-demo perturbed HDF5 to derive intermediate
#    sizes from.  Auto-build if missing.
P20="$SHARED_DATA/can_mass1p3_20demos.hdf5"
if [ ! -f "$P20" ]; then
  echo "[democount] missing $P20 -- collecting now (~10 min)" | tee -a "$LOG"
  LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
  mkdir -p "$LOCAL_DATA"
  python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
    --task can --mass-scale 1.3 --friction-scale 1.0 \
    --n-demos-target 20 \
    --output "$LOCAL_DATA/can_mass1p3_20demos.hdf5" \
    --combine-output "$LOCAL_DATA/can_mass1p3_220demos.hdf5" \
    --verbose 2>&1 | tee -a "$LOG"
  cp "$LOCAL_DATA"/can_mass1p3_{20,220}demos.hdf5 "$SHARED_DATA/"
fi

# -- Derive the demo-count ladder banks {201, 202, 203, 210}.
for N in 1 2 3 10; do
  TARGET="$SHARED_DATA/can_mass1p3_$((200 + N))demos.hdf5"
  if [ ! -f "$TARGET" ]; then
    echo "[democount] deriving 200 + $N bank -> $TARGET" | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
      --task can \
      --perturbed-20 "$P20" \
      --out-5 "$SHARED_DATA/_tmp_5.hdf5" \
      --out-205 "$TARGET" \
      --n-canonical 200 \
      --n-perturbed "$N" 2>&1 | tee -a "$LOG"
  fi
done
rm -f "$SHARED_DATA/_tmp_5.hdf5"
ls -lh "$SHARED_DATA"/can_mass1p3_*demos.hdf5 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Per-democount T2.1 cells.  Reuses T2.1 invocation (hybrid +
#    lambda=1 + extended bank), only varying the bank file.
run () {
  local N="$1"
  local BANK="$SHARED_DATA/can_mass1p3_$((200 + N))demos.hdf5"
  local TAG="T2_1_hybrid_${N}demo_mass1p3"
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 \
    --n-workers "$N_WORKERS" \
    --seed 0 \
    --mass-scale 1.3 \
    --prior demo_1nn --lambda-prior 1 \
    --prior-bank-dataset "$BANK" \
    --prior-bank-task can \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

run 1
run 2
run 3
run 10

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Combine with seed=0 T2.0 (0 demos), T2.1 (5 demos), T2.4 (20 demos)" | tee -a "$LOG"
echo "from results/slate_incremental_T2_2026_05_26/ to get the full {0,1,2,3,5,10,20} curve." | tee -a "$LOG"
