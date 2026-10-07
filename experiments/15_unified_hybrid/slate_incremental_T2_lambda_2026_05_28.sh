#!/usr/bin/env bash
# Slate Incremental T2 -- lambda sweep at T2.1 operating point.
#
# Sweeps the prior-strength lambda at the T2.1 configuration
# (hybrid + canonical 200 + 5 mass=1.3 demos @ Can mass=1.3).
# Tests whether the T2.1 dominance over T2.3 is specific to
# lambda=1 (the canonical headline operating point) or robust
# across a range of prior strengths.
#
# Lambda extremes:
#   lambda=0.5  weaker prior, MBD does more of the work
#   lambda=1.0  canonical headline -- on main as T2.1
#   lambda=2.0  stronger prior, bank guides more
#   lambda=5.0  very strong prior, MBD acts as light corrector
#
# Predicted outcome: T2.1-style adaptation gain is broad across
# lambda; lambda=5 may DROP because the bank's 200 canonical demos
# dominate the 5 perturbed ones, washing out the adaptation signal.
# lambda=0.5 may also drop because MBD needs a non-trivial prior
# pull to find the success-region modes.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (6)
# Branch:      feat/adapt-t2-lambda
# Worktree:    HRTG-lambda
# Wall-clock:  ~3 h (3 hybrid cells x ~50 min) + ~10 min collect.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_lambda_2026_05_28"
N_WORKERS="${N_WORKERS:-10}"        # override e.g. to 8 on a 10-core Mac
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate Incremental T2 lambda sweep at T2.1 operating point" | tee -a "$LOG"
echo "SHARED_DATA = $SHARED_DATA" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# -- Prereq check / build the T2.1 bank if missing -------------------
B205="$SHARED_DATA/can_mass1p3_205demos.hdf5"
if [ ! -f "$B205" ]; then
  echo "[lambda] missing $B205 -- collecting now (~10 min)" | tee -a "$LOG"
  LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
  mkdir -p "$LOCAL_DATA"
  python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
    --task can --mass-scale 1.3 --friction-scale 1.0 \
    --n-demos-target 20 \
    --output "$LOCAL_DATA/can_mass1p3_20demos.hdf5" \
    --combine-output "$LOCAL_DATA/can_mass1p3_220demos.hdf5" \
    --verbose 2>&1 | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
    --task can \
    --perturbed-20 "$LOCAL_DATA/can_mass1p3_20demos.hdf5" \
    --out-5 "$LOCAL_DATA/can_mass1p3_5demos.hdf5" \
    --out-205 "$LOCAL_DATA/can_mass1p3_205demos.hdf5" 2>&1 | tee -a "$LOG"
  cp "$LOCAL_DATA"/can_mass1p3_{20,205,220,5}demos.hdf5 "$SHARED_DATA/"
fi

# -- T2.1 at lambda in {0.5, 2.0, 5.0}.  lambda=1 already on main.
run () {
  local LAM="$1"
  local TAG_LAM="${LAM//./p}"
  local TAG="T2_1_lambda_${TAG_LAM}"
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  lambda=$LAM  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 \
    --n-workers "$N_WORKERS" \
    --seed 0 \
    --mass-scale 1.3 \
    --prior demo_1nn --lambda-prior "$LAM" \
    --prior-bank-dataset "$B205" \
    --prior-bank-task can \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  lambda=$LAM  done ===" | tee -a "$LOG"
}

run 0.5
run 2.0
run 5.0

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Combine with seed=0 T2.1 lambda=1 result on main for the lambda saturation curve." | tee -a "$LOG"
