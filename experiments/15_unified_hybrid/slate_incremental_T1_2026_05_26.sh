#!/usr/bin/env bash
# Slate Incremental T1 -- cross-task transfer prior.
#
# Tests whether the retrieval prior captures *task-generic* motion
# structure by pairing each eval task with a bank from a different
# task family.  6 cells, single seed=0, 10 demos x 3 rollouts = 30
# evaluations per cell.  All cells use the proprio-only retrieval
# variant (--prior demo_1nn_proprio), which is the only obs vocab
# that's task-agnostic across Can / Lift / Square (their object
# obs are task-specific and would otherwise dominate the 1-NN
# distance).
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 2 (T1)
# Theory:      docs/theory/01_soft_control.md sec 1 (p^data factor)
# Branch:      feat/adapt-incremental-prior
# Worktree:    HRTG-incremental
#
# Predicted outcomes (substantiates T1 if all hold):
#   - T1.0 Can/Can     ~74%  (canonical hybrid headline)
#   - T1.5 Lift/Lift   ~100% (canonical Lift headline; from G1 slate)
#   - T1.4 Can/no-prior 0%   (no retrieval baseline; lambda=0 control)
#   - T1.1 Can/Lift    10-50% (substantiates "task-generic motion")
#   - T1.2 Can/Square  10-50%
#   - T1.3 Lift/Can    10-50%
#
# Falsifies if T1.1/T1.2/T1.3 <= 5% -- retracts the "task-generic
# motion structure" framing; restricts claim to "reduces variance
# within a task family".
#
# Wall-clock estimate: 6 cells x ~50 min each = ~5 h.  All canonical
# dynamics (no --mass-scale).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T1_2026_05_26"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Canonical operating point per docs/adaptation/04_incremental_prior_spec.md.
# H=120 + R=1 + chunk=60 = 60-step closed-loop, matching the dynamics /
# cost-time slate structure for cross-comparison consistency.
COMMON=(
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers 10
  --seed 0
  --output-dir "$OUTDIR"
)

LIFT_HDF5=$(python -c "
from pathlib import Path
import robomimic
p = Path(robomimic.__file__).resolve().parent.parent / 'datasets' / 'lift' / 'ph' / 'low_dim_v141.hdf5'
print(p)")
SQUARE_HDF5=$(python -c "
from pathlib import Path
import robomimic
p = Path(robomimic.__file__).resolve().parent.parent / 'datasets' / 'square' / 'ph' / 'low_dim_v141.hdf5'
print(p)")
CAN_HDF5=$(python -c "
from pathlib import Path
import robomimic
p = Path(robomimic.__file__).resolve().parent.parent / 'datasets' / 'can' / 'ph' / 'low_dim_v141.hdf5'
print(p)")

run () {
  local TAG="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

date | tee "$LOG"
echo "Slate Incremental T1 - cross-task transfer prior" | tee -a "$LOG"
echo "Spec: docs/adaptation/04_incremental_prior_spec.md sec 2" | tee -a "$LOG"
echo "Branch: feat/adapt-incremental-prior  worktree: HRTG-incremental" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# T1.0 -- Can eval, Can bank (control).  Reproduces the canonical
# headline cell at the proprio-only retrieval variant.
run "T1_0_can_can_control" \
  --task can \
  --prior demo_1nn_proprio --lambda-prior 1

# T1.1 -- Can eval, Lift bank (cross-task).  HEADLINE T1 cell.
run "T1_1_can_lift" \
  --task can \
  --prior demo_1nn_proprio --lambda-prior 1 \
  --prior-bank-dataset "$LIFT_HDF5" \
  --prior-bank-task lift

# T1.2 -- Can eval, Square bank (cross-task).
run "T1_2_can_square" \
  --task can \
  --prior demo_1nn_proprio --lambda-prior 1 \
  --prior-bank-dataset "$SQUARE_HDF5" \
  --prior-bank-task square

# T1.3 -- Lift eval, Can bank (cross-task reverse).
run "T1_3_lift_can" \
  --task lift \
  --prior demo_1nn_proprio --lambda-prior 1 \
  --prior-bank-dataset "$CAN_HDF5" \
  --prior-bank-task can

# T1.4 -- Can eval, lambda=0 (no prior contribution).  MBD-only
# baseline; should land near 0% per the dynamics-slate's lam=0 cells.
run "T1_4_can_no_prior" \
  --task can \
  --prior demo_1nn_proprio --lambda-prior 0

# T1.5 -- Lift eval, Lift bank (control).  Reproduces the canonical
# Lift headline.
run "T1_5_lift_lift_control" \
  --task lift \
  --prior demo_1nn_proprio --lambda-prior 1

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Aggregate with experiments/15_unified_hybrid/aggregate_incremental.py" | tee -a "$LOG"
