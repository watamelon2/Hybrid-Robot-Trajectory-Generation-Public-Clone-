#!/usr/bin/env bash
# Slate Incremental T2 (eval cells) -- incremental adaptation Pareto plot.
#
# Tests the headline claim: hybrid + 5 fresh mass=1.3 demos beats
# BC retrained on 205 demos on the (wall-clock x adapted-succ) Pareto
# plot.  4 eval cells here, plus 1 BC-retrain+eval cell in the sister
# launcher slate_incremental_T2_bc_retrain_2026_05_26.sh.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3 (T2)
# Theory:      docs/theory/03_pareto_bias_variance.md (Corollary 1)
# Branch:      feat/adapt-incremental-prior
# Worktree:    HRTG-incremental
# Prereq:      slate_incremental_collect_2026_05_26.sh must have run
#              (writes can_mass1p3_5demos.hdf5 + can_mass1p3_205demos.hdf5
#              + can_mass1p3_220demos.hdf5).
#
# Cells:
#   T2.0  hybrid + canonical 200 Can bank @ mass=1.3 (degraded baseline)
#   T2.1  hybrid + canonical 200 + 5 mass=1.3 bank @ mass=1.3  -- HEADLINE
#   T2.2  BC unadapted @ mass=1.3
#   T2.4  hybrid + canonical 200 + 20 mass=1.3 bank @ mass=1.3 (saturation)
#
# (T2.3 = BC retrained on 205 -- separate slate due to ~30 min retrain.)
#
# Predicted outcomes (substantiates T2 if T2.1 sits upper-left of T2.3
# on the wall-clock x succ Pareto plot):
#   - T2.0 ~15-25%  (hybrid degrades under mass shift; from dynamics slate)
#   - T2.1 ~50-65%  (5 perfect mode anchors restore hybrid succ)
#   - T2.2 ~10%     (BC unadapted; from dynamics slate)
#   - T2.4 ~55-70%  (saturation; T2.1 + 15 more demos => modest +pp)
#
# Wall-clock estimate: 4 cells x ~50 min = ~3.5 h.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_2026_05_26"
DATA_DIR="$REPO/experiments/15_unified_hybrid/data"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Sanity: the prereq banks must exist.
for f in "$DATA_DIR/can_mass1p3_205demos.hdf5" "$DATA_DIR/can_mass1p3_220demos.hdf5"; do
  if [ ! -f "$f" ]; then
    echo "[T2-eval] FATAL: prereq bank missing: $f" >&2
    echo "[T2-eval] Run slate_incremental_collect_2026_05_26.sh first." >&2
    exit 3
  fi
done

# Canonical operating point per docs/adaptation/04_incremental_prior_spec.md.
COMMON=(
  --task can
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers 10
  --seed 0
  --mass-scale 1.3
  --output-dir "$OUTDIR"
)

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
echo "Slate Incremental T2 (eval cells) - incremental adaptation Pareto plot" | tee -a "$LOG"
echo "Spec: docs/adaptation/04_incremental_prior_spec.md sec 3" | tee -a "$LOG"
echo "All cells @ mass_scale=1.3" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# T2.0 -- hybrid + canonical 200 Can bank @ mass=1.3 (degraded baseline).
run "T2_0_hybrid_canonical_bank_mass1p3" \
  --prior demo_1nn --lambda-prior 1

# T2.1 -- HEADLINE.  Hybrid + canonical 200 + 5 mass=1.3 demos.
run "T2_1_hybrid_5demo_adapted_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$DATA_DIR/can_mass1p3_205demos.hdf5" \
  --prior-bank-task can

# T2.2 -- BC unadapted @ mass=1.3 (baseline).
run "T2_2_bc_unadapted_mass1p3" \
  --prior bc --lambda-prior 5

# T2.4 -- saturation probe: hybrid + 20 mass=1.3 demos.
run "T2_4_hybrid_20demo_saturation_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$DATA_DIR/can_mass1p3_220demos.hdf5" \
  --prior-bank-task can

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Run slate_incremental_T2_bc_retrain_2026_05_26.sh for T2.3" | tee -a "$LOG"
echo "Aggregate with experiments/15_unified_hybrid/aggregate_incremental.py" | tee -a "$LOG"
