#!/usr/bin/env bash
# Slate Lift Incremental T2 -- cross-task generalisation of T2 (Can)
#                              incremental-adaptation Pareto headline.
#
# Repeats the hybrid arm of the Can T2 ladder on Lift @ mass=1.3.
# BC cells are SKIPPED because run_can.py's --prior=bc is hard-wired
# to the Can-specific OBS_KEYS in experiments/03_bc_baseline/dataset.py
# (see run_can.py line 848 "BC / flow stay on the default (Can)").
# Hybrid cells use task-aware obs keys via priors_zerotrain so they
# work on Lift out of the box.
#
# Cells:
#   T2.0_lift  hybrid + canonical 200 Lift bank @ mass=1.3 (degraded baseline)
#   T2.1_lift  hybrid + canonical 200 + 5 mass=1.3 Lift demos @ mass=1.3 -- HEADLINE
#   T2.4_lift  hybrid + canonical 200 + 20 mass=1.3 Lift demos @ mass=1.3 (saturation)
#
# Expected outcomes (success):
#   T2.0_lift  Lift is robust; could already be high.  If T2.0 ~95%+,
#              the perturbation is too mild and we'd want mass=1.5
#              for a meaningful demonstration.
#   T2.1_lift  >= T2.0_lift if adaptation generalises.
#   T2.4_lift  ~ T2.1_lift if N=5 already saturates.
#
# Wall: ~10 min collect + 3 cells x ~15 min = ~55 min.
#
# Spec: docs/adaptation/04_incremental_prior_spec.md sec 5.1 mentions
#       Lift mass=1.3 as a generalisation probe.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_lift_incremental_T2_2026_05_28"
DATA_DIR="$REPO/experiments/15_unified_hybrid/data"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR" "$DATA_DIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate Lift Incremental T2 (cross-task perturbed adaptation)" | tee -a "$LOG"
echo "Spec: docs/adaptation/04_incremental_prior_spec.md sec 5.1" | tee -a "$LOG"
echo "All cells @ mass_scale=1.3 on Lift" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Stage 1: collect 20 mass=1.3 Lift demos + combined 220 bank ----
if [ ! -f "$DATA_DIR/lift_mass1p3_220demos.hdf5" ]; then
  echo "=== $(date '+%H:%M:%S')  collecting 20 mass=1.3 Lift demos  ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
      --task lift \
      --mass-scale 1.3 \
      --friction-scale 1.0 \
      --n-demos-target 20 \
      --output "$DATA_DIR/lift_mass1p3_20demos.hdf5" \
      --combine-output "$DATA_DIR/lift_mass1p3_220demos.hdf5" \
      --verbose 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
else
  echo "=== Stage 1 skipped: lift_mass1p3_220demos.hdf5 exists ===" | tee -a "$LOG"
fi

# ---- Stage 2: derive 5-demo + 205-demo banks from the 20-demo HDF5 ----
if [ ! -f "$DATA_DIR/lift_mass1p3_205demos.hdf5" ]; then
  echo "=== $(date '+%H:%M:%S')  deriving 5-demo / 205-demo Lift banks  ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
      --task lift \
      --perturbed-20 "$DATA_DIR/lift_mass1p3_20demos.hdf5" \
      --out-5 "$DATA_DIR/lift_mass1p3_5demos.hdf5" \
      --out-205 "$DATA_DIR/lift_mass1p3_205demos.hdf5" 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
else
  echo "=== Stage 2 skipped: lift_mass1p3_205demos.hdf5 exists ===" | tee -a "$LOG"
fi

# Sanity: banks must exist before the eval cells.
for f in "$DATA_DIR/lift_mass1p3_205demos.hdf5" "$DATA_DIR/lift_mass1p3_220demos.hdf5"; do
  if [ ! -f "$f" ]; then
    echo "[T2-lift] FATAL: prereq bank missing: $f" >&2
    exit 3
  fi
done

# Canonical operating point (matches Can T2 slate; H is fine for Lift).
COMMON=(
  --task lift
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

# ---- Stage 3: T2.0_lift -- hybrid + canonical 200 Lift bank @ mass=1.3 ----
run "T2_0_lift_hybrid_canonical_bank_mass1p3" \
  --prior demo_1nn --lambda-prior 1

# ---- Stage 4: T2.1_lift -- HEADLINE.  Hybrid + canonical 200 + 5 mass=1.3 Lift demos ----
run "T2_1_lift_hybrid_5demo_adapted_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$DATA_DIR/lift_mass1p3_205demos.hdf5" \
  --prior-bank-task lift

# ---- Stage 5: T2.4_lift -- saturation probe ----
run "T2_4_lift_hybrid_20demo_saturation_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$DATA_DIR/lift_mass1p3_220demos.hdf5" \
  --prior-bank-task lift

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Aggregate by hand or extend aggregate_incremental.py with a Lift section." | tee -a "$LOG"
