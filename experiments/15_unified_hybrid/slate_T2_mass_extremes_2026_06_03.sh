#!/usr/bin/env bash
# Slate T2-MASS-EXTREMES -- push the mass axis to the extremes to
# locate the adaptation "bite zone".
#
# Follow-up #1a from the 2026-06-03 future-directions review.  The
# mass sweep (slate_incremental_T2_magnitude) covered mass in
# [0.7, 1.5] and the n200 extension reached 2.0; all of it was
# Outcome C -- T2.0 (no adaptation) ~= T2.1 (5-demo adapted), no bite
# zone.  The friction sweep {0.5..2.0} gave the same null.  This slate
# pushes mass to {0.5, 2.5} (more extreme than anything run so far) to
# test whether T2.0 finally COLLAPSES and T2.1 RESCUES it -- the only
# regime where the Slide-7 adaptation story could still be true.  If
# T2.0 still holds here, the contribution should be reframed as
# "robustness without adaptation".
#
# Cloned from slate_T2_friction_sweep_2026_06_01.sh (same operating
# point, same cell structure) with the perturbation axis swapped from
# friction to mass, so the two sweeps are directly comparable.
#
# Cells (6 = 2 mass values x (T2.0, T2.1, T2.3))
# ----------------------------------------------
#   mass in {0.5, 2.5}
#   per mass value:
#     T2.0  hybrid + canonical 200 Can bank  (no adaptation)
#     T2.1  hybrid + 200 + 5 mass-X Can demos  (5-demo adaptation)
#     T2.3  BC retrained on 205 mass-X Can demos  (78x-slower compare)
#
# Wall-clock: ~10 min collect + ~80 s BC + 3 x ~12 min eval per mass
#             value; two mass values ~= 1.5 h.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_mass_extremes_2026_06_03"
N_WORKERS="${N_WORKERS:-10}"
MASSES="${MASSES:-0.5 2.5}"   # extremes beyond the [0.7, 2.0] already run
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-MASS-EXTREMES: push the mass axis to locate the bite zone" | tee -a "$LOG"
echo "SHARED_DATA  = $SHARED_DATA" | tee -a "$LOG"
echo "MASSES       = $MASSES" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_mass () {
  local MASS="$1"                  # e.g. 0.5
  local TAG="${MASS//./p}"         # e.g. 0p5
  local P20="$SHARED_DATA/can_mass${TAG}_20demos.hdf5"
  local B205="$SHARED_DATA/can_mass${TAG}_205demos.hdf5"
  local B220="$SHARED_DATA/can_mass${TAG}_220demos.hdf5"
  local B5="$SHARED_DATA/can_mass${TAG}_5demos.hdf5"

  echo "" | tee -a "$LOG"
  echo "================ MASS mass=$MASS ================" | tee -a "$LOG"

  # 1. collect if missing.  Mass-only perturbations -- friction stays
  #    at the canonical 1.0.
  if [ ! -f "$B220" ] || [ ! -f "$B205" ] || [ ! -f "$B5" ]; then
    echo "[mass $MASS] collecting + deriving banks..." | tee -a "$LOG"
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

  # 2. BC retrain on 205 mass-X demos (cpu, ~80 s).
  local CKPT="$OUTDIR/policy_mass${TAG}_205.pt"
  local TRAIN_LOG="$OUTDIR/training_log_mass${TAG}_205.json"
  if [ ! -f "$CKPT" ]; then
    echo "[mass $MASS] BC retrain on 205 mass-$MASS demos..." | tee -a "$LOG"
    python "$REPO/experiments/03_bc_baseline/train.py" \
      --dataset "$B205" --task can \
      --output "$CKPT" --log "$TRAIN_LOG" \
      --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
      --seed 0 --device cpu 2>&1 | tee -a "$LOG"
  fi

  # 3. the 3 eval cells (T2.0 canonical, T2.1 adapted, T2.3 BC).
  local COMMON=(
    --task can --mpc-replans 1 --mpc-chunk-steps 60
    --K 10 --M 16 --H 120
    --sigma-max 0.25 --sigma-min 0.05
    --n-demos 10 --n-rollouts 3
    --n-workers "$N_WORKERS" --seed 0
    --mass-scale "$MASS"
    --friction-scale 1.0
    --output-dir "$OUTDIR"
  )
  # Parallel bash arrays (not a colon-joined string) so the
  # space-containing shared-data path under ".../ECE M237/..." passed
  # via --prior-bank-dataset / --bc-checkpoint survives word-splitting.
  local CELL_TAGS=(
    "T2_0_hybrid_canonical"
    "T2_1_hybrid_5demo_adapted"
    "T2_3_bc_retrained_205demos"
  )
  local -a CELL_0_ARGS=(--prior demo_1nn --lambda-prior 1)
  local -a CELL_1_ARGS=(--prior demo_1nn --lambda-prior 1 --prior-bank-dataset "$B205" --prior-bank-task can)
  local -a CELL_2_ARGS=(--prior bc --lambda-prior 5 --bc-checkpoint "$CKPT")

  local i
  for i in 0 1 2; do
    local CELL_TAG="${CELL_TAGS[$i]}"
    local TAG_FULL="${CELL_TAG}_mass${TAG}"
    if ls "$OUTDIR"/unified_*"_${TAG_FULL}".json >/dev/null 2>&1; then
      echo "[mass $MASS] $CELL_TAG already done; skipping." | tee -a "$LOG"
      continue
    fi
    local -a CELL_ARGS=()
    case "$i" in
      0) CELL_ARGS=("${CELL_0_ARGS[@]}") ;;
      1) CELL_ARGS=("${CELL_1_ARGS[@]}") ;;
      2) CELL_ARGS=("${CELL_2_ARGS[@]}") ;;
    esac
    echo "" | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  starting ===" | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --output-name-suffix "_$TAG_FULL" \
      "${CELL_ARGS[@]}" 2>&1 | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  done ===" | tee -a "$LOG"
  done
}

for MASS in $MASSES; do
  run_mass "$MASS"
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-MASS-EXTREMES  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUTDIR")
print()
print(f"{'mass':>6}  {'cell':<32}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 65)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    mass = d.get("mass_scale", "?")
    tag = js.stem.split("_R-1_")[-1] if "_R-1_" in js.stem else js.stem
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    print(f"{str(mass):>6}  {tag:<32}  {ns:>7}  {n:>4}  {sr:>6.1f}%")
print()
print("Bite zone = where T2.0 collapses but T2.1 holds.  Compare to the")
print("friction extremes {0.3, 3.0} for the 2-axis robustness envelope.")
PY
