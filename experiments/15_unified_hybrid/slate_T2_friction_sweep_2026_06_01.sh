#!/usr/bin/env bash
# Slate T2-FRICTION-SWEEP -- adaptation along the orthogonal dynamics
# axis (friction, not mass).
#
# Follow-up B6 from the 2026-06-01 backlog research.  The Slide 7
# headline adaptation result currently lives on the *mass* axis only
# (mass-scale 1.1 -> 1.5 from slate_incremental_T2_magnitude_2026_05_28
# + the wider 1.6 -> 2.0 extension on feat/adapt-t2-n200-magnitude).
# The brief's §8.1 #5 limitation explicitly calls out "no friction
# perturbation studies".  This slate fills that gap on the orthogonal
# dynamics axis.
#
# Why friction is orthogonal to mass
# ----------------------------------
# Mass perturbations change inertia (slower / faster response to the
# same torque).  Friction perturbations change the contact dynamics
# at the gripper / object interface -- they don't affect free-space
# motion, only the grasp + manipulation phases.  A method that adapts
# to mass via the analytical cost gradient may or may not adapt to
# friction; running both is what makes the §7 "adaptation" story
# robust.
#
# Cells (16 total = 4 friction values x (T2.0, T2.1, T2.3) + smokes)
# -----------------------------------------------------------------
#   friction in {0.5, 0.7, 1.5, 2.0}  (symmetric around canonical 1.0)
#   per friction value:
#     T2.0  hybrid + canonical 200 Can bank  (baseline at perturbed
#           friction, no adaptation)
#     T2.1  hybrid + 200 + 5 friction-X Can demos  (5-demo adaptation
#           via retrieval bank)
#     T2.3  BC retrained on 205 friction-X Can demos  (the 78x-slower
#           comparison)
#
# Wall-clock
# ----------
#   per friction:  ~10 min collect + ~80 s BC retrain + 3 x ~12 min
#                  hybrid eval = ~50 min total.
#   four friction values:  ~3.5 h.
#   Optional friction in {0.3, 3.0} extreme cells: +1 h.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_friction_sweep_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
FRICTIONS="${FRICTIONS:-0.5 0.7 1.5 2.0}"   # override e.g. "0.3 3.0" for extremes
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-FRICTION-SWEEP: orthogonal dynamics axis to the mass sweep" | tee -a "$LOG"
echo "SHARED_DATA  = $SHARED_DATA" | tee -a "$LOG"
echo "FRICTIONS    = $FRICTIONS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_friction () {
  local FRIC="$1"                  # e.g. 0.5
  local TAG="${FRIC//./p}"         # e.g. 0p5
  local P20="$SHARED_DATA/can_fric${TAG}_20demos.hdf5"
  local B205="$SHARED_DATA/can_fric${TAG}_205demos.hdf5"
  local B220="$SHARED_DATA/can_fric${TAG}_220demos.hdf5"
  local B5="$SHARED_DATA/can_fric${TAG}_5demos.hdf5"

  echo "" | tee -a "$LOG"
  echo "================ FRICTION fric=$FRIC ================" | tee -a "$LOG"

  # 1. collect if missing.  Friction-only perturbations -- mass stays
  #    at the canonical 1.0.  collect_perturbed_demos.py supports
  #    --friction-scale natively.
  if [ ! -f "$B220" ] || [ ! -f "$B205" ] || [ ! -f "$B5" ]; then
    echo "[fric $FRIC] collecting + deriving banks..." | tee -a "$LOG"
    LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
    mkdir -p "$LOCAL_DATA"
    python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
      --task can --mass-scale 1.0 --friction-scale "$FRIC" \
      --n-demos-target 20 \
      --output "$LOCAL_DATA/can_fric${TAG}_20demos.hdf5" \
      --combine-output "$LOCAL_DATA/can_fric${TAG}_220demos.hdf5" \
      --verbose 2>&1 | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
      --task can \
      --perturbed-20 "$LOCAL_DATA/can_fric${TAG}_20demos.hdf5" \
      --out-5 "$LOCAL_DATA/can_fric${TAG}_5demos.hdf5" \
      --out-205 "$LOCAL_DATA/can_fric${TAG}_205demos.hdf5" 2>&1 | tee -a "$LOG"
    cp "$LOCAL_DATA"/can_fric${TAG}_{20,205,220,5}demos.hdf5 "$SHARED_DATA/"
  fi

  # 2. BC retrain on 205 friction-X demos (cpu, ~80 s).
  local CKPT="$OUTDIR/policy_fric${TAG}_205.pt"
  local TRAIN_LOG="$OUTDIR/training_log_fric${TAG}_205.json"
  if [ ! -f "$CKPT" ]; then
    echo "[fric $FRIC] BC retrain on 205 friction-$FRIC demos..." | tee -a "$LOG"
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
    --mass-scale 1.0
    --friction-scale "$FRIC"
    --output-dir "$OUTDIR"
  )
  # NOTE: cell args are kept in parallel bash arrays (not a colon-joined
  # string) so that paths containing spaces -- e.g. the shared-data dir
  # under ".../ECE M237/..." passed via --prior-bank-dataset / --bc-checkpoint
  # -- survive intact.  An earlier string-split + unquoted $CELL_ARGS
  # expansion word-split on that space and crashed (same bug as the #2
  # n200-magnitude slate, fixed 2026-06-01).
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
    local TAG_FULL="${CELL_TAG}_fric${TAG}"
    if ls "$OUTDIR"/unified_*"_${TAG_FULL}".json >/dev/null 2>&1; then
      echo "[fric $FRIC] $CELL_TAG already done; skipping." | tee -a "$LOG"
      continue
    fi
    local -a CELL_ARGS=()
    case "$i" in
      0) CELL_ARGS=("${CELL_0_ARGS[@]}") ;;
      1) CELL_ARGS=("${CELL_1_ARGS[@]}") ;;
      2) CELL_ARGS=("${CELL_2_ARGS[@]}") ;;
    esac
    echo "" | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  fric=$FRIC  $CELL_TAG  starting ===" | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --output-name-suffix "_$TAG_FULL" \
      "${CELL_ARGS[@]}" 2>&1 | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  fric=$FRIC  $CELL_TAG  done ===" | tee -a "$LOG"
  done
}

for FRIC in $FRICTIONS; do
  run_friction "$FRIC"
done

# ============================================================
# Final summary table (friction-axis bite-zone reading)
# ============================================================
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-FRICTION-SWEEP  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

out = Path("$OUTDIR")
print()
print(f"{'fric':>6}  {'cell':<32}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 65)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    fric = d.get("friction_scale", "?")
    tag = js.stem.split("_R-1_")[-1] if "_R-1_" in js.stem else js.stem
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    print(f"{fric:>6}  {tag:<32}  {ns:>7}  {n:>4}  {sr:>6.1f}%")
print()
print("Read T2.1 - T2.0 gap at each friction.  Bite zone is where")
print("T2.0 collapses but T2.1 holds up.  T2.3 BC-retrained gives")
print("the 78x-slower comparison reference.")
PY

echo "" | tee -a "$LOG"
echo "Next:" | tee -a "$LOG"
echo "  - Combine with the mass-axis slates via aggregate_incremental.py" | tee -a "$LOG"
echo "    to produce a 2-axis (mass, friction) bite-zone plot." | tee -a "$LOG"
echo "  - Update brief §8.1 #5: strike 'no friction perturbation'." | tee -a "$LOG"
echo "  - Update presentation/SLIDE_SPEC.md Slide 7 right panel to" | tee -a "$LOG"
echo "    add the friction axis." | tee -a "$LOG"
