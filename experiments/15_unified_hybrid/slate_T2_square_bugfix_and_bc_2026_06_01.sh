#!/usr/bin/env bash
# Slate T2-SQUARE-BUGFIX-BC -- re-run the invalid T2.0_square cell and
# add the previously-blocked BC arm.
#
# Follow-up B5 from the 2026-06-01 backlog research.  Two issues with
# the original slate_incremental_T2_square_hybrid_2026_05_28.sh that
# need correction:
#
#   1. INVALID T2.0_square cell.  The original launcher omitted
#      --prior-bank-dataset on the T2.0 cell.  Per priors_zerotrain
#      semantics this should fall back to the canonical 200 Square
#      bank, but the rollout JSON recorded prior_bank_dataset=None
#      and the cell came in at 1/30 = 3.3% (consistent with an
#      empty / wrong bank, NOT a faithful T2.0 measurement).  See
#      docs/results_writeups/t2_square.md "Diagnostic narrative".
#      Fix: re-run with --prior-bank-dataset set explicitly.
#
#   2. MISSING BC ARM.  T2.2_square (BC unadapted) and T2.3_square
#      (BC retrained on 205 demos) were originally blocked on
#      feat/adapt-bc-multitask-obs, which has since merged
#      (2026-05-28).  The 03_bc_baseline pipeline now supports
#      --task square, so the BC arm can finally land.
#
# Two stages
# ----------
#
#   stage A  T2.0 re-run with explicit canonical bank ('bugfix').
#            One cell, 30 rollouts, ~50 min.
#
#   stage B  BC arm (T2.2 + T2.3).  Two BC trains + two eval cells.
#            ~80 s/train + ~50 min/eval = ~2 h.
#
# Honest framing
# --------------
# Even after the bugfix, the Square hybrid pipeline is independently
# known to floor at 0-8% under every Slate G perturbation knob, so
# T2.0_square's re-measured number may still be near floor.  That's
# acceptable -- the goal here is to RECORD an honest cell and close
# the data-validity gap, not to demonstrate positive T2 transfer.
# The structural Square fix lives on feat/contrastive-square-slate.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_square_bugfix_and_bc_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-SQUARE-BUGFIX-BC: re-run T2.0 + land BC arm" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Prereq: canonical Square /ph dataset + mass=1.3 banks
# ============================================================
SQUARE_CANONICAL="$REPO/experiments/15_unified_hybrid/data/square_ph.hdf5"
if [ ! -f "$SQUARE_CANONICAL" ]; then
  SQUARE_CANONICAL_ALT="$(find "$REPO" -name 'square*.hdf5' -path '*ph*' 2>/dev/null | head -1)"
  if [ -z "$SQUARE_CANONICAL_ALT" ]; then
    echo "FATAL: canonical Square /ph dataset not found." | tee -a "$LOG"
    echo "       Sync the robomimic Square /ph dataset first." | tee -a "$LOG"
    exit 2
  fi
  SQUARE_CANONICAL="$SQUARE_CANONICAL_ALT"
fi
echo "Square canonical dataset: $SQUARE_CANONICAL" | tee -a "$LOG"

B205="$SHARED_DATA/square_mass1p3_205demos.hdf5"
B220="$SHARED_DATA/square_mass1p3_220demos.hdf5"
if [ ! -f "$B205" ] || [ ! -f "$B220" ]; then
  echo "[prereq] Square mass=1.3 banks missing; collecting..." | tee -a "$LOG"
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

# ============================================================
# Shared run helper
# ============================================================
COMMON=(--task square --mpc-replans 1 --mpc-chunk-steps 60
        --K 10 --M 16 --H 120
        --sigma-max 0.25 --sigma-min 0.05
        --n-demos 10 --n-rollouts 3
        --n-workers "$N_WORKERS" --seed 0
        --mass-scale 1.3
        --output-dir "$OUTDIR")

run () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# ============================================================
# Stage A: T2.0_square bugfix re-run
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE A: T2.0_square bugfix re-run ##############" | tee -a "$LOG"
echo "Adds explicit --prior-bank-dataset = canonical Square (was MISSING" | tee -a "$LOG"
echo "in original 2026-05-28 launcher; recorded prior_bank_dataset=None" | tee -a "$LOG"
echo "in the now-invalidated 1/30 = 3.3% result)." | tee -a "$LOG"

run "A_T2_0_square_hybrid_canonical_FIXED_mass1p3" \
  --prior demo_1nn --lambda-prior 1 \
  --prior-bank-dataset "$SQUARE_CANONICAL" \
  --prior-bank-task square

# ============================================================
# Stage B: BC arm (T2.2 unadapted, T2.3 retrained on 205)
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE B: BC arm (now unblocked) ##############" | tee -a "$LOG"

CKPT_UNADAPTED="$OUTDIR/policy_square_canonical.pt"
CKPT_ADAPTED="$OUTDIR/policy_square_mass1p3_205.pt"

# B1: BC train on canonical 200 Square demos (cpu, ~80 s).
if [ ! -f "$CKPT_UNADAPTED" ]; then
  echo "[stage B] BC train on canonical 200 Square demos..." | tee -a "$LOG"
  python "$REPO/experiments/03_bc_baseline/train.py" \
    --dataset "$SQUARE_CANONICAL" --task square \
    --output "$CKPT_UNADAPTED" \
    --log "$OUTDIR/training_log_square_canonical.json" \
    --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
    --seed 0 --device cpu 2>&1 | tee -a "$LOG"
fi

# B2: BC retrain on 205 Square mass=1.3 demos (cpu, ~80 s).
if [ ! -f "$CKPT_ADAPTED" ]; then
  echo "[stage B] BC retrain on 205 Square mass=1.3 demos..." | tee -a "$LOG"
  python "$REPO/experiments/03_bc_baseline/train.py" \
    --dataset "$B205" --task square \
    --output "$CKPT_ADAPTED" \
    --log "$OUTDIR/training_log_square_mass1p3_205.json" \
    --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
    --seed 0 --device cpu 2>&1 | tee -a "$LOG"
fi

# B3: T2.2_square -- BC unadapted, evaluated at mass=1.3.
run "B_T2_2_square_bc_unadapted_mass1p3" \
  --prior bc --lambda-prior 5 \
  --bc-checkpoint "$CKPT_UNADAPTED"

# B4: T2.3_square -- BC retrained on 205, evaluated at mass=1.3.
run "B_T2_3_square_bc_retrained_205_mass1p3" \
  --prior bc --lambda-prior 5 \
  --bc-checkpoint "$CKPT_ADAPTED"

# ============================================================
# Final summary
# ============================================================
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-SQUARE-BUGFIX-BC  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

out = Path("$OUTDIR")
print()
print(f"{'cell':<55}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 80)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    tag = js.stem.split("_demo_1nn_")[-1] if "_demo_1nn_" in js.stem else js.stem
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    print(f"{tag:<55}  {ns:>7}  {n:>4}  {sr:>6.1f}%")
PY

echo "" | tee -a "$LOG"
echo "Next: update docs/results_writeups/t2_square.md with the corrected" | tee -a "$LOG"
echo "T2.0_square number + the now-landed BC arm cells.  Replace the" | tee -a "$LOG"
echo "invalid 1/30 measurement and the '__' placeholders." | tee -a "$LOG"
