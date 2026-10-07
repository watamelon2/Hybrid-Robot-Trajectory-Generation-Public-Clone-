#!/usr/bin/env bash
# Slate Incremental T2 -- multi-seed validation.
#
# Re-runs the T2 ladder (T2.0, T2.1, T2.2, T2.3, T2.4) at seeds {1, 2}
# to upgrade the headline from n=1 seed to n=3 seeds (combining with
# the seed=0 results already on main).
#
# Predicted outcome (substantiates T2 if it holds across seeds):
#   - T2.0 succ should remain near the seed=0 value (~13%)
#   - T2.1 succ should remain non-trivially above T2.0 (~50-80%)
#   - T2.3 succ should remain near 0%  (BC catastrophic on +5 demos)
#   - Pooled 3-seed CIs across T2.1 vs T2.3 should NOT overlap.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Plan ref:    docs/adaptation/05_followup_plan.md step (1)
# Branch:      feat/adapt-t2-multiseed
# Worktree:    HRTG-multiseed
# Wall-clock:  ~7 h (10 hybrid cells x ~40 min each).  Pure hybrid;
#              T2.3 BC retrain is the only non-hybrid cell and only
#              needs to happen ONCE per seed (BC training is
#              deterministic given the dataset, so we set the BC
#              torch seed = run seed).
#
# Data prereq: $SHARED_DATA/can_mass1p3_{205,220}demos.hdf5
#              Built on-demand if missing (~10 min).
#
# Usage:       ./slate_incremental_T2_multiseed_2026_05_28.sh
#              SHARED_DATA=/custom/path ./slate_...sh    # override

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR_ROOT="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_multiseed_2026_05_28"
mkdir -p "$OUTDIR_ROOT" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG_MAIN="$OUTDIR_ROOT/run.log"
date | tee "$LOG_MAIN"
echo "Slate Incremental T2 multi-seed" | tee -a "$LOG_MAIN"
echo "SHARED_DATA = $SHARED_DATA" | tee -a "$LOG_MAIN"
echo "" | tee -a "$LOG_MAIN"

# -- Prereq check / build perturbed banks if missing -----------------
for f in "$SHARED_DATA/can_mass1p3_205demos.hdf5" "$SHARED_DATA/can_mass1p3_220demos.hdf5"; do
  if [ ! -f "$f" ]; then
    echo "[multiseed] missing prereq: $f -- collecting now (~10 min)" | tee -a "$LOG_MAIN"
    LOCAL_DATA="$REPO/experiments/15_unified_hybrid/data"
    mkdir -p "$LOCAL_DATA"
    python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
      --task can --mass-scale 1.3 --friction-scale 1.0 \
      --n-demos-target 20 \
      --output "$LOCAL_DATA/can_mass1p3_20demos.hdf5" \
      --combine-output "$LOCAL_DATA/can_mass1p3_220demos.hdf5" \
      --verbose 2>&1 | tee -a "$LOG_MAIN"
    python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
      --task can \
      --perturbed-20 "$LOCAL_DATA/can_mass1p3_20demos.hdf5" \
      --out-5   "$LOCAL_DATA/can_mass1p3_5demos.hdf5" \
      --out-205 "$LOCAL_DATA/can_mass1p3_205demos.hdf5" 2>&1 | tee -a "$LOG_MAIN"
    cp "$LOCAL_DATA"/can_mass1p3_{5,20,205,220}demos.hdf5 "$SHARED_DATA/"
    echo "[multiseed] banks copied to $SHARED_DATA" | tee -a "$LOG_MAIN"
    break
  fi
done
ls -lh "$SHARED_DATA"/can_mass1p3*.hdf5 | tee -a "$LOG_MAIN"
echo "" | tee -a "$LOG_MAIN"

# -- Per-seed slate --------------------------------------------------
run_seed_cell () {
  local SEED="$1"; local TAG="$2"; shift 2
  local OUTDIR="$OUTDIR_ROOT/seed_$SEED"
  mkdir -p "$OUTDIR"
  local LOG="$OUTDIR/run.log"
  echo "" | tee -a "$LOG_MAIN" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  seed=$SEED  $TAG  starting ===" | tee -a "$LOG_MAIN" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 10 \
    --seed "$SEED" \
    --mass-scale 1.3 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_${TAG}_seed${SEED}" \
    "$@" 2>&1 | tee -a "$LOG" | tee -a "$LOG_MAIN"
  echo "=== $(date '+%H:%M:%S')  seed=$SEED  $TAG  done ===" | tee -a "$LOG_MAIN" | tee -a "$LOG"
}

for SEED in 1 2; do
  echo "" | tee -a "$LOG_MAIN"
  echo "=========== SEED $SEED ===========" | tee -a "$LOG_MAIN"

  # T2.0 -- hybrid + canonical 200 Can bank @ mass=1.3 (degraded baseline)
  run_seed_cell "$SEED" "T2_0_hybrid_canonical_bank_mass1p3" \
    --prior demo_1nn --lambda-prior 1

  # T2.1 -- HEADLINE.  Hybrid + 200 + 5 mass=1.3 demos
  run_seed_cell "$SEED" "T2_1_hybrid_5demo_adapted_mass1p3" \
    --prior demo_1nn --lambda-prior 1 \
    --prior-bank-dataset "$SHARED_DATA/can_mass1p3_205demos.hdf5" \
    --prior-bank-task can

  # T2.2 -- BC unadapted @ mass=1.3
  run_seed_cell "$SEED" "T2_2_bc_unadapted_mass1p3" \
    --prior bc --lambda-prior 5

  # T2.4 -- saturation probe (hybrid + 200 + 20 mass=1.3 demos)
  run_seed_cell "$SEED" "T2_4_hybrid_20demo_saturation_mass1p3" \
    --prior demo_1nn --lambda-prior 1 \
    --prior-bank-dataset "$SHARED_DATA/can_mass1p3_220demos.hdf5" \
    --prior-bank-task can

  # T2.3 -- BC retrained @ mass=1.3 (500 epochs, batch 512) AT THIS SEED.
  # For multi-seed validity the BC training itself must be seeded with
  # $SEED so the BC pipeline gets its own seed-sample of the retrain
  # process; sharing one policy across seeds would only multi-seed the
  # eval (MBD rollouts) and would underestimate variance.
  RETRAIN_CKPT="$OUTDIR_ROOT/seed_$SEED/policy_mass1p3_205_seed${SEED}.pt"
  RETRAIN_LOG="$OUTDIR_ROOT/seed_$SEED/training_log_mass1p3_205_seed${SEED}.json"
  if [ ! -f "$RETRAIN_CKPT" ]; then
    echo "[T2.3 seed=$SEED] retraining BC ~80s..." | tee -a "$LOG_MAIN"
    TRAIN_T0=$(date +%s)
    python "$REPO/experiments/03_bc_baseline/train.py" \
        --dataset "$SHARED_DATA/can_mass1p3_205demos.hdf5" \
        --task can \
        --output "$RETRAIN_CKPT" \
        --log "$RETRAIN_LOG" \
        --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
        --seed "$SEED" --device cpu 2>&1 | tee -a "$LOG_MAIN"
    echo "[T2.3 seed=$SEED] retrain wall: $(($(date +%s) - TRAIN_T0))s" | tee -a "$LOG_MAIN"
  fi
  run_seed_cell "$SEED" "T2_3_bc_retrained_205demos_mass1p3" \
    --prior bc --lambda-prior 5 \
    --bc-checkpoint "$RETRAIN_CKPT"
done

echo "" | tee -a "$LOG_MAIN"
echo "=== $(date '+%H:%M:%S')  ALL CELLS DONE ===" | tee -a "$LOG_MAIN"
echo "Aggregate per-seed JSONs with aggregate_incremental.py;" | tee -a "$LOG_MAIN"
echo "pool 3-seed CIs in docs/results_writeups/t2_multiseed.md." | tee -a "$LOG_MAIN"
