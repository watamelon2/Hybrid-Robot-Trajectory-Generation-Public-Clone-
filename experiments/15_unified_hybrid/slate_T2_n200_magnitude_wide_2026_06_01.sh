#!/usr/bin/env bash
# Slate T2-N200-MAG-WIDE -- push the adaptation experiment harder.
#
# Future-work item 2 from the Slide 9 conclusion deck (2026-06-01).
# The headline adaptation result on Slide 7 is "+62 pp gap held
# across 5 demos and mass scales 1.1 -> 1.5".  This slate stress-
# tests both axes so we can identify where the method actually
# breaks (the "bite zone" boundary).
#
# Two axes, run in two stages
# ---------------------------
#   stage A (democount, mass=1.3, ~5 cells, ~5 h)
#     Extend the existing slate_incremental_T2_democount sweep
#     beyond N=20.  N in {30, 50, 100, 150, 200}.
#     Hypothesis: succ rate saturates by N ~ 20-30; pushing further
#     either confirms saturation or uncovers a second knee where
#     the bank starts to *lose* signal (over-coverage washing out
#     the retrieval prior).
#
#   stage B (magnitude, ~16 cells, ~10 h)
#     Extend the existing slate_incremental_T2_magnitude sweep
#     beyond mass=1.5.  Mass in {1.6, 1.7, 1.8, 2.0}.  Three
#     cells per magnitude (T2.0 canonical bank, T2.1 5-demo
#     adapted bank, T2.3 BC-retrained-on-205).
#     Hypothesis: T2.1 stops dominating T2.3 by mass ~ 1.7-1.8
#     where 5 demos are no longer enough to bridge the dynamics
#     gap.  T2.0 collapses earlier (mass ~ 1.5-1.6).  Mapping the
#     upper bound of the bite zone closes the "where does this
#     stop working?" gap that the deck currently glosses over.
#
# What this slate does NOT do
#   - It does not re-run mass in {1.1, 1.2, 1.3, 1.4, 1.5} -- those
#     cells already live in slate_incremental_T2_2026_05_26 and
#     slate_incremental_T2_magnitude_2026_05_28.  Aggregation is
#     a post-slate step (re-run aggregate_incremental.py with
#     --combine-with-existing).
#   - It does not extend N=200 + mass>=1.6 (no headroom for the
#     two extensions to combine on a budgeted Mac; the cross-
#     product is in a follow-on slate if budget allows).
#
# Wall-clock estimates (10-worker Mac, can task, H=120)
#   Stage A demo-count cells (each is ~12 min hybrid):    ~1 h
#   Stage A larger-N bank derivations + collection:       ~1 h
#   Stage B per magnitude (collect + 3 cells + retrain):  ~2.5 h
#   Stage B 4 magnitudes:                                  ~10 h
#   Total: ~12-15 h sequential.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_n200_magnitude_wide_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
STAGE="${STAGE:-both}"                          # both | A | B
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-N200-MAG-WIDE: push adaptation harder" | tee -a "$LOG"
echo "SHARED_DATA = $SHARED_DATA  STAGE=$STAGE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Stage A: demo-count beyond N=20, fixed mass=1.3
# ============================================================
stage_A () {
  echo "" | tee -a "$LOG"
  echo "############## STAGE A: democount up to N=200 ##############" | tee -a "$LOG"

  # We need to collect a 200-demo perturbed bank.  The existing
  # collect_perturbed_demos.py was tuned to collect 20; we just
  # call it with --n-demos-target 200 (linear in N).
  local BIG="$SHARED_DATA/can_mass1p3_200pert.hdf5"
  if [ ! -f "$BIG" ]; then
    echo "[A] collecting 200 perturbed mass=1.3 demos (~2 h)..." | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/collect_perturbed_demos.py" \
      --task can --mass-scale 1.3 --friction-scale 1.0 \
      --n-demos-target 200 \
      --output "$BIG" \
      --combine-output "$SHARED_DATA/can_mass1p3_400demos.hdf5" \
      --verbose 2>&1 | tee -a "$LOG"
  fi

  # Derive intermediate banks via derive_perturbed_banks (slices
  # the 200-bank into 30 / 50 / 100 / 150 sub-banks, each appended
  # to the canonical 200 to form 230 / 250 / 300 / 350).
  for N in 30 50 100 150 200; do
    local TARGET="$SHARED_DATA/can_mass1p3_$((200 + N))demos.hdf5"
    if [ ! -f "$TARGET" ]; then
      echo "[A] deriving 200 + $N bank -> $TARGET" | tee -a "$LOG"
      python "$REPO/experiments/15_unified_hybrid/derive_perturbed_banks.py" \
        --task can \
        --perturbed-20 "$BIG" \
        --out-5 "$SHARED_DATA/_tmp_5.hdf5" \
        --out-205 "$TARGET" \
        --n-canonical 200 \
        --n-perturbed "$N" 2>&1 | tee -a "$LOG"
    fi
  done
  rm -f "$SHARED_DATA/_tmp_5.hdf5"

  # Per-N hybrid cell (T2.1-style: hybrid + extended bank).
  local COMMON=(
    --task can --mpc-replans 1 --mpc-chunk-steps 60
    --K 10 --M 16 --H 120
    --sigma-max 0.25 --sigma-min 0.05
    --n-demos 10 --n-rollouts 3
    --n-workers "$N_WORKERS" --seed 0
    --mass-scale 1.3
    --prior demo_1nn --lambda-prior 1
    --output-dir "$OUTDIR"
  )

  for N in 30 50 100 150 200; do
    local BANK="$SHARED_DATA/can_mass1p3_$((200 + N))demos.hdf5"
    local TAG="A_T2_1_hybrid_${N}demo_mass1p3"
    if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
      echo "[A] $TAG already done; skipping." | tee -a "$LOG"
      continue
    fi
    echo "" | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --prior-bank-dataset "$BANK" --prior-bank-task can \
      --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
  done
}

# ============================================================
# Stage B: magnitude beyond mass=1.5, all three T2 cells per mass
# ============================================================
stage_B () {
  echo "" | tee -a "$LOG"
  echo "############## STAGE B: magnitude > 1.5 ##############" | tee -a "$LOG"

  run_magnitude () {
    local MASS="$1"                  # e.g. 1.6
    local TAG="${MASS//./p}"         # e.g. 1p6
    local P20="$SHARED_DATA/can_mass${TAG}_20demos.hdf5"
    local B205="$SHARED_DATA/can_mass${TAG}_205demos.hdf5"
    local B220="$SHARED_DATA/can_mass${TAG}_220demos.hdf5"
    local B5="$SHARED_DATA/can_mass${TAG}_5demos.hdf5"

    echo "" | tee -a "$LOG"
    echo "================ MAGNITUDE mass=$MASS ================" | tee -a "$LOG"

    # 1. collect if missing.  At high mass, the canonical BC policy
    #    is less likely to succeed in the collection rollout, so
    #    collect_perturbed_demos may need extra attempts.  The
    #    --n-demos-target=20 + script's internal retry logic handles
    #    this; just be patient.
    if [ ! -f "$B220" ] || [ ! -f "$B205" ] || [ ! -f "$B5" ]; then
      echo "[B mass=$MASS] collecting + deriving banks..." | tee -a "$LOG"
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
      echo "[B mass=$MASS] BC retrain on 205 demos..." | tee -a "$LOG"
      python "$REPO/experiments/03_bc_baseline/train.py" \
        --dataset "$B205" --task can \
        --output "$CKPT" --log "$TRAIN_LOG" \
        --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
        --seed 0 --device cpu 2>&1 | tee -a "$LOG"
    fi

    # 3. the 3 eval cells (T2.0 canonical bank, T2.1 adapted, T2.3 BC).
    local COMMON=(
      --task can --mpc-replans 1 --mpc-chunk-steps 60
      --K 10 --M 16 --H 120
      --sigma-max 0.25 --sigma-min 0.05
      --n-demos 10 --n-rollouts 3
      --n-workers "$N_WORKERS" --seed 0
      --mass-scale "$MASS"
      --output-dir "$OUTDIR"
    )
    # Cells defined as parallel arrays so each cell's args survive
    # path expansion (paths under SHARED_DATA contain a space in
    # "ECE M237" and would otherwise be split into separate args by
    # bash's unquoted-variable splitting -- which is what crashed
    # this slate's first run at 21:14 PDT 2026-06-01).
    local -a CELL_TAGS=(
      "B_T2_0_hybrid_canonical"
      "B_T2_1_hybrid_5demo_adapted"
      "B_T2_3_bc_retrained_205demos"
    )
    local -a CELL_0_ARGS=( --prior demo_1nn --lambda-prior 1 )
    local -a CELL_1_ARGS=( --prior demo_1nn --lambda-prior 1 --prior-bank-dataset "$B205" --prior-bank-task can )
    local -a CELL_2_ARGS=( --prior bc --lambda-prior 5 --bc-checkpoint "$CKPT" )

    run_one_b_cell () {
      local CELL_TAG="$1"; shift
      local TAG_FULL="${CELL_TAG}_mass${TAG}"
      if ls "$OUTDIR"/unified_*"_${TAG_FULL}".json >/dev/null 2>&1; then
        echo "[B mass=$MASS] $CELL_TAG already done; skipping." | tee -a "$LOG"
        return 0
      fi
      echo "" | tee -a "$LOG"
      echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  starting ===" | tee -a "$LOG"
      python "$REPO/experiments/15_unified_hybrid/run_can.py" \
        "${COMMON[@]}" \
        --output-name-suffix "_$TAG_FULL" \
        "$@" 2>&1 | tee -a "$LOG"
      echo "=== $(date '+%H:%M:%S')  mass=$MASS  $CELL_TAG  done ===" | tee -a "$LOG"
    }

    run_one_b_cell "${CELL_TAGS[0]}" "${CELL_0_ARGS[@]}"
    run_one_b_cell "${CELL_TAGS[1]}" "${CELL_1_ARGS[@]}"
    run_one_b_cell "${CELL_TAGS[2]}" "${CELL_2_ARGS[@]}"
  }

  run_magnitude 1.6
  run_magnitude 1.7
  run_magnitude 1.8
  run_magnitude 2.0
}

# ============================================================
# Dispatch
# ============================================================
case "$STAGE" in
  both) stage_A; stage_B ;;
  A)    stage_A ;;
  B)    stage_B ;;
  *)    echo "FATAL: STAGE must be 'both', 'A', or 'B' (got: $STAGE)"; exit 2 ;;
esac

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-N200-MAG-WIDE  ALL DONE ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next: combine with existing T2 cells via" | tee -a "$LOG"
echo "  python experiments/15_unified_hybrid/aggregate_incremental.py \\" | tee -a "$LOG"
echo "    --include-dirs results/slate_incremental_T2_{2026_05_26,democount_2026_05_28,magnitude_2026_05_28} \\" | tee -a "$LOG"
echo "                   results/slate_T2_n200_magnitude_wide_2026_06_01" | tee -a "$LOG"
echo "to refresh the full democount + magnitude curves for Slide 7." | tee -a "$LOG"
