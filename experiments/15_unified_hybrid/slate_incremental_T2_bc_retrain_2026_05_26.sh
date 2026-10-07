#!/usr/bin/env bash
# Slate Incremental T2.3 -- BC retrained on the combined bank.
#
# Two stages:
#   Stage 1: re-train BC on canonical 200 Can demos + 5 mass=1.3 demos
#            using the SAME hyperparameters as the canonical
#            policy.pt baseline (500 epochs, batch=512, lr=1e-4) so
#            the only nuisance variable is the training set itself.
#            ~60 s on M-series CPU.  Output:
#            experiments/03_bc_baseline/results/policy_mass1p3_205.pt
#   Stage 2: evaluate that re-trained BC at mass=1.3 (one cell, ~13 min).
#
# Bug fix (2026-05-28): the original slate used the train.py default
# (100 epochs, batch=256) which is 5x undertrained relative to the
# canonical policy.pt (500 epochs, batch=512 -- see ck['extra']).
# That asymmetric budget caused the retrained policy's last-layer
# weight norm to drop to 2.39 (vs canonical 3.53) and the closed-loop
# success rate at mass=1.3 to collapse to 0/30, masking the actual
# BC-retrain effect.
#
# This is the T2.3 cell of the Pareto plot:
#   (wall_clock = 30 min retrain + 50 min eval ~ 1800 s, succ = ~25%).
# Compare with T2.1's (~1 s adaptation, ~55% succ) from the sister
# eval slate.
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md sec 3
# Branch:      feat/adapt-incremental-prior
# Worktree:    HRTG-incremental
# Prereq:      slate_incremental_collect_2026_05_26.sh must have run
#              (writes can_mass1p3_205demos.hdf5).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_bc_retrain_2026_05_26"
DATA_DIR="$REPO/experiments/15_unified_hybrid/data"
BC_RESULTS="$REPO/experiments/03_bc_baseline/results"
RETRAINED_CKPT="$BC_RESULTS/policy_mass1p3_205.pt"
RETRAIN_LOG="$BC_RESULTS/training_log_mass1p3_205.json"
LOG="$OUTDIR/run.log"

mkdir -p "$OUTDIR" "$BC_RESULTS"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Sanity: prereq.
if [ ! -f "$DATA_DIR/can_mass1p3_205demos.hdf5" ]; then
  echo "[T2.3] FATAL: prereq bank missing: $DATA_DIR/can_mass1p3_205demos.hdf5" >&2
  echo "[T2.3] Run slate_incremental_collect_2026_05_26.sh first." >&2
  exit 3
fi

date | tee "$LOG"
echo "Slate Incremental T2.3 - BC retrained on 200+5 mass=1.3 demos" | tee -a "$LOG"
echo "Spec: docs/adaptation/04_incremental_prior_spec.md sec 3" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Stage 1: BC retrain on the combined bank ----
echo "=== $(date '+%H:%M:%S')  Stage 1: BC retrain (~60s, 500 epochs)  ===" | tee -a "$LOG"
TRAIN_T0=$(date +%s)
python "$REPO/experiments/03_bc_baseline/train.py" \
    --dataset "$DATA_DIR/can_mass1p3_205demos.hdf5" \
    --task can \
    --output "$RETRAINED_CKPT" \
    --log "$RETRAIN_LOG" \
    --epochs 500 \
    --batch-size 512 \
    --lr 1e-4 \
    --hidden 256 \
    --seed 0 \
    --device cpu 2>&1 | tee -a "$LOG"
TRAIN_WALL=$(($(date +%s) - TRAIN_T0))
echo "[T2.3] BC retrain wall: ${TRAIN_WALL}s ($((TRAIN_WALL / 60))m $((TRAIN_WALL % 60))s)" | tee -a "$LOG"
echo "[T2.3] checkpoint: $RETRAINED_CKPT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---- Stage 2: eval retrained BC at mass=1.3 ----
echo "=== $(date '+%H:%M:%S')  Stage 2: eval retrained BC @ mass=1.3  ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 10 \
    --seed 0 \
    --mass-scale 1.3 \
    --prior bc --lambda-prior 5 \
    --bc-checkpoint "$RETRAINED_CKPT" \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_T2_3_bc_retrained_mass1p3" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "BC retrain wall: ${TRAIN_WALL}s (this is the T2.3 x-axis on the Pareto plot)" | tee -a "$LOG"
echo "Aggregate with experiments/15_unified_hybrid/aggregate_incremental.py" | tee -a "$LOG"
