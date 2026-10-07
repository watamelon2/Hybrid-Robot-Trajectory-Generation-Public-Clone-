#!/usr/bin/env bash
# T2.3 control: retrain BC on JUST the 200 canonical demos, CPU, seed=0,
# 500 epochs, then eval at mass=1.3.  Isolates "device" (CUDA -> CPU)
# from "data composition" (+5 mass=1.3 demos) as the cause of the
# T2.3 0/30 collapse.
#
# Expected runtime: ~80 s train + ~920 s eval = ~17 min.
#
# Decision tree on the result:
#   succ ~ 30%  -> CPU/CUDA difference is benign; +5 demos cause the drop.
#                   Next step: shift-statistics diff or gripper-bias hypothesis.
#   succ ~ 0%   -> CPU retrain produces a worse policy than canonical CUDA.
#                   T2.3's "BC retraining is bad" claim becomes contaminated
#                   by an experimental artifact (device choice).  Need to
#                   either re-train both on the same device for fair compare
#                   or downgrade the T2.3 claim to "BC retraining on CPU
#                   doesn't help".
#   succ ~ 15%  -> Both effects contribute roughly equally.

set -euo pipefail
REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
DATA="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/can/ph/low_dim_v141.hdf5"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_bc_retrain_2026_05_26"
LOG="$OUTDIR/control_canonical_cpu.log"
CKPT="$REPO/experiments/03_bc_baseline/results/policy_canonical_200_cpu.pt"
TLOG="$REPO/experiments/03_bc_baseline/results/training_log_canonical_200_cpu.json"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "=== T2.3 CONTROL: BC retrain on canonical 200 demos, CPU, seed=0 ===" | tee -a "$LOG"

# Stage 1: retrain BC on canonical 200 demos.
echo "Stage 1: BC retrain (~80s)" | tee -a "$LOG"
cd "$REPO"
T0=$(date +%s)
python experiments/03_bc_baseline/train.py \
    --dataset "$DATA" \
    --task can \
    --output "$CKPT" \
    --log "$TLOG" \
    --epochs 500 \
    --batch-size 512 \
    --lr 1e-4 \
    --hidden 256 \
    --seed 0 \
    --device cpu 2>&1 | tee -a "$LOG"
TWALL=$(($(date +%s) - T0))
echo "[CONTROL] BC retrain wall: ${TWALL}s" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Stage 2: eval at mass=1.3.
echo "Stage 2: eval @ mass=1.3 (~15 min, 10 workers)" | tee -a "$LOG"
E0=$(date +%s)
python experiments/15_unified_hybrid/run_can.py \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --n-demos 10 --n-rollouts 3 \
    --n-workers 10 \
    --seed 0 \
    --mass-scale 1.3 \
    --prior bc --lambda-prior 5 \
    --bc-checkpoint "$CKPT" \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_CONTROL_canonical_200_cpu_mass1p3" 2>&1 | tee -a "$LOG"
EWALL=$(($(date +%s) - E0))
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  CONTROL DONE  train=${TWALL}s  eval=${EWALL}s ===" | tee -a "$LOG"
