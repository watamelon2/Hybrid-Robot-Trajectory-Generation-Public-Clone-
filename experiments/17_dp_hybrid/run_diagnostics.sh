#!/usr/bin/env bash
# Phase 1 diagnostic ladder for the DP + pull-MBD regression
# (hybrid produced reward ~1-2 on demos where standalone hit 25-33).
#
# Each cell uses **1 demo (demo_0) × 3 rollouts** = 3 trials.
# Demo 0 is the cleanest reference: standalone DP hits 3/3 there.
# Expected wall: ~18 min/cell.
#
# Cells:
#   D0   lambda=0,   sigma=0.25, K=10  -> MBD-without-pull (tests if MBD alone corrupts)
#   D1   lambda=1.0, sigma=0.10, K=10  -> narrower MBD perturbations (tests sigma)
#   D2   lambda=10,  sigma=0.25, K=10  -> stronger pull anchor (tests pull weight)
#   D3   lambda=1.0, sigma=0.25, K=3   -> fewer MBD inner steps (tests K)
#
# Usage: ./run_diagnostics.sh <cell_name>
# Where cell_name ∈ {D0, D1, D2, D3}.

set -euo pipefail
cd "$(dirname "$0")/../.."

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

CELL="${1:?usage: $0 <D0|D1|D2|D3>}"

case "$CELL" in
    D0) LAM=0.0;  SMAX=0.25; K=10; LABEL="lam0_K10_smax25"   ;;  # MBD without pull (CONFIRMED: regresses)
    D1) LAM=1.0;  SMAX=0.10; K=10; LABEL="lam1_K10_smax10"   ;;  # narrower MBD sigma
    D2) LAM=10.0; SMAX=0.25; K=10; LABEL="lam10_K10_smax25"  ;;  # strong pull anchor
    D3) LAM=1.0;  SMAX=0.25; K=3;  LABEL="lam1_K3_smax25"    ;;  # fewer MBD iters
    D4) LAM=1.0;  SMAX=0.05; K=10; LABEL="lam1_K10_smax05"   ;;  # tightest sigma, pull on
    D5) LAM=1.0;  SMAX=0.10; K=3;  LABEL="lam1_K3_smax10"    ;;  # tight sigma + few iters
    D6) LAM=0.0;  SMAX=0.05; K=10; LABEL="lam0_K10_smax05"   ;;  # tightest sigma, no pull
    D7) LAM=0.0;  SMAX=0.25; K=0;  LABEL="K0_nombd_control"  ;;  # K=0 = NO MBD (delivers DP plan as-is)
    *)  echo "unknown cell $CELL"; exit 1 ;;
esac

OUT="experiments/17_dp_hybrid/results/abl_${CELL}_${LABEL}"
mkdir -p "$OUT"
LOG="$OUT/run.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] starting diagnostic cell $CELL ($LABEL)"
echo "  lam=$LAM sigma_max=$SMAX K=$K"
echo "  output: $OUT"

python -u experiments/15_unified_hybrid/run_can.py \
    --task can \
    --prior flow --score-mode pull \
    --prior-checkpoint experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16.pt \
    --H 16 --mpc-chunk-steps 8 --mpc-replans 29 \
    --K "$K" --M 16 --lambda-prior "$LAM" \
    --sigma-max "$SMAX" --sigma-min 0.05 --prior-n-steps 100 \
    --deterministic-prior-warmstart \
    --n-demos 1 --n-rollouts 3 \
    --n-workers 4 --seed 0 --device cpu \
    --output-dir "$OUT" \
    --output-name-suffix "_${CELL}_${LABEL}"

echo "[$(date)] cell $CELL ($LABEL) finished"
touch "$OUT/done"
