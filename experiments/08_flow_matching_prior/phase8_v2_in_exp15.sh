#!/usr/bin/env bash
# Phase 8 follow-up: plug v_theta_v2 into exp 15's unified driver at the
# Phase-5 best cell (sigma=0.02, lambda=0) with replanning.
#
# Motivation: Phase 5 found v_theta_v2 + MBD single-shot peaks at
# 10/60 = 16.7 % at sigma=0.02 lambda=0.  Exp 15's BC-prior baseline
# at K=10 R=0 was 0/60 single-shot but jumped to 25/60 = 41.7 % with
# R=1 (mid-episode replan).  This phase tests whether replanning
# multiplies success for the flow prior the same way.
#
# Cells (both at sigma_max=0.02, sigma_min=0.01, lambda_prior=0,
# K=10, M=16, H=120, lambda_softmax=0.5, n_demos=20, n_rollouts=3,
# seed=0, n_workers=10):
#
#   R=0  (control)  -> must reproduce Phase 5's 10/60 within +/-1
#                      (validates exp 15's flow-prior driver matches
#                      exp 07's at single-shot).
#   R=1  (headline)  -> the answer to the headline question.
#
# Notes
# -----
# Lambda=0 means unified_loop.py never calls score_fn_factory, so
# the score-bridge formula is irrelevant (Phase 7 also confirmed it
# isn't the blocker).  No CLI flag for --bridge-formula is needed.
#
# Expected wall:
#   R=0  ~17 min  (60 plans * 1 chunk each, 10 workers)
#   R=1  ~35 min  (60 plans * 2 chunks each)
#   Total ~52 min.

set -e
source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PRIOR_CKPT="$REPO/experiments/08_flow_matching_prior/results/v_theta_v2.pt"
DATASET="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/can/ph/low_dim_v141.hdf5"

OUT="$REPO/experiments/08_flow_matching_prior/results/v2_followup_2026_05_16"
mkdir -p "$OUT"
LOG="$OUT/followup.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log ""
log "=== Phase 8 follow-up: v_theta_v2 in exp 15 driver, sigma=0.02 lambda=0 ==="
log "prior_checkpoint: $PRIOR_CKPT"
log "output_dir:       $OUT"

run_cell() {
    local R="$1"
    local TAG="exp15_R${R}"
    local FN="$OUT/unified_flow_lam-0.0_K-10_R-${R}_${TAG}.json"
    if [ -f "$FN" ]; then
        log "  R=${R} already done at $FN, skipping"
        return
    fi
    log ""
    log "  --- R=${R} cell ---"
    python experiments/15_unified_hybrid/run_can.py \
        --task can \
        --dataset "$DATASET" \
        --prior flow \
        --prior-checkpoint "$PRIOR_CKPT" \
        --warm-start \
        --lambda-prior 0.0 \
        --mpc-replans "$R" \
        --K 10 --M 16 --H 120 \
        --sigma-max 0.02 --sigma-min 0.01 \
        --lambda_ 0.5 \
        --n-demos 20 --n-rollouts 3 \
        --seed 0 \
        --n-workers 10 \
        --output-dir "$OUT" \
        --output-name-suffix "_${TAG}" \
        2>&1 | tee -a "$LOG"
}

run_cell 0
run_cell 1

# ---- Quick summary -----------------------------------------------------------
log ""
log "=== Phase 8 follow-up summary ==="
python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUT")
for R in (0, 1):
    p = out / f"unified_flow_lam-0.0_K-10_R-{R}_exp15_R{R}.json"
    if not p.exists():
        print(f"R={R}: MISSING ({p.name})")
        continue
    d = json.load(open(p))
    n = d["n_total"]; s = d["n_success"]; pct = 100 * s / max(n, 1)
    wall = d["wall_s"]
    print(f"R={R}: {s}/{n} = {pct:.1f}%  wall={wall:.0f}s  ({p.name})")
print()
print("Phase 5 reference (single-shot, no replanning): 10/60 = 16.7%")
print("Exp 15 BC-prior reference at K=10 R=1 lambda=5 sigma=0.25: 25/60 = 41.7%")
PY

log ""
log "=== Phase 8 follow-up DONE ==="
