#!/usr/bin/env bash
# Phase 7 contingency: score-bridge formula fix.
#
# Motivation: adapter_v2.make_prior_score_fn ships with a "linear"
# denominator (1 - t) in the score-bridge, while the theoretical bridge
# under linear-interp + N(0,I) noising is (1 - t)^2.  All of Phase 1, 2,
# 5, 6 ran with bridge="linear", and within Phase 5 every cell with
# lambda > 0 underperformed the lambda=0 control at sigma=0.02
# (10/60 vs 2/60).  This is consistent with the linear bridge
# under-weighting the prior near t=1 in a way that makes positive
# lambda actively harmful at small sigma.
#
# Phase 7 re-runs the BEST sigma cell (sigma_max=0.02) with the
# "quadratic" bridge at lambda ∈ {0.1, 0.5, 1.0}.  Direct comparison
# to Phase 5 sigma_max=0.02:
#
#   linear    lambda=0.1 -> 2/60       quadratic lambda=0.1 -> ?
#   linear    lambda=0.5 -> 2/60       quadratic lambda=0.5 -> ?
#                                      quadratic lambda=1.0 -> ?
#
# Expected wall: ~51 min (3 lambdas × ~17 min @ 10 workers, sequential).

set -e
source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PRIOR_CKPT="$REPO/experiments/08_flow_matching_prior/results/v_theta_v2.pt"
DATASET="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/can/ph/low_dim_v141.hdf5"

OUT="$REPO/experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16"
LOG="$OUT/sweep.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

PHASE7_DIR="$OUT/phase7_bridge_quadratic"
mkdir -p "$PHASE7_DIR"

log ""
log "=== Phase 7: bridge=quadratic at sigma=0.02 (~51 min) ==="

if [ -f "$PHASE7_DIR/summary.json" ]; then
    log "  skipping -- $PHASE7_DIR/summary.json exists"
else
    SIG=0.02
    SIG_MIN=$(python -c "print(max(0.01, $SIG * 0.1))")
    python experiments/07_lambda_sweep/run_can.py \
        --task can \
        --dataset "$DATASET" \
        --prior flow \
        --prior-checkpoint "$PRIOR_CKPT" \
        --bridge-formula quadratic \
        --warm-start \
        --lambda-prior 0.1 0.5 1.0 \
        --n-demos 20 --n-rollouts 3 \
        --H 120 --M 16 --K 10 \
        --sigma-max "$SIG" --sigma-min "$SIG_MIN" \
        --seed 0 \
        --n-workers 10 \
        --output-dir "$PHASE7_DIR" \
        --no-live-view 2>&1 | tee -a "$LOG"
fi

# Re-aggregate including the new phase.
log ""
log "=== Re-running aggregator with Phase 7 ==="
python experiments/08_flow_matching_prior/sweep_v2_aggregate.py \
    "$OUT" 2>&1 | tee -a "$LOG"

log ""
log "=== Phase 7 DONE ==="
