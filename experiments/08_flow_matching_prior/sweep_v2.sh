#!/usr/bin/env bash
# Master sweep runner for v_theta_v2.pt
#
# Phase 0: prior-only replay (no MBD)               ~5 min
# Phase 1: lambda-sweep with warm-start (8 values)  ~96 min
# Phase 2: lambda-sweep no-warm-start (3 values)    ~36 min
# Phase 3: best-lambda + K-extension (K=20, K=30)   ~52 min (contingent)
# Phase 4: best-lambda + sigma-max sweep           ~26 min (contingent)
#
# Total wall: ~3.5 hr.  Idempotent -- each phase skips itself if its
# summary JSON already exists.
#
# Run under `screen -dmS sweep bash experiments/08_flow_matching_prior/sweep_v2.sh`.

set -e
source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PRIOR_CKPT="$REPO/experiments/08_flow_matching_prior/results/v_theta_v2.pt"
BC_CKPT="$REPO/experiments/03_bc_baseline/results/policy.pt"
DATASET="/Users/szh/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/can/ph/low_dim_v141.hdf5"

OUT="$REPO/experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16"
mkdir -p "$OUT"
LOG="$OUT/sweep.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log "=== sweep_v2 START ==="
log "out:     $OUT"
log "prior:   $PRIOR_CKPT  ($(ls -la "$PRIOR_CKPT" | awk '{print $5}') bytes)"
log "bc:      $BC_CKPT      ($(ls -la "$BC_CKPT" | awk '{print $5}') bytes)"
log "dataset: $DATASET"


# ---------- Phase 0 ----------------------------------------------------------
PHASE0_OUT="$OUT/phase0_prior_only_replay.json"
log ""
log "=== Phase 0: prior-only replay (no MBD; tests if warm-start alone succeeds) ==="
if [ -f "$PHASE0_OUT" ]; then
    log "  skipping -- $PHASE0_OUT exists"
else
    python experiments/08_flow_matching_prior/replay_prior.py \
        --checkpoint "$PRIOR_CKPT" \
        --dataset "$DATASET" \
        --n-demos 20 --n-rollouts 3 \
        --H 120 --seed 0 \
        --n-workers 10 \
        --output "$PHASE0_OUT" 2>&1 | tee -a "$LOG"
fi


# ---------- Phase 1 ----------------------------------------------------------
PHASE1_DIR="$OUT/phase1_lambda_sweep_warmstart"
mkdir -p "$PHASE1_DIR"
log ""
log "=== Phase 1: lambda sweep with WARM-START (8 values, ~96 min) ==="
log "    lambdas = {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0}"
log "    paired noise across lambdas via exp07's hash-seeded rng_lam"
if [ -f "$PHASE1_DIR/summary.json" ]; then
    log "  skipping -- $PHASE1_DIR/summary.json exists"
else
    python experiments/07_lambda_sweep/run_can.py \
        --task can \
        --dataset "$DATASET" \
        --prior flow \
        --prior-checkpoint "$PRIOR_CKPT" \
        --warm-start \
        --lambda-prior 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 \
        --n-demos 20 --n-rollouts 3 \
        --H 120 --M 16 --K 10 \
        --sigma-max 0.5 --sigma-min 0.05 \
        --seed 0 \
        --n-workers 10 \
        --output-dir "$PHASE1_DIR" \
        --no-live-view 2>&1 | tee -a "$LOG"
fi


# ---------- Phase 2 ----------------------------------------------------------
PHASE2_DIR="$OUT/phase2_lambda_sweep_no_warmstart"
mkdir -p "$PHASE2_DIR"
log ""
log "=== Phase 2: lambda sweep WITHOUT warm-start (~36 min) ==="
log "    lambdas = {0.0, 0.5, 2.0}  (0.0 = pure MBD baseline)"
if [ -f "$PHASE2_DIR/summary.json" ]; then
    log "  skipping -- $PHASE2_DIR/summary.json exists"
else
    python experiments/07_lambda_sweep/run_can.py \
        --task can \
        --dataset "$DATASET" \
        --prior flow \
        --prior-checkpoint "$PRIOR_CKPT" \
        --no-warm-start \
        --lambda-prior 0.0 0.5 2.0 \
        --n-demos 20 --n-rollouts 3 \
        --H 120 --M 16 --K 10 \
        --sigma-max 0.5 --sigma-min 0.05 \
        --seed 0 \
        --n-workers 10 \
        --output-dir "$PHASE2_DIR" \
        --no-live-view 2>&1 | tee -a "$LOG"
fi


# ---------- Phase 3 (DISABLED at runtime) -----------------------------------
# Skipped because Phase 0's 8.3 % prior-only finding promoted Phase 5
# (sigma-preservation cross lambda) to highest priority.  K-extension at
# the wrong sigma is unlikely to help.  See sweep_v2_ext.sh.
PHASE3_DIR="$OUT/phase3_best_lambda_K_extension"
mkdir -p "$PHASE3_DIR"
log ""
log "=== Phase 3: SKIPPED at runtime (low priority given Phase 0 finding) ==="
if true; then
    echo "[]" > "$PHASE3_DIR/summary.json"
elif [ -f "$PHASE3_DIR/summary.json" ]; then
    log "  skipping -- $PHASE3_DIR/summary.json exists"
elif [ ! -f "$PHASE1_DIR/summary.json" ]; then
    log "  SKIPPING -- Phase 1 summary not available"
else
    BEST_LAM=$(python - <<PY
import json
summary = json.load(open("$PHASE1_DIR/summary.json"))
# Pick the lambda with the highest n_success, breaking ties by smallest lambda.
best = max(summary, key=lambda r: (r["n_success"], -r["lambda_prior"]))
print(best["lambda_prior"])
PY
)
    log "  best lambda from Phase 1: $BEST_LAM"
    for K in 20 30; do
        K_DIR="$PHASE3_DIR/K_${K}"
        mkdir -p "$K_DIR"
        if [ -f "$K_DIR/summary.json" ]; then
            log "  K=$K already done; skipping"
            continue
        fi
        log "  --- K=$K at lambda=$BEST_LAM ---"
        python experiments/07_lambda_sweep/run_can.py \
            --task can \
            --dataset "$DATASET" \
            --prior flow \
            --prior-checkpoint "$PRIOR_CKPT" \
            --warm-start \
            --lambda-prior "$BEST_LAM" \
            --n-demos 20 --n-rollouts 3 \
            --H 120 --M 16 --K "$K" \
            --sigma-max 0.5 --sigma-min 0.05 \
            --seed 0 \
            --n-workers 10 \
            --output-dir "$K_DIR" \
            --no-live-view 2>&1 | tee -a "$LOG"
    done
    # Synthesize a Phase 3 summary covering both K values for convenience.
    python - <<PY
import json
from pathlib import Path
parts = []
for K in (20, 30):
    s = Path("$PHASE3_DIR") / f"K_{K}" / "summary.json"
    if s.exists():
        for row in json.load(open(s)):
            row["K"] = K
            parts.append(row)
json.dump(parts, open("$PHASE3_DIR/summary.json", "w"), indent=2)
PY
fi


# ---------- Phase 4 (DISABLED at runtime) -----------------------------------
# Skipped because Phase 5 (sweep_v2_ext.sh) covers sigma_max ∈ {0.02, 0.05, 0.1}
# crossed with lambda ∈ {0.0, 0.1, 0.5} -- strictly better than Phase 4's
# sigma_max ∈ {0.1, 0.25} at a single best-lambda.
PHASE4_DIR="$OUT/phase4_best_lambda_sigma_sweep"
mkdir -p "$PHASE4_DIR"
log ""
log "=== Phase 4: SKIPPED at runtime (superseded by Phase 5 in sweep_v2_ext.sh) ==="
if true; then
    echo "[]" > "$PHASE4_DIR/summary.json"
elif [ -f "$PHASE4_DIR/summary.json" ]; then
    log "  skipping -- $PHASE4_DIR/summary.json exists"
elif [ ! -f "$PHASE1_DIR/summary.json" ]; then
    log "  SKIPPING -- Phase 1 summary not available"
else
    BEST_LAM=$(python - <<PY
import json
summary = json.load(open("$PHASE1_DIR/summary.json"))
best = max(summary, key=lambda r: (r["n_success"], -r["lambda_prior"]))
print(best["lambda_prior"])
PY
)
    log "  best lambda from Phase 1: $BEST_LAM"
    for SIG in 0.1 0.25; do
        SIG_DIR="$PHASE4_DIR/sigma_${SIG}"
        mkdir -p "$SIG_DIR"
        if [ -f "$SIG_DIR/summary.json" ]; then
            log "  sigma_max=$SIG already done; skipping"
            continue
        fi
        log "  --- sigma_max=$SIG at lambda=$BEST_LAM ---"
        python experiments/07_lambda_sweep/run_can.py \
            --task can \
            --dataset "$DATASET" \
            --prior flow \
            --prior-checkpoint "$PRIOR_CKPT" \
            --warm-start \
            --lambda-prior "$BEST_LAM" \
            --n-demos 20 --n-rollouts 3 \
            --H 120 --M 16 --K 10 \
            --sigma-max "$SIG" --sigma-min 0.02 \
            --seed 0 \
            --n-workers 10 \
            --output-dir "$SIG_DIR" \
            --no-live-view 2>&1 | tee -a "$LOG"
    done
    python - <<PY
import json
from pathlib import Path
parts = []
for SIG in (0.1, 0.25):
    s = Path("$PHASE4_DIR") / f"sigma_{SIG}" / "summary.json"
    if s.exists():
        for row in json.load(open(s)):
            row["sigma_max"] = SIG
            parts.append(row)
json.dump(parts, open("$PHASE4_DIR/summary.json", "w"), indent=2)
PY
fi


# ---------- Phase 5: Aggregate -----------------------------------------------
log ""
log "=== Phase 5: aggregate -> v2_sweep_report.md ==="
python experiments/08_flow_matching_prior/sweep_v2_aggregate.py "$OUT" 2>&1 | tee -a "$LOG"

log ""
log "=== sweep_v2 MAIN DONE -- chaining to sweep_v2_ext.sh ==="

# Chain to extension (Phase 5+).  sweep_v2_ext.sh is idempotent.
if [ -f "$REPO/experiments/08_flow_matching_prior/sweep_v2_ext.sh" ]; then
    bash "$REPO/experiments/08_flow_matching_prior/sweep_v2_ext.sh"
else
    log "  no extension script found; stopping after main sweep."
fi

log ""
log "=== sweep_v2 ALL DONE ==="
