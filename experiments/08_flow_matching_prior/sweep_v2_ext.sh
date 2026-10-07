#!/usr/bin/env bash
# Extension to sweep_v2.sh -- runs AFTER the main sweep completes.
# Triggered by sweep_v2.sh's tail line. Idempotent.
#
# Phase 5: sigma-preservation sweep (sigma_max ∈ {0.02, 0.05, 0.1})
#   The Phase 0 finding (prior-only replay = 8.3 %) showed MBD's
#   default sigma_max=0.5 destroys the prior's good warm-starts.
#   This phase explores sigma values low enough to preserve them.
#   Crossed with lambda ∈ {0.0, 0.1, 0.5} since the sigma optimum
#   could shift with lambda.
#
# Phase 6: best-config (lambda, sigma) + K-extension
#   If Phase 5 finds a (lambda, sigma) > 8.3 %, run that at K=20.
#
# Phase 7: aggregate everything into v2_sweep_report.md.

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

log ""
log "=== sweep_v2_EXT START ==="

# ---------- Phase 5: sigma-preservation sweep -------------------------------
PHASE5_DIR="$OUT/phase5_sigma_preservation"
mkdir -p "$PHASE5_DIR"
log ""
log "=== Phase 5: sigma-preservation sweep (sigma_max ∈ {0.02, 0.05, 0.1}; ~110 min) ==="
log "    Each sigma cell runs lambda-sweep ∈ {0.0, 0.1, 0.5} (paired noise)."
log "    Motivated by Phase 0 finding: prior alone = 8.3 %, MBD@sigma=0.5 kills it."
if [ -f "$PHASE5_DIR/summary.json" ]; then
    log "  skipping -- $PHASE5_DIR/summary.json exists"
else
    for SIG in 0.02 0.05 0.1; do
        SIG_DIR="$PHASE5_DIR/sigma_${SIG}"
        mkdir -p "$SIG_DIR"
        if [ -f "$SIG_DIR/summary.json" ]; then
            log "  sigma_max=$SIG already done; skipping"
            continue
        fi
        log "  --- sigma_max=$SIG, lambda ∈ {0.0, 0.1, 0.5} ---"
        # sigma_min = sigma_max * 0.1 (same shape ratio as the default schedule)
        SIG_MIN=$(python -c "print(max(0.01, $SIG * 0.1))")
        python experiments/07_lambda_sweep/run_can.py \
            --task can \
            --dataset "$DATASET" \
            --prior flow \
            --prior-checkpoint "$PRIOR_CKPT" \
            --warm-start \
            --lambda-prior 0.0 0.1 0.5 \
            --n-demos 20 --n-rollouts 3 \
            --H 120 --M 16 --K 10 \
            --sigma-max "$SIG" --sigma-min "$SIG_MIN" \
            --seed 0 \
            --n-workers 10 \
            --output-dir "$SIG_DIR" \
            --no-live-view 2>&1 | tee -a "$LOG"
    done
    # Synthesise phase summary
    python - <<PY
import json
from pathlib import Path
parts = []
for SIG in ("0.02", "0.05", "0.1"):
    s = Path("$PHASE5_DIR") / f"sigma_{SIG}" / "summary.json"
    if s.exists():
        for row in json.load(open(s)):
            row["sigma_max"] = float(SIG)
            parts.append(row)
json.dump(parts, open("$PHASE5_DIR/summary.json", "w"), indent=2)
print(f"wrote $PHASE5_DIR/summary.json ({len(parts)} rows)")
PY
fi


# ---------- Phase 6: best (lambda, sigma) + K=20 ----------------------------
PHASE6_DIR="$OUT/phase6_best_k_extension"
mkdir -p "$PHASE6_DIR"
log ""
log "=== Phase 6: best (lambda, sigma) from Phase 5 + K-extension (K=20; ~24 min) ==="
if [ -f "$PHASE6_DIR/summary.json" ]; then
    log "  skipping -- $PHASE6_DIR/summary.json exists"
elif [ ! -f "$PHASE5_DIR/summary.json" ]; then
    log "  SKIPPING -- Phase 5 summary not available"
else
    BEST=$(python - <<PY
import json
rows = json.load(open("$PHASE5_DIR/summary.json"))
best = max(rows, key=lambda r: (r["n_success"], -r["lambda_prior"]))
print(best["sigma_max"], best["lambda_prior"], best["n_success"])
PY
)
    BEST_SIG=$(echo "$BEST" | awk '{print $1}')
    BEST_LAM=$(echo "$BEST" | awk '{print $2}')
    BEST_N=$(echo "$BEST" | awk '{print $3}')
    log "  best (sigma, lambda) from Phase 5: sigma=$BEST_SIG  lambda=$BEST_LAM  (n_success=$BEST_N)"
    if [ "$BEST_N" -gt 5 ]; then
        SIG_MIN=$(python -c "print(max(0.01, $BEST_SIG * 0.1))")
        python experiments/07_lambda_sweep/run_can.py \
            --task can \
            --dataset "$DATASET" \
            --prior flow \
            --prior-checkpoint "$PRIOR_CKPT" \
            --warm-start \
            --lambda-prior "$BEST_LAM" \
            --n-demos 20 --n-rollouts 3 \
            --H 120 --M 16 --K 20 \
            --sigma-max "$BEST_SIG" --sigma-min "$SIG_MIN" \
            --seed 0 \
            --n-workers 10 \
            --output-dir "$PHASE6_DIR" \
            --no-live-view 2>&1 | tee -a "$LOG"
    else
        log "  Phase 5 best <= 5 (= Phase 0 baseline); K-extension unlikely to help, skipping."
        # Write empty summary so we don't retry
        echo "[]" > "$PHASE6_DIR/summary.json"
    fi
fi


# ---------- Aggregate -------------------------------------------------------
log ""
log "=== Final aggregate ==="
python experiments/08_flow_matching_prior/sweep_v2_aggregate.py "$OUT" 2>&1 | tee -a "$LOG"

log ""
log "=== sweep_v2_EXT DONE ==="
