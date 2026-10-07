#!/usr/bin/env bash
# Sigma-max warm-start strength curve.
#
# Sweeps the cosine schedule's top sigma over a small range while
# keeping everything else identical to Exp 05's headline. Tests the
# "BC structure preservation vs. planner freedom" trade-off
# narrated in the Exp 05 README. One JSON per sigma_max value;
# ../06_failure_taxonomy/run.py turns them into the bucketised
# table for the report.
#
# Wall: ~80 min/value with --n-demos 20 --n-rollouts 3 (sequential
# sim). Set N_DEMOS=2 N_ROLLOUTS=1 for a wiring smoke (~5 min total).

set -euo pipefail
cd "$(dirname "$0")/../.."

SIGMAS=${SIGMAS:-"0.05 0.10 0.25 0.40 0.50"}
N_DEMOS=${N_DEMOS:-20}
N_ROLLOUTS=${N_ROLLOUTS:-3}
SEED=${SEED:-0}
N_WORKERS=${N_WORKERS:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}
OUT_DIR=${OUT_DIR:-experiments/09_sigma_sweep/results}
mkdir -p "$OUT_DIR"

for sigma in $SIGMAS; do
    OUT_JSON="$OUT_DIR/sigma_${sigma}.json"
    if [[ "$SKIP_EXISTING" == "1" && -f "$OUT_JSON" ]]; then
        echo "=== sigma_max=$sigma  SKIPPED (output exists) ==="
        continue
    fi
    echo "=== sigma_max=$sigma ==="
    python experiments/05_hybrid_method/run_can.py \
        --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" \
        --seed "$SEED" --no-live-view \
        --sigma-max "$sigma" \
        --n-workers "$N_WORKERS" \
        --deterministic-bc-reset \
        --output "$OUT_JSON" \
        --video-dir ""
done

python experiments/06_failure_taxonomy/run.py \
    --inputs "$OUT_DIR"/sigma_*.json \
    --output "$OUT_DIR/taxonomy.json"
echo "[exp09] done"
