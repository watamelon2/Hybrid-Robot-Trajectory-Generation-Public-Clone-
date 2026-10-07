#!/usr/bin/env bash
# K x n_replans grid at fixed wall-clock budget.
#
# The Exp 05b README poses the question "is wall-time better spent
# denoising deeper (K up) or replanning more often (n_replans up)?"
# but doesn't have data to answer it. This script runs the small
# product:
#
#   K        in {5, 10, 20}
#   replans  in {0, 1, 2}
#
# at fixed (M=16, H=120, sigma_max=0.25). Approximately
# proportional-to-(K*(replans+1)) wall time. Smoke mode keeps
# K=5 only and n_demos=2 so the grid completes in a few minutes.

set -euo pipefail
cd "$(dirname "$0")/../.."

KS=${KS:-"5 10 20"}
REPLANS=${REPLANS:-"0 1 2"}
N_DEMOS=${N_DEMOS:-20}
N_ROLLOUTS=${N_ROLLOUTS:-3}
SEED=${SEED:-0}
SIGMA_MAX=${SIGMA_MAX:-0.25}
OUT_DIR=${OUT_DIR:-experiments/10_kxreplans_grid/results}
mkdir -p "$OUT_DIR"

for K in $KS; do
    for R in $REPLANS; do
        # n_replans=0 means single-shot; for fairness with the MPC
        # path we still pass --mpc-replans 0 (which exp 05 routes
        # through the single-shot branch automatically).
        out="$OUT_DIR/K${K}_R${R}.json"
        echo "=== K=$K replans=$R ==="
        python experiments/05_hybrid_method/run_can.py \
            --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" \
            --seed "$SEED" --no-live-view \
            --K "$K" \
            --sigma-max "$SIGMA_MAX" \
            --mpc-replans "$R" \
            --deterministic-bc-reset \
            --output "$out" \
            --video-dir ""
    done
done

python experiments/06_failure_taxonomy/run.py \
    --inputs "$OUT_DIR"/K*.json \
    --output "$OUT_DIR/taxonomy.json"
echo "[exp10] done"
