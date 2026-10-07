#!/usr/bin/env bash
# 3-seed repeat for Exp 04, 05, 05b. Tier 2c in docs/plans.md.
#
# Pools to n=180 per method; tightens Wilson 95% CI enough to
# claim p<0.05 separation from 0/180 if the published rates hold.
# Approx wall: 3 * (78 + 81 + 222) = ~19 h sequential. Easy
# overnight.

set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=${SEEDS:-"0 1 2"}
OUT=experiments/14_seed_sweep/results
mkdir -p "$OUT"

for seed in $SEEDS; do
    echo "=== seed=$seed ==="

    # Exp 04: MBD only.
    python experiments/04_mbd_baseline/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" --no-live-view \
        --output "$OUT/exp04_seed${seed}.json" \
        --video-dir ""

    # Exp 05: hybrid single-shot, with the BC reset bug fixed.
    python experiments/05_hybrid_method/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" --no-live-view \
        --deterministic-bc-reset \
        --output "$OUT/exp05_seed${seed}.json" \
        --video-dir ""

    # Exp 05b: hybrid + 1 MPC replan.
    python experiments/05_hybrid_method/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" --no-live-view \
        --mpc-replans 1 --mpc-chunk-steps 60 \
        --deterministic-bc-reset \
        --output "$OUT/exp05b_seed${seed}.json" \
        --video-dir ""
done

python experiments/14_seed_sweep/pool.py --in "$OUT" --out "$OUT/summary.json"
python experiments/06_failure_taxonomy/run.py \
    --inputs "$OUT"/*.json --output "$OUT/taxonomy.json"
echo "[exp14] done"
