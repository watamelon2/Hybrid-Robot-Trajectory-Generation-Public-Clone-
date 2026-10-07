#!/usr/bin/env bash
# Run the methods that already work on Square (02 random, 04 MBD).
# 03/05/07 require the OBS_KEYS shim documented in this dir's README.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=experiments/13_square_task/results
mkdir -p "$OUT"

python -m robomimic.scripts.download_datasets \
    --tasks square --hdf5_types low_dim || true

python experiments/02_random_baseline/random_actions.py \
    --task square --n-demos 20 --n-rollouts 3 \
    --output "$OUT/exp02_square.json"

python experiments/04_mbd_baseline/run_can.py \
    --task square --n-demos 20 --n-rollouts 3 \
    --seed 0 --no-live-view \
    --output "$OUT/exp04_square.json" \
    --video-dir ""

python experiments/06_failure_taxonomy/run.py \
    --inputs "$OUT"/*.json --output "$OUT/taxonomy.json"
echo "[exp13] done"
