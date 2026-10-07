#!/bin/bash
# Slate O (2026-05-25, Tier-1 follow-up to Slate N) -- pure v6 standalone, multi-seed.
#
# WHY:
# Slate L/M/N closed on "v6 + MBD-pull is statistically indistinguishable from
# v6 standalone within seed noise" using K3 (K=10 M=16 sigma=0.25 lam=1 pull) as
# the v6-standalone proxy.  This is theoretically correct (pull at lam=1 cancels
# MBD noise to first order) but mixes in an M=16 rank-and-pick selection that
# could contribute a 5-10 pp lift.  Slate N's K=1 M=1 sigma=0.05 cells were
# intended as standalone proxies but the sigma=0.05 perturbation introduced
# 27 pp of cell-to-cell variance, making them unusable as a clean reference.
#
# Slate O measures TRUE pure v6 standalone:
#   K=1, M=1, lambda=0, sigma_max=sigma_min=1e-4 (effectively zero)
#   prior-n-steps=18, max-env-steps=240, deterministic-prior-warmstart
#   seeds = 0, 1, 2
#
# Decision rule:
#   If pool ~ 81 % (matches v6+pull 3-seed pool): parity claim is fully clean,
#     MBD-pull adds nothing over standalone.
#   If pool < 75 %: v6+pull's 81.1 % includes a non-negligible M=16 rank-and-pick
#     lift over true standalone; Slate M parity table needs updating to credit
#     MBD with that lift.
#
# Cost: 3 cells * ~5 min = ~15 min total.

set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PY="/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
OUT="experiments/08_flow_matching_prior/results/eval_dp_ema_e300"
V6_CKPT="experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt"
QLOG="/tmp/wave_queue_slate_o.log"

echo "[$(date +'%H:%M:%S')] Slate O queue start (PID $$)" > "$QLOG"

# Wait for any other live run_can.py in HRTG.
echo "[$(date +'%H:%M:%S')] waiting for any live run_can.py in HRTG" >> "$QLOG"
while [ "$(ps -axo command | grep -E 'run_can\.py' | grep -cE 'Hybrid-Robot-Trajectory-Generation')" -gt 0 ]; do
    sleep 30
done
echo "[$(date +'%H:%M:%S')] all clear; starting Slate O" >> "$QLOG"

v6_pure_standalone() {
    local seed="$1"; shift
    local tag="slateO_pure_standalone_seed${seed}"
    local logf="$OUT/run_${tag}.log"
    echo "[$(date +'%H:%M:%S')] launching: $tag" >> "$QLOG"
    "$PY" -u experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint "$V6_CKPT" \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 1 --M 1 \
        --mpc-replans 0 --mpc-chunk-steps 120 \
        --max-env-steps 240 \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max 0.0001 --sigma-min 0.0001 \
        --prior-n-steps 18 \
        --lambda-prior 0.0 \
        --lambda-schedule constant \
        --score-mode pull \
        --warmstart-noise-sigma 0.0 \
        --n-workers 4 --device cpu --seed "$seed" \
        --output-dir "$OUT" \
        --output-name-suffix "_${tag}" \
        < /dev/null \
        > "$logf" 2>&1
    local rc=$?
    echo "[$(date +'%H:%M:%S')] done: $tag (rc=$rc)" >> "$QLOG"
    grep "wrote" "$logf" | tail -1 >> "$QLOG"
}

for seed in 0 1 2; do
    v6_pure_standalone "$seed"
done

echo "===================================================================" >> "$QLOG"
echo "SLATE O COMPLETE -- pure v6 standalone, 3 seeds" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
for seed in 0 1 2; do
    tag="slateO_pure_standalone_seed${seed}"
    printf "%-40s " "$tag" >> "$QLOG"
    grep "wrote" "$OUT/run_${tag}.log" 2>/dev/null | tail -1 >> "$QLOG"
done
echo "[$(date +'%H:%M:%S')] Slate O queue done" >> "$QLOG"
