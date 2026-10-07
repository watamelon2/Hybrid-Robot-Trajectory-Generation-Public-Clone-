#!/bin/bash
# Slate N (2026-05-25, draft -- DO NOT launch until Slate L + M close
# and user reviews).
#
# Slate L + Slate K together establish that v6+pull at canonical config
# (sigma=0.25, lam=1, K=10, M=16, R=0) is exactly 83.3% = v6 standalone.
# Slate L L1-L4 confirm no MBD configuration around that point exceeds
# parity.
#
# Slate N asks the *complementary* question: is the 83.3% actually
# v6's task ceiling, or a sampler / horizon artifact?  If we can
# push v6 standalone past 83.3% by changing the sampler step count
# or the execution budget, the parity claim is preserved but the
# ceiling moves up.  If we can't, 83.3% is the actual task ceiling.
#
# Cells:
#   N1: v6 standalone, prior-n-steps in {28, 50}, seed=0  (sampler density)
#   N2: v6 standalone, max-env-steps in {300, 360}, seed=0 (execution budget)
#   N3: v6+pull MBD at the WINNER of N1/N2 (does MBD also lift?)
#
# Per cell: 30 trials.  Run sequentially.
#
# Branch: feat/v6-pull-slate-l (commits cac9fd9, c47aa73).

set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PY="/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
OUT="experiments/08_flow_matching_prior/results/eval_dp_ema_e300"
V6_CKPT="experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt"
QLOG="/tmp/wave_queue_slate_n.log"

echo "[$(date +'%H:%M:%S')] Slate N queue start (PID $$)" > "$QLOG"

# Wait for ALL other run_can.py in HRTG to be done before starting.
echo "[$(date +'%H:%M:%S')] waiting for any live run_can.py in HRTG" >> "$QLOG"
while [ "$(ps -axo command | grep -E 'run_can\.py' | grep -cE 'Hybrid-Robot-Trajectory-Generation')" -gt 0 ]; do
    sleep 60
done
echo "[$(date +'%H:%M:%S')] all clear; starting Slate N" >> "$QLOG"

# Common arg set for v6 STANDALONE (lambda-prior 0, K=1, M=1).
# This disables MBD: K=1 with M=1 has a single perturbation candidate
# that gets averaged with itself; the warm-start passes through with
# a tiny noise perturbation but is otherwise the standalone trajectory.
v6_standalone() {
    local tag="$1"; shift
    local prior_n_steps="$1"; shift
    local max_env_steps="$1"; shift
    local seed="$1"; shift
    local logf="$OUT/run_${tag}.log"
    echo "[$(date +'%H:%M:%S')] launching: $tag  (n_steps=$prior_n_steps max_env=$max_env_steps seed=$seed)" >> "$QLOG"
    "$PY" -u experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint "$V6_CKPT" \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 1 --M 1 \
        --mpc-replans 0 --mpc-chunk-steps 120 \
        --max-env-steps "$max_env_steps" \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max 0.05 --sigma-min 0.01 \
        --prior-n-steps "$prior_n_steps" \
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

v6_pull_canonical() {
    local tag="$1"; shift
    local prior_n_steps="$1"; shift
    local max_env_steps="$1"; shift
    local seed="$1"; shift
    local logf="$OUT/run_${tag}.log"
    echo "[$(date +'%H:%M:%S')] launching: $tag  (PULL canonical, n_steps=$prior_n_steps max_env=$max_env_steps seed=$seed)" >> "$QLOG"
    "$PY" -u experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint "$V6_CKPT" \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 10 --M 16 \
        --mpc-replans 0 --mpc-chunk-steps 120 \
        --max-env-steps "$max_env_steps" \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max 0.25 --sigma-min 0.05 \
        --prior-n-steps "$prior_n_steps" \
        --lambda-prior 1.0 \
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

# N1: sampler density sweep on v6 standalone.  prior-n-steps=18 is
# canonical (already 83.3%); 28 and 50 test if a denser ODE
# integration of the Heun sampler gives a meaningfully better
# warm-start.
v6_standalone slateN1_nsteps28 28 240 0
v6_standalone slateN1_nsteps50 50 240 0

# N2: execution-budget sweep on v6 standalone.  max-env-steps=240
# is the K3 cap (matches DP-paper-faithful eval); 300 and 360 test
# if some of the 5/30 failures at canonical are timeout-bound rather
# than sample-quality-bound.
v6_standalone slateN2_maxenv300 18 300 0
v6_standalone slateN2_maxenv360 18 360 0

# N3: v6+pull MBD at (n_steps=50, max_env=360, seed=0) -- the
# union of N1 and N2's best directions.  Tests whether the parity
# ceiling moves up when the underlying standalone improves.
v6_pull_canonical slateN3_n50_max360_pull 50 360 0

echo "===================================================================" >> "$QLOG"
echo "SLATE N COMPLETE -- summary" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
for tag in slateN1_nsteps28 slateN1_nsteps50 slateN2_maxenv300 slateN2_maxenv360 slateN3_n50_max360_pull; do
    printf "%-32s " "$tag" >> "$QLOG"
    grep "wrote" "$OUT/run_${tag}.log" 2>/dev/null | tail -1 >> "$QLOG"
done
echo "[$(date +'%H:%M:%S')] Slate N queue done" >> "$QLOG"
