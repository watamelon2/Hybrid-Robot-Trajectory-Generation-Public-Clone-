#!/bin/bash
# Wave queue for Slate L (2026-05-25): four levers on v6+pull to test
# whether anything pushes past the 83.3% standalone / parity ceiling.
#
# Background: K3 found v6+pull at lam=1, sigma_max=0.25, K=10, M=16,
# R=1 reaches *exact* parity with v6 standalone (25/30 = 83.3%).
# K4 lambda-sweep monotone-decreased below lam=1; K5 K/M doubling
# made things worse.  This queue tests four untried axes:
#
#   Wave A: sigma sweep (sigma_max < 0.25)  -- less noise variance
#   Wave B: chunked v6 R=4 + pull           -- closed-loop feedback
#   Wave C: time-varying lambda             -- search early, anchor late
#   Wave D: warm-start corruption probe     -- rescue regime test
#
# Per cell: 30 trials = 10 demos x 3 rollouts.  Cells run in pairs
# (2 at a time) for parallelism on the 12-core M4 Max.  Each cell
# uses --n-workers 4.

set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PY="/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
OUT="experiments/08_flow_matching_prior/results/eval_dp_ema_e300"
V6_CKPT="experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt"

QLOG="/tmp/wave_queue_v6.log"
echo "[$(date +'%H:%M:%S')] queue start (PID $$)" > "$QLOG"

# Common args for a single-shot v6+pull cell at H=120 R=1.
v6_pull_singleshot() {
    local tag="$1"; shift
    local sigma_max="$1"; shift
    local sigma_min="$1"; shift
    local lam="$1"; shift
    local lamschedule="$1"; shift
    local noise_sigma="$1"; shift
    local logf="$OUT/run_${tag}.log"
    echo "[$(date +'%H:%M:%S')] launching: $tag  (sigma_max=$sigma_max lam=$lam schedule=$lamschedule noise=$noise_sigma)" >> "$QLOG"
    "$PY" -u experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint "$V6_CKPT" \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 10 --M 16 \
        --mpc-replans 0 --mpc-chunk-steps 120 \
        --max-env-steps 240 \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max "$sigma_max" --sigma-min "$sigma_min" \
        --prior-n-steps 18 \
        --lambda-prior "$lam" \
        --lambda-schedule "$lamschedule" \
        --score-mode pull \
        --warmstart-noise-sigma "$noise_sigma" \
        --n-workers 4 --device cpu --seed 0 \
        --output-dir "$OUT" \
        --output-name-suffix "_${tag}" \
        < /dev/null \
        > "$logf" 2>&1
    local rc=$?
    echo "[$(date +'%H:%M:%S')] done: $tag (rc=$rc)" >> "$QLOG"
    grep "wrote" "$logf" | tail -1 >> "$QLOG"
}

# Chunked closed-loop variant.
v6_pull_chunked() {
    local tag="$1"; shift
    local replans="$1"; shift
    local chunk_steps="$1"; shift
    local lam="$1"; shift
    local sigma_max="$1"; shift
    local sigma_min="$1"; shift
    local logf="$OUT/run_${tag}.log"
    echo "[$(date +'%H:%M:%S')] launching: $tag  (R=$replans chunk=$chunk_steps lam=$lam)" >> "$QLOG"
    "$PY" -u experiments/15_unified_hybrid/run_can.py \
        --task can --prior flow \
        --prior-checkpoint "$V6_CKPT" \
        --n-demos 10 --n-rollouts 3 \
        --H 120 --K 10 --M 16 \
        --mpc-replans "$replans" --mpc-chunk-steps "$chunk_steps" \
        --max-env-steps 240 \
        --warm-start --deterministic-prior-warmstart \
        --sigma-max "$sigma_max" --sigma-min "$sigma_min" \
        --prior-n-steps 18 \
        --lambda-prior "$lam" \
        --lambda-schedule constant \
        --score-mode pull \
        --warmstart-noise-sigma 0.0 \
        --n-workers 4 --device cpu --seed 0 \
        --output-dir "$OUT" \
        --output-name-suffix "_${tag}" \
        < /dev/null \
        > "$logf" 2>&1
    local rc=$?
    echo "[$(date +'%H:%M:%S')] done: $tag (rc=$rc)" >> "$QLOG"
    grep "wrote" "$logf" | tail -1 >> "$QLOG"
}

# dppull_lam2p0 already finished by the time this restart launches;
# just confirm no run_can.py is alive before starting.
echo "[$(date +'%H:%M:%S')] checking for any live run_can.py" >> "$QLOG"
while [ "$(ps -axo command | grep -cE '^[^|]*run_can\.py' 2>/dev/null)" -gt 0 ]; do
    sleep 30
done
echo "[$(date +'%H:%M:%S')] no run_can.py alive; starting parallel waves" >> "$QLOG"

###########################################################################
# Wave A: sigma_max sweep on v6+pull
###########################################################################
echo "===================================================================" >> "$QLOG"
echo "WAVE A: v6+pull sigma sweep (lam=1, K=10, M=16, R=1)" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
v6_pull_singleshot waveA_sig005 0.05 0.01 1.0 constant 0.0 &
PID1=$!
v6_pull_singleshot waveA_sig010 0.10 0.02 1.0 constant 0.0 &
PID2=$!
wait $PID1
wait $PID2
v6_pull_singleshot waveA_sig015 0.15 0.03 1.0 constant 0.0

###########################################################################
# Wave C: time-varying lambda (sigma_max=0.25 canonical, lam=1, R=1)
###########################################################################
echo "===================================================================" >> "$QLOG"
echo "WAVE C: v6+pull time-varying lambda schedules" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
v6_pull_singleshot waveC_lin     0.25 0.05 1.0 linear         0.0 &
PID1=$!
v6_pull_singleshot waveC_invlin  0.25 0.05 1.0 inverse_linear 0.0 &
PID2=$!
wait $PID1
wait $PID2
v6_pull_singleshot waveC_cosine  0.25 0.05 1.0 cosine         0.0

###########################################################################
# Wave D: warm-start corruption probe (canonical sigma + lam=1, R=1)
###########################################################################
echo "===================================================================" >> "$QLOG"
echo "WAVE D: v6+pull warm-start corruption probe" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
v6_pull_singleshot waveD_n0p3 0.25 0.05 1.0 constant 0.3 &
PID1=$!
v6_pull_singleshot waveD_n0p5 0.25 0.05 1.0 constant 0.5 &
PID2=$!
wait $PID1
wait $PID2
v6_pull_singleshot waveD_n1p0 0.25 0.05 1.0 constant 1.0

###########################################################################
# Wave B: chunked v6 + pull (closed-loop feedback test)
###########################################################################
echo "===================================================================" >> "$QLOG"
echo "WAVE B: v6+pull chunked closed-loop (R=4 chunk=24 = exec all 120 steps)" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
v6_pull_chunked waveB_R4_lam1 4 24 1.0 0.25 0.05

###########################################################################
echo "===================================================================" >> "$QLOG"
echo "ALL WAVES COMPLETE -- summary" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
for tag in waveA_sig005 waveA_sig010 waveA_sig015 \
           waveC_lin waveC_invlin waveC_cosine \
           waveD_n0p3 waveD_n0p5 waveD_n1p0 \
           waveB_R4_lam1; do
    printf "%-22s " "$tag" >> "$QLOG"
    grep "wrote" "$OUT/run_${tag}.log" 2>/dev/null | tail -1 >> "$QLOG"
done
echo "[$(date +'%H:%M:%S')] queue done" >> "$QLOG"
