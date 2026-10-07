#!/bin/bash
# Slate M (2026-05-25, queued to chain after Slate L wave queue).
#
# Slate L exhausts the lever space *around the canonical v6+pull cell*
# and confirms parity is the ceiling.  Slate M asks the two questions
# that remain after Slate L:
#
#   M1-M2  Is the 83.3% v6+pull "parity" headline an artefact of
#          seed=0, or robust across seeds?  Two extra seeds at the
#          canonical config.  If the std-of-mean is wide, the
#          K3-K5 parity claims need a CI.  If tight, "parity is
#          the absolute ceiling" lands cleanly.
#
#   M3     Where exactly does the rescue-vs-parity crossover happen?
#          Slate E had demo_1nn (11.7% standalone -> 73.8% +MBD),
#          Slate K has v6 (83.3% standalone -> 83.3% +MBD).  An
#          intermediate point (demo_knn k=5) at the same canonical
#          config gives us a third (x, y) on the lift curve so we
#          can claim the crossover is monotone, not a v6-specific
#          artefact.
#
# Per cell: 30 trials = 10 demos x 3 rollouts.  Single-process
# (n_workers=4) to avoid contention with the user's HRTG-slate
# slate_demoscale that may still be running in parallel.

set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
cd "$REPO"

PY="/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
OUT="experiments/08_flow_matching_prior/results/eval_dp_ema_e300"
V6_CKPT="experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt"
QLOG="/tmp/wave_queue_slate_m.log"

echo "[$(date +'%H:%M:%S')] Slate M queue start (PID $$)" > "$QLOG"

# --- Step 0: wait for Slate L queue to finish (if it's still running)
echo "[$(date +'%H:%M:%S')] waiting for any live run_can.py" >> "$QLOG"
while [ "$(ps -axo command | grep -cE '^[^|]*Hybrid-Robot-Trajectory-Generation/.*run_can\.py')" -gt 0 ]; do
    sleep 60
done
echo "[$(date +'%H:%M:%S')] no live run_can.py in HRTG; starting Slate M" >> "$QLOG"

# --- M1: v6+pull seed=1 (reproducibility for the 83.3% parity headline)
echo "[$(date +'%H:%M:%S')] launching: slateM1_v6pull_seed1" >> "$QLOG"
"$PY" -u experiments/15_unified_hybrid/run_can.py \
    --task can --prior flow \
    --prior-checkpoint "$V6_CKPT" \
    --n-demos 10 --n-rollouts 3 \
    --H 120 --K 10 --M 16 \
    --mpc-replans 0 --mpc-chunk-steps 120 \
    --max-env-steps 240 \
    --warm-start --deterministic-prior-warmstart \
    --sigma-max 0.25 --sigma-min 0.05 \
    --prior-n-steps 18 \
    --lambda-prior 1.0 \
    --lambda-schedule constant \
    --score-mode pull \
    --warmstart-noise-sigma 0.0 \
    --n-workers 4 --device cpu --seed 1 \
    --output-dir "$OUT" \
    --output-name-suffix "_slateM1_v6pull_seed1" \
    < /dev/null \
    > "$OUT/run_slateM1_v6pull_seed1.log" 2>&1
echo "[$(date +'%H:%M:%S')] done: slateM1_v6pull_seed1 (rc=$?)" >> "$QLOG"
grep "wrote" "$OUT/run_slateM1_v6pull_seed1.log" | tail -1 >> "$QLOG"

# --- Re-check: wait for ALL other run_can.py (Slate L cells) to be
# done before M2-M4 launch.  M1 was launched too eagerly; this
# gate ensures M2-M4 do not pile on top of late Slate L cells.
echo "[$(date +'%H:%M:%S')] M1 done; waiting for any other live run_can.py before M2" >> "$QLOG"
while [ "$(ps -axo command | grep -E 'run_can\.py' | grep -cE 'Hybrid-Robot-Trajectory-Generation')" -gt 0 ]; do
    sleep 60
done
echo "[$(date +'%H:%M:%S')] all clear; launching M2" >> "$QLOG"

# --- M2: v6+pull seed=2 (reproducibility, third seed)
echo "[$(date +'%H:%M:%S')] launching: slateM2_v6pull_seed2" >> "$QLOG"
"$PY" -u experiments/15_unified_hybrid/run_can.py \
    --task can --prior flow \
    --prior-checkpoint "$V6_CKPT" \
    --n-demos 10 --n-rollouts 3 \
    --H 120 --K 10 --M 16 \
    --mpc-replans 0 --mpc-chunk-steps 120 \
    --max-env-steps 240 \
    --warm-start --deterministic-prior-warmstart \
    --sigma-max 0.25 --sigma-min 0.05 \
    --prior-n-steps 18 \
    --lambda-prior 1.0 \
    --lambda-schedule constant \
    --score-mode pull \
    --warmstart-noise-sigma 0.0 \
    --n-workers 4 --device cpu --seed 2 \
    --output-dir "$OUT" \
    --output-name-suffix "_slateM2_v6pull_seed2" \
    < /dev/null \
    > "$OUT/run_slateM2_v6pull_seed2.log" 2>&1
echo "[$(date +'%H:%M:%S')] done: slateM2_v6pull_seed2 (rc=$?)" >> "$QLOG"
grep "wrote" "$OUT/run_slateM2_v6pull_seed2.log" | tail -1 >> "$QLOG"

# --- M3: demo_knn k=5 + MBD-pull canonical (intermediate prior strength)
# Retrieval priors are ALWAYS pull-mode (no --score-mode plumbing needed),
# so this is just the canonical demo_knn cell at the same hyper-params
# v6+pull used.  Gives us the (standalone_demo_knn, MBD_demo_knn) point
# for the lift curve.
echo "[$(date +'%H:%M:%S')] launching: slateM3_demo_knn5_mbd" >> "$QLOG"
"$PY" -u experiments/15_unified_hybrid/run_can.py \
    --task can --prior demo_knn --knn-k 5 \
    --n-demos 10 --n-rollouts 3 \
    --H 120 --K 10 --M 16 \
    --mpc-replans 0 --mpc-chunk-steps 120 \
    --max-env-steps 240 \
    --warm-start --deterministic-prior-warmstart \
    --sigma-max 0.25 --sigma-min 0.05 \
    --lambda-prior 1.0 \
    --lambda-schedule constant \
    --warmstart-noise-sigma 0.0 \
    --n-workers 4 --device cpu --seed 0 \
    --output-dir "$OUT" \
    --output-name-suffix "_slateM3_knn5_mbd" \
    < /dev/null \
    > "$OUT/run_slateM3_knn5_mbd.log" 2>&1
echo "[$(date +'%H:%M:%S')] done: slateM3_demo_knn5_mbd (rc=$?)" >> "$QLOG"
grep "wrote" "$OUT/run_slateM3_knn5_mbd.log" | tail -1 >> "$QLOG"

# --- M4 (companion): demo_knn k=5 STANDALONE (lambda=0, no MBD pull) so
# we have the (standalone, +MBD) pair for M3.  Reuses the same warm-start
# generator (knn k=5) but disables the per-step pull score by setting
# lambda-prior to 0.  This is the "pure prior" baseline for demo_knn.
# Equivalent to executing the demo_knn warm-start under MBD with no
# pull anchor; K=1 + M=1 + lambda=0 is approximately "execute the
# warm-start" so 30 trials should pin standalone strength.
echo "[$(date +'%H:%M:%S')] launching: slateM4_demo_knn5_standalone" >> "$QLOG"
"$PY" -u experiments/15_unified_hybrid/run_can.py \
    --task can --prior demo_knn --knn-k 5 \
    --n-demos 10 --n-rollouts 3 \
    --H 120 --K 1 --M 1 \
    --mpc-replans 0 --mpc-chunk-steps 120 \
    --max-env-steps 240 \
    --warm-start --deterministic-prior-warmstart \
    --sigma-max 0.05 --sigma-min 0.01 \
    --lambda-prior 0.0 \
    --lambda-schedule constant \
    --warmstart-noise-sigma 0.0 \
    --n-workers 4 --device cpu --seed 0 \
    --output-dir "$OUT" \
    --output-name-suffix "_slateM4_knn5_standalone" \
    < /dev/null \
    > "$OUT/run_slateM4_knn5_standalone.log" 2>&1
echo "[$(date +'%H:%M:%S')] done: slateM4_demo_knn5_standalone (rc=$?)" >> "$QLOG"
grep "wrote" "$OUT/run_slateM4_knn5_standalone.log" | tail -1 >> "$QLOG"

# --- Summary
echo "===================================================================" >> "$QLOG"
echo "SLATE M COMPLETE -- summary" >> "$QLOG"
echo "===================================================================" >> "$QLOG"
for tag in slateM1_v6pull_seed1 slateM2_v6pull_seed2 slateM3_knn5_mbd slateM4_knn5_standalone; do
    printf "%-32s " "$tag" >> "$QLOG"
    grep "wrote" "$OUT/run_${tag}.log" 2>/dev/null | tail -1 >> "$QLOG"
done
echo "[$(date +'%H:%M:%S')] Slate M queue done" >> "$QLOG"
