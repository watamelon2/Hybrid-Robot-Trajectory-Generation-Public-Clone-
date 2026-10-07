#!/usr/bin/env bash
# Slate J -- tool_hang + transport recovery with dense reward shaping
# (May 22 daytime, manual fire).
#
# Motivation
# ----------
# Slate H H1 (2026-05-21 16:04 - 19:19) returned 0/40 on tool_hang.
# Mid-slate investigation found that both robosuite-1.5's `ToolHang`
# and `TwoArmTransport` have **hard-coded sparse** rewards
# (the `reward_shaping=True` flag is honored at __init__ but never
# consulted in reward(); both files carry literal `# TODO!` comments
# where the dense branch would live).
#
# Fix landed in commit <to-be-tagged>:
#   - experiments/15_unified_hybrid/env_dense_rewards.py:
#     monkey-patches env.step to compute a staged distance +
#     subgoal-indicator shaping for tool_hang and transport.
#   - experiments/15_unified_hybrid/run_can.py main():
#     calls wrap_env_with_dense_reward(env, args.task) right
#     after make_env_for_dataset(...).  No-op for can / lift /
#     square (which actually have dense shaping upstream).
#
# Slate J validates the fix and produces the first scored
# tool_hang + transport cells in the project.
#
# Two tracks, 4 cells, ~10-12 h wall:
#
#   J1 (~5-6 h, 2 cells)
#     demo_1nn lambda-sweep on `tool_hang/ph` at the H1
#     operating point.  lambda in {0, 1} at sigma=0.25 seed=0,
#     20 demos x 2 rollouts = 40 rollouts/cell, K=10 R=1 M=16
#     H=300, n-workers=10.
#
#     Acceptance:
#       - STRONG  : at least one cell with mean reward >= 5
#                   and any non-zero success.  Validates that
#                   the dense shaping is doing useful work and
#                   that the score-bridge can leverage it.
#       - WEAK    : mean reward in [2, 5], success 0.  Means
#                   the shaping is firing (early-phase signal)
#                   but H=300 is too short to reach hang state.
#                   Recommend H=500 retry in a follow-on slate.
#       - FAIL    : mean reward < 2 at both lambdas.  Indicates
#                   the shaping function is mis-calibrated
#                   (probably the weights are too small) or the
#                   prior is not informative on tool_hang.
#
#   J2 (~5-6 h, 2 cells)
#     demo_1nn lambda-sweep on `transport/ph` at the parallel
#     point.  Same operating point as J1.  Transport demos
#     average T~714 so H=300 covers a similar fraction (~42 %)
#     as tool_hang.
#
#     Acceptance:
#       - STRONG  : at least one cell with mean reward >= 3
#                   (transport shaping max per-step is 2 in
#                   the approach phase; 5 in either subgoal;
#                   22 at full success).  Validates dense
#                   shaping + cross-task transfer to dual-arm.
#       - WEAK    : mean reward in [1, 3].  Shaping fires for
#                   one subgoal but not both; expected when
#                   only one of the two arms can be driven by
#                   a single-arm prior.  Transport demos use
#                   a 14-d action (two 7-d arms); demo_1nn
#                   retrieves the full 14-d window so both
#                   arms get prior actions, but the prior may
#                   not coordinate them perfectly.
#       - FAIL    : mean reward < 1 at both lambdas.  Indicates
#                   the dual-arm dispatch is broken or the
#                   demo_1nn retrieval is not finding viable
#                   neighbours (transport bank has only 200
#                   demos and they're 700+ steps each, so the
#                   bank is large but the obs space is also
#                   the largest we've seen at ~70 dims).
#
# Engineering posture:
#   - Wall ~10-12 h sequential.  Fire AFTER both Slate H and
#     Slate I have finished (the box has 16 cores and Slate I
#     uses 10 workers; J would contend).
#   - The cells are written as TWO separate run_can invocations
#     (one per task) so a partial failure on tool_hang doesn't
#     skip transport.
#   - Lambda sweep within each task is a SINGLE invocation
#     (lambdas in {0, 1}) so the bank + env load only once per
#     task.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_j_2026_05_22"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate J: tool_hang (J1) + transport (J2) recovery with dense reward shaping" | tee -a "$LOG"
echo "Total ~10-12 h wall, 4 cells.  Fire AFTER Slate H + Slate I have finished." | tee -a "$LOG"

# Safety: refuse to start if other slates are still hot.  The box has
# 16 cores; concurrent slates would contend.
for SCREEN in slate_h slate_i; do
  if screen -ls 2>/dev/null | grep -qE "\.${SCREEN}[[:space:]]"; then
    echo "FATAL: $SCREEN screen still alive.  Wait for it to finish " | tee -a "$LOG"
    echo "       (bash experiments/15_unified_hybrid/results/slate_g_2026_05_21/monitor.sh)." | tee -a "$LOG"
    exit 2
  fi
done
echo "No competing slates; OK to start." | tee -a "$LOG"

# ----- J1: tool_hang lambda-sweep at H=300 ---------------------------
echo "##### J1: tool_hang demo_1nn lambda-sweep (dense reward) #####" | tee -a "$LOG"
TAG="J1_toolhang_1nn_lamsweep_dense"
echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task tool_hang \
  --mpc-replans 1 --mpc-chunk-steps 60 \
  --K 10 --M 16 --H 300 \
  --sigma-max 0.25 --sigma-min 0.01 \
  --n-demos 20 --n-rollouts 2 \
  --n-workers 10 \
  --output-dir "$OUTDIR" \
  --output-name-suffix "_$TAG" \
  --prior demo_1nn --lambda-prior 0 1 \
  --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

# ----- J2: transport lambda-sweep at H=300 ---------------------------
echo "##### J2: transport demo_1nn lambda-sweep (dense reward) #####" | tee -a "$LOG"
TAG="J2_transport_1nn_lamsweep_dense"
echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task transport \
  --mpc-replans 1 --mpc-chunk-steps 60 \
  --K 10 --M 16 --H 300 \
  --sigma-max 0.25 --sigma-min 0.01 \
  --n-demos 20 --n-rollouts 2 \
  --n-workers 10 \
  --output-dir "$OUTDIR" \
  --output-name-suffix "_$TAG" \
  --prior demo_1nn --lambda-prior 0 1 \
  --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
