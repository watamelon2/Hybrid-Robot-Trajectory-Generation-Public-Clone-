#!/usr/bin/env bash
# Slate H -- tool_hang end-to-end + A1 generalisation on Can/Lift
# (May 21, queued for post-Slate-G launch).
#
# Pre-staged at Slate G engineering time.  DO NOT LAUNCH until Slate
# G finishes -- this slate exercises features Slate G added but did
# not fully cross-validate.  Total ~4.6 h wall (7 cells).
#
# Open questions Slate G does NOT address:
#
#   1. Does the new task `tool_hang` run end-to-end?  Added to
#      dataset.py during Slate G prep but the G launcher only
#      uses `lift`.  Tool_hang demos are very long (T~681), so
#      Slate G's H=120 would trivially fail; H1 uses H=300.
#      `transport` (also wired in dataset.py) is DEFERRED from
#      this slate -- the smoke at 2:26 PT surfaced a robosuite
#      warning: "No dense reward currently implemented for this
#      task. Forcing sparse rewards."  With sparse rewards and
#      demos T~714, an H=300 scored cell would near-certainly
#      hit 0/40 = 0 %; the wall (~3.8 h) is not justified by
#      the signal.  Transport needs a dense-reward shaping
#      function (Slate I prerequisite).
#
#   2. Is the A1 action-history augmentation a regression on the
#      existing headline tasks?  Slate G's G4 only tests A1 on
#      Square (3.3 % mechanism-limited ceiling), so the headline-
#      task control is missing.  H3 (Can) and H4 (Lift) sweep
#      `--action-history-len` at the canonical operating point;
#      acceptance is "within +/- 5 pp of N=0 baseline" -- any
#      large drop would indicate the augmented obs hurts more
#      than it helps when retrieval already works.
#
# Three tracks, 7 cells, ~4.6 h:
#
#   H1 (~2 h, 1 cell): tool_hang demo_1nn lambda=1 sigma=0.25
#       seed=0 H=300, 20 demos x 2 rollouts = 40 rollouts.
#       Validates the task end-to-end.  Acceptance: wall < 3 h
#       and mean reward > 0 (any non-trivial progress).  Tool_hang
#       has dense shaped reward in robosuite, so progress should
#       show even without full success.
#
#   H3 (~1.3 h, 3 cells): A1 control on Can.  demo_1nn lambda=1
#       sigma=0.25 seed=0, --action-history-len in {1, 5, 10}.
#       Baseline at N=0: 75.0 % single-seed (Slate D D2 / Slate F
#       F3 reproductions).  Acceptance: all three cells within
#       +/- 5 pp of baseline -- A1 should be near-no-op on a task
#       where retrieval already works without contact-phase
#       disambiguation.  A big drop would falsify Slate G's G4
#       hypothesis (would indicate action-history is generically
#       harmful, not Square-specific helpful).
#
#   H4 (~1.3 h, 3 cells): A1 control on Lift.  Same shape as H3.
#       Lift baseline depends on G1 result; acceptance set
#       relative to G1's N=0 cell (i.e., demo_1nn lambda=1 on
#       Lift).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_h_2026_05_21"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate H: tool_hang (H1) + A1-on-Can (H3) + A1-on-Lift (H4) -- transport DEFERRED (sparse reward)" | tee -a "$LOG"
echo "Total ~4.6 h wall, 7 cells.  PRECONDITION: Slate G must be done." | tee -a "$LOG"

# Defensive check: refuse to start if Slate G screen still exists.
if screen -ls 2>/dev/null | grep -q "slate_g"; then
  echo "ABORT: 'slate_g' screen session still running.  Wait for Slate G to finish." | tee -a "$LOG"
  exit 1
fi

# ----- H1: tool_hang end-to-end at canonical (long-horizon) ----------
# H=300 is conservative; tool_hang demos avg 681 steps so even 300
# only covers ~44 % of a typical demo, but should be enough for the
# "pick up the tool" phase.  20 demos x 2 rollouts = 40 rollouts.
echo "##### H1: tool_hang demo_1nn lambda=1 #####" | tee -a "$LOG"
TAG="H1_toolhang_1nn_lam1_H300"
echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
python "$REPO/experiments/15_unified_hybrid/run_can.py" \
  --task tool_hang \
  --mpc-replans 1 --mpc-chunk-steps 150 \
  --K 10 --M 16 --H 300 \
  --sigma-max 0.25 --sigma-min 0.01 \
  --n-demos 20 --n-rollouts 2 \
  --n-workers 10 \
  --output-dir "$OUTDIR" \
  --output-name-suffix "_$TAG" \
  --prior demo_1nn --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"

# H2 (transport) DEFERRED:  robosuite has no dense reward for
# transport (verified 2026-05-21 02:26 PT).  Sparse-reward at
# H=300 with demos T~714 would produce 0/40 with high probability;
# wall ~3.8 h is not justified.  Re-enable when a dense reward
# shaping function lands (Slate I prerequisite).

# ----- H3: A1 control on Can at lambda=1 -----------------------------
# Bank rebuilt once per action-history-len (cache key includes it).
echo "##### H3: A1 on Can #####" | tee -a "$LOG"
for HIST in 1 5 10; do
  TAG="H3_can_1nn_lam1_hist${HIST}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 \
    --action-history-len "$HIST" --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- H4: A1 control on Lift at lambda=1 ----------------------------
echo "##### H4: A1 on Lift #####" | tee -a "$LOG"
for HIST in 1 5 10; do
  TAG="H4_lift_1nn_lam1_hist${HIST}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task lift \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 \
    --action-history-len "$HIST" --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
