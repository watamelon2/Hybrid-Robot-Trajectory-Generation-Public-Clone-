#!/usr/bin/env bash
# Watcher: polls for the `slate_h` screen to disappear, then runs
# Slate I (the C1 contrastive-prior follow-up on Square + Can).
# Launch this in its own detached screen so it survives terminal
# exits.
#
# Usage:
#   screen -dmS slate_i_watcher bash experiments/15_unified_hybrid/slate_i_watcher.sh
#
# It is safe to launch this BEFORE slate_h finishes; the watcher
# just polls every 120 s.  If slate_h is already gone when the
# watcher starts, Slate I fires immediately.
#
# To cancel Slate I (e.g. you want to inspect Slate H results
# first), kill the watcher screen:
#   screen -X -S slate_i_watcher quit

set -u

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
SLATE_I="$REPO/experiments/15_unified_hybrid/slate_i_2026_05_22.sh"
LOG="$REPO/experiments/15_unified_hybrid/results/slate_i_2026_05_22/watcher.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"

{
  echo "=== watcher start  $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  # Poll until slate_h screen is gone.  Don't rely on screen's exit
  # code (BSD quirk); use a non-empty match instead.
  while screen -ls 2>/dev/null | grep -E '\.slate_h[[:space:]]' >/dev/null; do
    echo "[watch $(date '+%H:%M:%S')] slate_h still running, sleeping 120s"
    sleep 120
  done
  echo "=== slate_h finished at $(date '+%H:%M:%S'); launching Slate I ==="
  # Run inside its own detached screen so the watcher screen can
  # exit cleanly after handing off.  This mirrors the slate_h_watcher
  # convention but with a fresh slate_i screen for monitor.sh to
  # track.
  screen -dmS slate_i bash "$SLATE_I"
  echo "=== watcher handed off to slate_i screen at $(date '+%H:%M:%S') ==="
} >> "$LOG" 2>&1
