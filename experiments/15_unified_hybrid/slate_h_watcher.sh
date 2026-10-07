#!/usr/bin/env bash
# Watcher: polls for the `slate_g` screen to disappear, then runs
# Slate H.  Launch this in its own detached screen so it survives
# terminal exits.
#
# Usage:
#   screen -dmS slate_h_watcher bash experiments/15_unified_hybrid/slate_h_watcher.sh

set -u

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
SLATE_H="$REPO/experiments/15_unified_hybrid/slate_h_2026_05_21.sh"
LOG="$REPO/experiments/15_unified_hybrid/results/slate_h_2026_05_21/watcher.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"

{
  echo "=== watcher start  $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  # Poll until slate_g screen is gone.  Don't rely on screen's exit
  # code (BSD quirk); use a non-empty match instead.
  while screen -ls 2>/dev/null | grep -E '\.slate_g[[:space:]]' >/dev/null; do
    echo "[watch $(date '+%H:%M:%S')] slate_g still running, sleeping 120s"
    sleep 120
  done
  echo "=== slate_g finished at $(date '+%H:%M:%S'); launching Slate H ==="
  bash "$SLATE_H"
  echo "=== watcher done at $(date '+%H:%M:%S') ==="
} >> "$LOG" 2>&1
