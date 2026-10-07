#!/usr/bin/env bash
# Watcher: polls for the `slate_f` screen to disappear, then runs
# the Square budget-retry cell.  Launch this in its own detached
# screen so it survives terminal exits.
#
# Usage:
#   screen -dmS slate_f_retry bash experiments/15_unified_hybrid/slate_f_followup_watcher.sh

set -u

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
RETRY="$REPO/experiments/15_unified_hybrid/slate_f_square_retry_2026_05_20.sh"
LOG="$REPO/experiments/15_unified_hybrid/results/slate_f_square_retry_2026_05_20/watcher.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO"

{
  echo "=== watcher start  $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  # Poll until slate_f screen is gone.  Don't rely on screen's exit
  # code (BSD quirk); use a non-empty match instead.
  while screen -ls 2>/dev/null | grep -E '\.slate_f[[:space:]]' >/dev/null; do
    echo "[watch $(date '+%H:%M:%S')] slate_f still running, sleeping 60s"
    sleep 60
  done
  echo "=== slate_f finished at $(date '+%H:%M:%S'); launching retry ==="
  bash "$RETRY"
  echo "=== watcher done at $(date '+%H:%M:%S') ==="
} >> "$LOG" 2>&1
