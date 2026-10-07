#!/usr/bin/env bash
# probe_slate_costtime.sh
# Read-only probe: how far along is slate_costtime in the HRTG-slate
# worktree?  Reports per-cell completion + whether the screen is alive,
# and exits 0 if the slate has finished (all 8 expected JSONs landed),
# 1 if still running, 2 if the screen is gone but the slate isn't done.
#
# Safe to invoke from any worktree; does not mutate state anywhere.
# Suggested use: `watch -n 60 bash scripts/local/probe_slate_costtime.sh`

# Intentionally no `set -e` / `set -o pipefail`: `screen -ls` exits 1
# when there are screens (counterintuitively), which would break the
# detection logic.  We check exit codes explicitly where they matter.
set -u

SLATE_WT="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-slate"
SLATE_DIR="$SLATE_WT/experiments/15_unified_hybrid/results/slate_costtime_2026_05_25"

if [ ! -d "$SLATE_DIR" ]; then
    echo "ERROR: slate dir not found: $SLATE_DIR"
    exit 3
fi

echo "============================================"
echo "slate_costtime probe @ $(date +'%Y-%m-%d %H:%M:%S %Z')"
echo "============================================"

# Screen status -- `screen -ls` exits 1 when screens exist; capture
# its full output first, then grep the captured text (avoids pipefail
# false positives).
SCREEN_OUT="$(screen -ls 2>/dev/null || true)"
if echo "$SCREEN_OUT" | grep -q "slate_costtime"; then
    SCREEN_STATE="alive"
else
    SCREEN_STATE="GONE"
fi

# Live processes
LIVE_PIDS=$(ps -A -o pid,command 2>/dev/null | grep "run_can\.py" | grep "HRTG-slate" | grep -v grep | awk '{print $1}' | tr '\n' ' ')

# Cells done (count of unified_*.json files in slate dir)
JSON_COUNT=$(find "$SLATE_DIR" -maxdepth 1 -name "unified_*.json" 2>/dev/null | wc -l | tr -d ' ')
SUMMARY_COUNT=$(find "$SLATE_DIR" -maxdepth 1 -name "unified_summary_*.json" 2>/dev/null | wc -l | tr -d ' ')

# Expected: 4 perturbations x {demo_1nn-via-list-lambda, bc} = 8 cells
# Each demo_1nn invocation writes 2 JSONs (lam=0 and lam=1), so JSON count = 4*2 + 4*1 = 12
# Or just 8 if BC writes once and demo_1nn list writes once.  Empirically, look at run.log "succ=" lines.

SUCC_LINES=$(grep -c "succ=" "$SLATE_DIR/run.log" 2>/dev/null || echo 0)

echo "Screen 'slate_costtime':  $SCREEN_STATE"
echo "Live run_can.py PIDs:     ${LIVE_PIDS:-none}"
echo "JSONs in slate dir:       $JSON_COUNT"
echo "Summary JSONs:            $SUMMARY_COUNT"
echo "Completion lines in log:  $SUCC_LINES"
echo

# Per-cell breakdown
echo "--- per-cell completion (from run.log) ---"
grep "succ=" "$SLATE_DIR/run.log" 2>/dev/null | tail -16
echo

# Latest progress
echo "--- latest rollout (last 3 [exp15] lines) ---"
grep "\[exp15\]" "$SLATE_DIR/run.log" 2>/dev/null | tail -3
echo

# Exit code logic
if [ "$SCREEN_STATE" = "alive" ] && [ -n "$LIVE_PIDS" ]; then
    echo "STATUS: still running"
    exit 1
elif [ "$SCREEN_STATE" = "GONE" ] && [ "$SUCC_LINES" -ge 8 ]; then
    echo "STATUS: DONE (>= 8 cell completion markers)"
    exit 0
else
    echo "STATUS: screen gone but only $SUCC_LINES completion markers in log -- inspect manually"
    exit 2
fi
