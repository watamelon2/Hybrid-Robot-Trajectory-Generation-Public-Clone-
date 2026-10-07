#!/usr/bin/env bash
# Queue the incremental meta-slate to fire AFTER the dynamics slate
# truly completes.  Four concerns this handles:
#
#   1. The dynamics slate has crashed 3x today (BrokenPipe on the
#      multiprocessing pool).  Each relaunch restarts from cell 1, so
#      "screen dead" is NOT a reliable completion signal -- it usually
#      means a crash mid-cell.  Gate on artifact count instead.
#   2. Dynamics slate lives in a different worktree
#      (/private/tmp/HRTG-dynamics-tmp).  We poll for ITS screen and
#      ITS output dir, not ours.
#   3. Loop survives crash->relaunch cycles.  If the screen dies but
#      not all artifacts landed, we keep polling (the user will
#      relaunch dynamics).  We only fire when EVERY expected output
#      JSON is present in the live OUTDIR (excluding attempt*_dead_*).
#   4. Idempotent on the meta-slate side -- if a slate_incremental
#      screen already exists, we don't double-fire.
#
# Completion gate: 12 result JSONs at the top level of
#   /private/tmp/HRTG-dynamics-tmp/experiments/15_unified_hybrid/
#     results/slate_dynamics_2026_05_26/
# matching unified_*_lam-*_K-10_R-1_*.json.
# That's 4 mass scales x 3 lambdas (lam-0.0, lam-1.0, lam-5.0).
#
# Polling: every 120 s, log to watcher.log.
#
# Usage (run in its own detached screen so it survives terminal exits):
#
#   screen -dmS slate_incremental_watcher bash \
#     experiments/15_unified_hybrid/queue_incremental_after_dynamics.sh
#
# To cancel before completion (e.g. you change your mind):
#
#   screen -X -S slate_incremental_watcher quit
#
# The legacy --no-resume flag is kept for backwards compatibility but
# is a no-op now (dynamics is no longer SIGSTOPped -- it's running).

set -u

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
META_SLATE="$REPO/experiments/15_unified_hybrid/slate_incremental_all_2026_05_26.sh"
WATCH_LOG="$REPO/experiments/15_unified_hybrid/results/slate_incremental_meta_2026_05_26/watcher.log"
mkdir -p "$(dirname "$WATCH_LOG")"

# The dynamics slate's live output dir (NOT attempt*_dead_* subdirs).
DYN_OUTDIR="/private/tmp/HRTG-dynamics-tmp/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26"
EXPECTED_CELLS=12   # 4 masses x 3 lambdas (lam-0.0, lam-1.0, lam-5.0)

# Parse flags (--no-resume is now a no-op; preserved for back-compat).
for arg in "$@"; do
  case "$arg" in
    --no-resume) ;;  # no-op: dynamics is no longer SIGSTOPped
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Helpers ------------------------------------------------------------

count_landed_cells () {
  # Count result JSONs at the top level of DYN_OUTDIR (maxdepth 1
  # excludes attempt*_dead_* subdirs).  Returns 0 if dir doesn't exist.
  if [ ! -d "$DYN_OUTDIR" ]; then echo 0; return; fi
  find "$DYN_OUTDIR" -maxdepth 1 -type f \
    -name 'unified_*_lam-*_K-10_R-1_*.json' 2>/dev/null \
    | wc -l | tr -d ' '
}

screen_alive () {
  screen -ls 2>/dev/null | grep -qE '\.slate_dynamics([[:space:]]|$)'
}

incremental_already_running () {
  screen -ls 2>/dev/null | grep -qE '\.slate_incremental([[:space:]]|$)'
}

# Main loop ----------------------------------------------------------

{
  echo "=== watcher start  $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "[watch] gate: $EXPECTED_CELLS result JSONs in $DYN_OUTDIR"
  echo "[watch] initial count: $(count_landed_cells)/$EXPECTED_CELLS"
  echo "[watch] initial screen state: $(screen_alive && echo ALIVE || echo dead)"

  loop_n=0
  while true; do
    loop_n=$((loop_n + 1))
    landed=$(count_landed_cells)
    alive=$(screen_alive && echo 1 || echo 0)

    # Verbose status every 5th poll (~10 min).
    if [ $((loop_n % 5)) -eq 1 ]; then
      if [ "$alive" -eq 1 ]; then
        progress=$(
          ps -A -o pid,etime,%cpu,command \
            | awk '/HRTG-dynamics-tmp.*run_can\.py/ {print $1,$2,$3"%CPU"}' \
            | head -1
        )
        echo "[watch $(date '+%H:%M:%S')] dyn ALIVE  cells=$landed/$EXPECTED_CELLS  parent: ${progress:-(none)}"
      else
        echo "[watch $(date '+%H:%M:%S')] dyn dead   cells=$landed/$EXPECTED_CELLS  (waiting for relaunch or completion)"
      fi
    fi

    # FIRE condition: all expected cells landed AND screen is dead
    # (don't fire while the last cell is still being written).
    if [ "$landed" -ge "$EXPECTED_CELLS" ] && [ "$alive" -eq 0 ]; then
      echo "=== gate satisfied at $(date '+%Y-%m-%d %H:%M:%S'): $landed/$EXPECTED_CELLS cells + screen dead ==="
      break
    fi

    sleep 120
  done

  # Idempotency check before firing.
  if incremental_already_running; then
    echo "[watch] slate_incremental screen already exists; NOT double-firing"
    echo "=== watcher done at $(date '+%Y-%m-%d %H:%M:%S') (no-op) ==="
    exit 0
  fi

  echo "[watch] launching incremental meta-slate ..."
  screen -dmS slate_incremental bash "$META_SLATE"

  # Brief sanity: is the screen actually up?
  sleep 5
  if incremental_already_running; then
    echo "[watch] hand-off OK -- slate_incremental screen is alive"
  else
    echo "[watch] FATAL: slate_incremental screen failed to start"
    echo "[watch] try manually:  screen -dmS slate_incremental bash $META_SLATE"
    exit 5
  fi

  echo "=== watcher done at $(date '+%Y-%m-%d %H:%M:%S') ==="
} >> "$WATCH_LOG" 2>&1
