#!/usr/bin/env bash
# One-shot status snapshot for slate_demoscale (2026-05-25).
#
# Usage:
#   bash experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25/monitor.sh
#
# Or live-refreshing:
#   watch -n 30 -c bash experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25/monitor.sh
#
# Single-shot, read-only, macOS BSD-userland-safe.  Side-effect-free
# so watch can hammer it.
#
# Reports for each of the 6 cells (N_b in {5, 10, 20, 50, 100, 200}):
#   - cell state         (RUNNING / DONE / PENDING)
#   - rollouts logged    (parsed from run.log)
#   - successes so far   (parsed from run.log)
#   - in-bank vs OOB     (a quick diagnostic per the Theorem 1' prediction)

set -u

LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$LOG_DIR/run.log"
# Match both the original (slate_demoscale) and the resume
# (slate_demoscale_resume) screen sessions.
SCREEN_NAME="slate_demoscale"

# All 6 expected cells, in order
CELLS=(Nb5 Nb10 Nb20 Nb50 Nb100 Nb200)

if [[ ! -f "$LOG" ]]; then
  echo "no run.log at $LOG"
  exit 1
fi

NOW=$(date +"%Y-%m-%dT%H:%M:%S%z")

# Screen state
SCREEN_LINE=$(screen -ls 2>/dev/null | awk -v n="$SCREEN_NAME" '$0 ~ n {print}' | head -1)
if [[ -n "$SCREEN_LINE" ]]; then
  SCREEN_STATE=$(echo "$SCREEN_LINE" | grep -oE '\(Detached\)|\(Attached\)|\(Dead' | head -1)
else
  SCREEN_STATE="(NOT FOUND)"
fi

# Python pid (any run_can.py under this slate's output dir)
PY_LINE=$(ps -A -o pid,etime,command | grep "run_can.py" | grep "slate_demoscale" | grep -v grep | head -1)
if [[ -n "$PY_LINE" ]]; then
  PY_PID=$(echo "$PY_LINE" | awk '{print $1}')
  PY_ETIME=$(echo "$PY_LINE" | awk '{print $2}')
else
  PY_PID="-"
  PY_ETIME="-"
fi

echo "============================================================"
echo "slate_demoscale monitor                $NOW"
echo "============================================================"
echo "screen : $SCREEN_NAME $SCREEN_STATE"
echo "python : pid=$PY_PID elapsed=$PY_ETIME"
echo "log    : $LOG  ($(wc -l < "$LOG") lines)"
echo "------------------------------------------------------------"
printf "%-7s  %-9s  %-13s  %-6s  %-7s  %s\n" \
  "cell" "state" "rollouts/60" "succ" "rate" "in-bank-vs-OOB"

OVERALL_DONE=0
for CELL in "${CELLS[@]}"; do
  # Did this cell finish?  The slate emits "=== HH:MM:SS  CELL  done ==="
  # (note: two spaces around CELL, from the slate's `=== ... $TAG ... ===` echo).
  DONE=$(grep -c "  ${CELL}  done ===" "$LOG" 2>/dev/null; true)
  STARTED=$(grep -c "  ${CELL}  starting ===" "$LOG" 2>/dev/null; true)
  DONE=${DONE:-0}
  STARTED=${STARTED:-0}

  if [[ "$DONE" -gt 0 ]]; then
    STATE="DONE"
  elif [[ "$STARTED" -gt 0 ]]; then
    STATE="RUNNING"
  else
    STATE="PENDING"
  fi

  # Count rollouts logged after this cell's "starting" line and before
  # either its "done" line or end-of-log.
  if [[ "$STARTED" -gt 0 ]]; then
    # Extract the slice of the log corresponding to this cell.
    # Note: the slate's per-cell banner uses TWO spaces around the
    # cell tag ("=== HH:MM:SS  $TAG  starting ===") because of the
    # double-space separator in the shell echo.  Match with two
    # spaces to avoid Nb5 also matching a hypothetical Nb50 prefix.
    SLICE=$(awk '
      /=== .*  '"$CELL"'  starting ===/ {in_cell=1; next}
      /=== .* done ===/ {if (in_cell) {in_cell=0}}
      in_cell {print}
    ' "$LOG")

    ROLLOUTS=$(echo "$SLICE" | grep -cE "^\[exp15\] prior=demo_1nn"; true)
    SUCC=$(echo "$SLICE" | grep -c "success=True"; true)
    ROLLOUTS=${ROLLOUTS:-0}
    SUCC=${SUCC:-0}
    if [[ "$ROLLOUTS" -gt 0 ]]; then
      RATE=$(awk "BEGIN {printf \"%.0f%%\", 100*$SUCC/$ROLLOUTS}")
      # In-bank/OOB diagnostic: demo_N where N < bank_size is "in-bank"
      NB_NUM=$(echo "$CELL" | sed 's/Nb//')
      IN_BANK_SUCC=$(echo "$SLICE" | grep -E "demo_[0-$((NB_NUM-1))]#" | grep -c "success=True"; true)
      IN_BANK_TOT=$(echo "$SLICE" | grep -cE "demo_[0-$((NB_NUM-1))]#"; true)
      OOB_SUCC=$(echo "$SLICE" | grep -vE "demo_[0-$((NB_NUM-1))]#" | grep -E "^\[exp15\] prior=demo_1nn" | grep -c "success=True"; true)
      IN_BANK_SUCC=${IN_BANK_SUCC:-0}
      IN_BANK_TOT=${IN_BANK_TOT:-0}
      OOB_SUCC=${OOB_SUCC:-0}
      OOB_TOT=$(( ROLLOUTS - IN_BANK_TOT ))
      if [[ "$NB_NUM" -ge 20 ]]; then
        DIAG="(full coverage)"
      else
        DIAG="IB ${IN_BANK_SUCC}/${IN_BANK_TOT}  OOB ${OOB_SUCC}/${OOB_TOT}"
      fi
    else
      RATE="-"
      DIAG="(no rollouts yet)"
    fi
  else
    ROLLOUTS=0
    SUCC=0
    RATE="-"
    DIAG="-"
  fi

  printf "%-7s  %-9s  %-13s  %-6s  %-7s  %s\n" \
    "$CELL" "$STATE" "$ROLLOUTS/60" "$SUCC" "$RATE" "$DIAG"

  if [[ "$STATE" == "DONE" ]]; then
    OVERALL_DONE=$(( OVERALL_DONE + 1 ))
  fi
done

echo "------------------------------------------------------------"
echo "overall: $OVERALL_DONE / ${#CELLS[@]} cells complete"
echo "------------------------------------------------------------"
echo "results JSONs in $LOG_DIR/ :"
ls -1 "$LOG_DIR"/*.json 2>/dev/null | sed 's|.*/||' | head -8 \
  || echo "  (no .json yet -- cell 1 still in flight)"
echo "============================================================"
