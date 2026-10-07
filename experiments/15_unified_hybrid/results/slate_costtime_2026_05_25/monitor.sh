#!/usr/bin/env bash
# One-shot status snapshot for slate_costtime (2026-05-25).
#
# Usage:
#   bash experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/monitor.sh
#
# Or live-refreshing:
#   watch -n 30 -c bash experiments/15_unified_hybrid/results/slate_costtime_2026_05_25/monitor.sh
#
# Single-shot, read-only, macOS BSD-userland-safe.  Side-effect-free
# so watch can hammer it.
#
# Reports for each of the 8 cells (4 perturbations x 2 invocations:
# demo_1nn lambda-sweep + bc bridge), the state, rollouts logged,
# successes so far, and the rate.  Each demo_1nn cell internally
# runs 2 lambdas (0 + 1) so its expected rollout count is 120;
# each bc cell runs a single lambda so its expected count is 60.

set -u

LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$LOG_DIR/run.log"
SCREEN_NAME="slate_costtime"

# All 8 expected cells, in firing order (4 dx perturbations x
# 2 invocation flavours: demo_1nn lambda-sweep + bc bridge).
CELLS=(
  demo1nn_dx000  bc_dx000
  demo1nn_dx005  bc_dx005
  demo1nn_dx010  bc_dx010
  demo1nn_dx020  bc_dx020
)

if [[ ! -f "$LOG" ]]; then
  echo "no run.log at $LOG"
  exit 1
fi

NOW=$(date +"%Y-%m-%dT%H:%M:%S%z")

# Screen state.
SCREEN_LINE=$(screen -ls 2>/dev/null | awk -v n="$SCREEN_NAME" '$0 ~ n {print}' | head -1)
if [[ -n "$SCREEN_LINE" ]]; then
  SCREEN_STATE=$(echo "$SCREEN_LINE" | grep -oE '\(Detached\)|\(Attached\)|\(Dead' | head -1)
else
  SCREEN_STATE="(NOT FOUND)"
fi

# Python pid (any run_can.py under this slate's output dir).
PY_LINE=$(ps -A -o pid,etime,command | grep "run_can.py" | grep "slate_costtime" | grep -v grep | head -1)
if [[ -n "$PY_LINE" ]]; then
  PY_PID=$(echo "$PY_LINE" | awk '{print $1}')
  PY_ETIME=$(echo "$PY_LINE" | awk '{print $2}')
else
  PY_PID="-"
  PY_ETIME="-"
fi

echo "============================================================"
echo "slate_costtime monitor                $NOW"
echo "============================================================"
echo "screen : $SCREEN_NAME $SCREEN_STATE"
echo "python : pid=$PY_PID elapsed=$PY_ETIME"
echo "log    : $LOG  ($(wc -l < "$LOG") lines)"
echo "------------------------------------------------------------"
printf "%-16s  %-9s  %-13s  %-6s  %-7s\n" \
  "cell" "state" "rollouts" "succ" "rate"

OVERALL_DONE=0
for CELL in "${CELLS[@]}"; do
  # Per-invocation expected rollout count.  demo_1nn cells sweep
  # 2 lambdas (60 + 60 = 120); bc cells run a single lambda (60).
  case "$CELL" in
    demo1nn_*) EXPECTED=120 ;;
    bc_*)      EXPECTED=60  ;;
    *)         EXPECTED=60  ;;
  esac

  # Cell start / done markers (matched with double spaces around the
  # cell tag, matching the slate's `=== ... $TAG ... ===` echo format).
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

  if [[ "$STARTED" -gt 0 ]]; then
    SLICE=$(awk '
      /=== .*  '"$CELL"'  starting ===/ {in_cell=1; next}
      /=== .* done ===/ {if (in_cell) {in_cell=0}}
      in_cell {print}
    ' "$LOG")

    # exp15's per-rollout line format: `[exp15] prior=<prior> lam=<L>`
    ROLLOUTS=$(echo "$SLICE" | grep -cE "^\[exp15\] prior="; true)
    SUCC=$(echo "$SLICE" | grep -c "success=True"; true)
    ROLLOUTS=${ROLLOUTS:-0}
    SUCC=${SUCC:-0}
    if [[ "$ROLLOUTS" -gt 0 ]]; then
      RATE=$(awk "BEGIN {printf \"%.0f%%\", 100*$SUCC/$ROLLOUTS}")
    else
      RATE="-"
    fi
  else
    ROLLOUTS=0
    SUCC=0
    RATE="-"
  fi

  printf "%-16s  %-9s  %-13s  %-6s  %-7s\n" \
    "$CELL" "$STATE" "${ROLLOUTS}/${EXPECTED}" "$SUCC" "$RATE"

  if [[ "$STATE" == "DONE" ]]; then
    OVERALL_DONE=$(( OVERALL_DONE + 1 ))
  fi
done

echo "------------------------------------------------------------"
echo "overall: $OVERALL_DONE / ${#CELLS[@]} cells complete"
echo "------------------------------------------------------------"
echo "results JSONs in $LOG_DIR/ :"
ls -1 "$LOG_DIR"/*.json 2>/dev/null | sed 's|.*/||' | head -12 \
  || echo "  (no .json yet -- cell 1 still in flight)"
echo "============================================================"
