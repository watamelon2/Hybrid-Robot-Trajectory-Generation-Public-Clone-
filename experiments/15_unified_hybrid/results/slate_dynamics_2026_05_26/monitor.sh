#!/usr/bin/env bash
# One-shot status snapshot for slate_dynamics (2026-05-26).
#
# Usage (single-shot):
#   bash /private/tmp/HRTG-dynamics-tmp/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26/monitor.sh
#
# Or live-refreshing:
#   watch -n 30 -c bash /private/tmp/HRTG-dynamics-tmp/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26/monitor.sh
#
# Single-shot, read-only, macOS BSD-userland-safe.  Side-effect-free
# so `watch` can hammer it.
#
# Reports for each of the 8 cells (4 mass scales x 2 invocations:
# demo_1nn lambda-sweep + bc bridge), the state, rollouts logged,
# successes so far, and per-lambda rates.  Each demo_1nn cell
# internally runs 2 lambdas (0 + 1) so its expected rollout count
# is 120; each bc cell runs a single lambda so its expected count
# is 60.
#
# Adapted from slate_costtime_2026_05_25/monitor.sh -- key changes:
#   - cell-tag scheme: m100 / m070 / m130 / m150 (mass percent)
#   - firing order: 1.0 control first, then 0.7, 1.3, 1.5
#   - per-lambda succ breakdown for demo_1nn cells so hybrid (lam=1)
#     and pure-MBD (lam=0) rates can be eyeballed without waiting
#     for the cell to close
#   - control-row anchor: shows the m100 (mass=1.0) per-lambda rates
#     at the bottom so perturbation cells can be compared at-a-glance

set -u

LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$LOG_DIR/run.log"
SCREEN_NAME="slate_dynamics"

# Watcher (queue_incremental_after_dynamics.sh) integration.
WATCHER_SCREEN="slate_incremental_watcher"
WATCHER_LOG="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental/experiments/15_unified_hybrid/results/slate_incremental_meta_2026_05_26/watcher.log"
EXPECTED_JSONS=12   # gate the watcher fires on: 4 masses x 3 lambdas

# All 8 expected cells, in firing order (mass=1.0 control first,
# then the three perturbed cells; demo_1nn lambda-sweep then bc
# bridge inside each mass).
CELLS=(
  demo1nn_m100  bc_m100
  demo1nn_m070  bc_m070
  demo1nn_m130  bc_m130
  demo1nn_m150  bc_m150
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
PY_LINE=$(ps -A -o pid,etime,command | grep "run_can.py" | grep "slate_dynamics_2026_05_26" | grep -v grep | head -1)
if [[ -n "$PY_LINE" ]]; then
  PY_PID=$(echo "$PY_LINE" | awk '{print $1}')
  PY_ETIME=$(echo "$PY_LINE" | awk '{print $2}')
  # Active cell tag = the last --output-name-suffix _<TAG> in the cmdline.
  ACTIVE_CELL=$(echo "$PY_LINE" | grep -oE -- "--output-name-suffix +_[a-z0-9_]+" \
                                | tail -1 | awk '{print $2}' | sed 's/^_//')
  ACTIVE_CELL=${ACTIVE_CELL:--}
else
  PY_PID="-"
  PY_ETIME="-"
  ACTIVE_CELL="-"
fi

# Attempt history: count attempt*_dead_* sibling dirs (each is a
# prior crash that was rotated out of the live OUTDIR).
DEAD_ATTEMPTS=$(find "$LOG_DIR" -maxdepth 1 -type d -name 'attempt*_dead_*' 2>/dev/null | wc -l | tr -d ' ')
DEAD_ATTEMPTS=${DEAD_ATTEMPTS:-0}
CURRENT_ATTEMPT=$((DEAD_ATTEMPTS + 1))

# Gate progress: how many of the 12 result JSONs have landed in the
# live OUTDIR (NOT inside attempt*_dead_* subdirs).
LANDED_JSONS=$(find "$LOG_DIR" -maxdepth 1 -type f \
  -name 'unified_*_lam-*_K-10_R-1_*.json' 2>/dev/null | wc -l | tr -d ' ')
LANDED_JSONS=${LANDED_JSONS:-0}

echo "============================================================"
echo "slate_dynamics monitor                $NOW"
echo "============================================================"
echo "attempt: #$CURRENT_ATTEMPT today ($DEAD_ATTEMPTS prior crash(es) archived under attempt*_dead_*/)"
echo "screen : $SCREEN_NAME $SCREEN_STATE"
echo "python : pid=$PY_PID elapsed=$PY_ETIME"
echo "active : $ACTIVE_CELL"
echo "log    : $LOG  ($(wc -l < "$LOG") lines)"
echo "gate   : $LANDED_JSONS/$EXPECTED_JSONS result JSONs landed (watcher fires when 12/12 AND screen dead)"
echo "------------------------------------------------------------"

printf "%-15s  %-9s  %-9s  %-19s  %-7s\n" \
  "cell" "state" "rollouts" "succ (lam=0|lam=1|bc)" "wall"

OVERALL_DONE=0
# Cache the m100 control-row rates for the comparison block below.
CTRL_L0_RATE="-"
CTRL_L1_RATE="-"
CTRL_BC_RATE="-"

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
    # Extract just the lines for this cell.
    SLICE=$(awk '
      /=== .*  '"$CELL"'  starting ===/ {in_cell=1; next}
      /=== .* done ===/ {if (in_cell) {in_cell=0}}
      in_cell {print}
    ' "$LOG")

    # exp15's per-rollout line format:
    #   `[exp15] prior=<prior> lam=  <L>  R=...  reward=...  success=True/False`
    ROLLOUTS=$(echo "$SLICE" | grep -cE "^\[exp15\] prior= *(demo_1nn|bc)"; true)
    ROLLOUTS=${ROLLOUTS:-0}

    # Per-lambda counts (demo_1nn cells).
    L0_N=$(echo "$SLICE" | grep -cE "^\[exp15\] prior= *demo_1nn lam= *0\.0"; true); L0_N=${L0_N:-0}
    L0_S=$(echo "$SLICE" | grep -E   "^\[exp15\] prior= *demo_1nn lam= *0\.0" | grep -c "success=True"; true); L0_S=${L0_S:-0}
    L1_N=$(echo "$SLICE" | grep -cE "^\[exp15\] prior= *demo_1nn lam= *1\.0"; true); L1_N=${L1_N:-0}
    L1_S=$(echo "$SLICE" | grep -E   "^\[exp15\] prior= *demo_1nn lam= *1\.0" | grep -c "success=True"; true); L1_S=${L1_S:-0}
    # bc cells (single lambda=5.0).
    BC_N=$(echo "$SLICE" | grep -cE "^\[exp15\] prior= *bc"; true); BC_N=${BC_N:-0}
    BC_S=$(echo "$SLICE" | grep -E   "^\[exp15\] prior= *bc" | grep -c "success=True"; true); BC_S=${BC_S:-0}

    fmt_rate () {
      local n=$1; local s=$2
      if [[ "$n" -gt 0 ]]; then
        awk -v s="$s" -v n="$n" 'BEGIN {printf "%d/%-2d=%2.0f%%", s, n, 100*s/n}'
      else
        printf '%s' "-       "
      fi
    }
    L0_F=$(fmt_rate "$L0_N" "$L0_S")
    L1_F=$(fmt_rate "$L1_N" "$L1_S")
    BC_F=$(fmt_rate "$BC_N" "$BC_S")
    SUCC_FIELD=$(printf "%-8s %-8s %-8s" "$L0_F" "$L1_F" "$BC_F")

    # Cache control-row values (mass=1.0) for the comparison block.
    if [[ "$CELL" == "demo1nn_m100" ]]; then
      [[ "$L0_N" -gt 0 ]] && CTRL_L0_RATE=$(awk -v s="$L0_S" -v n="$L0_N" 'BEGIN {printf "%2.0f%%", 100*s/n}')
      [[ "$L1_N" -gt 0 ]] && CTRL_L1_RATE=$(awk -v s="$L1_S" -v n="$L1_N" 'BEGIN {printf "%2.0f%%", 100*s/n}')
    fi
    if [[ "$CELL" == "bc_m100" && "$BC_N" -gt 0 ]]; then
      CTRL_BC_RATE=$(awk -v s="$BC_S" -v n="$BC_N" 'BEGIN {printf "%2.0f%%", 100*s/n}')
    fi

    # Per-cell wall time (start -> last log line OR done line).
    START_TS=$(grep -m1 "  ${CELL}  starting ===" "$LOG" | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}")
    if [[ "$STATE" == "DONE" ]]; then
      END_TS=$(grep "  ${CELL}  done ===" "$LOG" | tail -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}")
    else
      END_TS=$(date "+%H:%M:%S")
    fi
    if [[ -n "$START_TS" && -n "$END_TS" ]]; then
      WALL=$(awk -v s="$START_TS" -v e="$END_TS" '
        function tos(t,   a) {split(t,a,":"); return a[1]*3600+a[2]*60+a[3]}
        BEGIN {d = tos(e) - tos(s); if (d < 0) d += 86400;
               printf "%dm%02ds", d/60, d%60}')
    else
      WALL="-"
    fi
  else
    ROLLOUTS=0
    SUCC_FIELD=$(printf "%-8s %-8s %-8s" "-" "-" "-")
    WALL="-"
  fi

  printf "%-15s  %-9s  %-9s  %-19s  %-7s\n" \
    "$CELL" "$STATE" "${ROLLOUTS}/${EXPECTED}" "$SUCC_FIELD" "$WALL"

  if [[ "$STATE" == "DONE" ]]; then
    OVERALL_DONE=$(( OVERALL_DONE + 1 ))
  fi
done

echo "------------------------------------------------------------"
echo "overall : $OVERALL_DONE / ${#CELLS[@]} cells complete"

# At-a-glance comparison: each perturbation cell's hybrid (lam=1)
# rate vs the control (mass=1.0).  Surfaces the headline gap-vs-mass
# trend before the AGGREGATE is generated.
echo "------------------------------------------------------------"
echo "control anchor (mass=1.0): pure-MBD=$CTRL_L0_RATE  hybrid=$CTRL_L1_RATE  BC=$CTRL_BC_RATE"
echo "(perturbation cells' rates above; eyeball drift across mass scales)"
echo "------------------------------------------------------------"

# Failure-mode probe: any tracebacks since slate start?
# (grep -c always prints '0' on no-match but exits 1; guard with || true so
# pipefail-safe shells don't blow up here.)
ERR_CT=$(grep -cE "^Traceback|^FATAL|raise [A-Z]" "$LOG" 2>/dev/null || true)
ERR_CT=${ERR_CT:-0}
if [[ "$ERR_CT" -gt 0 ]]; then
  echo "WARNING: $ERR_CT error lines in run.log -- inspect:"
  grep -nE "^Traceback|^FATAL|raise [A-Z]" "$LOG" | head -3
  echo "------------------------------------------------------------"
fi

# Result JSONs landed so far (gate counter for the watcher).
echo "result JSONs in $LOG_DIR/ ($LANDED_JSONS/$EXPECTED_JSONS for the queue gate):"
ls -1 "$LOG_DIR"/*.json 2>/dev/null | sed 's|.*/||' | head -20 \
  || echo "  (no .json yet -- cell 1 still in flight)"
echo "------------------------------------------------------------"

# Queue / watcher status: is the incremental meta-slate queued behind
# this one?  Shows watcher screen state + last poll line.
WATCHER_LINE=$(screen -ls 2>/dev/null | awk -v n="$WATCHER_SCREEN" '$0 ~ n {print}' | head -1)
if [[ -n "$WATCHER_LINE" ]]; then
  WATCHER_STATE=$(echo "$WATCHER_LINE" | grep -oE '\(Detached\)|\(Attached\)|\(Dead' | head -1)
  echo "queue  : $WATCHER_SCREEN $WATCHER_STATE   (will fire incremental meta-slate when gate clears)"
else
  # Distinguish "never armed" from "already fired and exited".
  if screen -ls 2>/dev/null | grep -qE '\.slate_incremental([[:space:]]|$)'; then
    echo "queue  : watcher EXITED + slate_incremental is ALIVE  (incremental phase running now)"
  else
    echo "queue  : NOT armed  (run: screen -dmS $WATCHER_SCREEN bash <REPO>/experiments/15_unified_hybrid/queue_incremental_after_dynamics.sh)"
  fi
fi
if [[ -f "$WATCHER_LOG" ]]; then
  LAST_POLL=$(grep -E '^\[watch ' "$WATCHER_LOG" | tail -1)
  [[ -n "$LAST_POLL" ]] && echo "         last: $LAST_POLL"
fi
echo "============================================================"
