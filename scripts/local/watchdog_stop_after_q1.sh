#!/bin/bash
# Watchdog (belt-and-suspenders):
#   When Q1 finishes, terminate the `phase_d_all_queues` screen and kill
#   any leftover python children that might belong to Q2/Q3 (which we
#   moved to Colab).
#
# Why this exists: the master chain script was edited mid-run to insert
# an `exit 0` after Q1.  Bash usually re-reads scripts as it executes,
# but to be safe this watchdog forcibly stops the chain at the boundary.
#
# Usage:
#   screen -dmS phase_d_watchdog_q1 bash scripts/local/watchdog_stop_after_q1.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/chain_all_queues.log"
WD_LOG="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/watchdog_q1.log"
CHAIN_SCREEN="phase_d_all_queues"
POLL_SEC=30

# Sentinel phrase the master chain emits after Q1's launcher returns.
SENTINEL="\[queue 1\] finished"

log() {
    echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$WD_LOG"
}

log "starting; watching for sentinel '$SENTINEL' in $LOG"
log "will then kill screen '$CHAIN_SCREEN' + reap stray python children"

# Poll until sentinel appears, OR the chain screen disappears (already done).
while true; do
    # Master log written?
    if [ -f "$LOG" ] && grep -qE "$SENTINEL" "$LOG"; then
        log "sentinel found in master log"
        break
    fi
    # Chain screen gone?
    sout=$(screen -ls 2>/dev/null || true)
    if ! echo "$sout" | grep -q "$CHAIN_SCREEN"; then
        log "chain screen '$CHAIN_SCREEN' is gone already; nothing to do"
        log "watchdog exiting cleanly"
        exit 0
    fi
    sleep "$POLL_SEC"
done

# Sentinel found.  Give the master chain ~10 s to exit on its own via
# the `exit 0` we patched in.
log "giving chain 10 s to exit naturally ..."
sleep 10

sout=$(screen -ls 2>/dev/null || true)
if echo "$sout" | grep -q "$CHAIN_SCREEN"; then
    log "chain '$CHAIN_SCREEN' still alive after 10 s; forcibly killing"
    screen -X -S "$CHAIN_SCREEN" quit 2>/dev/null || true
    sleep 2
    # Also reap any bash subshells from chain_phase_d_all_queues.sh
    for pid in $(pgrep -f 'chain_phase_d_all_queues' 2>/dev/null); do
        log "  killing chain bash pid=$pid"
        kill -KILL "$pid" 2>/dev/null || true
    done
else
    log "chain '$CHAIN_SCREEN' exited cleanly on its own; good"
fi

# Reap any Q2/Q3 python children that might have launched before bash
# saw the script edit.  Q2/Q3 use --output-dir under lambda_sweep_local
# or k_sweep_local.
log "scanning for stray Q2/Q3 python children ..."
stray=$(ps -A -ww -o pid,command 2>/dev/null \
    | grep -E 'lambda_sweep_local|k_sweep_local' \
    | grep -v grep \
    | awk '{print $1}')
if [ -n "$stray" ]; then
    log "found stray Q2/Q3 children: $stray"
    for pid in $stray; do
        log "  TERM $pid"
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $stray; do
        if kill -0 "$pid" 2>/dev/null; then
            log "  KILL $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
else
    log "no stray Q2/Q3 children"
fi

# Also clean up any empty Q2/Q3 output dirs we may have accidentally
# created -- so the Colab notebook can write fresh outputs without
# colliding.
for d in "$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/lambda_sweep_local" \
         "$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/k_sweep_local"; do
    if [ -d "$d" ]; then
        # Only blow away if it contains no result JSON.
        if [ -z "$(find "$d" -name 'unified_flow_*.json' -type f 2>/dev/null)" ]; then
            log "removing empty local Q2/Q3 dir: $d"
            rm -rf "$d"
        else
            log "Q2/Q3 local dir has results (kept): $d"
        fi
    fi
done

log "watchdog DONE at $(date)"
exit 0
