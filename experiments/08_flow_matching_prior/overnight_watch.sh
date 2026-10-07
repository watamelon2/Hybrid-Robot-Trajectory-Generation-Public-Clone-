#!/usr/bin/env bash
# Auto-refresh wrapper around overnight_monitor.sh + overnight_results.sh.
# Refreshes every 30s. Ctrl-C to exit.
#
# Usage: bash experiments/08_flow_matching_prior/overnight_watch.sh [interval_seconds]
set -u

INTERVAL=${1:-30}
EXP08="$(cd "$(dirname "$0")" && pwd)"

trap 'echo; echo "watch stopped."; exit 0' INT

while true; do
    clear
    echo "==============================================================================="
    echo " overnight queue WATCH   refresh=${INTERVAL}s   $(date '+%Y-%m-%d %H:%M:%S')   (Ctrl-C to stop)"
    echo "==============================================================================="
    bash "$EXP08/overnight_monitor.sh" 2>&1 | sed -e '1,3d'
    echo
    bash "$EXP08/overnight_results.sh" 2>&1
    sleep "$INTERVAL"
done
