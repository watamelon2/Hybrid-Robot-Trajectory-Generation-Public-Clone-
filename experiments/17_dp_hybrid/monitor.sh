#!/usr/bin/env bash
# One-shot status snapshot for the DP-hybrid Phase-1 cells.
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root

cyan="\033[36m"; green="\033[32m"; yellow="\033[33m"; red="\033[31m"; r="\033[0m"
hdr() { printf "${cyan}==> %s${r}\n" "$1"; }

hdr "wall clock"
date

hdr "tracked python procs"
ps -A -o pid,etime,cputime,%cpu,command | \
    awk '/standalone_dp\.py.*v_theta_dp_tobs2_h16|run_can\.py.*--score-mode pull.*v_theta_dp_tobs2_h16/' \
    | grep -v awk || echo "  (none — both finished or not started)"

hdr "load average"
uptime

hdr "cell: baseline_standalone_dp"
OUT=experiments/17_dp_hybrid/results/baseline_standalone_dp
if [ -f "$OUT/standalone_dp_repro.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT/standalone_dp_repro.json'))
print(f'  STATUS: ${green}done${r}'.replace('\\\\033','\033'))
print(f'  n_total = {d[\"n\"]}, n_succ = {d[\"n_succ\"]}, succ_rate = {d[\"succ_rate\"]:.3f}')
print(f'  wall_s = {d[\"wall_s\"]:.1f} ({d[\"wall_s\"]/60:.1f} min)')
"
else
    printf "  STATUS: ${yellow}running${r}\n"
    if [ -f "$OUT/run.log" ]; then
        N_DONE=$(grep -c '^\[standalone_dp\]  ' "$OUT/run.log" 2>/dev/null || echo 0)
        N_SUCC=$(grep '^\[standalone_dp\]  ' "$OUT/run.log" 2>/dev/null | grep -c 'success=True' || echo 0)
        echo "  trials done: $N_DONE / 30  (successes so far: $N_SUCC)"
        echo "  last 3 log lines:"
        grep '^\[standalone_dp\]' "$OUT/run.log" | tail -3 | sed 's/^/    /'
    fi
fi

hdr "cell: dp + pull-MBD"
OUT=experiments/17_dp_hybrid/results/cell_dp_pullmbd
SUM=$(ls $OUT/unified_summary_*.json 2>/dev/null | head -1)
if [ -n "$SUM" ] && [ -f "$SUM" ]; then
    python3 -c "
import json
d = json.load(open('$SUM'))
print(f'  STATUS: done')
print(f'  n_total = {d[\"n_total\"]}, n_succ = {d[\"n_success\"]}, succ_rate = {d[\"success_rate\"]:.3f}')
print(f'  wall_s = {d.get(\"wall_s\",\"?\")}')
"
else
    printf "  STATUS: ${yellow}running${r}\n"
    if [ -f "$OUT/run.log" ]; then
        N_DONE=$(grep -c '^\[exp15\] prior=' "$OUT/run.log" 2>/dev/null || echo 0)
        N_SUCC=$(grep '^\[exp15\] prior=' "$OUT/run.log" 2>/dev/null | grep -c 'success=True' || echo 0)
        echo "  trials done: $N_DONE / 30  (successes so far: $N_SUCC)"
        echo "  last 3 log lines:"
        grep '^\[exp15\]' "$OUT/run.log" | tail -3 | sed 's/^/    /'
    fi
fi

hdr "summary table (live)"
printf "  %-30s %-10s %-10s %-12s\n" "cell" "n_done" "n_succ" "rate"
for cell in baseline_standalone_dp cell_dp_pullmbd; do
    OUT=experiments/17_dp_hybrid/results/$cell
    LOG=$OUT/run.log
    N_DONE=0
    N_SUCC=0
    if [ ! -f "$LOG" ]; then
        printf "  %-30s %-10s %-10s %-12s\n" "$cell" "-" "-" "not started"
        continue
    fi
    case $cell in
        baseline_standalone_dp)
            pat='^\[standalone_dp\]  '
            ;;
        cell_dp_pullmbd)
            pat='^\[exp15\] prior=flow_pull '
            ;;
    esac
    if grep -q "$pat" "$LOG" 2>/dev/null; then
        N_DONE=$(grep -c "$pat" "$LOG")
        N_SUCC=$(grep "$pat" "$LOG" | grep -c 'success=True' || true)
    fi
    # Strip newlines defensively
    N_DONE=${N_DONE//$'\n'/}
    N_SUCC=${N_SUCC//$'\n'/}
    if [ "$N_DONE" -gt 0 ] 2>/dev/null; then
        RATE=$(awk "BEGIN{printf \"%.1f%%\", 100*$N_SUCC/$N_DONE}")
    else
        RATE="—"
    fi
    printf "  %-30s %-10s %-10s %-12s\n" "$cell" "$N_DONE/30" "$N_SUCC" "$RATE"
done
