#!/usr/bin/env bash
# One-shot status snapshot for the full BC-swap re-run (exp 10 then exp 07).
#
# Designed to be wrapped with `watch` for a refreshing dashboard:
#
#   watch -n 10 -c bash scripts/local/monitor_bc_swap_rerun.sh
#
# `-c` preserves ANSI colors. `-n 10` = refresh every 10 s.
#
# What it shows:
#   1. The latest chain-script step (waiting on exp 10 / running
#      exp 10 analyzer / launched exp 07 / etc.).
#   2. Exp 03 BC eval headline (already done -- 21.7 %).
#   3. Exp 10 K x R pareto: per-cell state (RUNNING / DONE / not started)
#      with success rate when DONE, alive PID + elapsed + CPU + RSS
#      when RUNNING.
#   4. Exp 07 lambda-BC sweep: same per-cell breakdown for the 6
#      lambda values once exp 10 finishes.
#   5. Aggregate CPU / RSS / process count.
#   6. Tail of chain_status.txt (last 5 lines).
#
# Rules:
#   * Single-shot: NO infinite loop, NO clear. Run-and-exit.
#   * Read-only:  touches nothing under logs/ or experiments/*/results/.
#   * Fast:       under 1 s on a warm cache, no network I/O.
#   * Portable:   macOS BSD userland (no GNU-only flags).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

EXP10_OUT="experiments/10_kxreplans_grid/results"
EXP10_LOG="logs/local_pareto_full"
EXP07_OUT="experiments/07_lambda_sweep/results"
EXP07_LOG="logs/local_lambda_bc_full"
CHAIN_STATUS="logs/bc_swap_rerun/chain_status.txt"
EXP03_JSON="experiments/03_bc_baseline/results/can_ph.json"

# Pareto cells: "K R"
PARETO_CELLS=(
    "5 0" "5 1" "5 2"
    "10 0" "10 1" "10 2"
    "20 0" "20 1" "20 2"
)
# Lambda cells (lambda strings as used in lambda_bc_${lam}.json filenames)
LAMBDA_CELLS=(0.0 0.1 0.5 1.0 5.0 inf)

# ---- colors ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; MAGENTA=$'\033[35m'
else
    BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; MAGENTA=""
fi

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RULE="==========================================================================="
SUB="---------------------------------------------------------------------------"

echo "$RULE"
printf "%sbc-swap-rerun monitor%s    refreshed: %s\n" "$BOLD" "$RESET" "$NOW"
echo "$RULE"

# ---- chain step (most recent line of chain_status.txt) ----
if [[ -f "$CHAIN_STATUS" ]]; then
    last_step="$(tail -1 "$CHAIN_STATUS")"
    printf "%schain step :%s %s\n" "$BOLD" "$RESET" "$last_step"
else
    printf "%schain step :%s %s(chain_status.txt not found)%s\n" \
        "$BOLD" "$RESET" "$DIM" "$RESET"
fi

# ---- chain process alive? ----
chain_pid="$(pgrep -f 'chain_bc_swap_rerun.sh' 2>/dev/null | head -1)"
if [[ -n "$chain_pid" ]] && kill -0 "$chain_pid" 2>/dev/null; then
    chain_etime="$(ps -p "$chain_pid" -o etime= 2>/dev/null | tr -d ' ')"
    printf "%schain pid  :%s %s%-6s%s elapsed %s\n" \
        "$BOLD" "$RESET" "$GREEN" "$chain_pid" "$RESET" "$chain_etime"
else
    printf "%schain pid  :%s %s(not running)%s\n" \
        "$BOLD" "$RESET" "$DIM" "$RESET"
fi
echo ""

# ---- Exp 03 (already done; just show the headline) ----
printf "%s[exp 03 BC eval, new [256, 256]]%s\n" "$BOLD" "$RESET"
if [[ -f "$EXP03_JSON" ]]; then
    python3 - <<PY 2>/dev/null
import json, sys
try:
    d = json.load(open("$EXP03_JSON"))
    n = d.get("n_total", "?")
    s = d.get("n_success", "?")
    r = d.get("success_rate", 0)
    print(f"  result : {s}/{n} = {100*r:.1f} %  (wall {d.get('total_wall_s', 0):.1f} s)")
except Exception as e:
    print(f"  (failed to parse {sys.argv[0]}: {e})")
PY
else
    printf "  %s(not run yet)%s\n" "$DIM" "$RESET"
fi
echo ""

# ---- helper: ps -o etime,pcpu,rss for a PID; empty if dead ----
ps_alive() {
    # Print "etime pcpu rss" if alive, else nothing
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ps -p "$pid" -o etime=,pcpu=,rss= 2>/dev/null | sed 's/^ *//;s/ \{1,\}/ /g'
    fi
}
# helper: format KB -> "X.XG"
fmt_gb() {
    awk -v kb="$1" 'BEGIN { printf "%.1fG", kb/1024/1024 }'
}

# ---- Exp 10 pareto status ----
printf "%s[exp 10 K x R pareto, 9 cells]%s  out=%s  log=%s\n" \
    "$BOLD" "$RESET" "$EXP10_OUT" "$EXP10_LOG"
printf "  %-7s %-9s %-9s %-18s %-7s %-6s\n" \
    "cell" "state" "succ" "pid / elapsed" "cpu%" "rss"
echo "  $SUB"

n_done_pareto=0
n_running_pareto=0
total_cpu_pareto=0.0
total_rss_pareto=0
for spec in "${PARETO_CELLS[@]}"; do
    read -r K R <<< "$spec"
    slug="k${K}-r${R}"
    json="$EXP10_OUT/K${K}_R${R}.json"
    # find a pid file for this cell
    pid_files=("$EXP10_LOG/k${K}_r${R}_"*.pid)
    pid=""
    if [[ -f "${pid_files[0]}" ]]; then
        pid="$(cat "${pid_files[0]}" 2>/dev/null)"
    fi

    if [[ -f "$json" ]]; then
        # DONE: extract success rate
        sr="$(python3 -c "import json; d=json.load(open('$json')); print(f'{d[\"n_success\"]}/{d[\"n_total\"]}')" 2>/dev/null || echo "?")"
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$GREEN" "DONE" "$RESET" "$sr" "-" "-" "-"
        n_done_pareto=$((n_done_pareto + 1))
    elif info=$(ps_alive "$pid"); [[ -n "$info" ]]; then
        # RUNNING
        etime="$(echo "$info" | awk '{print $1}')"
        pcpu="$(echo "$info" | awk '{print $2}')"
        rss="$(echo "$info" | awk '{print $3}')"
        rss_gb="$(fmt_gb "$rss")"
        pid_etime="$pid / $etime"
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$YELLOW" "RUNNING" "$RESET" "-" "$pid_etime" "$pcpu" "$rss_gb"
        n_running_pareto=$((n_running_pareto + 1))
        total_cpu_pareto=$(awk -v a="$total_cpu_pareto" -v b="$pcpu" 'BEGIN { print a + b }')
        total_rss_pareto=$((total_rss_pareto + rss))
    else
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$DIM" "pending" "$RESET" "-" "-" "-" "-"
    fi
done
echo "  $SUB"
printf "  done: %s%d/9%s    running: %s%d%s    total CPU: %.1f%%    total RSS: %s\n" \
    "$GREEN" "$n_done_pareto" "$RESET" \
    "$YELLOW" "$n_running_pareto" "$RESET" \
    "$total_cpu_pareto" "$(fmt_gb "$total_rss_pareto")"
echo ""

# ---- Exp 07 lambda-BC status ----
printf "%s[exp 07 lambda-BC sweep, 6 cells]%s  out=%s  log=%s\n" \
    "$BOLD" "$RESET" "$EXP07_OUT" "$EXP07_LOG"
printf "  %-7s %-9s %-9s %-18s %-7s %-6s\n" \
    "lambda" "state" "succ" "pid / elapsed" "cpu%" "rss"
echo "  $SUB"

n_done_lam=0
n_running_lam=0
total_cpu_lam=0.0
total_rss_lam=0
for lam in "${LAMBDA_CELLS[@]}"; do
    slug="lam=$lam"
    json="$EXP07_OUT/lambda_bc_${lam}.json"
    fslug="lam$(echo "$lam" | tr '.' '-')"
    pid_files=("$EXP07_LOG/${fslug}_"*.pid)
    pid=""
    if [[ -f "${pid_files[0]}" ]]; then
        pid="$(cat "${pid_files[0]}" 2>/dev/null)"
    fi

    if [[ -f "$json" ]]; then
        sr="$(python3 -c "import json; d=json.load(open('$json')); print(f'{d[\"n_success\"]}/{d[\"n_total\"]}')" 2>/dev/null || echo "?")"
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$GREEN" "DONE" "$RESET" "$sr" "-" "-" "-"
        n_done_lam=$((n_done_lam + 1))
    elif info=$(ps_alive "$pid"); [[ -n "$info" ]]; then
        etime="$(echo "$info" | awk '{print $1}')"
        pcpu="$(echo "$info" | awk '{print $2}')"
        rss="$(echo "$info" | awk '{print $3}')"
        rss_gb="$(fmt_gb "$rss")"
        pid_etime="$pid / $etime"
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$YELLOW" "RUNNING" "$RESET" "-" "$pid_etime" "$pcpu" "$rss_gb"
        n_running_lam=$((n_running_lam + 1))
        total_cpu_lam=$(awk -v a="$total_cpu_lam" -v b="$pcpu" 'BEGIN { print a + b }')
        total_rss_lam=$((total_rss_lam + rss))
    else
        printf "  %-7s %s%-9s%s %-9s %-18s %-7s %-6s\n" \
            "$slug" "$DIM" "pending" "$RESET" "-" "-" "-" "-"
    fi
done
echo "  $SUB"
printf "  done: %s%d/6%s    running: %s%d%s    total CPU: %.1f%%    total RSS: %s\n" \
    "$GREEN" "$n_done_lam" "$RESET" \
    "$YELLOW" "$n_running_lam" "$RESET" \
    "$total_cpu_lam" "$(fmt_gb "$total_rss_lam")"
echo ""

# ---- system load + aggregate ----
sys_load="$(uptime | awk -F'load average[s]?: ' '{print $2}')"

# macOS free memory via vm_stat. The trailing period on values
# breaks naive awk; strip it explicitly. Page size is the first
# number on the header line.
sys_mem_used_gb=""
if command -v vm_stat >/dev/null 2>&1; then
    eval "$(vm_stat | awk '
        /page size of/        { for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) { print "PGSZ="$i; break } }
        /Pages active/        { v=$NF; sub(/\.$/,"",v); print "PA="v }
        /Pages wired down/    { v=$NF; sub(/\.$/,"",v); print "PW="v }
        /Pages occupied by compressor/ { v=$NF; sub(/\.$/,"",v); print "PC="v }
    ')"
    if [[ -n "${PGSZ:-}" && -n "${PA:-}" && -n "${PW:-}" && -n "${PC:-}" ]]; then
        sys_mem_used_gb="$(awk -v p="$PGSZ" -v a="$PA" -v w="$PW" -v c="$PC" \
            'BEGIN { printf "%.1fG", (a+w+c)*p/1024/1024/1024 }')"
    fi
fi

printf "%ssystem%s     load avg : %s\n" "$BOLD" "$RESET" "$sys_load"
[[ -n "$sys_mem_used_gb" ]] && \
    printf "           mem used : %s (of 64 GB)\n" "$sys_mem_used_gb"
total_procs=$((n_running_pareto + n_running_lam))
total_all_cpu=$(awk -v a="$total_cpu_pareto" -v b="$total_cpu_lam" 'BEGIN { print a + b }')
total_all_rss=$((total_rss_pareto + total_rss_lam))
printf "           workers  : %s%d%s active (%.0f%% CPU, %s RSS)\n" \
    "$CYAN" "$total_procs" "$RESET" "$total_all_cpu" "$(fmt_gb "$total_all_rss")"
echo ""

# ---- chain log tail ----
if [[ -f "$CHAIN_STATUS" ]]; then
    printf "%schain log (tail 5)%s\n" "$BOLD" "$RESET"
    tail -5 "$CHAIN_STATUS" | sed 's/^/  /'
fi

echo "$RULE"
