#!/usr/bin/env bash
# One-shot status snapshot for the local pareto runner.
#
# Designed to be wrapped with `watch` so the user sees a live
# refreshing dashboard:
#
#   watch -n 10 -c bash scripts/local/monitor_pareto.sh
#
# The `-c` flag preserves ANSI colors. -n 10 = refresh every 10 s
# (fine; per-demo plan times are 1-3 min so 10 s is plenty fast).
#
# Rules for this script:
#   * Single-shot: NO infinite loop, NO clear. Run-and-exit.
#   * Read-only:   touch nothing in $LOG_DIR or $OUT_DIR. Side-effect-
#                  free so `watch` can hammer it safely.
#   * Fast:        finish in well under 1 s on a warm cache. Avoid
#                  shelling out to gsutil/gcloud or doing network I/O.
#   * Portable to macOS BSD userland: no GNU-only flags (e.g. no
#                                     `stat --format`, no `date -d`,
#                                     no `ps -o cmd`).
#
# Output layout (one screen, ~24 rows max):
#
#   ============================================================
#   local pareto monitor                      2026-05-14T05:55:12Z
#   ============================================================
#   cell    progress    last plan    rate     ETA            state
#   k10-r2     3/60       121 s    121 s/d   1h 54m left    RUNNING
#   k20-r1     2/60       156 s    156 s/d   2h 30m left    RUNNING
#   k20-r2     1/60       198 s    198 s/d   3h 14m left    RUNNING
#   ------------------------------------------------------------
#   workers active : 6      aggregate CPU :   570 %
#   master runner  : pid=1714 elapsed=00:11:42
#   ============================================================
#   results in experiments/10_kxreplans_grid/results/:
#     (no JSON files yet)
#   ============================================================

set -u

# ---- repo root (resolve regardless of CWD) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-experiments/10_kxreplans_grid/results}"
LOG_DIR="${LOG_DIR:-logs/local_pareto}"

# Cells that the local runner is responsible for. Keep in sync with
# scripts/local/run_remaining_pareto.sh::CELLS .
CELLS=(
    "10 2"
    "20 1"
    "20 2"
)

# ---- ANSI helpers (no-op if stdout is not a tty / NO_COLOR is set) ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; RESET=""
fi

# Format a duration (seconds) as "Nh Mm" or "Mm Ss" depending on size.
fmt_dur() {
    local s="$1"
    if (( s <= 0 )); then echo "-"; return; fi
    if (( s >= 3600 )); then
        printf "%dh %02dm" $(( s / 3600 )) $(( (s % 3600) / 60 ))
    elif (( s >= 60 )); then
        printf "%dm %02ds" $(( s / 60 )) $(( s % 60 ))
    else
        printf "%ds" "$s"
    fi
}

# Find the most-recent log file for a given (K, R). The local runner
# stamps a UTC timestamp into the filename so multiple invocations
# accumulate; we want the latest run.
latest_log_for() {
    local K="$1" R="$2"
    # shellcheck disable=SC2012  # ls -t is intentional here
    ls -t "$LOG_DIR"/k${K}_r${R}_*.log 2>/dev/null | head -1
}

# Parse a single per-cell log and emit a tab-separated record:
#   progress_done<TAB>last_plan_s<TAB>state<TAB>log_path
parse_cell_log() {
    local log="$1"

    if [[ -z "$log" || ! -f "$log" ]]; then
        printf "0\t0\tNOLOG\t-\n"
        return
    fi

    # The run_can.py output line we anchor on:
    #   [exp05]  N/60  demo_X#Y  bc_rew=... (123.4s plan)
    # Robust to whitespace; capture N (demos done) and the trailing
    # plan-time in seconds.
    #
    # We use awk to keep this single-process (faster than grep | tail | sed).
    # Note: 3-arg match() with capture array is gawk-only, so we use the
    # POSIX 2-arg form + RSTART/RLENGTH for macOS BSD awk compatibility.
    awk -v log="$log" '
        # match lines like:
        #   [exp05]  3/60  demo_0#2  ...  (121.4s plan)
        /^\[exp05\][[:space:]]+[0-9]+\/[0-9]+[[:space:]]+demo_/ {
            # extract the N (progress count) -- 2nd field is "N/60"
            split($2, a, "/")
            done = a[1]
            # extract last "(XXX.Xs plan)" by matching the parenthesized tail
            if (match($0, /\([0-9]+(\.[0-9]+)?s plan\)/)) {
                chunk = substr($0, RSTART, RLENGTH)
                # strip "(" prefix and "s plan)" suffix
                sub(/^\(/, "", chunk)
                sub(/s plan\)$/, "", chunk)
                last_plan_s = chunk
            }
        }
        /^Traceback/      { traceback = 1 }
        /OverflowError/   { ofe = 1 }
        END {
            state = "RUNNING"
            if (traceback) state = "ERROR"
            if (ofe)       state = "ERROR_OVERFLOW"
            # Truncate last_plan_s to integer for column alignment.
            lp = (last_plan_s == "") ? 0 : int(last_plan_s + 0.5)
            printf "%d\t%d\t%s\t%s\n", done, lp, state, log
        }
    ' "$log" 2>/dev/null
}

# ---- header ----
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${BOLD}============================================================${RESET}"
echo "${BOLD}local pareto monitor${RESET}                      ${DIM}${NOW_UTC}${RESET}"
echo "${BOLD}============================================================${RESET}"
printf "${BOLD}%-7s  %-9s  %-9s  %-9s  %-13s  %s${RESET}\n" \
    "cell" "progress" "last plan" "rate" "ETA" "state"

# ---- per-cell rows ----
worst_eta_s=0   # track the slowest remaining cell for "all done by" line

for spec in "${CELLS[@]}"; do
    read -r K R <<< "$spec"
    slug="k${K}-r${R}"
    json="$OUT_DIR/K${K}_R${R}.json"

    # If the result JSON exists, the cell is DONE (period).
    if [[ -f "$json" ]]; then
        size=$(wc -c < "$json" | tr -d ' ')
        printf "${GREEN}%-7s  %-9s  %-9s  %-9s  %-13s  %s${RESET}\n" \
            "$slug" "60/60" "-" "-" "-" "DONE (${size}B)"
        continue
    fi

    log="$(latest_log_for "$K" "$R")"
    IFS=$'\t' read -r done last_plan state _log_path < <(parse_cell_log "$log")

    # Compute rate (s/demo) and ETA based on observed last_plan.
    if (( done > 0 )) && (( last_plan > 0 )); then
        remaining=$(( 60 - done ))
        eta_s=$(( remaining * last_plan ))
        rate_str="${last_plan} s/d"
        eta_str="$(fmt_dur "$eta_s") left"
        last_str="${last_plan} s"
        if (( eta_s > worst_eta_s )); then worst_eta_s=$eta_s; fi
    elif (( done == 0 )) && [[ "$state" == "RUNNING" ]]; then
        rate_str="-"
        eta_str="${DIM}warming up${RESET}"
        last_str="-"
    else
        rate_str="-"
        eta_str="-"
        last_str="-"
    fi

    color=""
    case "$state" in
        RUNNING)         color="${CYAN}";  display="$state" ;;
        ERROR|ERROR_OVERFLOW) color="${RED}";   display="$state" ;;
        NOLOG)           color="${YELLOW}";display="NO LOG YET" ;;
        *)               color="";         display="$state" ;;
    esac

    printf "%-7s  %-9s  %-9s  %-9s  %-13b  ${color}%s${RESET}\n" \
        "$slug" "${done}/60" "$last_str" "$rate_str" "$eta_str" "$display"
done

echo "${DIM}------------------------------------------------------------${RESET}"

# ---- aggregate process info ----
# Count multiprocessing worker processes (parents = run_can.py mains).
# Use macOS-compatible ps fields.
WORKERS=$(pgrep -f 'multiprocessing.spawn' 2>/dev/null | wc -l | tr -d ' ')
MAINS=$(pgrep -f 'run_can\.py' 2>/dev/null | wc -l | tr -d ' ')
TOTAL_CPU=$(ps -A -o pcpu,command 2>/dev/null \
    | awk '/multiprocessing\.spawn|run_can\.py/ && !/grep/ {sum+=$1} END {printf "%.0f", sum}')
[[ -z "$TOTAL_CPU" ]] && TOTAL_CPU=0

printf "${DIM}workers active : %s      aggregate CPU : %5s %%${RESET}\n" \
    "$WORKERS" "$TOTAL_CPU"

# Master runner: parent of all main run_can.py processes.
# (Find a main, then look up its ppid.)
MAIN_PID="$(pgrep -f 'run_can\.py' 2>/dev/null | head -1)"
if [[ -n "$MAIN_PID" ]]; then
    MASTER_PID="$(ps -o ppid= -p "$MAIN_PID" 2>/dev/null | tr -d ' ')"
    if [[ -n "$MASTER_PID" ]]; then
        ETIME="$(ps -o etime= -p "$MASTER_PID" 2>/dev/null | tr -d ' ')"
        printf "${DIM}master runner  : pid=%s  elapsed=%s${RESET}\n" \
            "$MASTER_PID" "${ETIME:-?}"
    fi
fi

# ---- overall ETA + result file listing ----
echo "${BOLD}============================================================${RESET}"
if (( worst_eta_s > 0 )); then
    # Use python to add seconds to "now" -- portable across BSD/GNU date.
    finish_utc="$(python3 -c "
import datetime as dt
now = dt.datetime.utcnow()
eta = now + dt.timedelta(seconds=$worst_eta_s)
print(eta.strftime('%H:%M UTC'))
" 2>/dev/null)"
    echo "slowest cell ETA: $(fmt_dur "$worst_eta_s") (~$finish_utc) -- all 3 done at that point"
fi
echo "results in $OUT_DIR:"
if compgen -G "$OUT_DIR/K*_R*.json" >/dev/null 2>&1; then
    # shellcheck disable=SC2012  # ls is fine for human-readable size
    ls -la "$OUT_DIR"/K*_R*.json | awk '{ printf "  %s  %s\n", $5, $NF }'
else
    echo "${DIM}  (no JSON files yet)${RESET}"
fi
echo "${BOLD}============================================================${RESET}"
