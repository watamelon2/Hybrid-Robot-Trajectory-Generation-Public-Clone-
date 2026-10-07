#!/usr/bin/env bash
# One-shot status snapshot for the local lambda-BC sweep runner.
# Sibling to scripts/local/monitor_pareto.sh -- same usage pattern:
#
#   watch -n 10 -c bash scripts/local/monitor_lambda_bc.sh
#
# To watch BOTH pareto and lambda-bc on a single screen:
#
#   watch -n 10 -c 'bash scripts/local/monitor_pareto.sh; \
#                   echo; \
#                   bash scripts/local/monitor_lambda_bc.sh'
#
# Output format mirrors monitor_pareto.sh so both are visually
# consistent and parseable by the same eye. The progress line we
# anchor on in the exp 07 logs is:
#
#   [exp07] lam=  0.1    3  demo_0#2  exec=+1.42  success=False
#
# We extract:
#   * $4 = rollout count so far (1..60)
#   * elapsed time of the process (via ps) to compute rate
#
# Note: exp 07 does NOT print per-rollout plan time, so unlike the
# pareto monitor we estimate rate from (elapsed / count) instead of
# the most-recent plan time. The estimate is more stable but slower
# to react to current-rate changes.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-experiments/07_lambda_sweep/results}"
LOG_DIR="${LOG_DIR:-logs/local_lambda_bc}"

# Lambdas tracked by this monitor. Keep in sync with
# scripts/local/run_remaining_lambda_bc.sh::LAMBDAS .
LAMBDAS=(0.1 0.5 1.0 5.0)

# ---- ANSI helpers ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; RESET=""
fi

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

# Convert "MM:SS" or "HH:MM:SS" or "DD-HH:MM:SS" (ps etime format) to seconds.
etime_to_s() {
    local et="$1"
    awk -v t="$et" '
        BEGIN {
            n = split(t, dh, "-")
            d = 0
            if (n == 2) { d = dh[1]; t = dh[2] } else { t = dh[1] }
            n = split(t, hms, ":")
            if (n == 3)      { secs = hms[1]*3600 + hms[2]*60 + hms[3] }
            else if (n == 2) { secs = hms[1]*60 + hms[2] }
            else             { secs = hms[1] }
            print d*86400 + secs
        }
    '
}

# Latest log file for a lambda value.
latest_log_for() {
    local lam="$1"
    local fslug
    fslug="$(echo "lam${lam}" | tr '.' '-')"
    # shellcheck disable=SC2012
    ls -t "$LOG_DIR"/${fslug}_*.log 2>/dev/null | head -1
}

# Parse a lambda-bc log; emit:
#   progress_done<TAB>state<TAB>main_pid
parse_cell_log() {
    local log="$1"
    if [[ -z "$log" || ! -f "$log" ]]; then
        printf "0\tNOLOG\t\n"
        return
    fi
    awk '
        # match per-rollout progress lines:
        #   [exp07] lam=  0.1    3  demo_0#2  exec=...  success=...
        /^\[exp07\][[:space:]]+lam=[[:space:]]*[0-9.]+[[:space:]]+[0-9]+[[:space:]]+demo_/ {
            done = $4 + 0
        }
        # match the "wrote" line that exp 07 emits at end of each lambda
        /^\[exp07\] wrote/ {
            wrote = 1
        }
        /^Traceback/    { traceback = 1 }
        /OverflowError/ { ofe = 1 }
        END {
            state = "RUNNING"
            if (traceback) state = "ERROR"
            if (ofe)       state = "ERROR_OVERFLOW"
            if (wrote)     state = "WROTE"
            printf "%d\t%s\n", done, state
        }
    ' "$log" 2>/dev/null
}

# ---- header ----
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${BOLD}============================================================${RESET}"
echo "${BOLD}local lambda-bc monitor${RESET}                   ${DIM}${NOW_UTC}${RESET}"
echo "${BOLD}============================================================${RESET}"
printf "${BOLD}%-9s  %-9s  %-9s  %-9s  %-13s  %s${RESET}\n" \
    "lambda" "progress" "elapsed" "rate" "ETA" "state"

worst_eta_s=0

for lam in "${LAMBDAS[@]}"; do
    json="$OUT_DIR/lambda_bc_${lam}.json"
    if [[ -f "$json" ]]; then
        size=$(wc -c < "$json" | tr -d ' ')
        printf "${GREEN}%-9s  %-9s  %-9s  %-9s  %-13s  %s${RESET}\n" \
            "lam=$lam" "60/60" "-" "-" "-" "DONE (${size}B)"
        continue
    fi

    log="$(latest_log_for "$lam")"
    IFS=$'\t' read -r done state < <(parse_cell_log "$log")

    # Find the main pid for this cell (by ppid of any matching worker, or via the log mtime).
    # Simpler: look for run_can.py + matching --lambda-prior $lam.
    main_pid="$(pgrep -f "07_lambda_sweep/run_can.py.*--lambda-prior ${lam}( |$)" 2>/dev/null | head -1)"
    elapsed_s=0
    if [[ -n "$main_pid" ]]; then
        etime_str="$(ps -o etime= -p "$main_pid" 2>/dev/null | tr -d ' ')"
        if [[ -n "$etime_str" ]]; then
            elapsed_s=$(etime_to_s "$etime_str")
        fi
    fi

    if (( done > 0 )) && (( elapsed_s > 0 )); then
        # Rate = elapsed / done (s per demo, averaged over the run so far).
        rate_s=$(( elapsed_s / done ))
        remaining=$(( 60 - done ))
        eta_s=$(( remaining * rate_s ))
        rate_str="${rate_s} s/d"
        eta_str="$(fmt_dur "$eta_s") left"
        elapsed_str="$(fmt_dur "$elapsed_s")"
        if (( eta_s > worst_eta_s )); then worst_eta_s=$eta_s; fi
    elif [[ "$state" == "RUNNING" ]] && (( done == 0 )); then
        rate_str="-"
        eta_str="${DIM}warming up${RESET}"
        elapsed_str="$(fmt_dur "$elapsed_s")"
    else
        rate_str="-"
        eta_str="-"
        elapsed_str="-"
    fi

    color=""
    case "$state" in
        RUNNING)              color="${CYAN}";   display="$state" ;;
        WROTE)                color="${GREEN}";  display="WROTE (waiting for next?)" ;;
        ERROR|ERROR_OVERFLOW) color="${RED}";    display="$state" ;;
        NOLOG)                color="${YELLOW}"; display="NO LOG YET" ;;
        *)                    color="";          display="$state" ;;
    esac

    printf "%-9s  %-9s  %-9s  %-9s  %-13b  ${color}%s${RESET}\n" \
        "lam=$lam" "${done}/60" "$elapsed_str" "$rate_str" "$eta_str" "$display"
done

echo "${DIM}------------------------------------------------------------${RESET}"

# Aggregate compute info for the lambda-bc cells only (ppid filter).
LAMBDA_MAIN_PIDS=$(pgrep -f "07_lambda_sweep/run_can.py" 2>/dev/null | tr '\n' '|' | sed 's/|$//')
if [[ -n "$LAMBDA_MAIN_PIDS" ]]; then
    LAMBDA_WORKERS=$(ps -eo pid,ppid 2>/dev/null \
        | awk -v pids="$LAMBDA_MAIN_PIDS" 'BEGIN{n=split(pids,a,"|"); for(i=1;i<=n;i++) p[a[i]]=1}
                                          NR>1 && p[$2] {c++} END{print c+0}')
    LAMBDA_CPU=$(ps -eo pid,ppid,pcpu 2>/dev/null \
        | awk -v pids="$LAMBDA_MAIN_PIDS" 'BEGIN{n=split(pids,a,"|"); for(i=1;i<=n;i++) p[a[i]]=1}
                                          NR>1 && p[$2] {s+=$3} END{printf "%.0f", s}')
    printf "${DIM}workers active : %s      aggregate CPU : %5s %%${RESET}\n" \
        "$LAMBDA_WORKERS" "$LAMBDA_CPU"
fi

# Master runner: parent of any lambda-bc main.
ANY_MAIN="$(pgrep -f "07_lambda_sweep/run_can.py" 2>/dev/null | head -1)"
if [[ -n "$ANY_MAIN" ]]; then
    MASTER_PID="$(ps -o ppid= -p "$ANY_MAIN" 2>/dev/null | tr -d ' ')"
    if [[ -n "$MASTER_PID" ]]; then
        ETIME="$(ps -o etime= -p "$MASTER_PID" 2>/dev/null | tr -d ' ')"
        printf "${DIM}master runner  : pid=%s  elapsed=%s${RESET}\n" \
            "$MASTER_PID" "${ETIME:-?}"
    fi
fi

# Overall ETA and result file listing.
echo "${BOLD}============================================================${RESET}"
if (( worst_eta_s > 0 )); then
    finish_utc="$(python3 -c "
import datetime as dt
now = dt.datetime.utcnow()
eta = now + dt.timedelta(seconds=$worst_eta_s)
print(eta.strftime('%H:%M UTC'))
" 2>/dev/null)"
    echo "slowest cell ETA: $(fmt_dur "$worst_eta_s") (~$finish_utc) -- all 4 done at that point"
fi
echo "results in $OUT_DIR (filtered to lambda_bc_*):"
if compgen -G "$OUT_DIR/lambda_bc_*.json" >/dev/null 2>&1; then
    # shellcheck disable=SC2012
    ls -la "$OUT_DIR"/lambda_bc_*.json | awk '{ printf "  %s  %s\n", $5, $NF }'
else
    echo "${DIM}  (no JSON files yet)${RESET}"
fi
echo "${BOLD}============================================================${RESET}"
