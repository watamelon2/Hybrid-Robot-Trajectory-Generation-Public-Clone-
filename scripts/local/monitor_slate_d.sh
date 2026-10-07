#!/usr/bin/env bash
# One-shot status snapshot for Slate D
# (experiments/15_unified_hybrid/slate_d_2026_05_18.sh).
#
# Sibling to scripts/local/monitor_sweep_v2.sh -- same usage pattern:
#
#   watch -n 10 -c bash scripts/local/monitor_slate_d.sh
#
# Layout this monitor anchors on:
#
#   results/slate_d_2026_05_18/
#     unified_bc_lam-0.0_K-10_R-1_D1_bc_lam0_seed{1,2,3}.json     (D1, 3 cells)
#     unified_demo_1nn_lam-{0.5,1.0,2.0,10.0,20.0}_K-10_R-1_D2_1nn_lamsweep.json
#                                                                  (D2, 5 cells)
#     unified_demo_1nn_lam-5.0_K-10_R-1_D3_1nn_sigma{0.50,0.75,1.00}.json
#                                                                  (D3, 3 cells)
#     unified_summary_*.json     (one summary per `run` invocation)
#     run.log                    (chained-blocks log)
#
# We anchor on the exp 15 progress line for in-flight cells:
#
#     [exp15] prior=demo_1nn lam=  0.0  R=1  K=10    9  demo_2#2  reward=+18.08  success=False
#
# extracting the rollout count ($7) and tallying success/total from
# the per-rollout `success=True/False` field.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SLATE_DIR="${SLATE_DIR:-experiments/15_unified_hybrid/results/slate_d_2026_05_18}"
LOG="${LOG:-$SLATE_DIR/run.log}"
TOTAL_CELLS=11
TOTAL_BUDGET_S=16200   # 4.5 h

# ---- ANSI helpers ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    CYAN=$'\033[36m'; MAGENTA=$'\033[35m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; MAGENTA=""; RESET=""
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

# Read n_success/n_total from a per-cell JSON. Echo "succ/total".
read_succ() {
    local json="$1"
    if [[ ! -f "$json" ]]; then echo ""; return; fi
    python3 -c "
import json
try:
    d = json.load(open('$json'))
    print(f\"{d['n_success']}/{d['n_total']}\")
except Exception:
    pass
" 2>/dev/null
}

# Highlight a succ string by absolute success count out of 60.
hl_succ() {
    local s="$1"
    if [[ -z "$s" ]]; then echo "-"; return; fi
    local n="${s%%/*}"
    if (( n >= 30 )); then echo "${GREEN}${BOLD}${s}${RESET}"
    elif (( n >= 15 )); then echo "${GREEN}${s}${RESET}"
    elif (( n >= 5 )); then echo "${YELLOW}${s}${RESET}"
    elif (( n > 0 )); then echo "${YELLOW}${DIM}${s}${RESET}"
    else                   echo "${DIM}${s}${RESET}"; fi
}

# Cache the currently-running run_can.py's --output-name-suffix once
# per monitor invocation (cheap; ps is sub-ms).  Empty if no process.
CUR_SUFFIX_CACHED=""
CUR_SUFFIX_RESOLVED=0
current_suffix() {
    if (( CUR_SUFFIX_RESOLVED == 1 )); then
        echo "$CUR_SUFFIX_CACHED"; return
    fi
    CUR_SUFFIX_RESOLVED=1
    local cmd
    cmd="$(pgrep -fl '15_unified_hybrid/run_can.py' 2>/dev/null | head -1 \
           | sed 's/^[0-9]* //')"
    if [[ -z "$cmd" ]]; then
        CUR_SUFFIX_CACHED=""
    else
        CUR_SUFFIX_CACHED="$(echo "$cmd" | grep -oE -- '--output-name-suffix [^ ]+' \
            | head -1 | awk '{print $2}')"
    fi
    echo "$CUR_SUFFIX_CACHED"
}

# In-flight live progress for the currently running cell.
# Returns "<count>/<total> succ=<n>" by reading the log from the last
# "[exp15] prior=" header line (which is emitted once per `run`
# invocation, i.e. once per cell except D2 which emits 5 cells in one
# invocation; for D2 we additionally scope on the lam= value).
#
# Args: prior, lambda (string, e.g. "0.0"), expected total (60)
in_flight_progress() {
    local prior="$1"
    local lam="$2"
    local total="$3"
    python3 - "$LOG" "$prior" "$lam" "$total" <<'PY' 2>/dev/null
import re, sys
log_path, prior, lam, total = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    lines = open(log_path, errors='replace').read().splitlines()
except FileNotFoundError:
    print("")
    sys.exit(0)
# Find the most recent HEADER line for this prior.  The header has
# the form `[exp15] prior=<p>  lambdas=(...)  K=...` -- distinct from
# the per-rollout progress lines which start with `[exp15] prior=<p>
# lam= ...`.
hdr_re = re.compile(r'^\[exp15\] prior=' + re.escape(prior) + r'\s+lambdas=')
# Last header for this prior.
hdr_idx = -1
for i, ln in enumerate(lines):
    if hdr_re.search(ln):
        hdr_idx = i
if hdr_idx < 0:
    print("")
    sys.exit(0)
# From hdr_idx forward, accumulate (count, success) for the given lam.
prog_re = re.compile(
    r'^\[exp15\] prior=' + re.escape(prior) +
    r'\s+lam=\s*([0-9.]+)\s+R=\d+\s+K=\d+\s+(\d+)\s+demo_\S+\s+reward=\S+\s+success=(True|False)'
)
last_count = 0
n_succ = 0
for ln in lines[hdr_idx:]:
    m = prog_re.match(ln)
    if not m:
        continue
    line_lam = m.group(1)
    # Normalise both numbers ("0.0" vs "0", "5.0" vs "5").
    try:
        if float(line_lam) != float(lam):
            continue
    except ValueError:
        continue
    c = int(m.group(2))
    if c < last_count:
        # Restart (next cell in a multi-lambda invocation); reset.
        last_count = c
        n_succ = 1 if m.group(3) == 'True' else 0
    else:
        last_count = c
        if m.group(3) == 'True':
            n_succ += 1
print(f"{last_count}/{total} succ={n_succ}")
PY
}

# Print a single cell row: "<label> <succ-or-progress>".
# Args: prior, lam, suffix-glob (relative to SLATE_DIR), label
print_cell() {
    local prior="$1"; local lam="$2"; local suffix="$3"; local label="$4"
    # Resolve lambda to the on-disk format: integers like 5 become 5.0.
    local lam_disk
    lam_disk="$(python3 -c "x='$lam'; v=float(x); print(f'{int(v)}.0' if v.is_integer() else f'{v}')")"
    local cell="$SLATE_DIR/unified_${prior}_lam-${lam_disk}_K-10_R-1${suffix}.json"
    if [[ -f "$cell" ]]; then
        printf "    %-32s  %b\n" "$label" "$(hl_succ "$(read_succ "$cell")")"
    else
        # Only show live progress if THIS cell's suffix matches the
        # currently-running run_can.py invocation.  Otherwise the cell
        # is either upcoming (no python process for it) or has been
        # superseded; show '-'.
        local cur_sfx prog
        cur_sfx="$(current_suffix)"
        prog=""
        if [[ -n "$cur_sfx" && "$cur_sfx" == "$suffix" ]]; then
            prog="$(in_flight_progress "$prior" "$lam_disk" 60)"
        fi
        if [[ -n "$prog" ]]; then
            # Color the live row cyan.
            local cnt succ
            cnt="$(echo "$prog" | awk '{print $1}')"
            succ="$(echo "$prog" | awk '{print $2}' | cut -d= -f2)"
            if [[ "$cnt" == "60/60" ]]; then
                printf "    %-32s  ${CYAN}%s${RESET}  ${DIM}(writing JSON...)${RESET}\n" \
                    "$label" "$succ/60"
            else
                printf "    %-32s  ${CYAN}%s${RESET}  ${DIM}(succ-so-far=%s)${RESET}\n" \
                    "$label" "$cnt" "$succ"
            fi
        else
            printf "    %-32s  ${DIM}-${RESET}\n" "$label"
        fi
    fi
}

# Count completed (non-summary) JSON cells in SLATE_DIR.
n_done_cells() {
    ls "$SLATE_DIR"/unified_*.json 2>/dev/null \
        | grep -v "/unified_summary_" | wc -l | tr -d ' '
}

# ---- header ----
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
NOW_LOC="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "${BOLD}===========================================================================${RESET}"
echo "${BOLD}Slate D monitor${RESET}    refreshed: ${DIM}${NOW_LOC}  (${NOW_UTC})${RESET}"
echo "${BOLD}===========================================================================${RESET}"

# ---- D1 -------------------------------------------------------------------
D1_DONE=0
for s in 1 2 3; do
    [[ -f "$SLATE_DIR/unified_bc_lam-0.0_K-10_R-1_D1_bc_lam0_seed${s}.json" ]] && D1_DONE=$((D1_DONE+1))
done
if (( D1_DONE == 3 )); then D1_STATE="${GREEN}DONE${RESET}"
elif (( D1_DONE > 0 )); then D1_STATE="${CYAN}${D1_DONE}/3${RESET}"
else D1_STATE="${CYAN}...${RESET}"; fi
echo "${BOLD}[D1]${RESET} BC λ=0 4-seed pool                                       ${D1_STATE}"
for s in 1 2 3; do
    print_cell "bc" "0" "_D1_bc_lam0_seed${s}" "seed=${s}"
done

# ---- D2 -------------------------------------------------------------------
D2_LAMS=(0.5 1 2 10 20)
D2_DONE=0
for lam in "${D2_LAMS[@]}"; do
    lam_disk="$(python3 -c "v=float('$lam'); print(f'{int(v)}.0' if v.is_integer() else f'{v}')")"
    [[ -f "$SLATE_DIR/unified_demo_1nn_lam-${lam_disk}_K-10_R-1_D2_1nn_lamsweep.json" ]] && D2_DONE=$((D2_DONE+1))
done
if (( D2_DONE == 5 )); then D2_STATE="${GREEN}DONE${RESET}"
elif (( D2_DONE > 0 )); then D2_STATE="${CYAN}${D2_DONE}/5${RESET}"
else D2_STATE="${DIM}waiting${RESET}"; fi
echo "${BOLD}[D2]${RESET} demo_1nn λ-sweep at σ=0.25 (seed=0)                      ${D2_STATE}"
for lam in "${D2_LAMS[@]}"; do
    print_cell "demo_1nn" "$lam" "_D2_1nn_lamsweep" "λ=${lam}"
done

# ---- D3 -------------------------------------------------------------------
D3_SIGS=(0.50 0.75 1.00)
D3_DONE=0
for sig in "${D3_SIGS[@]}"; do
    [[ -f "$SLATE_DIR/unified_demo_1nn_lam-5.0_K-10_R-1_D3_1nn_sigma${sig}.json" ]] && D3_DONE=$((D3_DONE+1))
done
if (( D3_DONE == 3 )); then D3_STATE="${GREEN}DONE${RESET}"
elif (( D3_DONE > 0 )); then D3_STATE="${CYAN}${D3_DONE}/3${RESET}"
else D3_STATE="${DIM}waiting${RESET}"; fi
echo "${BOLD}[D3]${RESET} demo_1nn σ-tail at λ=5 (seed=0)                          ${D3_STATE}"
for sig in "${D3_SIGS[@]}"; do
    print_cell "demo_1nn" "5" "_D3_1nn_sigma${sig}" "σ=${sig}"
done

# ---- Active processes -----------------------------------------------------
echo "${DIM}---------------------------------------------------------------------------${RESET}"

# Master slate runner.
SLATE_BASH="$(pgrep -fl 'bash .*slate_d_2026_05_18\.sh' 2>/dev/null | head -1)"
if [[ -n "$SLATE_BASH" ]]; then
    BASH_PID="$(echo "$SLATE_BASH" | awk '{print $1}')"
    ETIME="$(ps -o etime= -p "$BASH_PID" 2>/dev/null | tr -d ' ')"
    ETIME_S="$(etime_to_s "${ETIME:-0}")"
    printf "${DIM}runner    : pid=%s  elapsed=%s${RESET}\n" \
        "$BASH_PID" "$(fmt_dur "$ETIME_S")"
    # ETA based on cells-done / cells-total fraction of budget.
    DONE="$(n_done_cells)"
    if (( DONE > 0 )); then
        # Linear extrapolation from elapsed.
        REMAIN_S=$(( ETIME_S * (TOTAL_CELLS - DONE) / DONE ))
        ETA_EPOCH=$(( $(date +%s) + REMAIN_S ))
        ETA_HM="$(date -r "$ETA_EPOCH" '+%H:%M %Z' 2>/dev/null || date -d @"$ETA_EPOCH" '+%H:%M %Z' 2>/dev/null)"
        printf "${DIM}            cells_done=%d/%d  ETA finish=%s  (in %s)${RESET}\n" \
            "$DONE" "$TOTAL_CELLS" "${ETA_HM:-?}" "$(fmt_dur "$REMAIN_S")"
    else
        BUDGET_LEFT_S=$(( TOTAL_BUDGET_S - ETIME_S ))
        printf "${DIM}            cells_done=0/%d  budget remaining=%s${RESET}\n" \
            "$TOTAL_CELLS" "$(fmt_dur "$BUDGET_LEFT_S")"
    fi
else
    DONE="$(n_done_cells)"
    if (( DONE == TOTAL_CELLS )); then
        printf "${DIM}runner    : ${GREEN}FINISHED${RESET}${DIM}  (all %d cells written)${RESET}\n" \
            "$TOTAL_CELLS"
    else
        printf "${DIM}runner    : ${RED}NOT RUNNING${RESET}${DIM}  (%d/%d cells done -- check screen)${RESET}\n" \
            "$DONE" "$TOTAL_CELLS"
    fi
fi

# Currently-active run_can.py for exp 15.
CUR_RUN="$(pgrep -fl '15_unified_hybrid/run_can.py' 2>/dev/null | head -1)"
if [[ -n "$CUR_RUN" ]]; then
    CUR_PID="$(echo "$CUR_RUN" | awk '{print $1}')"
    CUR_ETIME="$(ps -o etime= -p "$CUR_PID" 2>/dev/null | tr -d ' ')"
    CUR_CMD="$(ps -o command= -p "$CUR_PID" 2>/dev/null)"
    # --output-name-suffix is the cell tag.
    CUR_TAG="$(echo "$CUR_CMD" | grep -oE -- '--output-name-suffix [^ ]+' \
        | head -1 | awk '{print $2}')"
    # --prior and --lambda-prior <list>.
    CUR_PRIOR="$(echo "$CUR_CMD" | grep -oE -- '--prior [^ ]+' | head -1 | awk '{print $2}')"
    CUR_LAMS="$(echo "$CUR_CMD" | sed -nE 's/.*--lambda-prior ([0-9. ]+).*/\1/p' \
        | head -1 | sed -E 's/ +-.*//')"
    printf "${DIM}exp15 run : pid=%s  elapsed=%s${RESET}\n" "$CUR_PID" "${CUR_ETIME:-?}"
    printf "${DIM}            tag=%s  prior=%s  lambdas=[%s]${RESET}\n" \
        "${CUR_TAG:-?}" "${CUR_PRIOR:-?}" "${CUR_LAMS:-?}"

    # Most recent rollout-progress line (filter out the header line).
    LAST_PLAN="$(grep -E '^\[exp15\] prior=.* demo_[0-9]+#[0-9]+' "$LOG" 2>/dev/null \
        | tail -1)"
    if [[ -n "$LAST_PLAN" ]]; then
        # Squeeze whitespace for one-line display.
        printf "${DIM}            last: %s${RESET}\n" \
            "$(echo "$LAST_PLAN" | sed 's/  */ /g')"
    fi

    # Aggregate worker count + cpu under this PID.
    W_COUNT=$(ps -eo ppid 2>/dev/null | awk -v p="$CUR_PID" '$1==p' | wc -l | tr -d ' ')
    W_CPU=$(ps -eo ppid,pcpu 2>/dev/null | awk -v p="$CUR_PID" '$1==p {s+=$2} END{printf "%.0f", s}')
    printf "${DIM}            workers=%s  aggregate CPU=%s%%${RESET}\n" "$W_COUNT" "${W_CPU:-0}"
fi

# Screen session.
SCREEN_LINE="$(screen -ls 2>/dev/null | grep -E 'slate_d' | head -1)"
if [[ -n "$SCREEN_LINE" ]]; then
    printf "${DIM}screen    : %s${RESET}\n" "$(echo "$SCREEN_LINE" | sed 's/^[[:space:]]*//')"
fi

# ---- Best so far across all completed Slate D cells ----------------------
echo "${BOLD}===========================================================================${RESET}"
BEST_LINE="$(python3 - "$SLATE_DIR" <<'PY' 2>/dev/null
import json, sys, glob, os
slate = sys.argv[1]
best = None
for j in sorted(glob.glob(os.path.join(slate, 'unified_*.json'))):
    if '/unified_summary_' in j: continue
    try:
        d = json.load(open(j))
    except Exception:
        continue
    n = d.get('n_success', 0); tot = d.get('n_total', 0)
    if tot == 0: continue
    if best is None or n > best[1]:
        prior = d.get('prior', '?'); lam = d.get('lambda_prior', '?')
        sigma = d.get('params', {}).get('sigma_max', '?')
        seed = d.get('seed', '?')
        label = f'prior={prior:10s} λ={lam:>4}  σ={sigma}  seed={seed}'
        best = (label, n, tot)
if best is None:
    print('no completed Slate D cells yet')
else:
    label, n, tot = best
    print(f'BEST so far : {label}  ->  {n}/{tot} = {n/tot*100:.1f}%')
PY
)"
if [[ -n "$BEST_LINE" ]]; then
    echo "${BOLD}${GREEN}${BEST_LINE}${RESET}"
fi

# ---- Tail of log ---------------------------------------------------------
echo "${BOLD}===========================================================================${RESET}"
echo "${DIM}run.log tail:${RESET}"
tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
echo "${BOLD}===========================================================================${RESET}"
