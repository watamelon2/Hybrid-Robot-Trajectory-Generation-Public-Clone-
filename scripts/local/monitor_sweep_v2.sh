#!/usr/bin/env bash
# One-shot status snapshot for the v_theta_v2 sweep
# (experiments/08_flow_matching_prior/sweep_v2.sh + sweep_v2_ext.sh).
#
# Sibling to scripts/local/monitor_lambda_bc.sh -- same usage pattern:
#
#   watch -n 10 -c bash scripts/local/monitor_sweep_v2.sh
#
# Layout this monitor anchors on:
#
#   results/sweep_v2_2026_05_16/
#     phase0_prior_only_replay.json            (Phase 0 -- single JSON)
#     phase1_lambda_sweep_warmstart/
#         lambda_{0.0,0.1,...,20.0}.json       (Phase 1 -- 8 lambdas)
#         summary.json
#     phase2_lambda_sweep_no_warmstart/
#         lambda_{0.0,0.5,2.0}.json            (Phase 2 -- 3 lambdas)
#         summary.json
#     phase5_sigma_preservation/
#         sigma_{0.02,0.05,0.1}/lambda_{0.0,0.1,0.5}.json   (Phase 5 -- 3x3 grid)
#         summary.json
#     phase6_best_k_extension/
#         lambda_*.json (Phase 6 -- one lambda value at K=20)
#     sweep.log                                  (chained-phases log)
#
# Phases 3 and 4 are skipped at runtime; this monitor displays a
# greyed-out "SKIPPED" row for them so you can confirm the skip
# took effect.
#
# We anchor on the exp 07 progress line for in-flight cells:
#
#     [exp07] lam=  0.5   21  demo_6#2  exec=+27.36  success=False
#
# extracting the rollout count ($4 of that line).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SWEEP_DIR="${SWEEP_DIR:-experiments/08_flow_matching_prior/results/sweep_v2_2026_05_16}"
LOG="${LOG:-$SWEEP_DIR/sweep.log}"

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

# Convert ps etime (MM:SS / HH:MM:SS / DD-HH:MM:SS) to seconds.
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
import json, sys
try:
    d = json.load(open('$json'))
    print(f\"{d['n_success']}/{d['n_total']}\")
except Exception:
    pass
" 2>/dev/null
}

# Highlight a succ string if >0/60.
hl_succ() {
    local s="$1"
    if [[ -z "$s" ]]; then echo "-"; return; fi
    local n="${s%%/*}"
    if (( n >= 10 )); then echo "${GREEN}${BOLD}${s}${RESET}"
    elif (( n >= 5 )); then echo "${GREEN}${s}${RESET}"
    elif (( n > 0 )); then echo "${YELLOW}${s}${RESET}"
    else                  echo "${DIM}${s}${RESET}"; fi
}

# Find the python run_can.py process running for a given output-dir suffix
# (e.g. phase5_sigma_preservation/sigma_0.1). Echo the pid or empty.
# NB: variable name is "pat" not "sub" -- "sub" is BSD awk's built-in
# substitution function and using it as -v name triggers a parse error.
find_main_pid_for_outdir() {
    local outdir_substr="$1"
    pgrep -fl "07_lambda_sweep/run_can.py" 2>/dev/null \
        | awk -v pat="$outdir_substr" '$0 ~ pat {print $1; exit}'
}

# In-flight plan count for a running cell -- the highest rollout-counter we've
# seen in the sweep log since the cell's output-dir flag last appeared.
in_flight_plan_count() {
    local outdir_substr="$1"
    awk -v pat="$outdir_substr" '
        # Anchor: "output_dir=...phaseN/cell" line written when the cell starts.
        index($0, pat) { in_cell = 1; cur = 0; next }
        # Sub-line of progress: [exp07] lam=  X.X   N  demo_K#R  ...
        in_cell && /^\[exp07\][[:space:]]+lam=[[:space:]]*[0-9.]+[[:space:]]+[0-9]+[[:space:]]+demo_/ {
            cur = $4 + 0
        }
        END { print cur+0 }
    ' "$LOG" 2>/dev/null
}

# ---- header ----
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${BOLD}===========================================================================${RESET}"
echo "${BOLD}sweep_v2 monitor${RESET}    refreshed: ${DIM}${NOW_UTC}${RESET}"
echo "${BOLD}===========================================================================${RESET}"

# ---- Phase 0 ---------------------------------------------------------------
P0="$SWEEP_DIR/phase0_prior_only_replay.json"
if [[ -f "$P0" ]]; then
    P0_SUCC="$(read_succ "$P0")"
    P0_RATE="$(python3 -c "
import json
d = json.load(open('$P0'))
print(f\"{d['success_rate']*100:.1f}%\")
" 2>/dev/null)"
    printf "${BOLD}[Phase 0]${RESET} prior-only replay (no MBD)        " \
        ; echo -e "$(hl_succ "$P0_SUCC")  rate=${P0_RATE}"
else
    echo "${BOLD}[Phase 0]${RESET} ${DIM}not yet started${RESET}"
fi

# ---- Phase 1 ---------------------------------------------------------------
P1_DIR="$SWEEP_DIR/phase1_lambda_sweep_warmstart"
P1_SUMMARY="$P1_DIR/summary.json"
P1_LAMS=(0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0)
if [[ -f "$P1_SUMMARY" ]]; then
    P1_STATE="DONE"; P1_COLOR="${GREEN}"
else
    P1_STATE="..."; P1_COLOR="${CYAN}"
fi
echo "${BOLD}[Phase 1]${RESET} lambda-sweep WITH warm-start, sigma_max=0.5  ${P1_COLOR}${P1_STATE}${RESET}"
printf "  "
for lam in "${P1_LAMS[@]}"; do
    cell="$P1_DIR/lambda_${lam}.json"
    if [[ -f "$cell" ]]; then
        printf "%s=%b  " "λ=$lam" "$(hl_succ "$(read_succ "$cell")")"
    else
        # In-flight? Probe live progress.
        pid="$(find_main_pid_for_outdir "phase1_lambda_sweep_warmstart")"
        if [[ -n "$pid" ]]; then
            now="$(grep -E "^\[exp07\] lam=[[:space:]]*${lam}[[:space:]]" "$LOG" 2>/dev/null \
                | tail -1 | awk '{print $4}')"
            now="${now:-0}"
            printf "%s=${CYAN}%s/60${RESET}  " "λ=$lam" "$now"
        else
            printf "${DIM}%s=-${RESET}     " "λ=$lam"
        fi
    fi
done
echo

# ---- Phase 2 ---------------------------------------------------------------
P2_DIR="$SWEEP_DIR/phase2_lambda_sweep_no_warmstart"
P2_SUMMARY="$P2_DIR/summary.json"
P2_LAMS=(0.0 0.5 2.0)
if [[ -f "$P2_SUMMARY" ]]; then
    P2_STATE="DONE"; P2_COLOR="${GREEN}"
else
    P2_STATE="..."; P2_COLOR="${CYAN}"
fi
echo "${BOLD}[Phase 2]${RESET} lambda-sweep NO warm-start, sigma_max=0.5    ${P2_COLOR}${P2_STATE}${RESET}"
printf "  "
for lam in "${P2_LAMS[@]}"; do
    cell="$P2_DIR/lambda_${lam}.json"
    if [[ -f "$cell" ]]; then
        printf "%s=%b  " "λ=$lam" "$(hl_succ "$(read_succ "$cell")")"
    else
        printf "${DIM}%s=-${RESET}     " "λ=$lam"
    fi
done
echo

# ---- Phases 3 & 4 (skipped) ------------------------------------------------
echo "${BOLD}[Phase 3]${RESET} ${DIM}K-extension at wrong sigma         SKIPPED${RESET}"
echo "${BOLD}[Phase 4]${RESET} ${DIM}sigma in {0.1, 0.25}              SKIPPED${RESET}"

# ---- Phase 5 (the headline) ------------------------------------------------
P5_DIR="$SWEEP_DIR/phase5_sigma_preservation"
P5_SUMMARY="$P5_DIR/summary.json"
P5_SIGS=(0.02 0.05 0.1)
P5_LAMS=(0.0 0.1 0.5)
if [[ -f "$P5_SUMMARY" ]]; then
    P5_STATE="DONE"; P5_COLOR="${GREEN}"
else
    P5_STATE="..."; P5_COLOR="${CYAN}"
fi
echo "${BOLD}[Phase 5]${RESET} ${MAGENTA}σ-preservation × λ cross sweep${RESET}                ${P5_COLOR}${P5_STATE}${RESET}"
# Grid header
printf "    %-9s" ""
for lam in "${P5_LAMS[@]}"; do printf "  ${BOLD}%-10s${RESET}" "λ=$lam"; done
echo
for sig in "${P5_SIGS[@]}"; do
    printf "    ${BOLD}σ=%-7s${RESET}" "$sig"
    for lam in "${P5_LAMS[@]}"; do
        cell="$P5_DIR/sigma_${sig}/lambda_${lam}.json"
        if [[ -f "$cell" ]]; then
            printf "  %-10b" "$(hl_succ "$(read_succ "$cell")")"
        else
            # Probe in-flight progress: this cell is "the one running" if a
            # run_can.py is alive with --output-dir matching sigma_${sig} AND
            # the last [exp07] lam= line matches ${lam}. We don't try to be
            # too clever; we just grep the log.
            pid="$(find_main_pid_for_outdir "phase5_sigma_preservation/sigma_${sig}")"
            if [[ -n "$pid" ]]; then
                # Last plan line for this lambda anywhere in the log.
                last_for_lam="$(grep -E "^\[exp07\][[:space:]]+lam=[[:space:]]*${lam}[[:space:]]" "$LOG" 2>/dev/null | tail -1)"
                # Last plan line for ANY lambda.
                last_any="$(grep -E "^\[exp07\][[:space:]]+lam=" "$LOG" 2>/dev/null | tail -1)"
                if [[ -n "$last_for_lam" && "$last_for_lam" == "$last_any" ]]; then
                    cnt="$(echo "$last_for_lam" | awk '{print $4}')"
                    cnt="${cnt:-0}"
                    printf "  ${CYAN}%-10s${RESET}" "${cnt}/60"
                else
                    printf "  ${DIM}%-10s${RESET}" "..."
                fi
            else
                printf "  ${DIM}%-10s${RESET}" "-"
            fi
        fi
    done
    echo
done

# ---- Phase 6 ---------------------------------------------------------------
P6_DIR="$SWEEP_DIR/phase6_best_k_extension"
P6_SUMMARY="$P6_DIR/summary.json"
if [[ -f "$P6_SUMMARY" ]]; then
    # Could be either a populated summary or "[]" (skipped because Phase 5 best <= 5).
    P6_SUMMARY_LEN="$(python3 -c "
import json
print(len(json.load(open('$P6_SUMMARY'))))
" 2>/dev/null)"
    if [[ "$P6_SUMMARY_LEN" == "0" ]]; then
        echo "${BOLD}[Phase 6]${RESET} ${DIM}K=20 best-cell extension       SKIPPED (Phase 5 best <= 5)${RESET}"
    else
        echo "${BOLD}[Phase 6]${RESET} ${GREEN}K=20 best-cell extension       DONE${RESET}"
        for cell in "$P6_DIR"/lambda_*.json; do
            [[ -f "$cell" ]] || continue
            base="$(basename "$cell" .json)"
            printf "    %-12s %b\n" "$base" "$(hl_succ "$(read_succ "$cell")")"
        done
    fi
else
    echo "${BOLD}[Phase 6]${RESET} ${DIM}K=20 best-cell extension          (waiting for Phase 5)${RESET}"
fi

# ---- Phase 7 ---------------------------------------------------------------
# Score-bridge formula contingency: sigma=0.02 with --bridge-formula quadratic
# at lambda in {0.1, 0.5, 1.0}.  Only shown if the dir exists.
P7_DIR="$SWEEP_DIR/phase7_bridge_quadratic"
P7_SUMMARY="$P7_DIR/summary.json"
if [[ -d "$P7_DIR" ]]; then
    if [[ -f "$P7_SUMMARY" ]]; then
        echo "${BOLD}[Phase 7]${RESET} ${GREEN}bridge=quadratic, σ=0.02            DONE${RESET}"
        # summary.json is the raw per-lambda list from run_can.py.
        python3 -c "
import json
rows = json.load(open('$P7_SUMMARY'))
for r in rows:
    lam = r.get('lambda_prior', '?')
    n = r.get('n_success', 0)
    tot = r.get('n_total', 60)
    print(f'    λ={lam}  {n}/{tot}')
" 2>/dev/null
    else
        # In flight: probe the log for the most recent plan line.
        echo "${BOLD}[Phase 7]${RESET} ${YELLOW}bridge=quadratic, σ=0.02            ...${RESET}"
        LAST_P7="$(grep -E '^\[exp07\].*flow\[quadratic\]|^\[exp07\] lam=' "$LOG" 2>/dev/null | tail -1)"
        if [[ -n "$LAST_P7" ]]; then
            printf "    %s\n" "$(echo "$LAST_P7" | sed 's/  */ /g')"
        fi
    fi
fi

# ---- Active processes ------------------------------------------------------
echo "${DIM}---------------------------------------------------------------------------${RESET}"

# Master sweep runner: sweep_v2.sh, sweep_v2_ext.sh, or phase7_bridge_fix.sh.
# Keep the '/' anchor or stray zsh state dumps that mention the script name
# inside their env get matched.
SWEEP_BASH="$(pgrep -fl 'bash .*/(sweep_v2(\.sh|_ext\.sh)|phase7_bridge_fix\.sh)' 2>/dev/null | head -1)"
if [[ -n "$SWEEP_BASH" ]]; then
    BASH_PID="$(echo "$SWEEP_BASH" | awk '{print $1}')"
    BASH_NAME="$(echo "$SWEEP_BASH" | awk '{print $NF}' | xargs basename 2>/dev/null)"
    ETIME="$(ps -o etime= -p "$BASH_PID" 2>/dev/null | tr -d ' ')"
    printf "${DIM}runner    : pid=%s  %s  elapsed=%s${RESET}\n" \
        "$BASH_PID" "$BASH_NAME" "${ETIME:-?}"
else
    echo "${DIM}runner    : ${RED}NOT RUNNING${RESET}${DIM} (sweep may have finished or crashed)${RESET}"
fi

# Currently-active exp07 run.
CUR_RUN="$(pgrep -fl '07_lambda_sweep/run_can.py' 2>/dev/null | head -1)"
if [[ -n "$CUR_RUN" ]]; then
    CUR_PID="$(echo "$CUR_RUN" | awk '{print $1}')"
    CUR_ETIME="$(ps -o etime= -p "$CUR_PID" 2>/dev/null | tr -d ' ')"
    # Extract the --output-dir value.  The repo path contains a space
    # ("ECE M237"), so we walk the argv token-by-token: find the
    # "--output-dir" flag and join following tokens until we hit a
    # token that starts with "--" (the next flag).
    CUR_CMD="$(ps -o command= -p "$CUR_PID" 2>/dev/null)"
    CUR_OUTDIR="$(echo "$CUR_CMD" | awk '
        {
            for (i = 1; i <= NF; i++) {
                if ($i == "--output-dir") {
                    out = ""
                    for (j = i + 1; j <= NF; j++) {
                        if (substr($j, 1, 2) == "--") break
                        if (out != "") out = out " "
                        out = out $j
                    }
                    print out
                    exit
                }
            }
        }
    ')"
    CUR_OUTDIR_REL="${CUR_OUTDIR#"$REPO_ROOT/"}"
    CUR_LAMS="$(echo "$CUR_CMD" \
        | grep -oE '\-\-lambda-prior [0-9. ]+' | head -1)"
    printf "${DIM}exp07 run : pid=%s  elapsed=%s${RESET}\n" "$CUR_PID" "${CUR_ETIME:-?}"
    printf "${DIM}            outdir: %s${RESET}\n" "${CUR_OUTDIR_REL:-?}"
    printf "${DIM}            %s${RESET}\n" "${CUR_LAMS:-?}"

    # Most recent plan line.
    LAST_PLAN="$(grep '^\[exp07\] lam=' "$LOG" 2>/dev/null | tail -1)"
    if [[ -n "$LAST_PLAN" ]]; then
        printf "${DIM}            last: %s${RESET}\n" \
            "$(echo "$LAST_PLAN" | sed 's/  */ /g')"
    fi

    # Aggregate worker count + cpu under this PID.
    W_COUNT=$(ps -eo ppid 2>/dev/null | awk -v p="$CUR_PID" '$1==p' | wc -l | tr -d ' ')
    W_CPU=$(ps -eo ppid,pcpu 2>/dev/null | awk -v p="$CUR_PID" '$1==p {s+=$2} END{printf "%.0f", s}')
    printf "${DIM}            workers=%s  aggregate CPU=%s%%${RESET}\n" "$W_COUNT" "${W_CPU:-0}"
fi

# Screen session.
SCREEN_LINE="$(screen -ls 2>/dev/null | grep -E "sweep_v2" | head -1)"
if [[ -n "$SCREEN_LINE" ]]; then
    printf "${DIM}screen    : %s${RESET}\n" "$(echo "$SCREEN_LINE" | sed 's/^[[:space:]]*//')"
fi

# ---- Highlight current best across all phases -----------------------------
echo "${BOLD}===========================================================================${RESET}"
BEST_LINE="$(python3 -c "
import json, glob, os
sweep = '$SWEEP_DIR'
best = None
# Phase 0 baseline
p0 = os.path.join(sweep, 'phase0_prior_only_replay.json')
if os.path.exists(p0):
    d = json.load(open(p0))
    best = ('Phase 0  prior-only', d['n_success'], d['n_total'], 'n/a')
# Phase 1, 2 cells
for phase_dir, label in [
    (os.path.join(sweep, 'phase1_lambda_sweep_warmstart'), 'Phase 1  warm+λ'),
    (os.path.join(sweep, 'phase2_lambda_sweep_no_warmstart'), 'Phase 2  zero+λ'),
]:
    for j in sorted(glob.glob(os.path.join(phase_dir, 'lambda_*.json'))):
        d = json.load(open(j))
        if d.get('n_total', 0) == 0: continue
        if best is None or d['n_success'] > best[1]:
            lam = os.path.basename(j).replace('lambda_', '').replace('.json', '')
            best = (f'{label}=λ{lam}', d['n_success'], d['n_total'], 'sigma=0.5')
# Phase 5 grid
for sig_dir in sorted(glob.glob(os.path.join(sweep, 'phase5_sigma_preservation/sigma_*'))):
    sig = os.path.basename(sig_dir).replace('sigma_', '')
    for j in sorted(glob.glob(os.path.join(sig_dir, 'lambda_*.json'))):
        d = json.load(open(j))
        if d.get('n_total', 0) == 0: continue
        if best is None or d['n_success'] > best[1]:
            lam = os.path.basename(j).replace('lambda_', '').replace('.json', '')
            best = (f'Phase 5  σ={sig} λ={lam}', d['n_success'], d['n_total'], 'sigma='+sig)
# Phase 6
for j in sorted(glob.glob(os.path.join(sweep, 'phase6_best_k_extension/lambda_*.json'))):
    d = json.load(open(j))
    if d.get('n_total', 0) == 0: continue
    if best is None or d['n_success'] > best[1]:
        lam = os.path.basename(j).replace('lambda_', '').replace('.json', '')
        best = (f'Phase 6  K=20 λ={lam}', d['n_success'], d['n_total'], 'K=20')
if best is None:
    print('no completed cells yet')
else:
    label, n_succ, n_total, extra = best
    rate = (n_succ / n_total * 100) if n_total > 0 else 0
    print(f'BEST so far : {label}  →  {n_succ}/{n_total} = {rate:.1f}%  ({extra})')
" 2>/dev/null)"
echo "${BOLD}${GREEN}${BEST_LINE}${RESET}"

# ---- Tail of log -----------------------------------------------------------
echo "${BOLD}===========================================================================${RESET}"
echo "${DIM}sweep.log tail:${RESET}"
tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
echo "${BOLD}===========================================================================${RESET}"
