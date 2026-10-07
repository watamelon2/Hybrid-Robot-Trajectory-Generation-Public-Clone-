#!/usr/bin/env bash
# One-shot status snapshot for the two overnight jobs:
#
#   1. A1.2 sigma-sweep (Branch A, feat/stronger-mbd-signal)
#      Sequential 7-value sweep via experiments/09_sigma_sweep/run.sh.
#      One PID running the bash script that forks one exp 05 python
#      subprocess per sigma value.
#
#   2. C Phase 1 regression (Branch C, feat/hybrid-coupling)
#      Sequential 2-cell test via
#      experiments/15_unified_hybrid/regression_test.py. One PID
#      running python that forks one run_can subprocess per cell.
#
# Designed to be wrapped with `watch` for a refreshing dashboard:
#
#   watch -n 15 -c bash scripts/local/monitor_overnight.sh
#
# `-c` preserves ANSI colors. `-n 15` = refresh every 15 s.
#
# Rules (matches scripts/local/monitor_bc_swap_rerun.sh conventions):
#   * Single-shot: NO infinite loop, NO clear. Run-and-exit.
#   * Read-only:  touches nothing under logs/ or experiments/*/results/.
#   * Fast:       under 1 s on a warm cache, no network I/O.
#   * Portable:   macOS BSD userland (no GNU-only flags).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Paths
SIGMA_LOG_DIR="logs/local_sigma_sweep"
SIGMA_OUT_DIR="experiments/09_sigma_sweep/results"
REGR_LOG_DIR="logs/local_phase1_regression"
REGR_OUT_DIR="experiments/15_unified_hybrid/results"
# regression_summary.json moved to _archive/ on 2026-05-15.
# Use the archived copy if the active one is gone (post-archive state).
if [[ -f "$REGR_OUT_DIR/regression_summary.json" ]]; then
    REGR_SUMMARY="$REGR_OUT_DIR/regression_summary.json"
else
    REGR_SUMMARY="$REGR_OUT_DIR/_archive/regression_summary.json"
fi
EXT_LOG_DIR="logs/local_sigma_sweep_extension"
# (Extension shares SIGMA_OUT_DIR with the main sweep.)

# Sigma values to monitor (must match the launch script's SIGMAS list).
SIGMAS=(0.05 0.10 0.25 0.40 0.50 0.75 1.00)
# Extension sigma values (the below-floor sweep launched separately).
EXT_SIGMAS=(0.01 0.02 0.03)

# Regression cells (label, target_succ, lam, K, R).
# These must match experiments/15_unified_hybrid/regression_test.py CELLS.
REGR_CELLS=(
    "exp07_lambda5_singleshot|20|5.0|10|0"
    "exp10_K10_R2_warmstart|11|0.0|10|2"
)

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
printf "%sovernight monitor%s    refreshed: %s\n" "$BOLD" "$RESET" "$NOW"
echo "$RULE"

# ---- helpers ----
# Print "etime pcpu rss" (space-separated) if PID alive, else nothing.
ps_alive() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        ps -p "$pid" -o etime=,pcpu=,rss= 2>/dev/null \
            | sed 's/^ *//;s/ \{1,\}/ /g'
    fi
}

fmt_gb() {
    awk -v kb="$1" 'BEGIN { printf "%.1fG", kb/1024/1024 }'
}

# Read a PID file and clean whitespace.
read_pid() {
    local f="$1"
    [[ -f "$f" ]] && tr -d '[:space:]' < "$f"
}

# For a parent PID, return "n_procs total_cpu total_rss busiest_pid"
# aggregated across ALL descendants (driver -> run_can -> pool
# workers). The driver and run_can parents are usually sleeping (CPU
# 0%); pool workers are the real compute. Walks 2 levels deep which
# covers bash -> python -> Pool worker AND python driver -> run_can
# -> Pool worker.
worker_metrics_for() {
    local parent="$1"
    [[ -n "$parent" ]] || return 0
    local children
    children="$(pgrep -P "$parent" 2>/dev/null)"
    local grandkids=""
    for c in $children; do
        gk="$(pgrep -P "$c" 2>/dev/null)"
        [[ -n "$gk" ]] && grandkids="$grandkids $gk"
    done
    local all_descs="$children $grandkids"
    local n=0 total_cpu=0 total_rss=0
    local best_pid="" best_cpu=0
    for d in $all_descs; do
        [[ -z "$d" ]] && continue
        local info
        info="$(ps -p "$d" -o pcpu=,rss= 2>/dev/null)"
        [[ -z "$info" ]] && continue
        local pcpu rss
        pcpu="$(echo "$info" | awk '{print $1}')"
        rss="$(echo "$info" | awk '{print $2}')"
        n=$((n + 1))
        total_cpu="$(awk -v a="$total_cpu" -v b="$pcpu" 'BEGIN { print a + b }')"
        total_rss=$((total_rss + rss))
        if awk -v a="$pcpu" -v b="$best_cpu" 'BEGIN { exit !(a > b) }'; then
            best_cpu="$pcpu"; best_pid="$d"
        fi
    done
    if (( n > 0 )); then
        echo "$n $total_cpu $total_rss $best_pid"
    fi
}

# Last "=== sigma_max=$val ===" line in the log; returns the sigma value or empty.
current_sigma_from_log() {
    local log="$1"
    [[ -f "$log" ]] || return 0
    # `tail -r` reverses on macOS; portable enough for this repo.
    grep -E '^=== sigma_max=' "$log" 2>/dev/null \
        | tail -1 \
        | sed -E 's/^=== sigma_max=([0-9.]+) ===$/\1/'
}

# Last "[regression_test] running $label" line in the log, if it
# has been flushed yet. Returns the label or empty.
current_regr_cell_from_log() {
    local log="$1"
    [[ -f "$log" ]] || return 0
    grep -E '^\[regression_test\] running ' "$log" 2>/dev/null \
        | tail -1 \
        | sed -E 's/^\[regression_test\] running ([^:]+): .*/\1/'
}

# Fallback: when stdout is block-buffered (nohup to a file on
# macOS), the [regression_test] log line may not be flushed yet
# even though the worker IS running. Infer the current cell from
# the worker process's live command-line args via ps.
current_regr_cell_from_proc() {
    local parent="$1"
    [[ -n "$parent" ]] || return 0
    # Find any descendant python process whose cmdline contains run_can.
    local descs
    descs="$(pgrep -P "$parent" 2>/dev/null)"
    for d in $descs; do
        gk="$(pgrep -P "$d" 2>/dev/null)"
        [[ -n "$gk" ]] && descs="$descs $gk"
    done
    for d in $descs; do
        local cmd
        cmd="$(ps -p "$d" -o command= 2>/dev/null)"
        [[ "$cmd" != *run_can* ]] && continue
        # Match against the regression cells' distinguishing flag
        # combos. Order matters -- check cell 2 (lambda-prior=0 +
        # mpc-replans=2) before cell 1 (lambda-prior=5 +
        # mpc-replans=0) to disambiguate.
        if [[ "$cmd" == *"--lambda-prior=0 "*"--mpc-replans=2"* ]] \
           || [[ "$cmd" == *"--lambda-prior=0"*"--mpc-replans=2"* ]]; then
            echo "exp10_K10_R2_warmstart"
            return 0
        fi
        if [[ "$cmd" == *"--lambda-prior=5"* ]] && \
           [[ "$cmd" == *"--mpc-replans=0"* ]]; then
            echo "exp07_lambda5_singleshot"
            return 0
        fi
    done
}

# Did a regression cell's per-cell JSON get written?
# Args: lam, K, R. Returns the JSON path if found, else empty.
regr_cell_json_path() {
    local lam="$1" K="$2" R="$3"
    # Regression cell JSONs moved to _archive/ on 2026-05-15; check both.
    local active="$REGR_OUT_DIR/unified_bc_lam-${lam}_K-${K}_R-${R}_regress.json"
    local archive="$REGR_OUT_DIR/_archive/unified_bc_lam-${lam}_K-${K}_R-${R}_regress.json"
    if [[ -f "$active" ]]; then echo "$active"
    elif [[ -f "$archive" ]]; then echo "$archive"
    fi
}

# Read "n_success/n_total" from a result JSON, or "?".
# Recognises skip-marker JSONs (`_skipped: true`) and returns the
# literal token "SKIPPED" so callers can render those cells as
# intentionally-empty rather than as "DONE None/None".
read_succ_rate() {
    local json="$1"
    [[ -f "$json" ]] || { echo "?"; return; }
    python3 - "$json" <<'PY' 2>/dev/null || echo "?"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    if d.get("_skipped") is True:
        print("SKIPPED")
    else:
        print(f"{d['n_success']}/{d['n_total']}")
except Exception:
    print("?")
PY
}

# ---- 0. Overnight queue status (shown first because it's the active job) ----
QUEUE_DIR="logs/overnight_queue"
if [[ -f "$QUEUE_DIR/pid" ]]; then
    QPID="$(read_pid "$QUEUE_DIR/pid")"
    QSTATE="$(cat "$QUEUE_DIR/state.txt" 2>/dev/null || echo unknown)"

    printf "%s[overnight queue]%s  state=%s%s%s  " \
        "$BOLD" "$RESET" "$CYAN" "$QSTATE" "$RESET"
    if [[ -n "$QPID" ]] && kill -0 "$QPID" 2>/dev/null; then
        q_etime="$(ps -p "$QPID" -o etime= 2>/dev/null | tr -d ' ')"
        printf "PID=%s%s%s  elapsed=%s\n" \
            "$GREEN" "$QPID" "$RESET" "$q_etime"
    else
        printf "PID=%s%s%s  %s(not running)%s\n" \
            "$DIM" "${QPID:-?}" "$RESET" "$DIM" "$RESET"
    fi

    # ---- Stage 1: Phase 2 K x R grid (lam=5) ----
    PHASE2_K_VALS=(5 10 20)
    PHASE2_R_VALS=(0 1 2)
    # Phase 2 grid cells moved to _archive/ on 2026-05-15. Check both.
    PHASE2_DONE=0
    for K in "${PHASE2_K_VALS[@]}"; do
        for R in "${PHASE2_R_VALS[@]}"; do
            json="${REGR_OUT_DIR}/unified_bc_lam-5.0_K-${K}_R-${R}_phase2.json"
            json_archive="${REGR_OUT_DIR}/_archive/unified_bc_lam-5.0_K-${K}_R-${R}_phase2.json"
            ([[ -f "$json" ]] || [[ -f "$json_archive" ]]) && PHASE2_DONE=$((PHASE2_DONE + 1))
        done
    done
    printf "  Stage 1 Phase 2 K x R grid (lam=5):  %s%d/9%s done\n" \
        "$GREEN" "$PHASE2_DONE" "$RESET"

    # 3x3 grid rendering
    for K in "${PHASE2_K_VALS[@]}"; do
        printf "    "
        for R in "${PHASE2_R_VALS[@]}"; do
            json="${REGR_OUT_DIR}/unified_bc_lam-5.0_K-${K}_R-${R}_phase2.json"
            # Prefer active path (re-run lands fresh); fall back to _archive/.
            [[ -f "$json" ]] || json="${REGR_OUT_DIR}/_archive/unified_bc_lam-5.0_K-${K}_R-${R}_phase2.json"
            if [[ -f "$json" ]]; then
                sr="$(read_succ_rate "$json")"
                if [[ "$sr" == "SKIPPED" ]]; then
                    # Skip-marker JSON -- render as dim "SKIPPED" with no rate.
                    printf "K=%-2s R=%s %s%-8s%s %-6s   " \
                        "$K" "$R" "$DIM" "SKIPPED" "$RESET" "-"
                else
                    printf "K=%-2s R=%s %s%-8s%s %-6s   " \
                        "$K" "$R" "$GREEN" "DONE" "$RESET" "$sr"
                fi
            else
                # Determine if this is the currently-running cell from queue.log
                running=$(grep -E "RUN  unified K=$K R=$R" "$QUEUE_DIR/queue.log" 2>/dev/null | tail -1)
                done_line=$(grep -E "(DONE|FAIL|SKIP) unified K=$K R=$R" "$QUEUE_DIR/queue.log" 2>/dev/null | tail -1)
                if [[ -n "$running" && -z "$done_line" ]]; then
                    printf "K=%-2s R=%s %s%-8s%s %-6s   " \
                        "$K" "$R" "$YELLOW" "RUNNING" "$RESET" "-"
                else
                    printf "K=%-2s R=%s %s%-8s%s %-6s   " \
                        "$K" "$R" "$DIM" "pending" "$RESET" "-"
                fi
            fi
        done
        echo ""
    done

    # ---- Other stages (one-liner each) ----
    # Stage 2: sigma=0.05 seeds 1, 2
    # Files archived to _archive/ on 2026-05-15.
    s2_done=0
    for seed in 1 2; do
        ([[ -f "$SIGMA_OUT_DIR/sigma_0.05_seed${seed}.json" ]] || \
         [[ -f "$SIGMA_OUT_DIR/_archive/sigma_0.05_seed${seed}.json" ]]) && \
            s2_done=$((s2_done + 1))
    done
    printf "  Stage 2 sigma=0.05 seeds (1,2):       %s%d/2%s done\n" \
        "$GREEN" "$s2_done" "$RESET"

    # Stage 3: lambda sweep K=10 R=2
    # Files archived to _archive/ on 2026-05-15.
    s3_done=0
    for L in 0 1 2 5 10 20; do
        f="${REGR_OUT_DIR}/unified_bc_lam-${L}.0_K-10_R-2_lamsweep.json"
        f_archive="${REGR_OUT_DIR}/_archive/unified_bc_lam-${L}.0_K-10_R-2_lamsweep.json"
        ([[ -f "$f" ]] || [[ -f "$f_archive" ]]) && s3_done=$((s3_done + 1))
    done
    printf "  Stage 3 lambda-sweep K=10 R=2 (6 λs): %s%d/6%s done\n" \
        "$GREEN" "$s3_done" "$RESET"

    # Stage 4: best-cell seeds
    if [[ -f "$QUEUE_DIR/best_phase2_cell.json" ]]; then
        b_K="$(python3 -c "import json; print(json.load(open('$QUEUE_DIR/best_phase2_cell.json'))['K'])" 2>/dev/null)"
        b_R="$(python3 -c "import json; print(json.load(open('$QUEUE_DIR/best_phase2_cell.json'))['R'])" 2>/dev/null)"
        # Stage 4 seed files moved to _archive/ on 2026-05-15.
        s4_done=0
        for seed in 1 2; do
            f="${REGR_OUT_DIR}/unified_bc_lam-5.0_K-${b_K}_R-${b_R}_phase2_seed${seed}.json"
            f_archive="${REGR_OUT_DIR}/_archive/unified_bc_lam-5.0_K-${b_K}_R-${b_R}_phase2_seed${seed}.json"
            ([[ -f "$f" ]] || [[ -f "$f_archive" ]]) && s4_done=$((s4_done + 1))
        done
        printf "  Stage 4 best-cell seeds (K=%s R=%s):  %s%d/2%s done\n" \
            "$b_K" "$b_R" "$GREEN" "$s4_done" "$RESET"
    else
        printf "  Stage 4 best-cell seeds:              %snot started%s\n" \
            "$DIM" "$RESET"
    fi

    # Stage 5: M ablation
    # Files archived to _archive/ on 2026-05-15.
    s5_done=0
    for M in 64 256; do
        ([[ -f "$SIGMA_OUT_DIR/sigma_0.05_M${M}.json" ]] || \
         [[ -f "$SIGMA_OUT_DIR/_archive/sigma_0.05_M${M}.json" ]]) && \
            s5_done=$((s5_done + 1))
    done
    printf "  Stage 5 M-ablation M∈{64,256}:        %s%d/2%s done\n" \
        "$GREEN" "$s5_done" "$RESET"

    # ---- queue.log tail ----
    if [[ -f "$QUEUE_DIR/queue.log" ]]; then
        # Show last 3 non-empty / non-divider lines that contain RUN/DONE/FAIL/SKIP/STATE
        printf "  queue.log tail:\n"
        grep -E '(RUN|DONE|FAIL|SKIP|STATE|ABORT|WARN) ' "$QUEUE_DIR/queue.log" 2>/dev/null \
            | tail -3 | sed 's/^/    /'
    fi
    echo ""
fi

# ---- 0b. Parallel-extras queue status (Jobs A/B/C) ----
PARALLEL_DIR="logs/parallel_extras"
if [[ -f "$PARALLEL_DIR/pid" ]]; then
    PPID_PARALLEL="$(read_pid "$PARALLEL_DIR/pid")"
    PSTATE="$(cat "$PARALLEL_DIR/state.txt" 2>/dev/null || echo unknown)"

    printf "%s[parallel extras]%s state=%s%s%s  " \
        "$BOLD" "$RESET" "$MAGENTA" "$PSTATE" "$RESET"
    if [[ -n "$PPID_PARALLEL" ]] && kill -0 "$PPID_PARALLEL" 2>/dev/null; then
        p_etime="$(ps -p "$PPID_PARALLEL" -o etime= 2>/dev/null | tr -d ' ')"
        printf "PID=%s%s%s  elapsed=%s\n" "$GREEN" "$PPID_PARALLEL" "$RESET" "$p_etime"
    else
        printf "PID=%s%s%s  %s(not running)%s\n" \
            "$DIM" "${PPID_PARALLEL:-?}" "$RESET" "$DIM" "$RESET"
    fi

    # Job A: sigma-sweep at lambda=5 unified  (5 cells)
    # Files moved to results/_archive/ on 2026-05-15 after Job A closed.
    # Monitor still checks both locations so a hypothetical re-run lands cleanly.
    job_a_done=0
    for s in 0.01 0.03 0.05 0.10 0.25; do
        f="${REGR_OUT_DIR}/unified_bc_lam-5.0_K-10_R-0_sigmasweep_lam5_sigma${s}.json"
        f_archive="${REGR_OUT_DIR}/_archive/unified_bc_lam-5.0_K-10_R-0_sigmasweep_lam5_sigma${s}.json"
        ([[ -f "$f" ]] || [[ -f "$f_archive" ]]) && job_a_done=$((job_a_done + 1))
    done
    printf "  Job A sigma-sweep at lam=5 unified:   %s%d/5%s done\n" \
        "$GREEN" "$job_a_done" "$RESET"

    # Job B: 3-seed sigma=0.03 exp 05
    # Files archived to _archive/ on 2026-05-15.
    job_b_done=0
    for seed in 1 2 3; do
        ([[ -f "$SIGMA_OUT_DIR/sigma_0.03_seed${seed}.json" ]] || \
         [[ -f "$SIGMA_OUT_DIR/_archive/sigma_0.03_seed${seed}.json" ]]) && \
            job_b_done=$((job_b_done + 1))
    done
    printf "  Job B 3-seed sigma=0.03 exp 05:       %s%d/3%s done\n" \
        "$GREEN" "$job_b_done" "$RESET"

    # Job C: 3-seed best Phase 2 cell unified
    # Files archived to _archive/ on 2026-05-15.
    job_c_done=0
    for seed in 1 2 3; do
        f="${REGR_OUT_DIR}/unified_bc_lam-5.0_K-10_R-1_phase2_seed${seed}.json"
        f_archive="${REGR_OUT_DIR}/_archive/unified_bc_lam-5.0_K-10_R-1_phase2_seed${seed}.json"
        ([[ -f "$f" ]] || [[ -f "$f_archive" ]]) && job_c_done=$((job_c_done + 1))
    done
    printf "  Job C 3-seed K=10 R=1 lam=5 unified:  %s%d/3%s done\n" \
        "$GREEN" "$job_c_done" "$RESET"

    # parallel_extras log tail
    if [[ -f "$PARALLEL_DIR/queue.log" ]]; then
        printf "  parallel_extras.log tail:\n"
        grep -E '(RUN|DONE|FAIL|SKIP|STATE) ' "$PARALLEL_DIR/queue.log" 2>/dev/null \
            | tail -3 | sed 's/^/    /'
    fi
    echo ""
fi

# ---- 0c. Single-cell extras (one-off runs outside the queue scripts) ----
# Each entry: label | pid_file | stdout_log | result_json | n_total
EXTRA_CELLS=(
    "K=20 R=1 BC   lam=5|logs/k20_r1_extra/pid|logs/k20_r1_extra/stdout.log|experiments/15_unified_hybrid/results/_archive/unified_bc_lam-5.0_K-20_R-1_phase2.json|60"
    "K=10 R=1 FLOW lam=5|/tmp/feat-b-worktree/logs/flow_prior_extra/pid|/tmp/feat-b-worktree/logs/flow_prior_extra/stdout.log|/tmp/feat-b-worktree/experiments/15_unified_hybrid/results/unified_flow_lam-5.0_K-10_R-1_flow_phase2.json|60"
    "K=10 R=1 FLOW fixB|/tmp/feat-b-worktree/logs/flow_prior_fixB/pid|/tmp/feat-b-worktree/logs/flow_prior_fixB/stdout.log|/tmp/feat-b-worktree/experiments/15_unified_hybrid/results/unified_flow_lam-5.0_K-10_R-1_flow_fixB_phase2.json|60"
)

# Render section only if at least one entry has a live PID or a final result.
EXTRAS_ANY=0
for spec in "${EXTRA_CELLS[@]}"; do
    IFS='|' read -r _label pid_file _stdout_log result_json _n_total <<< "$spec"
    pid="$(read_pid "$pid_file")"
    if { [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; } || [[ -f "$result_json" ]]; then
        EXTRAS_ANY=1
        break
    fi
done

if (( EXTRAS_ANY == 1 )); then
    printf "%s[single-cell extras]%s  (one-off runs outside the queue scripts)\n" \
        "$BOLD" "$RESET"
    for spec in "${EXTRA_CELLS[@]}"; do
        IFS='|' read -r label pid_file stdout_log result_json n_total <<< "$spec"
        pid="$(read_pid "$pid_file")"

        # If a final JSON exists, prefer that.
        if [[ -f "$result_json" ]]; then
            succ_total="$(awk -F'[,:]' '
                /\"n_success\"/ { for (i=1;i<=NF;i++) if ($i ~ /\"n_success\"/) s=$(i+1) }
                /\"n_total\"/   { for (i=1;i<=NF;i++) if ($i ~ /\"n_total\"/)   t=$(i+1) }
                END { gsub(/[ ,}]/,"",s); gsub(/[ ,}]/,"",t); print s"/"t }
            ' "$result_json" 2>/dev/null)"
            printf "  %s%-22s%s %sDONE%s    %s\n" \
                "$BOLD" "$label" "$RESET" "$GREEN" "$RESET" "$succ_total"
            continue
        fi

        # Otherwise, render running / dead based on PID.
        if [[ -z "$pid" ]]; then
            continue
        fi
        if kill -0 "$pid" 2>/dev/null; then
            etime="$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')"
            # Count rollouts logged so far (one line per demo_X#Y).
            # grep -c always prints a number; just suppress stderr.
            # Note: grep -c exits 1 when zero matches, so `|| echo 0`
            # would double-append a "0" -- omit the OR fallback.
            rollouts_done=0
            succ_so_far=0
            if [[ -f "$stdout_log" ]]; then
                rollouts_done="$(grep -cE 'demo_[0-9]+#[0-9]+' "$stdout_log" 2>/dev/null)"
                succ_so_far="$(grep -cE 'demo_[0-9]+#[0-9]+.*success=True' "$stdout_log" 2>/dev/null)"
                : "${rollouts_done:=0}"
                : "${succ_so_far:=0}"
            fi
            # Workers count (descendants).
            n_workers="$(pgrep -P "$pid" 2>/dev/null | wc -l | tr -d ' ')"
            # Pct done.
            if (( rollouts_done > 0 )); then
                pct="$(awk -v a="$rollouts_done" -v b="$n_total" 'BEGIN { printf "%.0f", 100*a/b }')"
            else
                pct=0
            fi
            printf "  %s%-22s%s %sRUN%s   PID=%-6s elapsed=%-8s workers=%-2s  rollouts=%d/%d (%d%%)  interim=%d/%d\n" \
                "$BOLD" "$label" "$RESET" "$YELLOW" "$RESET" \
                "$pid" "$etime" "$n_workers" \
                "$rollouts_done" "$n_total" "$pct" \
                "$succ_so_far" "$rollouts_done"
        else
            printf "  %s%-22s%s %sDEAD%s   PID=%-6s (no result JSON yet -- check %s)\n" \
                "$BOLD" "$label" "$RESET" "$RED" "$RESET" "$pid" "$stdout_log"
        fi
    done
    echo ""
fi

# ---- 1. A1.2 sigma-sweep status ----
# Only render this section if the driver PID is currently alive.
# Once the sweep finishes, the section auto-hides (the headline
# numbers live in the result JSONs at $SIGMA_OUT_DIR/sigma_*.json).
SIGMA_PID="$(read_pid "$SIGMA_LOG_DIR/pid")"
if [[ -n "$SIGMA_PID" ]] && kill -0 "$SIGMA_PID" 2>/dev/null; then
printf "%s[A1.2 sigma-sweep, 7 cells]%s  log=%s  out=%s\n" \
    "$BOLD" "$RESET" "$SIGMA_LOG_DIR" "$SIGMA_OUT_DIR"
if [[ -n "$SIGMA_PID" ]] && info=$(ps_alive "$SIGMA_PID"); [[ -n "$info" ]]; then
    sigma_etime="$(echo "$info" | awk '{print $1}')"
    worker_info="$(worker_metrics_for "$SIGMA_PID")"
    if [[ -n "$worker_info" ]]; then
        n_procs="$(echo "$worker_info" | awk '{print $1}')"
        total_cpu="$(echo "$worker_info" | awk '{print $2}')"
        total_rss="$(echo "$worker_info" | awk '{print $3}')"
        busy_pid="$(echo "$worker_info" | awk '{print $4}')"
        total_rss_gb="$(fmt_gb "$total_rss")"
        printf "  driver PID : %s%-6s%s  elapsed %s\n" \
            "$GREEN" "$SIGMA_PID" "$RESET" "$sigma_etime"
        printf "  workers    : %s%d procs%s  CPU %s%.0f%%%s  RSS %s  (busiest PID %s)\n" \
            "$CYAN" "$n_procs" "$RESET" \
            "$CYAN" "$total_cpu" "$RESET" \
            "$total_rss_gb" "$busy_pid"
    else
        printf "  driver PID : %s%-6s%s  elapsed %s  %s(no active worker)%s\n" \
            "$GREEN" "$SIGMA_PID" "$RESET" "$sigma_etime" "$DIM" "$RESET"
    fi
    SIGMA_RUNNING=1
elif [[ -n "$SIGMA_PID" ]]; then
    printf "  driver PID : %s%-6s%s  %s(not running -- finished or died)%s\n" \
        "$DIM" "$SIGMA_PID" "$RESET" "$DIM" "$RESET"
    SIGMA_RUNNING=0
else
    printf "  driver PID : %s(no pid file at %s/pid)%s\n" \
        "$DIM" "$SIGMA_LOG_DIR" "$RESET"
    SIGMA_RUNNING=0
fi

cur_sigma="$(current_sigma_from_log "$SIGMA_LOG_DIR/stdout.log")"
printf "  %-9s %-9s %-9s %-9s\n" "sigma" "state" "succ" "result-path"
echo "  $SUB"

n_done_sigma=0
n_running_sigma=0
for sigma in "${SIGMAS[@]}"; do
    json="$SIGMA_OUT_DIR/sigma_${sigma}.json"
    if [[ -f "$json" ]]; then
        sr="$(read_succ_rate "$json")"
        printf "  %-9s %s%-9s%s %-9s %s\n" \
            "$sigma" "$GREEN" "DONE" "$RESET" "$sr" \
            "$(basename "$json")"
        n_done_sigma=$((n_done_sigma + 1))
    elif [[ "$sigma" == "$cur_sigma" && "$SIGMA_RUNNING" == "1" ]]; then
        printf "  %-9s %s%-9s%s %-9s %s\n" \
            "$sigma" "$YELLOW" "RUNNING" "$RESET" "-" "(in progress)"
        n_running_sigma=1
    else
        printf "  %-9s %s%-9s%s %-9s %s\n" \
            "$sigma" "$DIM" "pending" "$RESET" "-" "-"
    fi
done
echo "  $SUB"
printf "  done: %s%d/%d%s    running: %s%d%s    current sigma: %s\n" \
    "$GREEN" "$n_done_sigma" "${#SIGMAS[@]}" "$RESET" \
    "$YELLOW" "$n_running_sigma" "$RESET" \
    "${cur_sigma:-?}"
echo ""
fi  # end of section 1

# ---- 2. C Phase 1 regression status ----
# Same auto-hide rule as section 1.
REGR_PID="$(read_pid "$REGR_LOG_DIR/pid")"
if [[ -n "$REGR_PID" ]] && kill -0 "$REGR_PID" 2>/dev/null; then
printf "%s[C Phase 1 regression, 2 cells]%s  log=%s  out=%s\n" \
    "$BOLD" "$RESET" "$REGR_LOG_DIR" "$REGR_OUT_DIR"
if [[ -n "$REGR_PID" ]] && info=$(ps_alive "$REGR_PID"); [[ -n "$info" ]]; then
    regr_etime="$(echo "$info" | awk '{print $1}')"
    worker_info="$(worker_metrics_for "$REGR_PID")"
    if [[ -n "$worker_info" ]]; then
        n_procs="$(echo "$worker_info" | awk '{print $1}')"
        total_cpu="$(echo "$worker_info" | awk '{print $2}')"
        total_rss="$(echo "$worker_info" | awk '{print $3}')"
        busy_pid="$(echo "$worker_info" | awk '{print $4}')"
        total_rss_gb="$(fmt_gb "$total_rss")"
        printf "  driver PID : %s%-6s%s  elapsed %s\n" \
            "$GREEN" "$REGR_PID" "$RESET" "$regr_etime"
        printf "  workers    : %s%d procs%s  CPU %s%.0f%%%s  RSS %s  (busiest PID %s)\n" \
            "$CYAN" "$n_procs" "$RESET" \
            "$CYAN" "$total_cpu" "$RESET" \
            "$total_rss_gb" "$busy_pid"
    else
        printf "  driver PID : %s%-6s%s  elapsed %s  %s(no active worker)%s\n" \
            "$GREEN" "$REGR_PID" "$RESET" "$regr_etime" "$DIM" "$RESET"
    fi
    REGR_RUNNING=1
elif [[ -n "$REGR_PID" ]]; then
    printf "  driver PID : %s%-6s%s  %s(not running -- finished or died)%s\n" \
        "$DIM" "$REGR_PID" "$RESET" "$DIM" "$RESET"
    REGR_RUNNING=0
else
    printf "  driver PID : %s(no pid file at %s/pid)%s\n" \
        "$DIM" "$REGR_LOG_DIR" "$RESET"
    REGR_RUNNING=0
fi

cur_cell="$(current_regr_cell_from_log "$REGR_LOG_DIR/stdout.log")"
# Log may not be flushed yet -- fall back to inferring from the
# live worker process command-line.
if [[ -z "$cur_cell" && "$REGR_RUNNING" == "1" ]]; then
    cur_cell="$(current_regr_cell_from_proc "$REGR_PID")"
fi

printf "  %-28s %-9s %-9s %-9s %s\n" "cell" "state" "succ" "target" "verdict"
echo "  $SUB"

n_done_regr=0
n_running_regr=0
all_pass=1
for spec in "${REGR_CELLS[@]}"; do
    IFS='|' read -r label target lam K R <<< "$spec"
    json="$(regr_cell_json_path "$lam" "$K" "$R")"
    if [[ -n "$json" ]]; then
        # Cell done: compute succ + verdict.
        sr="$(read_succ_rate "$json")"
        succ="${sr%%/*}"
        n_total="${sr##*/}"
        diff=$(( succ - target ))
        abs_diff=${diff#-}
        if (( abs_diff <= 1 )); then
            verdict="${GREEN}PASS (|${succ}-${target}|=${abs_diff})${RESET}"
        else
            verdict="${RED}FAIL (|${succ}-${target}|=${abs_diff})${RESET}"
            all_pass=0
        fi
        printf "  %-28s %s%-9s%s %-9s %-9s %b\n" \
            "$label" "$GREEN" "DONE" "$RESET" "$sr" "$target/$n_total" "$verdict"
        n_done_regr=$((n_done_regr + 1))
    elif [[ "$label" == "$cur_cell" && "$REGR_RUNNING" == "1" ]]; then
        printf "  %-28s %s%-9s%s %-9s %-9s %s\n" \
            "$label" "$YELLOW" "RUNNING" "$RESET" "-" "$target/60" "(in progress)"
        n_running_regr=1
    else
        printf "  %-28s %s%-9s%s %-9s %-9s %s\n" \
            "$label" "$DIM" "pending" "$RESET" "-" "$target/60" "-"
    fi
done
echo "  $SUB"
printf "  done: %s%d/%d%s    running: %s%d%s    current cell: %s\n" \
    "$GREEN" "$n_done_regr" "${#REGR_CELLS[@]}" "$RESET" \
    "$YELLOW" "$n_running_regr" "$RESET" \
    "${cur_cell:-?}"

if [[ "$n_done_regr" -eq "${#REGR_CELLS[@]}" ]]; then
    if [[ -f "$REGR_SUMMARY" ]]; then
        overall="$(python3 -c "
import json
d = json.load(open('$REGR_SUMMARY'))
print('PASS' if d.get('all_pass') else 'FAIL')
" 2>/dev/null)"
        if [[ "$overall" == "PASS" ]]; then
            printf "  overall    : %sPHASE 1 ACCEPTANCE PASSED%s  (regression_summary.json)\n" \
                "$GREEN" "$RESET"
        else
            printf "  overall    : %sPHASE 1 ACCEPTANCE FAILED%s  (see regression_summary.json)\n" \
                "$RED" "$RESET"
        fi
    fi
fi
echo ""
fi  # end of section 2

# ---- 2b. A1.2 sigma-sweep EXTENSION (below-floor probe) ----
# Auto-hide once the extension PID is no longer alive (or never
# existed).
EXT_PID_PROBE="$(read_pid "$EXT_LOG_DIR/pid" 2>/dev/null)"
if [[ -f "$EXT_LOG_DIR/pid" ]] && [[ -n "$EXT_PID_PROBE" ]] && kill -0 "$EXT_PID_PROBE" 2>/dev/null; then
    printf "%s[A1.2 sigma-sweep EXTENSION, 3 cells]%s  log=%s\n" \
        "$BOLD" "$RESET" "$EXT_LOG_DIR"

    EXT_PID="$(read_pid "$EXT_LOG_DIR/pid")"
    if [[ -n "$EXT_PID" ]] && info=$(ps_alive "$EXT_PID"); [[ -n "$info" ]]; then
        ext_etime="$(echo "$info" | awk '{print $1}')"
        worker_info="$(worker_metrics_for "$EXT_PID")"
        if [[ -n "$worker_info" ]]; then
            n_procs="$(echo "$worker_info" | awk '{print $1}')"
            total_cpu="$(echo "$worker_info" | awk '{print $2}')"
            total_rss="$(echo "$worker_info" | awk '{print $3}')"
            busy_pid="$(echo "$worker_info" | awk '{print $4}')"
            total_rss_gb="$(fmt_gb "$total_rss")"
            printf "  driver PID : %s%-6s%s  elapsed %s\n" \
                "$GREEN" "$EXT_PID" "$RESET" "$ext_etime"
            printf "  workers    : %s%d procs%s  CPU %s%.0f%%%s  RSS %s  (busiest PID %s)\n" \
                "$CYAN" "$n_procs" "$RESET" \
                "$CYAN" "$total_cpu" "$RESET" \
                "$total_rss_gb" "$busy_pid"
        else
            printf "  driver PID : %s%-6s%s  elapsed %s  %s(no active worker)%s\n" \
                "$GREEN" "$EXT_PID" "$RESET" "$ext_etime" "$DIM" "$RESET"
        fi
        EXT_RUNNING=1
    elif [[ -n "$EXT_PID" ]]; then
        printf "  driver PID : %s%-6s%s  %s(not running -- finished or died)%s\n" \
            "$DIM" "$EXT_PID" "$RESET" "$DIM" "$RESET"
        EXT_RUNNING=0
    else
        EXT_RUNNING=0
    fi

    ext_cur_sigma="$(current_sigma_from_log "$EXT_LOG_DIR/stdout.log")"
    printf "  %-9s %-9s %-9s %-9s\n" "sigma" "state" "succ" "result-path"
    echo "  $SUB"

    n_done_ext=0
    n_running_ext=0
    for sigma in "${EXT_SIGMAS[@]}"; do
        json="$SIGMA_OUT_DIR/sigma_${sigma}.json"
        if [[ -f "$json" ]]; then
            sr="$(read_succ_rate "$json")"
            printf "  %-9s %s%-9s%s %-9s %s\n" \
                "$sigma" "$GREEN" "DONE" "$RESET" "$sr" \
                "$(basename "$json")"
            n_done_ext=$((n_done_ext + 1))
        elif [[ "$sigma" == "$ext_cur_sigma" && "$EXT_RUNNING" == "1" ]]; then
            printf "  %-9s %s%-9s%s %-9s %s\n" \
                "$sigma" "$YELLOW" "RUNNING" "$RESET" "-" "(in progress)"
            n_running_ext=1
        else
            printf "  %-9s %s%-9s%s %-9s %s\n" \
                "$sigma" "$DIM" "pending" "$RESET" "-" "-"
        fi
    done
    echo "  $SUB"
    printf "  done: %s%d/%d%s    running: %s%d%s    current sigma: %s\n" \
        "$GREEN" "$n_done_ext" "${#EXT_SIGMAS[@]}" "$RESET" \
        "$YELLOW" "$n_running_ext" "$RESET" \
        "${ext_cur_sigma:-?}"
    echo ""
fi

# ---- 3. System load + aggregate ----
sys_load="$(uptime | awk -F'load average[s]?: ' '{print $2}')"

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

# Count all run_can worker python subprocesses (the ones doing
# actual compute). The pattern is broad on purpose: catches both
# `python -m experiments.X.run_can` (regression) and
# `python experiments/X/run_can.py` (sigma sweep).
n_pyworkers="$(pgrep -fl 'run_can' 2>/dev/null | grep -v 'grep' | grep -c 'python')"
n_pyworkers="${n_pyworkers:-0}"

printf "%ssystem%s     load avg : %s\n" "$BOLD" "$RESET" "$sys_load"
[[ -n "$sys_mem_used_gb" ]] && \
    printf "           mem used : %s\n" "$sys_mem_used_gb"
printf "           workers  : %s%s%s python run_can processes active\n" \
    "$CYAN" "$n_pyworkers" "$RESET"
echo ""

# ---- 4. Tail of active log(s) ----
# Show each log tail ONLY when its driver PID is currently alive --
# the historical tails for finished jobs are pure noise.
SIGMA_PID_PROBE="$(read_pid "$SIGMA_LOG_DIR/pid" 2>/dev/null)"
if [[ -f "$SIGMA_LOG_DIR/stdout.log" ]] && [[ -n "$SIGMA_PID_PROBE" ]] && kill -0 "$SIGMA_PID_PROBE" 2>/dev/null; then
    printf "%sA1.2 sigma-sweep log (tail 3)%s\n" "$BOLD" "$RESET"
    tail -3 "$SIGMA_LOG_DIR/stdout.log" 2>/dev/null | sed 's/^/  /'
    echo ""
fi
REGR_PID_PROBE="$(read_pid "$REGR_LOG_DIR/pid" 2>/dev/null)"
if [[ -f "$REGR_LOG_DIR/stdout.log" ]] && [[ -n "$REGR_PID_PROBE" ]] && kill -0 "$REGR_PID_PROBE" 2>/dev/null; then
    printf "%sC Phase 1 regression log (tail 3)%s\n" "$BOLD" "$RESET"
    tail -3 "$REGR_LOG_DIR/stdout.log" 2>/dev/null | sed 's/^/  /'
fi

echo "$RULE"
