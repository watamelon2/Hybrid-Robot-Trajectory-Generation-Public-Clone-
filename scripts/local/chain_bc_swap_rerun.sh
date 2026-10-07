#!/usr/bin/env bash
# Chain the BC-swap re-run sequence: wait for exp 10 to finish,
# run analyzers, launch exp 07, run analyzer, mark done.
#
# Why this script exists:
#   The user wants exp 10 (9 cells, ~5-15h) and exp 07 (6 cells,
#   ~1-2h) re-run sequentially with the new [256, 256] BC, but
#   doesn't want to wake up to launch exp 07 manually. This script
#   waits for the 9 exp-10 PIDs to finish, then auto-launches the
#   exp-07 sweep and runs both analyzers.
#
# Usage:
#   nohup bash scripts/local/chain_bc_swap_rerun.sh PID1 PID2 ... &
#
# Arguments: the 9 PIDs of the exp 10 cells. Status reads:
#   STATUS_FILE = logs/bc_swap_rerun/chain_status.txt
# is updated at each step. Tail it to see progress overnight.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 PID1 PID2 ... (PIDs of exp 10 cells to wait for)" >&2
    exit 2
fi

EXP10_PIDS=("$@")
STATUS_FILE="${STATUS_FILE:-logs/bc_swap_rerun/chain_status.txt}"
CHAIN_LOG="${CHAIN_LOG:-logs/bc_swap_rerun/chain_$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "$(dirname "$STATUS_FILE")" "$(dirname "$CHAIN_LOG")"

# All output also goes to STATUS_FILE and CHAIN_LOG.
exec > >(tee -a "$CHAIN_LOG") 2>&1

log_step() {
    local msg="$1"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[chain $ts] $msg"
    echo "[$ts] $msg" >> "$STATUS_FILE"
}

# ---- conda activation (needed for analyzers and exp 07 launcher) ----
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
eval "$(conda shell.bash hook)"
conda activate hybrid-traj

log_step "chain start: waiting on ${#EXP10_PIDS[@]} exp-10 PIDs: ${EXP10_PIDS[*]}"

# ---- step 1: wait for all exp 10 PIDs ----
# `wait` only works on direct children of THIS shell. The exp-10
# cells are NOT children of this script; they were started by an
# earlier shell. So we poll instead.
for pid in "${EXP10_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log_step "exp10 pid=$pid done"
done

log_step "all 9 exp-10 cells finished"

# ---- step 2: count completed JSONs ----
PARETO_DIR="experiments/10_kxreplans_grid/results"
n_done=$(ls "$PARETO_DIR"/K*_R*.json 2>/dev/null | wc -l | tr -d ' ')
log_step "exp10 JSONs present: $n_done / 9"
if [[ "$n_done" -ne 9 ]]; then
    log_step "WARNING: not all 9 exp10 cells produced JSON; proceeding anyway"
fi

# ---- step 3: exp 10 analyzer ----
log_step "running exp 10 analyze.py"
if python experiments/10_kxreplans_grid/analyze.py >> "$CHAIN_LOG" 2>&1; then
    log_step "exp 10 analyze done"
else
    log_step "exp 10 analyze FAILED (rc=$?)"
fi

# ---- step 4: exp 10 failure taxonomy ----
log_step "running exp 06 failure taxonomy on exp 10 outputs"
if python experiments/06_failure_taxonomy/run.py \
        --inputs "$PARETO_DIR"/K*_R*.json \
        --output "$PARETO_DIR/taxonomy.json" >> "$CHAIN_LOG" 2>&1; then
    log_step "exp 10 taxonomy done"
else
    log_step "exp 10 taxonomy FAILED (rc=$?)"
fi

# ---- step 5: launch exp 07 lambda-bc full sweep ----
log_step "launching exp 07 lambda-bc full sweep (6 cells in parallel)"
nohup bash scripts/local/run_lambda_bc_full.sh \
    >> "$CHAIN_LOG" 2>&1 &
EXP07_LAUNCHER_PID=$!
log_step "exp 07 launcher pid=$EXP07_LAUNCHER_PID"

# Find the 6 cell PIDs once the launcher has spawned them.
sleep 30
EXP07_PIDS=()
for f in logs/local_lambda_bc_full/lam*_*.pid; do
    [[ -f "$f" ]] || continue
    p="$(cat "$f")"
    if kill -0 "$p" 2>/dev/null; then
        EXP07_PIDS+=("$p")
    fi
done
log_step "exp 07 cells running: ${#EXP07_PIDS[@]} (PIDs: ${EXP07_PIDS[*]:-none})"

# ---- step 6: wait for exp 07 to finish ----
for pid in "${EXP07_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
    log_step "exp07 pid=$pid done"
done

log_step "all exp-07 cells finished"

# ---- step 7: exp 07 analyzer ----
log_step "running exp 07 analyze.py"
if python experiments/07_lambda_sweep/analyze.py >> "$CHAIN_LOG" 2>&1; then
    log_step "exp 07 analyze done"
else
    log_step "exp 07 analyze FAILED (rc=$?)"
fi

# ---- final summary ----
log_step "CHAIN DONE"
log_step "exp 03 headline: $(python -c "import json; d=json.load(open('experiments/03_bc_baseline/results/can_ph.json')); print(f'{d[\"n_success\"]}/{d[\"n_total\"]} = {100*d[\"success_rate\"]:.1f}%')")"

log_step "exp 10 summary:"
[[ -f "$PARETO_DIR/summary.md" ]] && cat "$PARETO_DIR/summary.md" | sed 's/^/  /' >> "$STATUS_FILE"

log_step "exp 07 summary:"
[[ -f experiments/07_lambda_sweep/results/_archive/summary.md ]] && \
    cat experiments/07_lambda_sweep/results/_archive/summary.md | sed 's/^/  /' >> "$STATUS_FILE"

log_step "chain log: $CHAIN_LOG"
log_step "status file: $STATUS_FILE"
log_step "done"
