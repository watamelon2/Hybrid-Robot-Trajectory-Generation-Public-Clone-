#!/usr/bin/env bash
# Full 6-value BC-prior lambda sweep, run locally in parallel.
#
# Why this script exists (vs run_remaining_lambda_bc.sh):
#   run_remaining_lambda_bc.sh covers only the 4 middle lambdas
#   that were missing from GCS. After the BC-swap to Charles's
#   [256, 256] checkpoint the WHOLE sweep needs to be re-run
#   because every cell's BC prior path now uses a different
#   policy. This script does the full 6-cell sweep from scratch.
#
# Lambdas: 0.0, 0.1, 0.5, 1.0, 5.0, inf  (6 cells).
#
# Concurrency: each cell runs with n-workers=1 (sequential MBD),
# so 6 main python processes total. Fits comfortably in the M4
# Max's 12 P-cores. Cheap to run after exp 10 finishes (the Mac
# is otherwise idle), wall ~ 1-2 h.
#
# Usage:
#   bash scripts/local/run_lambda_bc_full.sh           # foreground
#   nohup bash scripts/local/run_lambda_bc_full.sh &   # background
#
# Env overrides:
#   N_DEMOS=20  N_ROLLOUTS=3  SEED=0
#   N_WORKERS=1
#   CONDA_ENV=hybrid-traj
#   OUT_DIR=experiments/07_lambda_sweep/results
#   LOG_DIR=logs/local_lambda_bc_full
#
# Resume:
#   Skips any cell whose output JSON already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults ----
N_DEMOS="${N_DEMOS:-20}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
SEED="${SEED:-0}"
N_WORKERS="${N_WORKERS:-1}"
CONDA_ENV="${CONDA_ENV:-hybrid-traj}"
OUT_DIR="${OUT_DIR:-experiments/07_lambda_sweep/results}"
LOG_DIR="${LOG_DIR:-logs/local_lambda_bc_full}"

# ---- 6 lambdas (full sweep) ----
LAMBDAS=(0.0 0.1 0.5 1.0 5.0 inf)

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- conda activation ----
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found on PATH and ~/anaconda3 not present" >&2
        exit 1
    fi
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

PY="$(command -v python)"
PY_VER="$(python --version 2>&1)"

# ---- pre-flight ----
RUN_SWEEP="experiments/07_lambda_sweep/run_can.py"
BC_CKPT="experiments/03_bc_baseline/results/policy.pt"
for f in "$RUN_SWEEP" "$BC_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2
        exit 1
    fi
done
BC_MD5="$(md5 -q "$BC_CKPT" 2>/dev/null || md5sum "$BC_CKPT" | cut -d' ' -f1)"

# ---- launcher ----
START_TS="$(date -u +%Y%m%dT%H%M%SZ)"
launch_cell() {
    local lam="$1"
    local out="$OUT_DIR/lambda_bc_${lam}.json"
    local fslug
    fslug="lam$(echo "$lam" | tr '.' '-')"
    local log="$LOG_DIR/${fslug}_${START_TS}.log"

    if [[ -f "$out" ]]; then
        echo "[skip] lam=$lam: $out already exists"
        return 0
    fi

    {
        echo "=== local lambda-bc FULL cell lam=$lam ==="
        echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host           : $(hostname)"
        echo "python         : $PY"
        echo "python version : $PY_VER"
        echo "conda env      : $CONDA_ENV"
        echo "BC checkpoint  : $BC_CKPT"
        echo "BC md5         : $BC_MD5 (expect 4c78fdab... for new [256,256])"
        echo "lambda-prior   : $lam"
        echo "prior          : bc"
        echo "n-demos        : $N_DEMOS"
        echo "n-rollouts     : $N_ROLLOUTS"
        echo "seed           : $SEED"
        echo "n-workers      : $N_WORKERS"
        echo "output         : $out"
        echo "==="
    } > "$log"

    nohup python "$RUN_SWEEP" \
        --prior bc \
        --bc-checkpoint "$BC_CKPT" \
        --lambda-prior "$lam" \
        --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" \
        --seed "$SEED" --no-live-view \
        --n-workers "$N_WORKERS" \
        --output-dir "$OUT_DIR" \
        >> "$log" 2>&1 &

    local pid=$!
    echo "[start] lam=$lam  pid=$pid  log=$log"
    echo "$pid" > "$LOG_DIR/${fslug}_${START_TS}.pid"
}

echo "==============================================================="
echo "local lambda-bc FULL runner -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo:    $REPO_ROOT"
echo "python:  $PY ($PY_VER)"
echo "BC ckpt: $BC_CKPT  (md5 $BC_MD5)"
echo "cells:   ${#LAMBDAS[@]} (lam = ${LAMBDAS[*]})"
echo "out:     $OUT_DIR"
echo "logs:    $LOG_DIR"
echo "==============================================================="

PIDS=()
for lam in "${LAMBDAS[@]}"; do
    launch_cell "$lam"
    fslug="lam$(echo "$lam" | tr '.' '-')"
    pid="$(cat "$LOG_DIR/${fslug}_${START_TS}.pid" 2>/dev/null || true)"
    [[ -n "$pid" ]] && PIDS+=("$pid")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[done] all 6 cells were already complete -- nothing to do"
    exit 0
fi

echo ""
echo "==============================================================="
echo "all ${#PIDS[@]} cells launched in parallel  (PIDs: ${PIDS[*]})"
echo "tail any log:  tail -f $LOG_DIR/lam*_${START_TS}.log"
echo "check ps:      ps -p ${PIDS[*]} -o pid,etime,pcpu,pmem,command"
echo "==============================================================="

FAILED=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "[ok] pid=$pid exited cleanly"
    else
        rc=$?
        echo "[FAIL] pid=$pid exited with code $rc"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "==============================================================="
echo "local lambda-bc FULL runner done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "outputs in $OUT_DIR:"
ls -la "$OUT_DIR"/lambda_bc_*.json 2>/dev/null | sed 's/^/  /'
echo "==============================================================="

exit $FAILED
