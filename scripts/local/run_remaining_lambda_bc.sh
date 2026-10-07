#!/usr/bin/env bash
# Run the 4 missing lambda-BC sweep cells locally on macOS.
#
# Why this script exists:
#   The overnight GCE launcher (scripts/gce/launch_overnight.sh) was
#   originally tasked with the full 6-value BC-prior lambda sweep in
#   experiments/07_lambda_sweep, but we killed the in-flight runs to
#   focus the GCE budget on the closed-vs-open-loop pareto grid.
#   Two values (lam=0, lam=inf) had already uploaded from earlier
#   sessions. The four middle values are still missing. This script
#   runs them on the M-series Mac, in parallel with whatever else is
#   running.
#
# Cells run by this script (must match the GCE lambda-bc emitter so
# results land in the same lambda_{lam}.json filenames):
#   lambda = 0.1, 0.5, 1.0, 5.0
#
# All four run in PARALLEL by default. With n-workers=2 each that's
#   - 1 main run_can.py per cell        = 4 mains
#   - 2 multiprocessing workers per cell = 8 workers
#   - 4 resource-tracker helpers
#   = 16 python processes added on top of any concurrent pareto runs.
#
# On an Apple M4 Max (12 perf + 4 efficiency cores, 64 GB) this
# cohabits cleanly with the existing 3-cell pareto runner: 14 active
# workers compete for 12 perf cores, with the extra 2 spilling to
# efficiency cores (~7% throughput penalty -- well worth not waiting).
#
# Usage:
#   bash scripts/local/run_remaining_lambda_bc.sh           # foreground
#   nohup bash scripts/local/run_remaining_lambda_bc.sh &   # background
#
# Env overrides (defaults match the GCE lambda-bc cells exactly):
#   N_DEMOS=20  N_ROLLOUTS=3  SEED=0
#   N_WORKERS=2
#   CONDA_ENV=hybrid-traj
#   OUT_DIR=experiments/07_lambda_sweep/results
#   LOG_DIR=logs/local_lambda_bc
#
# Resume:
#   The script SKIPS any cell whose output JSON already exists, so
#   re-running after a partial completion is safe.
#
# Note on numpy 2.x: experiments/07_lambda_sweep/run_can.py already
# masks its hash() seeds with & 0xFFFFFFFF (see line 280), so unlike
# exp 05 (which needed a fix on 2026-05-14) this script runs as-is.

set -euo pipefail

# ---- repo root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (match GCE lambda-bc cells) ----
N_DEMOS="${N_DEMOS:-20}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
SEED="${SEED:-0}"
N_WORKERS="${N_WORKERS:-2}"
CONDA_ENV="${CONDA_ENV:-hybrid-traj}"
OUT_DIR="${OUT_DIR:-experiments/07_lambda_sweep/results}"
LOG_DIR="${LOG_DIR:-logs/local_lambda_bc}"

# ---- lambdas to run (the 4 missing from GCS) ----
LAMBDAS=(0.1 0.5 1.0 5.0)

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

# ---- launcher function: starts one lambda cell in the background ----
START_TS="$(date -u +%Y%m%dT%H%M%SZ)"
launch_cell() {
    local lam="$1"
    # exp 07 writes lambda_bc_{lam}.json for prior=bc (since the
    # 2026-05-14 filename-disambiguation patch). Float-formatting must
    # match Python's repr: 0.1 -> "0.1", 1.0 -> "1.0", etc. The two
    # GCS-uploaded cells (lam=0, lam=inf) use this same naming.
    local out="$OUT_DIR/lambda_bc_${lam}.json"
    local slug="lam${lam}"
    # Sanitise slug for filenames (e.g. "0.5" -> "0-5" so it pairs with
    # logs/ entries from the GCE lambda-bc cells without name confusion).
    local fslug
    fslug="$(echo "$slug" | tr '.' '-')"
    local log="$LOG_DIR/${fslug}_${START_TS}.log"

    if [[ -f "$out" ]]; then
        echo "[skip] lam=$lam: $out already exists"
        return 0
    fi

    {
        echo "=== local lambda-bc cell lam=$lam ==="
        echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host           : $(hostname)"
        echo "python         : $PY"
        echo "python version : $PY_VER"
        echo "conda env      : $CONDA_ENV"
        echo "lambda-prior   : $lam"
        echo "prior          : bc"
        echo "bc-checkpoint  : $BC_CKPT"
        echo "n-demos        : $N_DEMOS"
        echo "n-rollouts     : $N_ROLLOUTS"
        echo "seed           : $SEED"
        echo "n-workers      : $N_WORKERS"
        echo "output         : $out"
        echo "==="
    } > "$log"

    # NOTE: flags must match scripts/gce/launch_overnight.sh::emit_lambda_sweep_bc
    # exactly so the resulting JSON is interchangeable with the GCE-
    # produced lam-bc-0 / lam-bc-inf cells.
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

# ---- launch all cells ----
echo "==============================================================="
echo "local lambda-bc runner -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo: $REPO_ROOT"
echo "python: $PY ($PY_VER)"
echo "cells: ${#LAMBDAS[@]} (lam = ${LAMBDAS[*]})"
echo "out:   $OUT_DIR"
echo "logs:  $LOG_DIR"
echo "==============================================================="

PIDS=()
for lam in "${LAMBDAS[@]}"; do
    launch_cell "$lam"
    fslug="$(echo "lam${lam}" | tr '.' '-')"
    pid="$(cat "$LOG_DIR/${fslug}_${START_TS}.pid" 2>/dev/null || true)"
    [[ -n "$pid" ]] && PIDS+=("$pid")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[done] all 4 cells were already complete -- nothing to do"
    exit 0
fi

echo ""
echo "==============================================================="
echo "all ${#PIDS[@]} cells launched in parallel  (PIDs: ${PIDS[*]})"
echo ""
echo "tail any log with:    tail -f $LOG_DIR/lam*_${START_TS}.log"
echo "check progress with:  ps -p ${PIDS[*]} -o pid,etime,pcpu,pmem,command"
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
echo "local lambda-bc runner done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "outputs in $OUT_DIR:"
ls -la "$OUT_DIR"/lambda_*.json 2>/dev/null | sed 's/^/  /'
echo "==============================================================="

exit $FAILED
