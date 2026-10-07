#!/usr/bin/env bash
# Full 9-cell K x replans pareto sweep, run locally in parallel.
#
# Why this script exists (vs run_remaining_pareto.sh):
#   run_remaining_pareto.sh covers only the 3 cells we killed on
#   GCE (k10-r2, k20-r1, k20-r2). After the BC-swap (the
#   [256, 256] checkpoint from Charles-test) we need to RE-RUN
#   all 9 cells from scratch because the warm-start is different
#   and the old JSONs are now apples-to-oranges. This script
#   does that.
#
# Cells: K in {5, 10, 20} x R in {0, 1, 2} = 9 cells.
#
# Concurrency: each cell runs with n-workers=1 (sequential MBD
# within the cell, so one main python process per cell). 9 cells
# parallel = 9 cores on the Mac M4 Max (12 P-cores, 4 E-cores)
# which fits cleanly without oversubscription. Cheap cells
# (K=5 R=0) finish in ~30 min, the longest (K=20 R=2) is the
# wall-time bottleneck at ~12-15 h. Approx peak RAM: ~5 GB.
#
# Usage:
#   bash scripts/local/run_pareto_full.sh           # foreground
#   nohup bash scripts/local/run_pareto_full.sh &   # background
#
# Env overrides:
#   N_DEMOS=20  N_ROLLOUTS=3  SEED=0  SIGMA_MAX=0.25
#   N_WORKERS=1                       # workers PER cell (default 1)
#   CONDA_ENV=hybrid-traj
#   OUT_DIR=experiments/10_kxreplans_grid/results
#   LOG_DIR=logs/local_pareto_full
#
# Resume:
#   Skips any cell whose output JSON already exists. Safe to
#   re-launch after partial completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (match the canonical pareto protocol) ----
N_DEMOS="${N_DEMOS:-20}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
SEED="${SEED:-0}"
SIGMA_MAX="${SIGMA_MAX:-0.25}"
N_WORKERS="${N_WORKERS:-1}"
CONDA_ENV="${CONDA_ENV:-hybrid-traj}"
OUT_DIR="${OUT_DIR:-experiments/10_kxreplans_grid/results}"
LOG_DIR="${LOG_DIR:-logs/local_pareto_full}"

# ---- 9 cells, "K R" per line ----
CELLS=(
    "5 0"
    "5 1"
    "5 2"
    "10 0"
    "10 1"
    "10 2"
    "20 0"
    "20 1"
    "20 2"
)

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

# ---- pre-flight: BC checkpoint must be the new [256, 256] one ----
RUN_CAN="experiments/05_hybrid_method/run_can.py"
BC_CKPT="experiments/03_bc_baseline/results/policy.pt"
for f in "$RUN_CAN" "$BC_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2
        exit 1
    fi
done

# Stamp the BC md5 in the launcher log so we can verify post-hoc
# which checkpoint was used. The new [256, 256] / 80k-param policy
# is md5 4c78fdab; the OLD [512, 512] / 290k-param one was
# md5 d6d0b895 and is archived under results/_archive/.
BC_MD5="$(md5 -q "$BC_CKPT" 2>/dev/null || md5sum "$BC_CKPT" | cut -d' ' -f1)"

# ---- launcher ----
START_TS="$(date -u +%Y%m%dT%H%M%SZ)"
launch_cell() {
    local K="$1" R="$2"
    local slug="k${K}_r${R}"
    local out="$OUT_DIR/K${K}_R${R}.json"
    local log="$LOG_DIR/${slug}_${START_TS}.log"

    if [[ -f "$out" ]]; then
        echo "[skip] $slug: $out already exists"
        return 0
    fi

    {
        echo "=== local pareto FULL cell $slug ==="
        echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host           : $(hostname)"
        echo "python         : $PY"
        echo "python version : $PY_VER"
        echo "conda env      : $CONDA_ENV"
        echo "BC checkpoint  : $BC_CKPT"
        echo "BC md5         : $BC_MD5 (expect 4c78fdab... for new [256,256])"
        echo "K              : $K"
        echo "R (mpc-replans): $R"
        echo "M              : 16 (default)"
        echo "sigma-max      : $SIGMA_MAX"
        echo "n-demos        : $N_DEMOS"
        echo "n-rollouts     : $N_ROLLOUTS"
        echo "seed           : $SEED"
        echo "n-workers      : $N_WORKERS"
        echo "output         : $out"
        echo "==="
    } > "$log"

    # Flags MUST match scripts/gce/launch_overnight.sh::emit_pareto
    # so resulting JSONs are interchangeable with GCE-produced cells.
    nohup python "$RUN_CAN" \
        --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" \
        --seed "$SEED" --no-live-view \
        --K "$K" --sigma-max "$SIGMA_MAX" \
        --mpc-replans "$R" --deterministic-bc-reset \
        --n-workers "$N_WORKERS" \
        --output "$out" \
        --video-dir '' \
        >> "$log" 2>&1 &

    local pid=$!
    echo "[start] $slug  pid=$pid  log=$log"
    echo "$pid" > "$LOG_DIR/${slug}_${START_TS}.pid"
}

# ---- header ----
echo "==============================================================="
echo "local pareto FULL runner -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo:    $REPO_ROOT"
echo "python:  $PY ($PY_VER)"
echo "BC ckpt: $BC_CKPT  (md5 $BC_MD5)"
echo "cells:   ${#CELLS[@]} (full 3 x 3 = 9 grid)"
echo "n-workers/cell: $N_WORKERS"
echo "out:     $OUT_DIR"
echo "logs:    $LOG_DIR"
echo "==============================================================="

PIDS=()
for spec in "${CELLS[@]}"; do
    read -r K R <<< "$spec"
    launch_cell "$K" "$R"
    pid="$(cat "$LOG_DIR/k${K}_r${R}_${START_TS}.pid" 2>/dev/null || true)"
    [[ -n "$pid" ]] && PIDS+=("$pid")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[done] all 9 cells were already complete -- nothing to do"
    exit 0
fi

echo ""
echo "==============================================================="
echo "all ${#PIDS[@]} cells launched in parallel  (PIDs: ${PIDS[*]})"
echo ""
echo "monitor with:         bash scripts/local/monitor_pareto_full.sh"
echo "tail any log:         tail -f $LOG_DIR/k*_${START_TS}.log"
echo "check ps:             ps -p ${PIDS[*]} -o pid,etime,pcpu,pmem,cmd"
echo ""
echo "Ctrl+C-ing this wait loop is safe; background pythons survive."
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
echo "local pareto FULL runner done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "outputs in $OUT_DIR:"
ls -la "$OUT_DIR"/K*_R*.json 2>/dev/null | sed 's/^/  /'
echo "==============================================================="

exit $FAILED
