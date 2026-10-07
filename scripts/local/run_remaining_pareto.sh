#!/usr/bin/env bash
# Run the 3 most-expensive K x replans pareto cells locally on macOS.
#
# Why this script exists:
#   The overnight GCE launcher (scripts/gce/launch_overnight.sh) ran
#   the 9-cell K in {5,10,20} x R in {0,1,2} grid, but the K=20 R>=1
#   and K=10 R=2 corners take 13-25h each on an e2-standard-2. To
#   free the GCE IP quota for the upcoming MJX (L4) experiment we
#   killed those 3 VMs and re-run the cells locally on the user's
#   Apple Silicon Mac.
#
# Cells run by this script (must match the GCE pareto emitter so
# results land in the same K{K}_R{R}.json filenames):
#   k10-r2  (K=10, R=2)   ~ 13h wall, longest on Mac M-series
#   k20-r1  (K=20, R=1)   ~ 9-11h wall
#   k20-r2  (K=20, R=2)   ~ 12-15h wall
#
# All three run in PARALLEL by default. Each spawns
#   - 1 main run_can.py process
#   - n-workers child processes (default 2, matches GCE)
# Total = 9 python processes for n-workers=2 (3 * (1+2)). Tested fine
# on 16 GB Apple Silicon; bump RAM check if you change n-workers.
#
# Usage:
#   bash scripts/local/run_remaining_pareto.sh           # foreground
#   nohup bash scripts/local/run_remaining_pareto.sh &   # background
#
# Env overrides (defaults match the GCE pareto cells exactly):
#   N_DEMOS=20  N_ROLLOUTS=3  SEED=0  SIGMA_MAX=0.25
#   N_WORKERS=2                       # workers per cell
#   CONDA_ENV=hybrid-traj
#   OUT_DIR=experiments/10_kxreplans_grid/results
#   LOG_DIR=logs/local_pareto
#
# Resume:
#   The script SKIPS any cell whose output JSON already exists, so
#   re-running after a partial completion is safe.

set -euo pipefail

# ---- repo root (resolve regardless of where this script is invoked) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (match GCE pareto cells) ----
N_DEMOS="${N_DEMOS:-20}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
SEED="${SEED:-0}"
SIGMA_MAX="${SIGMA_MAX:-0.25}"
N_WORKERS="${N_WORKERS:-2}"
CONDA_ENV="${CONDA_ENV:-hybrid-traj}"
OUT_DIR="${OUT_DIR:-experiments/10_kxreplans_grid/results}"
LOG_DIR="${LOG_DIR:-logs/local_pareto}"

# ---- cells to run (the 3 we killed on GCE) ----
# Format: "K R" per line
CELLS=(
    "10 2"
    "20 1"
    "20 2"
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- conda activation (POLICY: use python -m pip, never bare pip) ----
# The user's setup keeps anaconda3 at ~/anaconda3; if conda is not on
# PATH we source it explicitly. This matches the policy in
# docs/python_setup.md.
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found on PATH and ~/anaconda3 not present" >&2
        exit 1
    fi
fi
# `conda activate` requires `eval "$(conda shell.bash hook)"` in
# non-interactive scripts; this is the documented incantation.
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Sanity: which python are we about to use? (Stamped in every log
# header so the user can verify post-hoc that the Mac runs went
# through the right venv.)
PY="$(command -v python)"
PY_VER="$(python --version 2>&1)"

# ---- pre-flight: verify run_can.py and policy checkpoint exist ----
RUN_CAN="experiments/05_hybrid_method/run_can.py"
BC_CKPT="experiments/03_bc_baseline/results/policy.pt"
for f in "$RUN_CAN" "$BC_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file missing: $f" >&2
        exit 1
    fi
done

# ---- launcher function: starts one cell in the background ----
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
        echo "=== local pareto cell $slug ==="
        echo "started        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host           : $(hostname)"
        echo "python         : $PY"
        echo "python version : $PY_VER"
        echo "conda env      : $CONDA_ENV"
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

    # NOTE: flags match scripts/gce/launch_overnight.sh::emit_pareto
    # exactly so the resulting JSON is interchangeable with the GCE-
    # produced cells. Do NOT add or remove flags without updating both.
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

# ---- launch all cells ----
echo "==============================================================="
echo "local pareto runner -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo: $REPO_ROOT"
echo "python: $PY ($PY_VER)"
echo "cells: ${#CELLS[@]} (k10-r2, k20-r1, k20-r2)"
echo "out:   $OUT_DIR"
echo "logs:  $LOG_DIR"
echo "==============================================================="

PIDS=()
for spec in "${CELLS[@]}"; do
    read -r K R <<< "$spec"
    launch_cell "$K" "$R"
    pid="$(cat "$LOG_DIR/k${K}_r${R}_${START_TS}.pid" 2>/dev/null || true)"
    [[ -n "$pid" ]] && PIDS+=("$pid")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[done] all 3 cells were already complete -- nothing to do"
    exit 0
fi

echo ""
echo "==============================================================="
echo "all ${#PIDS[@]} cells launched in parallel  (PIDs: ${PIDS[*]})"
echo ""
echo "tail any log with:    tail -f $LOG_DIR/k*_${START_TS}.log"
echo "check progress with:  ps -p ${PIDS[*]} -o pid,etime,pcpu,pmem,cmd"
echo ""
echo "waiting for completion (you can safely Ctrl+C this wait loop;"
echo "the background python processes will keep running)..."
echo "==============================================================="

# Wait for all and report exit codes. `wait` is in this shell so the
# python processes parent here; if you Ctrl+C, they survive as orphans
# (intentional -- supports nohup-and-go workflow).
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
echo "local pareto runner done -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "outputs in $OUT_DIR:"
ls -la "$OUT_DIR"/K*_R*.json 2>/dev/null | sed 's/^/  /'
echo "==============================================================="

exit $FAILED
