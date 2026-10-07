#!/bin/bash
# Phase-D epoch-curve launcher (2026-05-30).
#
# Goal: build the headline plot for the "MBD lifts undertrained DP" story.
#       For each v3b training checkpoint (ep0049, ep0099, ep0149, ep0199),
#       run hybrid K=10 eval under the DP paper protocol, then compare
#       against Phase-B's standalone DP curve.
#
# Story expected:
#
#     test rate
#       80% |  *--*--*--*    hybrid K=10 (hopefully flat near 80%)
#       60% |  *
#           | /
#       40% | *
#           |/  *  *  *  *   standalone DP (Phase-B's learning curve)
#       20% | *
#           +________________
#             49  99  149 199  v3b epoch
#
# Parallel design:
#   - All four ckpts run concurrently as background jobs (`&`).
#   - Each uses --n-workers 2 (matching the killed Phase-C K-sweep).
#   - nice -n 10 deprioritises vs interactive macOS work.
#   - Per-ckpt log + result dir keeps output untangled.
#
# Capacity:
#   - 4 mains + 8 workers = ~12 cores on a 16-core M4 Max.
#   - Each python keeps ~2-3 GB RAM; total ~10-12 GB extra.
#
# Wall budget (CPU-only Mac, 4 in parallel):
#   - bottleneck = ep0049 (weakest prior, may need full max_env_steps=400)
#   - ~90-140 min walltime end to end
#   - per-ckpt CPU time = same as the K-sweep's K=10 cell (~90-140 min)
#
# Usage (from the HRTG-dp repo root):
#
#   bash scripts/local/run_phase_d_epoch_curve.sh
#
# Or to run only a subset (e.g. just the most undertrained):
#
#   CKPT_TAGS="ep0049 ep0099" \
#     bash scripts/local/run_phase_d_epoch_curve.sh
#
# Idempotency: any tag with a non-empty result JSON in its output dir
# is skipped (so re-running picks up after a crash or partial run).
#
# Output:
#   experiments/17_dp_hybrid/results/phase_d_2026_05_30/epoch_curve_local/
#     ep0049/run.log                + unified_flow_lam-0.5_K-10_R-49.json
#     ep0099/run.log                + ...
#     ep0149/run.log                + ...
#     ep0199/run.log                + ...
#     epoch_curve_launch.log        (overall driver log)
#
set -euo pipefail

# --- knobs the next agent might want to change -------------------------
# Find a working python.  Outside conda envs the Mac only has `python3`
# (no `python`); inside the hybrid-traj conda env both work.  Prefer
# the conda env's python if available, then fall back to `python3`.
_default_python() {
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif [ -x "/Users/szh/anaconda3/envs/hybrid-traj/bin/python" ]; then
        # Last-resort hard-coded path: matches the local Mac's conda env.
        echo "/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
    else
        echo ""
    fi
}
PYTHON="${PYTHON:-$(_default_python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: no python found.  Set PYTHON=/path/to/python and re-run." >&2
    exit 1
fi
DEVICE="${DEVICE:-cpu}"
CKPT_TAGS="${CKPT_TAGS:-ep0049 ep0099 ep0149 ep0199}"
K="${K:-10}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-0.5}"
N_WORKERS="${N_WORKERS:-2}"
NICE_LEVEL="${NICE_LEVEL:-10}"
# -----------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_DIR_DEFAULT="$REPO_ROOT/experiments/08_flow_matching_prior/results"
OUTROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/epoch_curve_local"
LOG="$OUTROOT/epoch_curve_launch.log"

mkdir -p "$OUTROOT"

# --- header --------------------------------------------------------------
{
    echo "=========================================================="
    echo "Phase-D epoch-curve launch:  $(date)"
    echo "python:                      $PYTHON"
    echo "device:                      $DEVICE"
    echo "ckpts:                       $CKPT_TAGS"
    echo "K (hybrid depth):            $K"
    echo "lambda-prior:                $LAMBDA_PRIOR"
    echo "n-workers per ckpt:          $N_WORKERS"
    echo "nice level:                  $NICE_LEVEL"
    echo "outroot:                     $OUTROOT"
    echo "=========================================================="
} | tee -a "$LOG"

# --- shared flags (mirrors run_phase_c_ksweep.sh COMMON_FLAGS) -----------
common_flags=(
    --task can --prior flow
    --H 16
    --mpc-replans 49 --mpc-chunk-steps 8
    --n-demos 10 --n-rollouts 1
    --n-test-rollouts 50 --test-start-seed 100000
    --max-env-steps 400
    --prior-n-steps 100
    --sigma-max 0.25 --sigma-min 0.05
    --lambda-prior "$LAMBDA_PRIOR"
    --device "$DEVICE"
    --seed 0
    --K "$K"
    --n-workers "$N_WORKERS"
)

# --- resolve and validate every ckpt before launching anything -----------
declare -a CKPT_PATHS=()
declare -a CKPT_TAG_ARR=()
for tag in $CKPT_TAGS; do
    # We accept both bare epoch tags ("ep0049") and full filenames.
    if [[ "$tag" == ep* ]]; then
        candidate="$CKPT_DIR_DEFAULT/v_theta_dp_tobs2_h16_v3b_${tag}.pt"
    else
        candidate="$CKPT_DIR_DEFAULT/$tag"
    fi
    if [ ! -f "$candidate" ] && [ ! -L "$candidate" ]; then
        echo "ERROR: ckpt missing: $candidate" | tee -a "$LOG"
        echo "Hint: sync from Drive via" | tee -a "$LOG"
        echo "  $PYTHON $REPO_ROOT/experiments/08_flow_matching_prior/sync_ckpts_from_drive.py \\" | tee -a "$LOG"
        echo "      --folder-url 'https://drive.google.com/drive/folders/<ID>?usp=sharing' \\" | tee -a "$LOG"
        echo "      --pattern v3b_ep" | tee -a "$LOG"
        exit 1
    fi
    real=$(readlink -f "$candidate" 2>/dev/null || $PYTHON -c "import os; print(os.path.realpath('$candidate'))")
    size=$(stat -f%z "$real" 2>/dev/null || stat -c%s "$real")
    if [ "$size" -lt 100000000 ]; then
        echo "ERROR: ckpt $real is only $size bytes; expected > 100 MB" | tee -a "$LOG"
        exit 1
    fi
    short_tag=$(basename "$candidate" .pt | sed -E 's/^v_theta_dp_tobs2_h16_v3b_?//')
    short_tag=${short_tag:-v3b}
    CKPT_PATHS+=("$candidate")
    CKPT_TAG_ARR+=("$short_tag")
    echo "[plan] $short_tag -> $candidate ($(echo "$size" | awk '{printf "%.1f MB", $1/1e6}'))" | tee -a "$LOG"
done

# --- per-ckpt runner (backgrounded; each invocation runs in its own subshell)
run_ckpt() {
    local tag="$1"
    local ckpt="$2"
    local out="$OUTROOT/$tag"
    mkdir -p "$out"

    local existing
    existing=$(ls "$out"/unified_flow_lam-*.json 2>/dev/null | head -1 || true)
    if [ -n "$existing" ]; then
        echo "[$tag] already has $existing, skipping." | tee -a "$LOG"
        return 0
    fi

    {
        echo ""
        echo "=========================================================="
        echo "[$tag] starting:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[$tag] ckpt:       $ckpt"
        echo "=========================================================="
    } | tee -a "$LOG" > "$out/run.log"

    nice -n "$NICE_LEVEL" "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
        "${common_flags[@]}" \
        --prior-checkpoint "$ckpt" \
        --output-dir "$out" 2>&1 \
        | sed -u "s/^/[$tag] /" \
        | tee -a "$out/run.log" \
        | grep --line-buffered -E "\[$tag\] \[exp15\]" \
        >> "$LOG" || true

    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[$tag] FAILED with rc=$rc" | tee -a "$LOG"
        return "$rc"
    fi

    local jsonf
    jsonf=$(ls "$out"/unified_flow_lam-*.json 2>/dev/null | head -1)
    if [ -z "$jsonf" ]; then
        echo "[$tag] no JSON produced (unexpected)" | tee -a "$LOG"
        return 1
    fi
    "$PYTHON" -c "
import json
d = json.load(open('$jsonf'))
print(f\"  [$tag] train: {d['train_n_succ']:>2}/{d['train_n']:<2} = {d['train_succ_rate']*100:>4.1f}%\")
print(f\"  [$tag] test : {d['test_n_succ']:>2}/{d['test_n']:<2} = {d['test_succ_rate']*100:>4.1f}%\")
print(f\"  [$tag] wall : {d['wall_s']/60:.1f} min\")
" | tee -a "$LOG"
}

# --- fan-out: launch one background subshell per ckpt --------------------
echo "" | tee -a "$LOG"
echo "[fan-out] launching ${#CKPT_TAG_ARR[@]} parallel ckpt evals" | tee -a "$LOG"

declare -a PIDS=()
for i in "${!CKPT_TAG_ARR[@]}"; do
    tag="${CKPT_TAG_ARR[$i]}"
    ckpt="${CKPT_PATHS[$i]}"
    (run_ckpt "$tag" "$ckpt") &
    PIDS+=($!)
    echo "[fan-out] [$tag] PID=$!  (background)" | tee -a "$LOG"
    # stagger by 2 s so the loaders don't collide on robosuite imports.
    sleep 2
done

echo "" | tee -a "$LOG"
echo "[fan-out] all launched.  PIDs=${PIDS[*]}" | tee -a "$LOG"
echo "[fan-out] tail the per-ckpt logs:" | tee -a "$LOG"
for tag in "${CKPT_TAG_ARR[@]}"; do
    echo "  tail -f $OUTROOT/$tag/run.log" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "[fan-out] waiting for all $(( ${#PIDS[@]} )) jobs to finish ..." | tee -a "$LOG"

# Wait for all and capture exit codes.
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$(( FAILED + 1 ))
    fi
done

# --- final summary -------------------------------------------------
{
    echo ""
    echo "=========================================================="
    echo "epoch-curve sweep done: $(date)"
    echo "  failed jobs: $FAILED"
    echo "=========================================================="
    echo ""
    "$PYTHON" -c "
import json, glob, re
from pathlib import Path

out = Path('$OUTROOT')
def _ep(s):
    m = re.search(r'ep(\d+)', s)
    return int(m.group(1)) if m else 0

print(f\"{'ckpt':<10}  {'epoch':>5}  {'train':>14}  {'test':>14}  {'total':>14}  {'wall':>8}\")
print('-' * 76)
for d in sorted(out.iterdir(), key=lambda p: _ep(p.name)):
    if not d.is_dir(): continue
    jsons = sorted(d.glob('unified_flow_lam-*.json'))
    if not jsons:
        print(f'{d.name:<10}  {_ep(d.name):>5}  {\"  (no result)  \":>14}')
        continue
    j = json.load(open(jsons[0]))
    print(
        f'{d.name:<10}  '
        f'{_ep(d.name):>5}     '
        + f\"{j['train_n_succ']}/{j['train_n']}={j['train_succ_rate']*100:.1f}%\".rjust(14)
        + '  '
        + f\"{j['test_n_succ']}/{j['test_n']}={j['test_succ_rate']*100:.1f}%\".rjust(14)
        + '  '
        + f\"{j['n_success']}/{j['n_total']}={j['success_rate']*100:.1f}%\".rjust(14)
        + '  '
        + f\"{j['wall_s']/60:.1f}m\".rjust(8)
    )
"
} | tee -a "$LOG"

exit "$FAILED"
