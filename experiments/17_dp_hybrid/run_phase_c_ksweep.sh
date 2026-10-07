#!/bin/bash
# Phase-C K-sweep launcher (corrected after the 2026-05-29 mpc-replans bug).
#
# Runs the hybrid pipeline at K = 0 / 10 / 20 against the Phase-B
# `v_theta_dp_tobs2_h16_v3b_ep0199.pt` prior using the DP-aligned eval
# protocol (10 train + 50 test rollouts, max_env_steps=400,
# n_action_steps=8, sigma_max=0.25, n_steps=100).
#
# Two-stage gating (pre-registered in `results/phase_c_2026_05_29/BRIEF.md`):
#
#   stage 1: K = 0 sanity gate
#     PASS  if test_succ_rate ∈ [0.75, 0.85]   (Phase-B 80 % ± 5 pp)
#     FAIL  otherwise -> halts; no K > 0 runs
#
#   stage 2: K = 10 and K = 20 (only if stage 1 PASSED)
#
# Wall budget per K (CPU-only Mac, see BRIEF "Wall budget" table):
#   K = 0   ~60 – 90 min
#   K = 10  ~90 – 140 min
#   K = 20  ~150 – 240 min
# Total worst case: ~7 hr.  Sized to fit a 6 hr unattended window if
# the per-rollout times come in at the lower end of the estimate.
#
# Usage (from the HRTG-dp repo root):
#
#   bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh
#
# Or with a different python binary / device:
#
#   PYTHON=/path/to/env/python \
#   DEVICE=cuda \
#     bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh
#
# Or to skip specific K values (e.g. to resume after K=0 already ran):
#
#   K_VALUES="10 20" \
#     bash experiments/17_dp_hybrid/run_phase_c_ksweep.sh
#
# Idempotency: any K with a non-empty K{K}/unified_flow_lam-*.json is
# skipped (so re-running the script picks up where a prior run died).
set -euo pipefail

# --- knobs the next agent might want to change -------------------------
PYTHON="${PYTHON:-$(command -v python)}"
DEVICE="${DEVICE:-cpu}"            # set to "cuda" if running on a GPU box
K_VALUES="${K_VALUES:-0 10 20}"    # which K to run, space-separated
LAMBDA_PRIOR="${LAMBDA_PRIOR:-0.5}"
# -----------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="$REPO_ROOT/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt"
OUTROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_c_2026_05_29"
LOG="$OUTROOT/ksweep.log"

if [ ! -f "$CKPT" ] && [ ! -L "$CKPT" ]; then
    echo "ERROR: ckpt not at $CKPT"
    echo "Hint: sync from Drive via"
    echo "  $PYTHON $REPO_ROOT/experiments/08_flow_matching_prior/sync_ckpts_from_drive.py \\"
    echo "      --folder-url 'https://drive.google.com/drive/folders/<ID>?usp=sharing' \\"
    echo "      --pattern v3b_ep0199"
    exit 1
fi

# Resolve symlink -> real file size sanity check.
REAL_CKPT="$(readlink -f "$CKPT" 2>/dev/null || python -c "import os; print(os.path.realpath('$CKPT'))")"
if [ ! -f "$REAL_CKPT" ]; then
    echo "ERROR: ckpt symlink at $CKPT does not resolve to a real file (got: $REAL_CKPT)"
    exit 1
fi
CKPT_SIZE=$(stat -f%z "$REAL_CKPT" 2>/dev/null || stat -c%s "$REAL_CKPT")
if [ "$CKPT_SIZE" -lt 100000000 ]; then
    echo "ERROR: ckpt at $REAL_CKPT is only $CKPT_SIZE bytes; expected > 100 MB"
    exit 1
fi

# Shared flags.  See BRIEF "Flag rationale" section for why each is set.
COMMON_FLAGS=(
    --task can --prior flow
    --prior-checkpoint "$CKPT"
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
)

mkdir -p "$OUTROOT"
{
    echo "=========================================================="
    echo "phase-c ksweep launch:  $(date)"
    echo "python:                 $PYTHON"
    echo "device:                 $DEVICE"
    echo "K values:               $K_VALUES"
    echo "lambda-prior:           $LAMBDA_PRIOR"
    echo "ckpt:                   $CKPT"
    echo "ckpt real path:         $REAL_CKPT  ($(echo "$CKPT_SIZE" | awk '{printf "%.1f MB", $1/1e6}'))"
    echo "outroot:                $OUTROOT"
    echo "=========================================================="
} | tee -a "$LOG"

# --- per-K runner ------------------------------------------------------
run_k() {
    local K=$1
    local out="$OUTROOT/K${K}"
    mkdir -p "$out"

    # Idempotency: skip K if a result JSON already exists.
    local existing
    existing=$(ls "$out"/unified_flow_lam-*.json 2>/dev/null | head -1 || true)
    if [ -n "$existing" ]; then
        echo "[ksweep] K=$K already has $existing, skipping." | tee -a "$LOG"
        return 0
    fi

    {
        echo ""
        echo "=========================================================="
        echo "K=$K starting:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================================="
    } | tee -a "$LOG"

    "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
        "${COMMON_FLAGS[@]}" \
        --K "$K" \
        --output-dir "$out" 2>&1 | tee "$out/run.log" | tee -a "$LOG" | \
        grep --line-buffered -E '^\[exp15\]' || true

    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[ksweep] K=$K FAILED with rc=$rc" | tee -a "$LOG"
        return "$rc"
    fi

    # Print one-line summary of this K's JSON.
    local jsonf
    jsonf=$(ls "$out"/unified_flow_lam-*.json 2>/dev/null | head -1)
    if [ -z "$jsonf" ]; then
        echo "[ksweep] K=$K: no JSON produced (unexpected)" | tee -a "$LOG"
        return 1
    fi
    "$PYTHON" -c "
import json
d = json.load(open('$jsonf'))
print(f\"  train: {d['train_n_succ']:>2}/{d['train_n']:<2} = {d['train_succ_rate']*100:>4.1f}%\")
print(f\"  test : {d['test_n_succ']:>2}/{d['test_n']:<2} = {d['test_succ_rate']*100:>4.1f}%  <-- DP apples-to-apples\")
print(f\"  total: {d['n_success']:>2}/{d['n_total']:<2} = {d['success_rate']*100:>4.1f}%\")
print(f\"  wall : {d['wall_s']/60:.1f} min\")
" | tee -a "$LOG"
}

# --- gate check ------------------------------------------------------
gate_passes() {
    local jsonf
    jsonf=$(ls "$OUTROOT/K0"/unified_flow_lam-*.json 2>/dev/null | head -1)
    if [ -z "$jsonf" ]; then
        return 1
    fi
    "$PYTHON" -c "
import json, sys
d = json.load(open('$jsonf'))
ts = d['test_succ_rate']
print(f'K=0 test rate: {ts*100:.1f}%  (band: 75-85%)')
sys.exit(0 if 0.75 <= ts <= 0.85 else 1)
"
}

# --- main flow ------------------------------------------------------
for K in $K_VALUES; do
    if [ "$K" != "0" ]; then
        # K > 0 only runs if the gate passed.
        if ls "$OUTROOT/K0"/unified_flow_lam-*.json >/dev/null 2>&1; then
            if ! gate_passes >>"$LOG" 2>&1; then
                echo "[ksweep] K=0 gate FAILED; not running K=$K" | tee -a "$LOG"
                continue
            fi
        else
            echo "[ksweep] K=0 has not run yet; not running K=$K" | tee -a "$LOG"
            continue
        fi
    fi
    run_k "$K"
done

# --- final summary -------------------------------------------------
{
    echo ""
    echo "=========================================================="
    echo "ksweep done: $(date)"
    echo "=========================================================="
    echo ""
    "$PYTHON" -c "
import json
from pathlib import Path
out = Path('$OUTROOT')
print(f\"{'K':>3}  {'train':>14}  {'test':>14}  {'total':>14}  {'wall':>8}\")
print('-' * 64)
for k in [0, 10, 20]:
    jsons = sorted((out / f'K{k}').glob('unified_flow_lam-*.json'))
    if not jsons: continue
    d = json.load(open(jsons[0]))
    print(f'{k:>3}  '
          f\"{d['train_n_succ']}/{d['train_n']}={d['train_succ_rate']*100:.1f}%\"
          .rjust(14)
          + '  '
          + f\"{d['test_n_succ']}/{d['test_n']}={d['test_succ_rate']*100:.1f}%\"
          .rjust(14)
          + '  '
          + f\"{d['n_success']}/{d['n_total']}={d['success_rate']*100:.1f}%\"
          .rjust(14)
          + '  '
          + f\"{d['wall_s']/60:.1f}m\".rjust(8))
"
} | tee -a "$LOG"
