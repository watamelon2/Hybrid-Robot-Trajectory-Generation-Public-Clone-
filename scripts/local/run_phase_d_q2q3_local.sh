#!/bin/bash
# Phase-D Q2 + Q3 local insurance pass against the GCP V2 long-pole.
#
# Context (2026-05-31 ~02:30 PDT):
#   GCP V2 launched at 08:43 UTC (~01:43 PDT) by the parallel agent.
#   Per-cell ETAs at GCP's 6-way share of c3-standard-22:
#     q3_K5    ~ +7 hr   (09:15 PDT)  -- fits the 10 hr deadline
#     q2_lam*  ~ +12 hr  (14:15 PDT)  -- SLIPS 2 hr past deadline
#     q3_K20   ~ +17 hr  (19:15 PDT)  -- SLIPS 7 hr past deadline
#
# This script runs the two slipping cells locally on full Mac CPU so the
# user has the full Q2 + Q3 story in hand within the 10-hour window
# regardless of GCP slippage:
#
#   WAVE 1 (alone, full box, ~2-3 hr):
#     q3_K20_local : K=20 lam=0.5 n_test=20 N_WORKERS=8
#
#   WAVE 2 (4-way parallel after Wave 1, ~5-6 hr):
#     q2_lam0p1_local  : K=10 lam=0.1  n_test=30 N_WORKERS=2
#     q2_lam0p25_local : K=10 lam=0.25 n_test=30 N_WORKERS=2
#     q2_lam1p0_local  : K=10 lam=1.0  n_test=30 N_WORKERS=2
#     q2_lam2p0_local  : K=10 lam=2.0  n_test=30 N_WORKERS=2
#
# Seed offset:  --test-start-seed 200000  (GCP uses 100000) so the local
# samples POOL with GCP rather than being redundant verification.
#
# Total wall (this Mac, M-series 10-core, after Q1's 21.3 h baseline):
#   Wave 1     : ~ 2.5 h
#   Wave 2     : ~ 5.5 h
#   GRAND TOTAL: ~ 8 h    (deadline = 10 h)
#
# Usage:
#   screen -dmS phase_d_q2q3_local bash scripts/local/run_phase_d_q2q3_local.sh
#
# Monitor:
#   tail -f experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_local/master.log

set -euo pipefail

# ---------- knobs ----------
NICE="${NICE:-10}"
DEVICE="${DEVICE:-cpu}"

# Hybrid pipeline shared knobs (must match GCP V2 + chain_phase_d_all_queues.sh).
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100
TEST_START_SEED=200000     # GCP uses 100000; offset so we don't collide.

# Wave 1: K=20 lam=0.5  n_test=20  (matches GCP V2 q3_K20)
W1_K=20
W1_LAM=0.5
W1_N_TEST=20
W1_N_WORKERS=8             # full box, single-cell

# Wave 2: K=10  4-way parallel  n_test=30 each  (matches GCP V2 q2_lam*)
W2_K=10
W2_N_TEST=30
W2_N_WORKERS=2             # 4 cells x 2 workers = 8 procs
W2_LAMBDAS=(0.1 0.25 1.0 2.0)

# Threading hygiene -- match the GCP screen.  Critical when forking
# 8 workers into a single robosuite env each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---------- paths ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_D_ROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30"
OUT_ROOT="$PHASE_D_ROOT/q2q3_local"
mkdir -p "$OUT_ROOT"
MASTER_LOG="$OUT_ROOT/master.log"

CKPT_EP0049="$REPO_ROOT/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0049.pt"
if [ ! -f "$CKPT_EP0049" ]; then
    echo "ERROR: ep0049 ckpt missing at $CKPT_EP0049" >&2
    exit 1
fi

# ---------- python ----------
_default_python() {
    if command -v python >/dev/null 2>&1; then command -v python; return; fi
    if command -v /Users/szh/anaconda3/envs/hybrid-traj/bin/python >/dev/null 2>&1; then
        echo /Users/szh/anaconda3/envs/hybrid-traj/bin/python; return
    fi
    return 1
}
PYTHON="${PYTHON:-$(_default_python)}"
if [ -z "${PYTHON:-}" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: no python found at $PYTHON" >&2; exit 1
fi

log() {
    echo "[q2q3-local] $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$MASTER_LOG"
}

# ---------- helpers ----------
# Run a single run_can.py invocation in foreground.  Caller wraps with `&` + `wait` for parallel.
run_one() {
    local tag="$1"
    local out_dir="$2"
    local K_val="$3"
    local lam="$4"
    local n_test="$5"
    local n_workers="$6"

    mkdir -p "$out_dir"
    local out_json="$out_dir/unified_flow_lam-${lam}_K-${K_val}_R-${MPC_REPLANS}.json"
    if [ -f "$out_json" ]; then
        local bytes
        bytes=$(stat -f%z "$out_json" 2>/dev/null || echo 0)
        if [ "$bytes" -gt 200 ]; then
            log "  [skip] $tag already has $out_json ($bytes B)"
            return 0
        fi
    fi

    local run_log="$out_dir/run.log"
    log "  [fire] $tag  K=$K_val lam=$lam n_test=$n_test n_workers=$n_workers -> $out_dir"

    cd "$REPO_ROOT"
    nice -n "$NICE" "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
        --task can \
        --prior flow \
        --H "$H" \
        --mpc-replans "$MPC_REPLANS" \
        --mpc-chunk-steps "$MPC_CHUNK_STEPS" \
        --n-demos "$N_DEMOS" \
        --n-rollouts "$N_ROLLOUTS" \
        --n-test-rollouts "$n_test" \
        --test-start-seed "$TEST_START_SEED" \
        --max-env-steps "$MAX_ENV_STEPS" \
        --prior-n-steps "$PRIOR_N_STEPS" \
        --sigma-max "$SIGMA_MAX" \
        --sigma-min "$SIGMA_MIN" \
        --lambda-prior "$lam" \
        --device "$DEVICE" \
        --seed 0 \
        --K "$K_val" \
        --n-workers "$n_workers" \
        --prior-checkpoint "$CKPT_EP0049" \
        --output-dir "$out_dir" \
        > "$run_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "  [done] $tag  -> $out_json"
    else
        log "  [FAIL] $tag  rc=$rc  see $run_log"
    fi
    return $rc
}

# ---------- master ----------
{
    echo "=========================================================="
    echo "[q2q3-local] master Q2+Q3 LOCAL insurance pass starting"
    echo "  date    : $(date)"
    echo "  python  : $PYTHON ($("$PYTHON" --version 2>&1))"
    echo "  ckpt    : $CKPT_EP0049"
    echo "  seed    : $TEST_START_SEED  (GCP=100000)"
    echo "  out     : $OUT_ROOT"
    echo "=========================================================="
} | tee -a "$MASTER_LOG"

# Smoke: confirm python can import torch + robosuite before launching.
log "[smoke] verify python env"
if ! "$PYTHON" -c "import torch, robosuite, robomimic; print('torch', torch.__version__, 'robosuite', robosuite.__version__)" >> "$MASTER_LOG" 2>&1; then
    log "[smoke FAIL] python env broken; bailing"
    exit 1
fi
log "[smoke] env OK"

# =========================================================================
# WAVE 1: Q3_K20 alone, full Mac CPU
# =========================================================================
log ""
log "=========================================================="
log "[wave 1] Q3_K20 alone -- K=$W1_K lam=$W1_LAM n=$W1_N_TEST  workers=$W1_N_WORKERS"
log "[wave 1] starting at $(date)"
log "=========================================================="

W1_OUT="$OUT_ROOT/q3_K20"
W1_T0=$(date +%s)
if ! run_one "q3_K20" "$W1_OUT" "$W1_K" "$W1_LAM" "$W1_N_TEST" "$W1_N_WORKERS"; then
    log "[wave 1 FAIL] continuing to Wave 2 anyway"
fi
W1_T1=$(date +%s)
W1_DT=$(( (W1_T1 - W1_T0) / 60 ))
log "[wave 1] DONE at $(date) (wall=${W1_DT} min)"

# =========================================================================
# WAVE 2: Q2 4-way parallel
# =========================================================================
log ""
log "=========================================================="
log "[wave 2] Q2 lambda-sweep 4-way parallel -- K=$W2_K n=$W2_N_TEST  workers=$W2_N_WORKERS each"
log "[wave 2] starting at $(date)"
log "=========================================================="

W2_T0=$(date +%s)
W2_PIDS=()
for lam in "${W2_LAMBDAS[@]}"; do
    # Slugify lambda for the dir name:  0.1 -> 0p1, 0.25 -> 0p25, 1.0 -> 1p0, 2.0 -> 2p0
    lam_slug=$(echo "$lam" | tr '.' 'p')
    w2_out="$OUT_ROOT/q2_lam${lam_slug}"
    run_one "q2_lam${lam_slug}" "$w2_out" "$W2_K" "$lam" "$W2_N_TEST" "$W2_N_WORKERS" &
    W2_PIDS+=("$!")
    log "  bg PID=$!  cell=q2_lam${lam_slug}"
    sleep 5    # stagger inits to avoid simultaneous robosuite C-ext load
done

log "[wave 2] all 4 cells launched: PIDs ${W2_PIDS[*]}"
log "[wave 2] waiting for completion ..."
W2_FAILS=0
for pid in "${W2_PIDS[@]}"; do
    if ! wait "$pid"; then
        W2_FAILS=$((W2_FAILS + 1))
    fi
done
W2_T1=$(date +%s)
W2_DT=$(( (W2_T1 - W2_T0) / 60 ))
log "[wave 2] DONE at $(date) (wall=${W2_DT} min, failed_jobs=$W2_FAILS)"

# =========================================================================
# Final summary
# =========================================================================
log ""
log "=========================================================="
log "[q2q3-local] MASTER COMPLETE at $(date)"
log "=========================================================="
log "  outputs:"
find "$OUT_ROOT" -name 'unified_summary_*.json' -type f | sort | tee -a "$MASTER_LOG"
log ""
log "  inspect with:"
log "    bash $REPO_ROOT/scripts/local/aggregate_q2q3.sh   (if landed)"
log ""

exit 0
