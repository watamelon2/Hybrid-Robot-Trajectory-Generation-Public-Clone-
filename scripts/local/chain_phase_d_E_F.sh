#!/bin/bash
# Phase-D E + F auto-chain.
#
# Waits for Q2 + Q3 (screens `phase_d_q2q3_local` and `phase_d_q3_K5`) to
# finish, then launches:
#
#   PHASE E  (3 cells parallel, ~3 h wall)  K-sweep across epoch curve:
#     E_K40_ep0049 : K=40 lam=0.5 n_test=15 N_WORKERS=2
#     E_K20_ep0149 : K=20 lam=0.5 n_test=20 N_WORKERS=2
#     E_K20_ep0199 : K=20 lam=0.5 n_test=20 N_WORKERS=2
#
#   PHASE F  (4 cells parallel, ~2 h wall)  extreme-lambda probe at mid+strong priors:
#     F_lam0p1_ep0149 : K=10 lam=0.1 n_test=20 N_WORKERS=2
#     F_lam2p0_ep0149 : K=10 lam=2.0 n_test=20 N_WORKERS=2
#     F_lam0p1_ep0199 : K=10 lam=0.1 n_test=20 N_WORKERS=2
#     F_lam2p0_ep0199 : K=10 lam=2.0 n_test=20 N_WORKERS=2
#
#   Then runs scripts/local/aggregate_q2q3.sh to write the full grid table.
#
# Seed offset 300000 to keep cells disjoint from local Q2/Q3 (200000) and GCP (100000).
#
# Total wall after Q2 + Q3 lands: ~5 h.  Target: done by ~12:00 PDT, well
# inside the Sunday afternoon deadline.
#
# Usage:
#   screen -dmS phase_d_E_F bash scripts/local/chain_phase_d_E_F.sh
#
# Monitor:
#   tail -f experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_local/chain_E_F.log

set -euo pipefail

# ---------- knobs ----------
NICE="${NICE:-10}"
DEVICE="${DEVICE:-cpu}"

# Hybrid pipeline shared knobs (must match Q2 + Q3 local + GCP V2).
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100
TEST_START_SEED=300000     # disjoint from local Q2/Q3 (200k) and GCP (100k)

# Threading hygiene -- match the GCP screen.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

POLL_SEC=60      # how often to check whether Q2 + Q3 screens are gone

# ---------- paths ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_D_ROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30"
OUT_ROOT="$PHASE_D_ROOT/q2q3_local"
mkdir -p "$OUT_ROOT"
CHAIN_LOG="$OUT_ROOT/chain_E_F.log"

CKPT_DIR="$REPO_ROOT/experiments/08_flow_matching_prior/results"

# bash 3.2 (macOS default) does not have associative arrays.  Use a function.
ckpt_for() {
    case "$1" in
        ep0049) echo "$CKPT_DIR/v_theta_dp_tobs2_h16_v3b_ep0049.pt" ;;
        ep0149) echo "$CKPT_DIR/v_theta_dp_tobs2_h16_v3b_ep0149.pt" ;;
        ep0199) echo "$CKPT_DIR/v_theta_dp_tobs2_h16_v3b_ep0199.pt" ;;
        *) echo "" ;;
    esac
}

# ---------- python ----------
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not executable at $PYTHON" >&2
    exit 1
fi

log() {
    echo "[chain-E-F] $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$CHAIN_LOG"
}

# ---------- helpers ----------
wait_for_screen_gone() {
    local name="$1"
    while true; do
        local out
        out=$(screen -ls 2>/dev/null || true)
        if echo "$out" | grep -q "[0-9]*\.${name}"; then
            log "  '$name' still running, sleeping ${POLL_SEC}s"
            sleep "$POLL_SEC"
        else
            sleep 5
            out=$(screen -ls 2>/dev/null || true)
            if echo "$out" | grep -q "[0-9]*\.${name}"; then continue; fi
            log "  '$name' confirmed gone"
            return 0
        fi
    done
}

# Run a single run_can.py invocation in foreground (caller wraps for parallel).
#
# Args:  tag  ckpt_key  K  lam  n_test  n_workers
run_one() {
    local tag="$1"
    local ckpt_key="$2"
    local K_val="$3"
    local lam="$4"
    local n_test="$5"
    local n_workers="$6"

    local ckpt
    ckpt=$(ckpt_for "$ckpt_key")
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        log "  [skip] $tag  ckpt missing for key '$ckpt_key': $ckpt"
        return 1
    fi

    local out_dir="$OUT_ROOT/$tag"
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
    log "  [fire] $tag  ckpt=$ckpt_key K=$K_val lam=$lam n_test=$n_test n_workers=$n_workers"

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
        --prior-checkpoint "$ckpt" \
        --output-dir "$out_dir" \
        > "$run_log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "  [done] $tag -> $out_json"
    else
        log "  [FAIL] $tag  rc=$rc  see $run_log"
    fi
    return $rc
}

# ---------- master ----------
{
    echo "=========================================================="
    echo "[chain-E-F] Phase-D E + F auto-chain starting"
    echo "  date    : $(date)"
    echo "  python  : $PYTHON ($("$PYTHON" --version 2>&1))"
    echo "  seed    : $TEST_START_SEED  (Q2/Q3 local=200000, GCP=100000)"
    echo "  out     : $OUT_ROOT"
    echo "=========================================================="
} | tee -a "$CHAIN_LOG"

# ---------- step 0: wait for Q2 + Q3 to finish ----------
log "[step 0] waiting for screen 'phase_d_q3_K5' to finish ..."
wait_for_screen_gone "phase_d_q3_K5"
log "[step 0a] q3_K5 done"

log "[step 0] waiting for screen 'phase_d_q2q3_local' (Wave 1 + Wave 2) to finish ..."
wait_for_screen_gone "phase_d_q2q3_local"
log "[step 0b] q2q3_local done"

log "[step 0] both Q2 + Q3 screens gone -- starting Phase E"
sleep 30  # let the box settle (memory pressure release, file flushes)

# ---------- PHASE E: K-sweep across epoch curve, 3-way parallel ----------
log ""
log "=========================================================="
log "[Phase E] K-sweep across epoch curve (3 cells parallel)"
log "  E_K40_ep0049 : K=40 lam=0.5 n_test=15"
log "  E_K20_ep0149 : K=20 lam=0.5 n_test=20"
log "  E_K20_ep0199 : K=20 lam=0.5 n_test=20"
log "[Phase E] starting at $(date)"
log "=========================================================="

E_T0=$(date +%s)
E_PIDS=()

run_one "E_K40_ep0049" "ep0049" 40 0.5 15 2 &
E_PIDS+=("$!"); log "  bg PID=$! cell=E_K40_ep0049"
sleep 5

run_one "E_K20_ep0149" "ep0149" 20 0.5 20 2 &
E_PIDS+=("$!"); log "  bg PID=$! cell=E_K20_ep0149"
sleep 5

run_one "E_K20_ep0199" "ep0199" 20 0.5 20 2 &
E_PIDS+=("$!"); log "  bg PID=$! cell=E_K20_ep0199"

log "[Phase E] all 3 cells launched: PIDs ${E_PIDS[*]}"
log "[Phase E] waiting for completion ..."
E_FAILS=0
for pid in "${E_PIDS[@]}"; do
    if ! wait "$pid"; then E_FAILS=$((E_FAILS + 1)); fi
done
E_T1=$(date +%s)
E_DT=$(( (E_T1 - E_T0) / 60 ))
log "[Phase E] DONE at $(date) (wall=${E_DT} min, failed_jobs=$E_FAILS)"

# Run a mid-chain aggregate so the user has the E table available even
# if Phase F gets killed.
log "[mid-aggregate] running aggregate_q2q3.sh after Phase E"
bash "$REPO_ROOT/scripts/local/aggregate_q2q3.sh" >> "$CHAIN_LOG" 2>&1 || \
    log "[mid-aggregate] failed; continuing to Phase F anyway"

# ---------- PHASE F: extreme-lambda probe at mid + strong priors, 4-way parallel ----------
log ""
log "=========================================================="
log "[Phase F] extreme-lambda probe (4 cells parallel)"
log "  F_lam0p1_ep0149 : K=10 lam=0.1 n_test=20"
log "  F_lam2p0_ep0149 : K=10 lam=2.0 n_test=20"
log "  F_lam0p1_ep0199 : K=10 lam=0.1 n_test=20"
log "  F_lam2p0_ep0199 : K=10 lam=2.0 n_test=20"
log "[Phase F] starting at $(date)"
log "=========================================================="

F_T0=$(date +%s)
F_PIDS=()
for cfg in "F_lam0p1_ep0149 ep0149 0.1" \
           "F_lam2p0_ep0149 ep0149 2.0" \
           "F_lam0p1_ep0199 ep0199 0.1" \
           "F_lam2p0_ep0199 ep0199 2.0"; do
    set -- $cfg
    tag="$1"; ckpt="$2"; lam="$3"
    run_one "$tag" "$ckpt" 10 "$lam" 20 2 &
    F_PIDS+=("$!")
    log "  bg PID=$! cell=$tag"
    sleep 5
done

log "[Phase F] all 4 cells launched: PIDs ${F_PIDS[*]}"
log "[Phase F] waiting for completion ..."
F_FAILS=0
for pid in "${F_PIDS[@]}"; do
    if ! wait "$pid"; then F_FAILS=$((F_FAILS + 1)); fi
done
F_T1=$(date +%s)
F_DT=$(( (F_T1 - F_T0) / 60 ))
log "[Phase F] DONE at $(date) (wall=${F_DT} min, failed_jobs=$F_FAILS)"

# ---------- final aggregate ----------
log ""
log "[final aggregate] running aggregate_q2q3.sh after Phase F"
bash "$REPO_ROOT/scripts/local/aggregate_q2q3.sh" >> "$CHAIN_LOG" 2>&1 || \
    log "[final aggregate] failed; inspect manually"

log ""
log "=========================================================="
log "[chain-E-F] MASTER COMPLETE at $(date)"
log "=========================================================="
log "  outputs:"
find "$OUT_ROOT" -name 'unified_summary_*.json' -type f | sort | tee -a "$CHAIN_LOG"
log ""
log "  inspect:"
log "    cat $PHASE_D_ROOT/q2q3_AGGREGATE.md"
log ""

exit 0
