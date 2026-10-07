#!/bin/bash
# Deferred relaunch of E_K40_ep0049 after the main Phase-D E+F chain finishes.
#
# Context:
#   The original E_K40_ep0049 cell (launched by chain_phase_d_E_F.sh at
#   11:46 PDT on 2026-05-31) was killed at 18:06 PDT after 12/25 rollouts
#   so Phase F could launch without waiting on the slow K=40 cell.
#   This script re-launches K40 standalone *after* the main chain finishes,
#   making it "truly last executed".  See chat 2026-05-31 17:57 PDT.
#
# What it does:
#   1. Waits for screen `phase_d_E_F` to disappear (= mid-/final-aggregate done,
#      Phase F lambdas at ep0149 + ep0199 landed).
#   2. Re-launches K40 with the same config as the chain script used
#      (ckpt=ep0049, K=40, lam=0.5, n_test=15, n_workers=2, seed=300000+).
#   3. After K40 finishes, runs aggregate_q2q3.sh to re-write the master
#      aggregate so the K-sweep table includes the K=40 row.
#
# Usage:
#   screen -dmS phase_d_K40_deferred bash scripts/local/relaunch_E_K40_after_chain.sh
#
# Monitor:
#   tail -f experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_local/k40_deferred.log

set -euo pipefail

# Match the chain script's knobs exactly.
NICE="${NICE:-10}"
DEVICE="${DEVICE:-cpu}"
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
N_TEST=15
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100
TEST_START_SEED=300000
K_VAL=40
LAM=0.5
POLL_SEC=60

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_D_ROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30"
OUT_ROOT="$PHASE_D_ROOT/q2q3_local"
CELL_DIR="$OUT_ROOT/E_K40_ep0049"
mkdir -p "$CELL_DIR"

DEFERRED_LOG="$OUT_ROOT/k40_deferred.log"
CELL_LOG="$CELL_DIR/run.log"
CKPT="$REPO_ROOT/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0049.pt"

PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"

log() {
    echo "[k40-deferred] $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$DEFERRED_LOG"
}

# ---------- step 0: wait for phase_d_E_F screen to be gone ----------
log ""
log "=========================================================="
log "Deferred K40 relaunch starting"
log "  out_dir : $CELL_DIR"
log "  ckpt    : $CKPT"
log "  K=$K_VAL  lam=$LAM  n_test=$N_TEST  seed_offset=$TEST_START_SEED"
log "=========================================================="
log "[step 0] waiting for screen 'phase_d_E_F' to be gone (= main chain done)"

while true; do
    out=$(screen -ls 2>/dev/null || true)
    if echo "$out" | grep -q "[0-9]*\.phase_d_E_F"; then
        log "  'phase_d_E_F' still running, sleeping ${POLL_SEC}s"
        sleep "$POLL_SEC"
    else
        sleep 5
        out=$(screen -ls 2>/dev/null || true)
        if echo "$out" | grep -q "[0-9]*\.phase_d_E_F"; then continue; fi
        log "  'phase_d_E_F' confirmed gone -- main chain has finished"
        break
    fi
done

# ---------- step 1: launch K40 standalone ----------
log ""
log "[step 1] launching standalone K40 at $(date)"
log "  this run will overwrite $CELL_LOG (the killed partial was archived as"
log "  run.log.killed_at_*)."

if [ ! -f "$CKPT" ]; then
    log "[FATAL] checkpoint missing: $CKPT"
    exit 1
fi

K40_T0=$(date +%s)

cd "$REPO_ROOT"
nice -n "$NICE" "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
    --task can \
    --prior flow \
    --H "$H" \
    --mpc-replans "$MPC_REPLANS" \
    --mpc-chunk-steps "$MPC_CHUNK_STEPS" \
    --n-demos "$N_DEMOS" \
    --n-rollouts "$N_ROLLOUTS" \
    --n-test-rollouts "$N_TEST" \
    --test-start-seed "$TEST_START_SEED" \
    --max-env-steps "$MAX_ENV_STEPS" \
    --prior-n-steps "$PRIOR_N_STEPS" \
    --sigma-max "$SIGMA_MAX" \
    --sigma-min "$SIGMA_MIN" \
    --lambda-prior "$LAM" \
    --device "$DEVICE" \
    --seed 0 \
    --K "$K_VAL" \
    --n-workers 2 \
    --prior-checkpoint "$CKPT" \
    --output-dir "$CELL_DIR" \
    > "$CELL_LOG" 2>&1
RC=$?

K40_T1=$(date +%s)
K40_DT=$(( (K40_T1 - K40_T0) / 60 ))

if [ $RC -eq 0 ]; then
    log "[step 1] K40 DONE  rc=0  wall=${K40_DT} min"
else
    log "[step 1] K40 FAIL  rc=$RC  wall=${K40_DT} min  see $CELL_LOG"
    log "[step 1] continuing to aggregator anyway (it will skip K40 if no JSON)"
fi

# ---------- step 2: re-run aggregator ----------
log ""
log "[step 2] re-running aggregate_q2q3.sh so K40 row lands in master aggregate"

if bash "$REPO_ROOT/scripts/local/aggregate_q2q3.sh" >> "$DEFERRED_LOG" 2>&1; then
    log "[step 2] aggregator done"
else
    log "[step 2] aggregator failed; inspect manually"
fi

log ""
log "=========================================================="
log "Deferred K40 relaunch finished at $(date)"
log "  K40 wall: ${K40_DT} min  rc=$RC"
log "=========================================================="
