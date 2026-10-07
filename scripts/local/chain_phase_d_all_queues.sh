#!/bin/bash
# Phase-D research queue: runs THREE queues SEQUENTIALLY after the
# currently-running epoch-curve sweep finishes.  All queues run
# locally (no Colab).
#
#   queue 1  over-trained ckpts hybrid eval (ep0249/0299/0349)
#            K=10, lambda=0.5, n_workers=2, parallel=3        ~3 hr
#
#   queue 2  lambda-sweep at ep0049 (the most-improvable prior)
#            lambda in {0.1, 0.25, 1.0, 2.0}, K=10
#            (lambda=0.5 already covered by the current sweep)
#            parallel=4                                       ~3 hr
#
#   queue 3  K-sweep at ep0049 (depth of MBD search budget)
#            K in {5, 20, 40}, lambda=0.5
#            (K=10 already covered by the current sweep)
#            parallel=3                                       ~3-4 hr
#
# Total wall after current sweep: ~9-10 hr.
#
# Usage:
#   screen -dmS phase_d_all_queues bash scripts/local/chain_phase_d_all_queues.sh

set -euo pipefail

# ---------- knobs ----------
WAIT_FOR="${WAIT_FOR:-phase_d_epoch_curve}"
POLL_SEC="${POLL_SEC:-60}"
N_WORKERS="${N_WORKERS:-2}"       # MBD perturbation workers (Phase-D safe path)
DEVICE="${DEVICE:-cpu}"
NICE="${NICE:-10}"

# Hybrid pipeline shared knobs (must match local run_phase_c_ksweep.sh).
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
N_TEST_ROLLOUTS=50
TEST_START_SEED=100000
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100

# ---------- paths ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE_D_ROOT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30"
LAUNCHER="$REPO_ROOT/scripts/local/run_phase_d_epoch_curve.sh"

OUT_Q1="$PHASE_D_ROOT/epoch_curve_local"          # existing dir (Q1 = more ckpts)
OUT_Q2="$PHASE_D_ROOT/lambda_sweep_local"         # new dir for Q2
OUT_Q3="$PHASE_D_ROOT/k_sweep_local"              # new dir for Q3

mkdir -p "$OUT_Q1" "$OUT_Q2" "$OUT_Q3"

MASTER_LOG="$PHASE_D_ROOT/chain_all_queues.log"

# Find a working python.
_default_python() {
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif [ -x "/Users/szh/anaconda3/envs/hybrid-traj/bin/python" ]; then
        echo "/Users/szh/anaconda3/envs/hybrid-traj/bin/python"
    fi
}
PYTHON="${PYTHON:-$(_default_python)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: no python found" >&2; exit 1
fi

# ep0049 ckpt path (used by Q2 and Q3).
CKPT_EP0049="$REPO_ROOT/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0049.pt"

# ---------- helpers ----------
log() {
    echo "[chain-all] $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$MASTER_LOG"
}

# Poll until a named screen session is gone.  Uses a variable-capture
# approach because `screen -ls | grep -q ...` interacts badly with
# `set -o pipefail`.
wait_for_screen_gone() {
    local name="$1"
    local poll="${2:-$POLL_SEC}"
    while true; do
        local out
        out=$(screen -ls 2>/dev/null || true)
        if echo "$out" | grep -q "$name"; then
            log "  '$name' still running, sleeping ${poll}s"
            sleep "$poll"
        else
            # Double-check after 5 s to avoid a transient false negative.
            sleep 5
            out=$(screen -ls 2>/dev/null || true)
            if echo "$out" | grep -q "$name"; then continue; fi
            log "  '$name' confirmed gone"
            return 0
        fi
    done
}

# Launch a single run_can.py with the given knobs.  Returns immediately
# (backgrounds the python process).  Caller is responsible for `wait`.
#
# Args:  tag  out_dir  ckpt_path  K  lambda_prior  extra_log_label
launch_one() {
    local tag="$1"
    local out_dir="$2"
    local ckpt="$3"
    local K_val="$4"
    local lam="$5"
    local label="$6"

    mkdir -p "$out_dir"

    # Idempotency:  skip if the output JSON already exists and is non-trivial.
    # run_can.py's natural filename:  unified_flow_lam-<L>_K-<K>_R-<R>.json
    local out_json="$out_dir/unified_flow_lam-${lam}_K-${K_val}_R-${MPC_REPLANS}.json"
    if [ -f "$out_json" ]; then
        local bytes
        bytes=$(stat -f%z "$out_json" 2>/dev/null || echo 0)
        if [ "$bytes" -gt 200 ]; then
            log "  [skip] $label/$tag already has $out_json ($bytes B)"
            return 0
        fi
    fi

    local run_log="$out_dir/run.log"
    log "  [fire] $label/$tag  K=$K_val lambda=$lam -> $out_dir"

    (
        nice -n "$NICE" "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
            --task can \
            --prior flow \
            --H "$H" \
            --mpc-replans "$MPC_REPLANS" \
            --mpc-chunk-steps "$MPC_CHUNK_STEPS" \
            --n-demos "$N_DEMOS" \
            --n-rollouts "$N_ROLLOUTS" \
            --n-test-rollouts "$N_TEST_ROLLOUTS" \
            --test-start-seed "$TEST_START_SEED" \
            --max-env-steps "$MAX_ENV_STEPS" \
            --prior-n-steps "$PRIOR_N_STEPS" \
            --sigma-max "$SIGMA_MAX" \
            --sigma-min "$SIGMA_MIN" \
            --lambda-prior "$lam" \
            --device "$DEVICE" \
            --seed 0 \
            --K "$K_val" \
            --n-workers "$N_WORKERS" \
            --prior-checkpoint "$ckpt" \
            --output-dir "$out_dir" \
            2>&1 | sed "s|^|[$label/$tag] |"
    ) > "$run_log" 2>&1 &

    log "    bg PID=$!  log=$run_log"
}

# ---------- master flow ----------

{
    echo "=========================================================="
    echo "[chain-all] master research queue starting"
    echo "  date    : $(date)"
    echo "  python  : $PYTHON"
    echo "  device  : $DEVICE"
    echo "  n_work  : $N_WORKERS"
    echo "  out_q1  : $OUT_Q1"
    echo "  out_q2  : $OUT_Q2"
    echo "  out_q3  : $OUT_Q3"
    echo "=========================================================="
} | tee -a "$MASTER_LOG"

# Step 0: wait for the currently-running epoch-curve sweep.
log "[step 0] waiting for screen '$WAIT_FOR' to finish ..."
wait_for_screen_gone "$WAIT_FOR" "$POLL_SEC"

# ========================================================================
# QUEUE 1:  hybrid eval on the over-trained ckpts (ep0249/0299/0349)
# ========================================================================
log ""
log "=========================================================="
log "[queue 1] over-trained ckpts hybrid eval @ K=10  lambda=0.5"
log "=========================================================="
log "[queue 1] starting at $(date)"

# Reuse the existing per-ckpt-parallel launcher with different CKPT_TAGS.
PYTHON="$PYTHON" \
    CKPT_TAGS="ep0249 ep0299 ep0349" \
    OUTROOT="$OUT_Q1" \
    bash "$LAUNCHER" 2>&1 | tee -a "$MASTER_LOG"
q1_rc=${PIPESTATUS[0]}

log "[queue 1] finished at $(date), rc=$q1_rc"

# ========================================================================
# Q2 and Q3 SKIPPED LOCALLY:  user moved these to Colab on 2026-05-30
# (after seeing Q0's 11.4 hr wall, local would have taken 30+ more hours).
#
# Q2 (lambda-sweep at ep0049 with lambda in {0.1, 0.25, 1.0, 2.0}, K=10)
# Q3 (K-sweep at ep0049 with K in {5, 20, 40}, lambda=0.5)
# both now live in the Colab notebook:
#   experiments/08_flow_matching_prior/colab_hybrid_lambda_k_probe.ipynb
#
# This `exit 0` makes the master chain terminate cleanly right after Q1.
# ========================================================================
log ""
log "[chain-all] Q2 + Q3 are now on Colab -- exiting locally after Q1."
log "[chain-all] master chain DONE at $(date)"
exit 0

# ========================================================================
# Done.
# ========================================================================
log ""
log "=========================================================="
log "[chain-all] ALL QUEUES COMPLETE at $(date)"
log "  q1 rc:    $q1_rc"
log "=========================================================="
log "  results:"
log "    Q1 (over-trained ckpts):  $OUT_Q1/ep0249,ep0299,ep0349"
log "    Q2 (lambda sweep):        $OUT_Q2"
log "    Q3 (K sweep):             $OUT_Q3"
log "  Aggregate JSONs to inspect:"
find "$OUT_Q1" "$OUT_Q2" "$OUT_Q3" -name 'unified_flow_lam-*_K-*_R-*.json' -type f | sort | tee -a "$MASTER_LOG"
log "=========================================================="

exit 0
