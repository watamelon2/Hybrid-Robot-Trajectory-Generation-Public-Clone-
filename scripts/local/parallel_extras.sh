#!/usr/bin/env bash
# parallel_extras.sh -- runs concurrently with overnight_queue.sh
#
# Three jobs, executed SEQUENTIALLY inside this script so we only use
# one n-workers=6 pool at a time. Combined with the main queue's
# n-workers=12 pool, total in-flight workers = 18 on a 16-core
# machine (~12% oversubscription -- Apple Silicon handles this fine).
#
# Job A: sigma-sweep at lambda=5 on the UNIFIED driver
#        Answers the open question after the lambda=0.5 sigma-sweep:
#        is the score-bridge at lambda=5 robust to large sigma the
#        way the exp 05 hybrid at lambda=0.5 is NOT?
#        Cells:  5  (sigma in {0.01, 0.03, 0.05, 0.10, 0.25})
#        Config: unified driver, K=10, R=0 (the exp 07 single-shot
#                config; matches Phase 1 regression cell exp07).
#        Expected wall (per cell, with main queue contention):
#                ~12-18 min  ->  total ~60-90 min
#
# Job B: 3-seed pool of sigma=0.03 on the EXP 05 driver
#        Validates the new 40% headline from the sigma-sweep
#        analysis (currently 1 seed at sigma=0.03 -> 24/60).
#        Cells:  3  (seeds 1, 2, 3)
#        Config: exp 05 driver, sigma=0.03, default K=10 lambda=0.5.
#        Expected wall (per cell): ~25-35 min  ->  total ~80-100 min
#
# Job C: 3-seed pool of the NEW BEST CELL discovered by the main
#        queue's Stage 1 (K=10, R=1, lambda=5, unified driver,
#        25/60 = 41.7%). Confirms whether 41.7% is real or seed-0
#        luck before we declare K=10 R=1 the canonical config.
#        Cells:  3  (seeds 1, 2, 3)
#        Config: unified driver, K=10 R=1 lambda=5, default sigma=0.25.
#        Expected wall (per cell): ~40-55 min  ->  total ~130-160 min
#
# Total estimated wall: ~5.0-5.8 h.
#
# Log/state location: logs/parallel_extras/
# Skip-existing semantics match the main queue: any cell whose result
# JSON already exists is skipped.
#
# To monitor: tail -f logs/parallel_extras/queue.log
# To abort:   kill $(cat logs/parallel_extras/pid)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
N_WORKERS_PARALLEL="${N_WORKERS_PARALLEL:-6}"
UNIFIED_OUT="experiments/15_unified_hybrid/results"
SIGMA_OUT="experiments/09_sigma_sweep/results"

LOG_DIR="logs/parallel_extras"
mkdir -p "$LOG_DIR"

# -------------- logging --------------
log() {
    local msg="$1"
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$msg" \
        | tee -a "$LOG_DIR/queue.log"
}

set_state() {
    echo "$1" > "$LOG_DIR/state.txt"
    log "STATE: $1"
}

# -------------- helper: run a unified cell --------------
run_unified_cell() {
    local K="$1" R="$2" lam="$3" sigma="$4" seed="$5" suffix="$6"
    local out_file="${UNIFIED_OUT}/unified_bc_lam-${lam}_K-${K}_R-${R}${suffix}.json"
    local stage_log="$LOG_DIR/cell_unified_lam${lam}_K${K}_R${R}_sigma${sigma}_seed${seed}${suffix}.log"

    if [[ -f "$out_file" ]]; then
        log "SKIP unified K=$K R=$R lam=$lam sigma=$sigma seed=$seed suffix=$suffix (exists)"
        return 0
    fi

    log "RUN  unified K=$K R=$R lam=$lam sigma=$sigma seed=$seed suffix=$suffix"
    local t0=$(date +%s)
    "$PY" experiments/15_unified_hybrid/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" \
        --n-workers "$N_WORKERS_PARALLEL" \
        --prior bc --warm-start \
        --lambda-prior "$lam" \
        --K "$K" --mpc-replans "$R" \
        --sigma-max "$sigma" \
        --output-dir "$UNIFIED_OUT" \
        --output-name-suffix "$suffix" \
        > "$stage_log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE unified K=$K R=$R lam=$lam sigma=$sigma seed=$seed suffix=$suffix in ${elapsed}s"
    else
        log "FAIL unified K=$K R=$R lam=$lam sigma=$sigma seed=$seed suffix=$suffix rc=$rc elapsed=${elapsed}s"
        log "     see $stage_log"
    fi
}

# -------------- helper: run an exp 05 sigma cell --------------
run_exp05_sigma_seed() {
    local sigma="$1" seed="$2"
    local out_file="${SIGMA_OUT}/sigma_${sigma}_seed${seed}.json"
    local stage_log="$LOG_DIR/cell_exp05_sigma${sigma}_seed${seed}.log"

    if [[ -f "$out_file" ]]; then
        log "SKIP exp05 sigma=$sigma seed=$seed (exists)"
        return 0
    fi

    log "RUN  exp05 sigma=$sigma seed=$seed"
    local t0=$(date +%s)
    "$PY" experiments/05_hybrid_method/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" \
        --no-live-view --n-workers "$N_WORKERS_PARALLEL" \
        --sigma-max "$sigma" \
        --deterministic-bc-reset \
        --output "$out_file" \
        > "$stage_log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE exp05 sigma=$sigma seed=$seed in ${elapsed}s"
    else
        log "FAIL exp05 sigma=$sigma seed=$seed rc=$rc elapsed=${elapsed}s"
        log "     see $stage_log"
    fi
}

# -------------- queue starts here --------------
log "==============================================================="
log "parallel_extras starting"
log "  N_WORKERS_PARALLEL=$N_WORKERS_PARALLEL"
log "  main queue PID:    $(cat logs/overnight_queue/pid 2>/dev/null || echo none)"
log "  this script PID:   $$"
log "==============================================================="

# Job A: sigma-sweep at lambda=5 unified  (5 cells)
set_state "job_a_sigma_sweep_lam5_unified"
log ""
log "##### Job A: sigma-sweep at lambda=5 on unified driver (K=10 R=0) #####"
for sigma in 0.01 0.03 0.05 0.10 0.25; do
    # IMPORTANT: include sigma in the suffix so the 5 cells don't
    # overwrite each other (the unified driver's filename template
    # is unified_bc_lam-X_K-Y_R-Z{suffix}.json -- no sigma slot).
    run_unified_cell 10 0 5.0 "$sigma" 0 "_sigmasweep_lam5_sigma${sigma}"
done
log "Job A done."

# Job B: 3-seed pool sigma=0.03 exp 05  (3 cells)
set_state "job_b_sigma003_3seed_exp05"
log ""
log "##### Job B: 3-seed pool sigma=0.03 on exp 05 driver #####"
for seed in 1 2 3; do
    run_exp05_sigma_seed 0.03 "$seed"
done
log "Job B done."

# Job C: 3-seed pool of new best cell (K=10 R=1 lam=5) unified  (3 cells)
set_state "job_c_best_cell_3seed_unified"
log ""
log "##### Job C: 3-seed pool of K=10 R=1 lam=5 (new best Phase 2 cell) #####"
for seed in 1 2 3; do
    run_unified_cell 10 1 5.0 0.25 "$seed" "_phase2_seed${seed}"
done
log "Job C done."

set_state "complete"
log "==============================================================="
log "parallel_extras COMPLETE"
log "==============================================================="
