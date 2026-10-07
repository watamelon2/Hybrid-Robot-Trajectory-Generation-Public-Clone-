#!/usr/bin/env bash
# scripts/local/overnight_queue.sh
#
# Sequentially run every queue-able experiment for the next ~12 h,
# starting after the currently-running σ-sweep, regression, and
# σ-extension jobs drain.
#
# See docs/archive/plan_overnight_queue.md for the full design rationale.
#
# Usage:
#   nohup bash scripts/local/overnight_queue.sh \
#       > logs/overnight_queue/stdout.log 2>&1 < /dev/null &
#
# Resume semantics: idempotent. Re-running picks up where it left off
# because every cell does a skip-if-exists check.

set -u   # NOT -e -- we want individual cells to fail without killing
         # the whole queue.

# ---- repo location ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- env ----------------------------------------------------------
HYBRID_BIN="/Users/szh/anaconda3/envs/hybrid-traj/bin"
PY="$HYBRID_BIN/python"
export PATH="$HYBRID_BIN:$PATH"

# ---- logging ------------------------------------------------------
QUEUE_DIR="logs/overnight_queue"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="$QUEUE_DIR/queue.log"
QUEUE_STATE="$QUEUE_DIR/state.txt"

log() {
    local msg="$1"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $msg" | tee -a "$QUEUE_LOG"
}

set_state() {
    local state="$1"
    echo "$state" > "$QUEUE_STATE"
    log "STATE: $state"
}

# ---- PIDs of the currently-running jobs ---------------------------
# These must match the launch-meta of the three running jobs.
CURRENT_SIGMA_PID="$(cat logs/local_sigma_sweep/pid 2>/dev/null || true)"
CURRENT_REGR_PID="$(cat logs/local_phase1_regression/pid 2>/dev/null || true)"
CURRENT_EXT_PID="$(cat logs/local_sigma_sweep_extension/pid 2>/dev/null || true)"

log "==============================================================="
log "overnight queue starting"
log "current jobs: sigma=$CURRENT_SIGMA_PID  regr=$CURRENT_REGR_PID  ext=$CURRENT_EXT_PID"
log "==============================================================="

# ===================================================================
# STAGE 0: wait for current jobs
# ===================================================================
set_state "stage_0_wait"

wait_for_pid() {
    local pid="$1" label="$2"
    if [[ -z "$pid" ]]; then
        log "no PID recorded for $label -- skipping wait"
        return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        log "$label (PID $pid) already finished"
        return
    fi
    log "waiting for $label (PID $pid) to finish"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "$label (PID $pid) finished"
}

wait_for_pid "$CURRENT_SIGMA_PID" "main_sigma_sweep"
wait_for_pid "$CURRENT_REGR_PID"  "regression"
wait_for_pid "$CURRENT_EXT_PID"   "sigma_extension"

log "all current jobs drained -- proceeding to acceptance gate"

# ===================================================================
# STAGE 0.5: Phase 1 acceptance gate
# ===================================================================
set_state "stage_0_5_acceptance"

# regression_summary.json was archived to results/_archive/ on 2026-05-15.
# Check both locations; prefer active (re-run lands fresh in results/).
REGR_SUMMARY_ACTIVE="experiments/15_unified_hybrid/results/regression_summary.json"
REGR_SUMMARY_ARCHIVE="experiments/15_unified_hybrid/results/_archive/regression_summary.json"
if [[ -f "$REGR_SUMMARY_ACTIVE" ]]; then
    REGR_SUMMARY="$REGR_SUMMARY_ACTIVE"
elif [[ -f "$REGR_SUMMARY_ARCHIVE" ]]; then
    REGR_SUMMARY="$REGR_SUMMARY_ARCHIVE"
else
    log "ABORT: regression_summary.json missing at $REGR_SUMMARY_ACTIVE or $REGR_SUMMARY_ARCHIVE"
    set_state "aborted_no_summary"
    exit 1
fi

PHASE1_PASS="$("$PY" -c "
import json, sys
d = json.load(open('$REGR_SUMMARY'))
# The regression_test.py writes 'all_pass' (true/false) at the top
# level. Fall back to per-cell observed/target/tolerance checks if
# 'all_pass' is missing (older summaries).
ok = d.get('all_pass')
if ok is None:
    tol = d.get('tolerance_successes', 1)
    cells = d.get('cells', [])
    ok = (len(cells) > 0) and all(
        abs(c.get('observed', -1) - c.get('target_succ', 1e9)) <= tol
        for c in cells
    )
print('PASS' if ok else 'FAIL')
" 2>/dev/null || echo "FAIL")"

if [[ "$PHASE1_PASS" != "PASS" ]]; then
    log "ABORT: Phase 1 acceptance FAILED (regression_summary says $PHASE1_PASS)"
    log "  see $REGR_SUMMARY for details"
    set_state "aborted_phase1_fail"
    exit 1
fi
log "Phase 1 acceptance PASSED -- proceeding"

# ===================================================================
# STAGE 1: C Phase 2 -- K x R grid at lambda=5
# ===================================================================
set_state "stage_1_phase2_grid"
log ""
log "##### Stage 1: Phase 2 K x R grid at lambda=5 #####"

UNIFIED_OUT="experiments/15_unified_hybrid/results"

run_unified_cell() {
    local K="$1" R="$2" lam="$3" suffix="$4" seed="${5:-0}"
    local out_file="${UNIFIED_OUT}/unified_bc_lam-${lam}_K-${K}_R-${R}${suffix}.json"
    local stage_log="${QUEUE_DIR}/cell_unified_lam${lam}_K${K}_R${R}${suffix}_seed${seed}.log"

    if [[ -f "$out_file" ]]; then
        log "SKIP unified K=$K R=$R lam=$lam suffix=$suffix seed=$seed (exists)"
        return 0
    fi

    log "RUN  unified K=$K R=$R lam=$lam suffix=$suffix seed=$seed"
    local t0=$(date +%s)
    "$PY" experiments/15_unified_hybrid/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" \
        --n-workers 12 \
        --prior bc --warm-start \
        --lambda-prior "$lam" \
        --K "$K" --mpc-replans "$R" \
        --output-dir "$UNIFIED_OUT" \
        --output-name-suffix "$suffix" \
        > "$stage_log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE unified K=$K R=$R lam=$lam suffix=$suffix seed=$seed in ${elapsed}s"
    else
        log "FAIL unified K=$K R=$R lam=$lam suffix=$suffix seed=$seed rc=$rc elapsed=${elapsed}s"
        log "     see $stage_log"
    fi
    return $rc
}

for K in 5 10 20; do
    for R in 0 1 2; do
        run_unified_cell "$K" "$R" "5.0" "_phase2" 0
    done
done
log "Stage 1 done."

# ===================================================================
# STAGE 2: 3-seed pool of sigma=0.05 (seeds 1, 2)
# ===================================================================
set_state "stage_2_sigma005_seeds"
log ""
log "##### Stage 2: 3-seed pool of sigma=0.05 #####"

SIGMA_OUT="experiments/09_sigma_sweep/results"

run_sigma_seed() {
    local sigma="$1" seed="$2"
    local out_file="${SIGMA_OUT}/sigma_${sigma}_seed${seed}.json"
    local stage_log="${QUEUE_DIR}/cell_sigma${sigma}_seed${seed}.log"

    if [[ -f "$out_file" ]]; then
        log "SKIP sigma=$sigma seed=$seed (exists)"
        return 0
    fi

    log "RUN  sigma=$sigma seed=$seed"
    local t0=$(date +%s)
    "$PY" experiments/05_hybrid_method/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed "$seed" \
        --no-live-view --n-workers 12 \
        --sigma-max "$sigma" \
        --deterministic-bc-reset \
        --output "$out_file" \
        > "$stage_log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE sigma=$sigma seed=$seed in ${elapsed}s"
    else
        log "FAIL sigma=$sigma seed=$seed rc=$rc elapsed=${elapsed}s"
        log "     see $stage_log"
    fi
    return $rc
}

for seed in 1 2; do
    run_sigma_seed "0.05" "$seed"
done
log "Stage 2 done."

# ===================================================================
# STAGE 3: lambda-sweep on unified driver at K=10, R=2
# ===================================================================
set_state "stage_3_lambda_sweep"
log ""
log "##### Stage 3: lambda-sweep on unified driver (K=10, R=2) #####"

# The unified driver supports a list of lambdas in one process and
# writes per-lambda JSONs. Use a one-shot invocation.
LAMSWEEP_STAGE_LOG="${QUEUE_DIR}/stage3_lambda_sweep.log"

# Check if all 6 lambda results exist; if so, skip.
LAMSWEEP_LAMS=(0 1 2 5 10 20)
LAMSWEEP_NEED=0
for L in "${LAMSWEEP_LAMS[@]}"; do
    # The unified driver writes 5.0 not 5; we need to match its filename
    # printf. Use a python helper for safety.
    fname="$("$PY" -c "print(f'unified_bc_lam-{float($L)}_K-10_R-2_lamsweep.json')")"
    # Files archived to _archive/ on 2026-05-15; check both locations.
    if [[ ! -f "$UNIFIED_OUT/$fname" ]] && [[ ! -f "$UNIFIED_OUT/_archive/$fname" ]]; then
        LAMSWEEP_NEED=1
        break
    fi
done

if [[ "$LAMSWEEP_NEED" -eq 1 ]]; then
    log "RUN  lambda-sweep (lams=0,1,2,5,10,20  K=10  R=2)"
    t0=$(date +%s)
    "$PY" experiments/15_unified_hybrid/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed 0 \
        --n-workers 12 \
        --prior bc --warm-start \
        --lambda-prior 0 1 2 5 10 20 \
        --K 10 --mpc-replans 2 \
        --output-dir "$UNIFIED_OUT" \
        --output-name-suffix "_lamsweep" \
        > "$LAMSWEEP_STAGE_LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE lambda-sweep in ${elapsed}s"
    else
        log "FAIL lambda-sweep rc=$rc elapsed=${elapsed}s"
        log "     see $LAMSWEEP_STAGE_LOG"
    fi
else
    log "SKIP lambda-sweep (all 6 lambda JSONs already exist)"
fi
log "Stage 3 done."

# ===================================================================
# STAGE 4: 3-seed pool of the best Phase 2 cell
# ===================================================================
set_state "stage_4_best_cell_seeds"
log ""
log "##### Stage 4: 3-seed pool of best Phase 2 cell #####"

# Identify the (K*, R*) cell with max success.
BEST_CELL_OUT="${QUEUE_DIR}/best_phase2_cell.json"
"$PY" - "$UNIFIED_OUT" "$BEST_CELL_OUT" <<'PYEOF' 2>>"$QUEUE_LOG"
import json, os, sys, glob, re
out_dir = sys.argv[1]
best_path = sys.argv[2]
pattern = os.path.join(out_dir, "unified_bc_lam-5.0_K-*_R-*_phase2.json")
best = None
for f in glob.glob(pattern):
    m = re.search(r"K-(\d+)_R-(\d+)", f)
    if not m:
        continue
    K, R = int(m.group(1)), int(m.group(2))
    d = json.load(open(f))
    succ = d.get("n_success", 0)
    total = d.get("n_total", 60)
    rate = succ / total if total else 0.0
    if best is None or rate > best["rate"]:
        best = {"K": K, "R": R, "succ": succ, "total": total, "rate": rate, "path": f}
if best is None:
    raise SystemExit("no Phase 2 cells found")
json.dump(best, open(best_path, "w"), indent=2)
print(f"BEST cell: K={best['K']} R={best['R']} succ={best['succ']}/{best['total']} ({100*best['rate']:.1f}%)")
PYEOF
BEST_RC=$?
if [[ $BEST_RC -ne 0 ]]; then
    log "WARN: best-cell picker exited rc=$BEST_RC -- Stage 4 will be skipped"
fi

if [[ -f "$BEST_CELL_OUT" ]]; then
    BEST_K="$("$PY" -c "import json; print(json.load(open('$BEST_CELL_OUT'))['K'])")"
    BEST_R="$("$PY" -c "import json; print(json.load(open('$BEST_CELL_OUT'))['R'])")"
    log "best Phase 2 cell: K=$BEST_K R=$BEST_R"
    for seed in 1 2; do
        run_unified_cell "$BEST_K" "$BEST_R" "5.0" "_phase2_seed${seed}" "$seed"
    done
else
    log "Stage 4 skipped (no best cell)"
fi
log "Stage 4 done."

# ===================================================================
# STAGE 5: A1.3 particle ablation at sigma=0.05
# ===================================================================
set_state "stage_5_particle_ablation"
log ""
log "##### Stage 5: A1.3 particle ablation (M in {64, 256}) #####"

run_M_cell() {
    local M="$1"
    local out_file="${SIGMA_OUT}/sigma_0.05_M${M}.json"
    local stage_log="${QUEUE_DIR}/cell_M${M}.log"

    if [[ -f "$out_file" ]]; then
        log "SKIP M=$M (exists)"
        return 0
    fi
    log "RUN  M=$M (sigma=0.05)"
    local t0=$(date +%s)
    "$PY" experiments/05_hybrid_method/run_can.py \
        --n-demos 20 --n-rollouts 3 --seed 0 \
        --no-live-view --n-workers 12 \
        --sigma-max 0.05 --M "$M" \
        --deterministic-bc-reset \
        --output "$out_file" \
        > "$stage_log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ $rc -eq 0 ]]; then
        log "DONE M=$M in ${elapsed}s"
    else
        log "FAIL M=$M rc=$rc elapsed=${elapsed}s"
        log "     see $stage_log"
    fi
    return $rc
}

# M=64 first (cheap), gate M=256 on a meaningful delta vs M=16 baseline.
run_M_cell 64

# Check if M=64 vs M=16 delta is worth investigating M=256.
M16_PATH="${SIGMA_OUT}/sigma_0.05.json"
M64_PATH="${SIGMA_OUT}/sigma_0.05_M64.json"
SHOULD_RUN_M256=1
if [[ -f "$M16_PATH" && -f "$M64_PATH" ]]; then
    DELTA="$("$PY" -c "
import json
d16 = json.load(open('$M16_PATH'))
d64 = json.load(open('$M64_PATH'))
r16 = d16.get('n_success', 0) / d16.get('n_total', 60)
r64 = d64.get('n_success', 0) / d64.get('n_total', 60)
print(f'{r64 - r16:.4f}')
")"
    # If delta < 0.02 (2 pp), M=256 likely won't move the needle.
    SHOULD_RUN_M256="$("$PY" -c "
delta = float('$DELTA')
print(1 if delta >= 0.02 else 0)
")"
    log "M=64 vs M=16 delta: $DELTA  ->  run M=256? $SHOULD_RUN_M256"
fi

if [[ "$SHOULD_RUN_M256" == "1" ]]; then
    run_M_cell 256
else
    log "SKIP M=256 (M=64 - M=16 delta below threshold)"
fi
log "Stage 5 done."

# ===================================================================
# STAGE 6: failure-taxonomy refresh
# ===================================================================
set_state "stage_6_taxonomy"
log ""
log "##### Stage 6: failure-taxonomy refresh #####"

TAX_LOG="${QUEUE_DIR}/stage6_taxonomy.log"
"$PY" experiments/06_failure_taxonomy/run.py \
    --inputs "$SIGMA_OUT"/sigma_*.json \
    --output "$SIGMA_OUT/taxonomy.json" \
    > "$TAX_LOG" 2>&1
if [[ $? -eq 0 ]]; then
    log "DONE taxonomy refresh"
else
    log "FAIL taxonomy refresh (see $TAX_LOG)"
fi

# ===================================================================
# done
# ===================================================================
set_state "complete"
log "==============================================================="
log "overnight queue COMPLETE"
log "==============================================================="
