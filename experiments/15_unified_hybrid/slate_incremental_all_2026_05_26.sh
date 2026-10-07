#!/usr/bin/env bash
# Slate Incremental (META) -- chains all four phases in sequence.
#
# Runs:
#   Phase 1 (~10 min)  slate_incremental_collect      (T2 bank prereq)
#   Phase 2 (~5 h)     slate_incremental_T1            (cross-task)
#   Phase 3 (~3.5 h)   slate_incremental_T2_eval       (T2.0/T2.1/T2.2/T2.4)
#   Phase 3' (~1.4 h)  slate_incremental_T2_bc_retrain (T2.3)
#   Phase 4            aggregate_incremental.py --write
#
# Total ~10 h sequential.  Each phase only fires if the previous
# phase exits 0.  Per-phase logs go to their own results/.../run.log;
# this meta-script writes a single top-level meta.log with stage
# transitions and ETAs.
#
# Usage (run in its own detached screen):
#
#   screen -dmS slate_incremental bash \
#     experiments/15_unified_hybrid/slate_incremental_all_2026_05_26.sh
#
# Spec:        docs/adaptation/04_incremental_prior_spec.md (T1 + T2)
# Branch:      feat/adapt-incremental-prior
# Worktree:    HRTG-incremental

set -uo pipefail   # NB: -e omitted so we can catch failures & log them

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-incremental"
META_DIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_meta_2026_05_26"
META_LOG="$META_DIR/meta.log"
mkdir -p "$META_DIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

log () { echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$META_LOG"; }

log "META-SLATE start"
log "Repo:    $REPO"
log "Branch:  $(git -C "$REPO" branch --show-current)"
log "HEAD:    $(git -C "$REPO" log -1 --oneline)"
log ""

# Bail early if any prior incremental run wrote outputs we'd
# overwrite (collect's banks, T1 cells, T2 cells).
T1_DIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T1_2026_05_26"
T2_DIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_2026_05_26"
T23_DIR="$REPO/experiments/15_unified_hybrid/results/slate_incremental_T2_bc_retrain_2026_05_26"
for d in "$T1_DIR" "$T2_DIR" "$T23_DIR"; do
  if [ -e "$d/run.log" ]; then
    log "WARNING: $d/run.log exists -- prior run will be APPENDED to, not overwritten."
  fi
done

run_phase () {
  local name="$1"; local script="$2"
  log ">>> Phase: $name  start"
  log "    script: $script"
  local t0=$(date +%s)
  bash "$script"
  local rc=$?
  local t1=$(date +%s)
  local dt=$((t1 - t0))
  log "<<< Phase: $name  end (rc=$rc, wall=${dt}s = $((dt / 60))m $((dt % 60))s)"
  if [ "$rc" -ne 0 ]; then
    log "FATAL: phase $name failed (rc=$rc); aborting subsequent phases."
    log "Inspect the phase's own run.log for the error."
    exit "$rc"
  fi
}

# ---- Phase 1: collect ----
run_phase "1-collect" \
  "$REPO/experiments/15_unified_hybrid/slate_incremental_collect_2026_05_26.sh"

# Sanity: the three bank files must exist before we proceed.
DATA_DIR="$REPO/experiments/15_unified_hybrid/data"
for f in "$DATA_DIR/can_mass1p3_5demos.hdf5" \
         "$DATA_DIR/can_mass1p3_205demos.hdf5" \
         "$DATA_DIR/can_mass1p3_220demos.hdf5"; do
  if [ ! -f "$f" ]; then
    log "FATAL: prereq bank missing after collect: $f"
    log "Subsequent phases need this; aborting."
    exit 4
  fi
done
log "Collect produced all 3 bank HDF5s; proceeding."

# ---- Phase 2: T1 cross-task ----
run_phase "2-T1-crosstask" \
  "$REPO/experiments/15_unified_hybrid/slate_incremental_T1_2026_05_26.sh"

# ---- Phase 3: T2 eval (4 cells) ----
run_phase "3-T2-eval" \
  "$REPO/experiments/15_unified_hybrid/slate_incremental_T2_eval_2026_05_26.sh"

# ---- Phase 3': T2.3 BC retrain + eval ----
run_phase "3prime-T2-3-bc-retrain" \
  "$REPO/experiments/15_unified_hybrid/slate_incremental_T2_bc_retrain_2026_05_26.sh"

# ---- Phase 4: aggregate ----
log ">>> Phase: 4-aggregate"
python "$REPO/experiments/15_unified_hybrid/aggregate_incremental.py" --write \
  2>&1 | tee -a "$META_LOG"
log "<<< Phase: 4-aggregate  done"

log ""
log "META-SLATE all phases done.  Summary:"
log "  $REPO/experiments/15_unified_hybrid/results/slate_incremental_aggregate/summary.md"
log "  $REPO/experiments/15_unified_hybrid/results/slate_incremental_aggregate/T1_bar_data.csv"
log "  $REPO/experiments/15_unified_hybrid/results/slate_incremental_aggregate/T2_pareto_data.csv"
