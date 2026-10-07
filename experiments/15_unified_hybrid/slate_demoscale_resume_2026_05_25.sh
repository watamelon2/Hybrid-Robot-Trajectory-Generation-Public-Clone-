#!/usr/bin/env bash
# Slate DemoScale -- RESUME (cells 2-6 only), running in the
# dedicated HRTG-slate worktree.
#
# Cell 1 (Nb5) completed cleanly in the main worktree at 15:30:59
# PT 2026-05-25 with 12/60 = 20.0% (IB 80% / OOB 0%).  JSON preserved
# in MAIN worktree's results/slate_demoscale_2026_05_25/ dir.
#
# Between Nb5 done and Nb10 start, the MAIN worktree's working tree
# drifted (stash-pop / cleanup activity), so the slate's
# python-invocation for Nb10 failed (argparse rejected
# --prior=demo_1nn).  The user is doing parallel branch cleanup in
# the main worktree, so this resume runs in the DEDICATED HRTG-slate
# worktree on feat/adapt-demo-scaling (clean checkout, no contention).
#
# Output dir = MAIN worktree's results dir, so Nb5 JSON and the new
# Nb10/20/50/100/200 JSONs sit alongside each other for aggregation.
#
# 5 cells, ~50 min each ~ 4 h wall.

set -euo pipefail

# REPO = THIS worktree (where the code lives).
REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-slate"

# OUTDIR = MAIN worktree's results dir (where Nb5 JSON already lives).
# Aggregation later just reads everything in this one dir.
OUTDIR="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation/experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

COMMON=(
  --task can
  --prior demo_1nn --lambda-prior 1
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.01
  --n-demos 20 --n-rollouts 3
  --n-workers 10
  --seed 0
  --output-dir "$OUTDIR"
)

run () {
  local TAG="$1"; shift
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

{
  echo ""
  echo "============================================================"
  echo "RESUMED $(date)  (in HRTG-slate worktree)"
  echo "Skipping Nb5 (already complete: 12/60 = 20.0%)"
  echo "Running cells: Nb10, Nb20, Nb50, Nb100, Nb200"
  echo "============================================================"
} | tee -a "$LOG"

for NB in 10 20 50 100 200; do
  run "Nb${NB}" --prior-bank-size "$NB"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE (RESUME) ===" | tee -a "$LOG"
echo "Full 6-cell scaling curve available in $OUTDIR" | tee -a "$LOG"
