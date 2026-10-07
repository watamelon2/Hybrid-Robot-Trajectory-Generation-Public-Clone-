#!/usr/bin/env bash
# T4 MBD-no-prior catch-up slate for Mac A.
#
# Background: the original T4 slate (slate_cross_task_baselines_2026_06_01.sh)
# at commit a1dd1e3 only committed 2/6 cells (lift_bc + lift_mbd_no_prior).
# Square + tool_hang cells were lost overnight on Mac B due to a chain
# `git clean -fd` that wiped uncommitted results after a slate failure.
#
# Mac B took the 2 BC cells (~5h, light wall).  This slate is for Mac A,
# the 16-core machine, to run the 2 MBD-no-prior cells which are heavier
# due to --no-warm-start --M=64 making each MBD step ~5-10x more expensive.
# Estimated wall on 16 cores: ~10-15h for both cells together.
#
# Cells:
#   - cross_square_mbd_no_prior      (H=400)
#   - cross_tool_hang_mbd_no_prior   (H=300)
#
# Outputs to:
#   experiments/15_unified_hybrid/results/slate_cross_task_baselines_2026_06_01/
#
# How to run on Mac A:
#   cd <YOUR_REPO>
#   git fetch origin feat/cross-task-baselines-fill
#   git checkout feat/cross-task-baselines-fill
#   git merge --ff-only origin/feat/cross-task-baselines-fill
#   bash experiments/15_unified_hybrid/slate_cross_task_baselines_mbd_only_macA.sh
#
# Then commit + push the produced JSONs back to feat/cross-task-baselines-fill.

set -uo pipefail

# Auto-detect REPO from this script's location (works regardless of where Mac A's repo lives).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_cross_task_baselines_2026_06_01"
N_WORKERS="${N_WORKERS:-12}"   # Mac A is 16-core; leave 4 cores for OS/scheduler
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_macA_mbd.log"

date 2>&1 | tee -a "$LOG"
echo "Slate T4 MBD-no-prior catch-up (Mac A)" | tee -a "$LOG"
echo "REPO=$REPO" | tee -a "$LOG"
echo "OUTDIR=$OUTDIR" | tee -a "$LOG"
echo "N_WORKERS=$N_WORKERS" | tee -a "$LOG"

# Activate conda env if not already.  Mac A's env path may differ.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "hybrid-traj" ]; then
  for CONDA_BASE in /Users/szh/anaconda3 /Users/szh/miniforge3 /Users/aaronwen/miniforge3 \
                    /opt/anaconda3 /opt/miniforge3 "$HOME/anaconda3" "$HOME/miniforge3" ; do
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate hybrid-traj && break
    fi
  done
fi
echo "conda env: ${CONDA_DEFAULT_ENV:-(none)}" | tee -a "$LOG"

cd "$REPO"

# Per-task params (same as original slate)
task_h ()         { case "$1" in lift) echo 120 ;; square) echo 400 ;; tool_hang) echo 300 ;; esac; }
task_ndemos ()    { case "$1" in lift) echo 20  ;; square) echo 10  ;; tool_hang) echo 10  ;; esac; }
task_nrollouts () { case "$1" in lift) echo 3   ;; square) echo 6   ;; tool_hang) echo 6   ;; esac; }

run_mbd_no_prior_cell () {
  local TASK="$1"
  local TAG="cross_${TASK}_mbd_no_prior"
  local H NDEMOS NROLLOUTS
  H=$(task_h "$TASK")
  NDEMOS=$(task_ndemos "$TASK")
  NROLLOUTS=$(task_nrollouts "$TASK")

  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1 ; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi

  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"

  # Note: --prior bc + --lambda-prior 0 + --no-warm-start = "MBD-no-prior"
  # (lambda=0 ignores the BC prior, so we measure pure MBD).  Matches
  # the exact invocation in the original slate.
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task "$TASK" \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H "$H" \
    --sigma-min 0.05 \
    --n-demos "$NDEMOS" --n-rollouts "$NROLLOUTS" \
    --n-workers "$N_WORKERS" \
    --seed 0 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior bc --lambda-prior 0 \
    --no-warm-start \
    --sigma-max 0.05 --M 64 2>&1 | tee -a "$LOG"

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

echo "##### task = square (MBD-no-prior, H=400) #####" | tee -a "$LOG"
run_mbd_no_prior_cell square

echo "" | tee -a "$LOG"
echo "##### task = tool_hang (MBD-no-prior, H=300) #####" | tee -a "$LOG"
run_mbd_no_prior_cell tool_hang

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T4 MBD-only Mac A  ALL DONE ===" | tee -a "$LOG"

# Optional: print headline summary at the end
echo "" | tee -a "$LOG"
echo "Cell summary:" | tee -a "$LOG"
for TAG in cross_square_mbd_no_prior cross_tool_hang_mbd_no_prior ; do
  for f in "$OUTDIR"/unified_*"_${TAG}".json ; do
    [ -f "$f" ] || continue
    python -c "import json,sys; d=json.load(open('$f')); print(' ', '$TAG', '->', d.get('per_lambda',[{}])[0].get('success_rate','?'))" 2>&1 | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "Now please:" | tee -a "$LOG"
echo "  git add -A && git add -f $OUTDIR/unified_*.json $OUTDIR/run_macA_mbd.log" | tee -a "$LOG"
echo "  git commit -m 'results(t4): MBD-no-prior cells from Mac A (square + tool_hang)'" | tee -a "$LOG"
echo "  git push origin feat/cross-task-baselines-fill" | tee -a "$LOG"
