#!/usr/bin/env bash
# Slate Dynamics v2 -- HARDENED launcher.
#
# Diff vs slate_dynamics_2026_05_26.sh:
#   1. SIGTERM trap at bash level: external pkill on the launcher itself
#      logs a warning and continues (children can still die; we recover).
#   2. set -e dropped so a single failed cell doesn't abort the whole slate.
#   3. cell_done() skips cells whose expected JSONs already exist.
#      Enables zero-cost resume after a kill (cells 1-2 from attempt 4
#      already on disk will be skipped on relaunch).
#   4. Per-cell rc tracking + 30s backoff on failure (gives any active
#      pkill burst time to settle before next cell).
#
# Scientific behavior identical to v1.  All other args, cell order, and
# JSON output paths unchanged.

set -uo pipefail   # NOTE: no -e; we want to survive single-cell failures

REPO="/private/tmp/HRTG-dynamics-tmp"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Trap SIGTERM at the bash level.  If an external pkill -f matches THIS
# script's command line, bash will log and ignore the signal.  Children
# (python masters) are still vulnerable to pkill matching their own
# command lines -- but on next cell iteration we recover automatically.
trap 'echo "[$(date)] WARN launcher (PID '"$$"') caught SIGTERM, ignoring" | tee -a "$LOG" >&2' TERM

COMMON=(
  --task can
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 20 --n-rollouts 3
  --n-workers 10
  --seed 0
  --output-dir "$OUTDIR"
)

# Skip helper: true (rc 0) if all expected per-lambda JSONs already
# exist for this cell -- means it ran to completion before.
cell_done () {
  local PRIOR_NAME="$1"    # "demo_1nn" or "bc"
  local TAG="$2"           # e.g., "demo1nn_m100"
  shift 2
  local LAM
  for LAM in "$@"; do
    local F="$OUTDIR/unified_${PRIOR_NAME}_lam-${LAM}_K-10_R-1_${TAG}.json"
    [[ -f "$F" ]] || return 1
  done
  return 0
}

# Run a cell unless its JSONs already exist.  Logs result of each
# attempt; never aborts the slate on a single-cell failure.
run_cell () {
  local TAG="$1"; shift
  local PRIOR_NAME="$1"; shift
  local LAMS_STR="$1"; shift   # space-separated lambdas, e.g. "0.0 1.0"
  # shellcheck disable=SC2206
  local LAMS=($LAMS_STR)

  if cell_done "$PRIOR_NAME" "$TAG" "${LAMS[@]}"; then
    echo "=== $(date '+%H:%M:%S')  $TAG  SKIP (all ${#LAMS[@]} JSONs already on disk) ===" | tee -a "$LOG"
    return 0
  fi

  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  local RC=${PIPESTATUS[0]}

  if [[ $RC -eq 0 ]]; then
    echo "=== $(date '+%H:%M:%S')  $TAG  done (rc=0) ===" | tee -a "$LOG"
  else
    echo "=== $(date '+%H:%M:%S')  $TAG  FAILED rc=$RC -- will be retried on next launch (skip-existing-JSONs) ===" | tee -a "$LOG"
    sleep 30   # backoff: give any active pkill burst time to settle
  fi
}

date | tee -a "$LOG"
echo "Slate Dynamics v2 (resilient): mass-scale adaptation on Can" | tee -a "$LOG"
echo "Hardening: SIGTERM trap + skip-existing-JSONs + no set -e + 30s backoff" | tee -a "$LOG"
echo "Spec: docs/adaptation/02_dynamics_spec.md (D1 mass ladder, minimal variant)" | tee -a "$LOG"
echo "Mass scales: {0.7, 1.0, 1.3, 1.5} on robot-link bodies + gripper" | tee -a "$LOG"
echo "Methods: hybrid (demo_1nn,lam=1) | pure-MBD (lam=0) | BC (bc,lam=5)" | tee -a "$LOG"
echo "Eval set: 20 demos x 3 rollouts = 60 evals/cell, seed=0" | tee -a "$LOG"

for MASS in 1.0 0.7 1.3 1.5; do
  CELL_TAG=$(printf "m%03d" $(echo "$MASS * 100 / 1" | bc))

  echo "" | tee -a "$LOG"
  echo "########## Mass scale $MASS  ($CELL_TAG) ##########" | tee -a "$LOG"

  # Hybrid + pure-MBD (single invocation, two lambdas: 0 and 1).
  run_cell "demo1nn_${CELL_TAG}" "demo_1nn" "0.0 1.0" \
    --prior demo_1nn \
    --lambda-prior 0 1 \
    --mass-scale "$MASS"

  # BC bridge.
  run_cell "bc_${CELL_TAG}" "bc" "5.0" \
    --prior bc \
    --lambda-prior 5 \
    --mass-scale "$MASS"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Headline output: per-cell JSONs in $OUTDIR" | tee -a "$LOG"
