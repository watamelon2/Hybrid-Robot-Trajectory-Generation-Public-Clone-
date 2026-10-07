#!/usr/bin/env bash
# Warm-start ablation at headline config K=10 R=1 sigma=0.25 M=16 H=120.
# All cells through the unified driver (exp 15). seed=0, 20 demos x 3 rollouts.
#
# Decomposes the BC + score-bridge 37.5 % headline:
#   - C1  BC warm-start, lambda=0  -> warm-start contribution
#   - C3  flow v2 warm-start, lambda=0 -> flow vs BC at headline sigma
#   - C4  flow v2 warm-start, lambda=5 (flow score-bridge) -> flow + bridge
#
# (Existing C0 = BC warm-start + lambda=5 score-bridge = 37.5 % 4-seed pool.)
# Written 2026-05-16 PM in service of the project's warm-start narrative.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/warmstart_ablation_2026_05_16"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

BC_CKPT="$REPO/experiments/03_bc_baseline/results/policy.pt"
FLOW_CKPT="$REPO/experiments/08_flow_matching_prior/results/v_theta_v2.pt"

# Headline-config hyperparameters (matched to MORNING_BRIEF Phase-2 row).
COMMON=(
  --task can
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.01
  --n-demos 20 --n-rollouts 3 --seed 0
  --n-workers 10
  --output-dir "$OUTDIR"
)

run_cell () {
  local LABEL="$1"; shift
  echo "=== $(date '+%H:%M:%S')  $LABEL  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$LABEL" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $LABEL  done ===" | tee -a "$LOG"
}

date | tee "$LOG"
echo "Slate: 3 cells, ~35 min each = ~1.75 h total" | tee -a "$LOG"

# C1  BC warm-start, no score-bridge (warm-start contribution to headline)
run_cell "C1_bc_lam0" \
  --prior bc --bc-checkpoint "$BC_CKPT" \
  --lambda-prior 0

# C3  flow v2 warm-start, no score-bridge (warm-start architecture at headline sigma)
run_cell "C3_flow_lam0" \
  --prior flow --prior-checkpoint "$FLOW_CKPT" \
  --lambda-prior 0

# C4  flow v2 warm-start + flow score-bridge (full flow method at headline sigma)
run_cell "C4_flow_lam5" \
  --prior flow --prior-checkpoint "$FLOW_CKPT" \
  --lambda-prior 5

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
