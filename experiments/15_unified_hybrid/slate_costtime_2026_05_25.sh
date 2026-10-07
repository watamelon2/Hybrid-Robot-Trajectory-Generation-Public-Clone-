#!/usr/bin/env bash
# Slate CostTime -- cost-time (goal-pose) adaptation experiment on Can.
#
# Spec:        docs/adaptation/01_cost_time_spec.md
# Theory:      docs/theory/01_soft_control.md sec 1 (composite score
#              under cost perturbation), 03_pareto_bias_variance.md.
# Branch:      feat/adapt-cost-time.
# Worker-prop patch: commit 716630a (enables this slate to fire).
#
# Tests whether the hybrid (demo_1nn + MBD) inherits MBD's automatic
# cost re-targeting, i.e. whether the hybrid-vs-BC success gap WIDENS
# with goal-pose perturbation magnitude.  Three methods x four
# perturbation magnitudes = 12 cells.
#
# Methods (canonical operating point from Slate E peak):
#
#   hybrid       --prior demo_1nn  --lambda-prior 1  --sigma-max 0.25
#   pure-MBD     --prior demo_1nn  --lambda-prior 0  --sigma-max 0.25
#                (lambda=0 zeroes the prior contribution to the score;
#                 the demo_1nn prior choice is irrelevant in this
#                 case but avoids loading the BC checkpoint)
#   BC bridge    --prior bc        --lambda-prior 5  --sigma-max 0.25
#                (canonical BC peak from Slate E E2 lambda-curve)
#
# Perturbations: dx in {0.00, 0.05, 0.10, 0.20} m, +x direction only.
# The +y perturbation extension lives in C2 (obstacle injection), not
# in this slate.
#
# Per cell: 20 demos x 3 rollouts = 60 evaluations, single seed=0.
# (Pool later if a gap is real; see spec sec 2.)
#
# Wall-clock estimate:
#   - hybrid+pure-MBD share one invocation per perturbation
#     (--lambda-prior 0 1 sweeps both inside the unified driver):
#     ~1 h per (perturbation x {0,1}) invocation, x 4 perturbations.
#   - BC: ~30 min per invocation, x 4 perturbations.
#   - Total: ~4 (h+m) + 4 (BC) = ~6 h.
#
# Start: 2026-05-25 evening.  Run in screen session "slate_costtime"
# from the HRTG-slate worktree (this branch).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-slate"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_costtime_2026_05_25"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Common args (must match docs/adaptation/01_cost_time_spec.md sec 2).
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

run () {
  local TAG="$1"; shift
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" \
    "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

date | tee "$LOG"
echo "Slate CostTime: goal-pose adaptation for hybrid vs BC vs pure-MBD on Can" | tee -a "$LOG"
echo "Spec: docs/adaptation/01_cost_time_spec.md" | tee -a "$LOG"
echo "Theory: docs/theory/01_soft_control.md sec 1, 03_pareto_bias_variance.md" | tee -a "$LOG"
echo "Worker-prop patch: 716630a (enables this slate)" | tee -a "$LOG"
echo "Perturbations: dx in {0.00, 0.05, 0.10, 0.20} m" | tee -a "$LOG"
echo "Methods: hybrid (demo_1nn,lam=1) | pure-MBD (lam=0) | BC (bc,lam=5)" | tee -a "$LOG"
echo "Eval set: 20 demos x 3 rollouts = 60 evals/cell, seed=0" | tee -a "$LOG"

# Iterate over the four perturbation magnitudes.  For each, fire
# the two-lambda sweep (hybrid + pure-MBD share workers) and then
# the BC bridge as a separate invocation.
for DX in 0.00 0.05 0.10 0.20; do
  # Pad the perturbation magnitude to two digits in the cell name
  # so the cells sort lexically in the right order across magnitudes.
  CELL_TAG=$(printf "dx%03d" $(echo "$DX * 100 / 1" | bc))

  echo "" | tee -a "$LOG"
  echo "########## Perturbation $DX m  ($CELL_TAG) ##########" | tee -a "$LOG"

  # Hybrid + pure-MBD (single invocation, two lambdas).
  run "demo1nn_${CELL_TAG}" \
    --prior demo_1nn \
    --lambda-prior 0 1 \
    --goal-perturbation-xyz $DX 0.00 0.00

  # BC bridge.
  run "bc_${CELL_TAG}" \
    --prior bc \
    --lambda-prior 5 \
    --goal-perturbation-xyz $DX 0.00 0.00
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Headline output: per-cell JSONs in $OUTDIR" | tee -a "$LOG"
echo "Aggregate with experiments/15_unified_hybrid/aggregate_costtime.py" | tee -a "$LOG"
