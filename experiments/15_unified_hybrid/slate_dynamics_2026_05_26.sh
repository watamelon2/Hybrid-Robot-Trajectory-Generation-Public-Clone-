#!/usr/bin/env bash
# Slate Dynamics -- dynamics (mass-scale D1) adaptation experiment on Can.
#
# Spec:        docs/adaptation/02_dynamics_spec.md
# Theory:      docs/theory/01_soft_control.md sec 1 (p^model factor
#              of the soft-control target -- this slate perturbs f),
#              03_pareto_bias_variance.md.
# Branch:      feat/adapt-dynamics.
# Worker-prop patch: commits 05edaf6 + 716630a (cherry-picked from
#              feat/adapt-cost-time; enables this slate to fire).
# Companion slate: slate_costtime_2026_05_25 (negative result;
#              identified the fixed-anchor mechanism on Can/ph).
#
# Tests whether the hybrid (demo_1nn + MBD) succeeds, fails, or
# degrades-with-mechanism under dynamics shift.  D2 friction ladder
# and the full 5-lambda heatmap from spec sec 2.1 are out of scope
# for this minimal D1' variant -- single canonical lambda per
# method, matching the cost-time slate's structure (12 cells total).
#
# Methods (canonical operating point from Slate E peak):
#
#   hybrid       --prior demo_1nn  --lambda-prior 1  --sigma-max 0.25
#   pure-MBD     --prior demo_1nn  --lambda-prior 0  --sigma-max 0.25
#                (lambda=0 zeroes the prior; demo_1nn choice is
#                 irrelevant in this case but avoids BC-checkpoint load)
#   BC bridge    --prior bc        --lambda-prior 5  --sigma-max 0.25
#                (canonical BC peak from Slate E E2 lambda-curve)
#
# Mass scales: {0.7, 1.0, 1.3, 1.5} on robot-link bodies (per the
# _ROBOT_LINK_BODY_PREFIXES registry: robot0_link*, robot0_right_hand,
# gripper0_*).  Scales body_mass AND body_inertia proportionally.
#
# Per cell: 20 demos x 3 rollouts = 60 evaluations, single seed=0.
# (Pool later if a signal is real; see spec sec 4 for the pool plan.)
#
# Wall-clock estimate (analogous to cost-time, which landed 8 cells
# in 5.5 h):
#   - hybrid+pure-MBD share one invocation per mass scale
#     (--lambda-prior 0 1 sweeps both inside the unified driver):
#     ~50 min per (mass x {0,1}) invocation, x 4 mass scales.
#   - BC: ~30 min per invocation, x 4 mass scales.
#   - Total: ~3.3 h + 2 h = ~5.5 h.
#
# Decision rules (post-hoc analysis):
#
#   Outcome A (mirror of cost-time negative):
#     hybrid-vs-BC gap NARROWS with |mass-1.0|.  The fixed-anchor
#     mechanism extends to dynamics.  Unifies adaptation story
#     into ONE finding: "the prior's retrieval bottleneck binds
#     before the cost-side mechanism can act".
#
#   Outcome B (positive, dynamics succeeds where cost-time failed):
#     hybrid-vs-BC gap WIDENS or stays flat with |mass-1.0|.  The
#     fixed-anchor mechanism is goal-pose-specific; mass/friction
#     are *structurally* outside the prior's obs vocab and so are
#     not constrained by the same retrieval bottleneck.  This is
#     the sharper, more interesting finding -- isolates the failure
#     to retrieve-from-obs perturbations.
#
#   Outcome C (universal collapse):
#     all 3 methods drop monotonically with no gap to BC.  Mass
#     shift is harder than Can's manifold can absorb at any prior.
#     Less interesting but still a publishable bound.
#
# Start: 2026-05-26 noon.  Run in screen session "slate_dynamics"
# from the HRTG-dynamics-tmp worktree (this branch).

set -euo pipefail

REPO="/private/tmp/HRTG-dynamics-tmp"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_dynamics_2026_05_26"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

# Common args (must match docs/adaptation/02_dynamics_spec.md sec 2.1
# AND the cost-time slate's operating point for cross-slate
# comparability).
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
echo "Slate Dynamics: mass-scale adaptation for hybrid vs BC vs pure-MBD on Can" | tee -a "$LOG"
echo "Spec: docs/adaptation/02_dynamics_spec.md (D1 mass ladder, minimal variant)" | tee -a "$LOG"
echo "Theory: docs/theory/01_soft_control.md sec 1, 03_pareto_bias_variance.md" | tee -a "$LOG"
echo "Worker-prop patches: 05edaf6 + 716630a (cherry-picked from feat/adapt-cost-time)" | tee -a "$LOG"
echo "Companion: slate_costtime_2026_05_25 (negative; fixed-anchor mechanism)" | tee -a "$LOG"
echo "Mass scales: {0.7, 1.0, 1.3, 1.5} on robot-link bodies + gripper" | tee -a "$LOG"
echo "Methods: hybrid (demo_1nn,lam=1) | pure-MBD (lam=0) | BC (bc,lam=5)" | tee -a "$LOG"
echo "Eval set: 20 demos x 3 rollouts = 60 evals/cell, seed=0" | tee -a "$LOG"

# Iterate over the four mass scales.  For each, fire the two-lambda
# sweep (hybrid + pure-MBD share workers) and then the BC bridge as
# a separate invocation.  mass=1.0 is the canonical control cell
# (no perturbation applied; matches Slate E E2 protocol).
for MASS in 1.0 0.7 1.3 1.5; do
  # Pad the mass scale into a 3-digit tag (e.g., m070, m100, m130, m150)
  # so cells sort lexically in a meaningful order.
  CELL_TAG=$(printf "m%03d" $(echo "$MASS * 100 / 1" | bc))

  echo "" | tee -a "$LOG"
  echo "########## Mass scale $MASS  ($CELL_TAG) ##########" | tee -a "$LOG"

  # Hybrid + pure-MBD (single invocation, two lambdas).
  run "demo1nn_${CELL_TAG}" \
    --prior demo_1nn \
    --lambda-prior 0 1 \
    --mass-scale "$MASS"

  # BC bridge.
  run "bc_${CELL_TAG}" \
    --prior bc \
    --lambda-prior 5 \
    --mass-scale "$MASS"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Headline output: per-cell JSONs in $OUTDIR" | tee -a "$LOG"
echo "Aggregate manually or write aggregate_dynamics.py once cells land." | tee -a "$LOG"
