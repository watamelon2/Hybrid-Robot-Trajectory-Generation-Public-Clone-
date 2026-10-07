#!/usr/bin/env bash
# Slate DemoScale -- retrieval-prior bank-size scaling law on Can.
#
# Companion to docs/adaptation/03_demo_scaling_spec.md.  Tests
# Corollary 1 of docs/theory/02_snis_bound.md: the variance-reduction
# ratio rho(bar_tau) of the anchored SNIS proposal saturates at the
# bank-coverage threshold N_b^cov.  Predicts a CONCAVE-SATURATING
# retrieval success curve in N_b, in contrast to BC's monotone
# non-saturating curve (BC retraining deferred to a follow-up).
#
# Cells (all demo_1nn at lambda=1 sigma=0.25 K=10 M=16 R=1 H=120,
# 20 demos x 3 rollouts = 60 evaluations, seed=0):
#
#   N_b = 5            (mode-undercoverage; sharply below plateau)
#   N_b = 10           (rising)
#   N_b = 20           (near canonical-headline regime)
#   N_b = 50
#   N_b = 100
#   N_b = 200          (full bank = the 73.8% canonical baseline)
#
# 6 cells, ~50 min/cell ~ 5 h wall.
#
# Important: --n-demos=20 is FIXED (controls the eval-rollout count =
# 60 rollouts/cell).  --prior-bank-size controls the RETRIEVAL bank
# size independently.  See docs/adaptation/03_demo_scaling_spec.md
# sec 2 for the design rationale.
#
# Started 2026-05-25.  Branch: feat/adapt-demo-scaling.

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_demoscale_2026_05_25"
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

date | tee "$LOG"
echo "Slate DemoScale: bank-size scaling law for demo_1nn on Can" | tee -a "$LOG"
echo "Theory companion: docs/theory/02_snis_bound.md Corollary 1" | tee -a "$LOG"
echo "Spec: docs/adaptation/03_demo_scaling_spec.md" | tee -a "$LOG"
echo "Cells: N_b in {5, 10, 20, 50, 100, 200}, 6 cells, ~5 h wall" | tee -a "$LOG"
echo "Eval set: --n-demos=20 x --n-rollouts=3 = 60 evals/cell (fixed)" | tee -a "$LOG"

for NB in 5 10 20 50 100 200; do
  run "Nb${NB}" --prior-bank-size "$NB"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
echo "Headline output: per-cell JSONs in $OUTDIR with prior_bank_size in params" | tee -a "$LOG"
