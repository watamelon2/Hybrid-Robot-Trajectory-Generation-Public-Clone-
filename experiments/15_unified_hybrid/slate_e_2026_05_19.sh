#!/usr/bin/env bash
# Slate E -- Story-B headline-tightening (May 19 overnight).
#
# Three additive blocks through the unified driver at the canonical
# operating point (K=10 R=1 M=16 H=120, 20 demos x 3 rollouts = 60
# rollouts per cell).  Each block closes one explicit follow-up
# called out in docs/results.md "Slate D" section (lines ~441-450)
# and the May-18 BRIEF.md.
#
#   E1: 4-seed pool at lambda=1 sigma=0.25 demo_1nn  (3 cells, ~105 min)
#       Highest priority follow-up.  Puts a CI on the May-18 single-
#       seed peak of 75.0 % at this exact config.  Combines with the
#       existing seed=0 cell (slate_d_2026_05_18/
#       unified_demo_1nn_lam-1.0_K-10_R-1_D2_1nn_lamsweep.json = 45/60).
#       Story-B falsifiable prediction: pool CI lower bound >= 63%
#       (above the bridge-dominated plateau) -- otherwise the
#       "hybrid synergy" claim collapses to seed-noise.
#
#   E2: BC lambda-sweep companion at sigma=0.25 seed=0  (5 cells in
#       one sweep invocation, ~175 min).  lambda in {0.5, 1, 2, 10, 20}.
#       Combines with existing BC seed=0 cells at lambda=0 (= 11.7 %,
#       warmstart_ablation C1) and lambda=5 (= 41.7 %, phase 2 K=10 R=1).
#       Mirrors D2 for the BC prior.  Story-B prediction: BC peaks at
#       lambda >= 5 (smoothed targets need stronger pull), while 1-NN
#       peaks at lambda=1 (clean targets need weaker pull).  If both
#       priors peak at the same lambda, the "prior-specific tuning"
#       claim weakens.
#
#   E3: demo_1nn sigma-tail at lambda=1 seed=0  (3 cells, ~105 min)
#       Tests whether the May-17/18 sigma-invariance claim
#       (sigma in [0.05, 1.00] all 63-68% at lambda=5) survives at
#       the new lambda=1 optimum.  sigma_max in {0.05, 0.40, 1.00}
#       gives a low/mid/high probe.  Combines with the existing
#       slate_d cell at sigma=0.25 lambda=1 = 45/60.
#
# Total: 11 cells, ~6.5 h wall.  Started 2026-05-19 ~02:40 PT in
# screen "slate_e".  ETA finish ~09:00 PT, 1 h before the 10:00
# gate.
#
# Branch: feat/zero-train-priors (HEAD = 3041466 at launch).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_e_2026_05_19"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

COMMON=(
  --task can
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.01
  --n-demos 20 --n-rollouts 3
  --n-workers 10
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
echo "Slate E: CI-on-peak (E1) + BC lambda companion (E2) + sigma-tail at lambda=1 (E3)" | tee -a "$LOG"
echo "Total ~6.5 h wall" | tee -a "$LOG"

# ----- E1: 4-seed pool at lambda=1 sigma=0.25 demo_1nn -----
# Seeds {1, 2, 3} to combine with the existing seed=0 = 45/60 cell.
echo "##### E1: 4-seed pool at lambda=1 sigma=0.25 demo_1nn #####" | tee -a "$LOG"
for SEED in 1 2 3; do
  run "E1_1nn_lam1_seed${SEED}" \
    --prior demo_1nn --lambda-prior 1 --seed "$SEED"
done

# ----- E2: BC lambda-sweep companion at sigma=0.25 seed=0 -----
# Single sweep invocation; run_can.py loops internally and writes
# one JSON per lambda.
echo "##### E2: BC lambda-sweep companion #####" | tee -a "$LOG"
run "E2_bc_lamsweep" \
  --prior bc --lambda-prior 0.5 1 2 10 20 --seed 0

# ----- E3: demo_1nn sigma-tail at lambda=1 seed=0 -----
# sigma-max is single-valued in run_can.py, so loop in shell.
echo "##### E3: demo_1nn sigma-tail at lambda=1 #####" | tee -a "$LOG"
for SIGMA in 0.05 0.40 1.00; do
  TAG="E3_1nn_lam1_sigma${SIGMA}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max "$SIGMA" --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 10 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
