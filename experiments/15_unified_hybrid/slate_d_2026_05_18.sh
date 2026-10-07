#!/usr/bin/env bash
# Slate D — Story-B completion (May 18 overnight).
#
# Three additive blocks through the unified driver at the
# canonical operating point (K=10 R=1 M=16 H=120, 20 demos x 3
# rollouts = 60 rollouts per cell), each closing one concrete gap
# left by the May 17 narrative follow-up:
#
#   D1: BC lambda=0 4-seed pool  (3 cells, ~75 min)
#       Fills the explicit gap from Slate 1's read-out --
#       "BC, which at lambda=0 we have not yet 4-seed pooled".
#       Predicts pool = 11.7 % +/- ~3 pp (matches demo_1nn lam=0
#       pool to within noise per Story B).
#
#   D2: demo_1nn lambda-sweep at sigma=0.25, seed=0   (5 cells, ~2 h)
#       Mirrors the BC lambda-sweep (exp 07) for the new headline
#       prior.  Story-B prediction: 1-NN should be less sensitive
#       to lambda than BC -- a broader optimum and a higher peak.
#       lambda in {0.5, 1, 2, 10, 20}; combines with existing
#       seed=0 cells at lambda=0 (= 11.7 %) and lambda=5 (= 63.3 %).
#
#   D3: demo_1nn sigma-tail at lambda=5, seed=0   (3 cells, ~75 min)
#       Extends Slate 3 past sigma=0.40 to find the failure point
#       of the sigma-invariance claim.  sigma_max in {0.50, 0.75,
#       1.00}; combines with existing seed=0 cells at sigma in
#       {0.05, 0.10, 0.15, 0.25, 0.40} (all 63 - 67 %).
#
# Total: 11 cells, ~4.5 h.  Started 2026-05-18 ~02:30 PT in screen
# "slate_d".  ETA finish ~07:00 PT, well before the 09:30 PT gate.
#
# Branch: feat/zero-train-priors (HEAD = c5e3d47 at launch).

set -euo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_d_2026_05_18"
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
echo "Slate D: BC lam=0 pool + demo_1nn lambda-sweep + demo_1nn sigma-tail" | tee -a "$LOG"
echo "Total ~4.5 h wall" | tee -a "$LOG"

# ----- D1: BC lambda=0 4-seed pool (seeds 1, 2, 3 to combine with seed 0 = C1) -----
echo "##### D1: BC lambda=0 4-seed pool #####" | tee -a "$LOG"
for SEED in 1 2 3; do
  run "D1_bc_lam0_seed${SEED}" \
    --prior bc --lambda-prior 0 --seed "$SEED"
done

# ----- D2: demo_1nn lambda-sweep at sigma=0.25, seed=0 -----
# Pass all 5 lambdas in one invocation -- run_can.py loops internally.
echo "##### D2: demo_1nn lambda-sweep at sigma=0.25 #####" | tee -a "$LOG"
run "D2_1nn_lamsweep" \
  --prior demo_1nn --lambda-prior 0.5 1 2 10 20 --seed 0

# ----- D3: demo_1nn sigma-tail at lambda=5, seed=0 -----
# sigma-max is single-valued, so loop in shell.
echo "##### D3: demo_1nn sigma-tail at lambda=5 #####" | tee -a "$LOG"
for SIGMA in 0.50 0.75 1.00; do
  TAG="D3_1nn_sigma${SIGMA}"
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
    --prior demo_1nn --lambda-prior 5 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== $(date '+%H:%M:%S')  ALL DONE ===" | tee -a "$LOG"
