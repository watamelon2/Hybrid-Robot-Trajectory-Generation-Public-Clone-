#!/usr/bin/env bash
# Slate G-aux -- "crack Square" parallel companion to Slate G.
# Runs concurrently in screen `slate_g_aux`, uses n-workers=5 so
# 10 (slate_g) + 5 (aux) = 15 of 16 cores -- safe margin of 1.
#
# Motivation:
#   Slate F (May 20) verdict: Square is mechanism-limited with hard
#   1-NN at sigma in {0.05, 0.25, 0.40, 1.0}.  Peak 3.3 % at lam=1
#   sigma=0.25, F2 sigma-tail had only 4 data points (big gaps),
#   and we never tried a non-degenerate non-1NN prior on Square.
#
#   Slate G G4 tests action-history on Square (A1 axis) but keeps
#   the prior as hard 1-NN.  Slate H drops Square entirely.  So
#   the "prior identity" axis on Square is unexplored.
#
#   Two tracks here exhaust the obvious remaining single-axis
#   ablations on Square at the F1 sweet spot:
#
#     AUX1 -- sigma-tail high resolution
#       Fill F2's gap between sigma=0.05 and sigma=0.25; F2 only
#       measured 4 points.  Hypothesis: sigma in {0.10, 0.15, 0.20,
#       0.30} resolves whether the optimum is sharper than 0.25 or
#       broader.  If the peak shifts left, mechanism is fine and
#       we were just on the wrong sigma.  If success stays <= 5 %
#       across the band, the F1 verdict (mechanism-limited)
#       hardens.
#
#     AUX2 -- soft-NN beta-sweep on Square
#       Slate G G2 ran this on Can.  Square is *more* multi-modal
#       than Can (square nut + peg geometry has more discrete
#       contact phases), so the bias-variance optimum might sit at
#       a different beta.  4 cells covering the same span as G2's
#       interior: beta in {0.5, 1, 2, 5}.  We skip beta=0 (uniform
#       k-NN; known terrible on hard tasks per Slate D) and the
#       beta>>1 limit (= hard 1-NN, already in F1).
#
# Total 8 cells, 60 rollouts each, n-workers=5.  At n-workers=10
# each Square cell took ~22 min (Slate F).  Halved workers -> wall
# scales roughly 1.7x (workers aren't pure speedup -- env construct
# + cache-warm overhead amortise), so ~38 min/cell.  8 cells ~
# 5.0 h wall.  Slate G ETA ~10:30 PDT (still ~7 h out from launch);
# slate_g_aux finishes comfortably inside that window.
#
# No precondition (does not depend on Slate G state).
# Output:  results/slate_g_aux_2026_05_21/

set -u
REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_g_aux_2026_05_21"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"
cd "$REPO"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj

date | tee "$LOG"
echo "Slate G-aux: Square sigma-tail (AUX1) + Square soft-NN beta (AUX2)" | tee -a "$LOG"
echo "8 cells, n-workers=5, runs PARALLEL to Slate G" | tee -a "$LOG"

# ----- AUX1: Square sigma-tail high-resolution at lambda=1 -----------
echo "##### AUX1: Square sigma-tail high-res (1-NN) #####" | tee -a "$LOG"
for SIGMA in 0.10 0.15 0.20 0.30; do
  SIGMA_TAG="${SIGMA/./p}"
  TAG="AUX1_square_1nn_lam1_sigma${SIGMA_TAG}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max "$SIGMA" --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 5 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_1nn --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

# ----- AUX2: Square soft-NN beta-sweep at lambda=1 sigma=0.25 --------
# Mirror of Slate G G2 (which is on Can), but on Square at the F1
# sweet spot.  At beta>>1 the soft-NN -> hard 1-NN (which already
# hit 3.3 % in F1, so we don't re-run that).  At beta=0 the
# soft-NN -> uniform k-NN k=5 (variance-dominated; known weak).
# The four cells span the interesting middle.
echo "##### AUX2: Square soft-NN beta-sweep #####" | tee -a "$LOG"
for BETA in 0.5 1 2 5; do
  BETA_TAG="${BETA/./p}"
  TAG="AUX2_square_softnn_k5_b${BETA_TAG}"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task square \
    --mpc-replans 1 --mpc-chunk-steps 60 \
    --K 10 --M 16 --H 120 \
    --sigma-max 0.25 --sigma-min 0.01 \
    --n-demos 20 --n-rollouts 3 \
    --n-workers 5 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" \
    --prior demo_softnn --knn-k 5 --retrieval-beta "$BETA" \
    --lambda-prior 1 --seed 0 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
done

echo "=== ALL DONE  $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee -a "$LOG"
