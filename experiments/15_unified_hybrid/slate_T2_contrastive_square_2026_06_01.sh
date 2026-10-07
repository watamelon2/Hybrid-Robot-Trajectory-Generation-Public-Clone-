#!/usr/bin/env bash
# Slate T2-CONTRASTIVE-SQUARE -- break the Square 8% representation
# ceiling via a learned retrieval embedding.
#
# Future-work item 4 from the Slide 9 conclusion deck (2026-06-01).
# Square's headline is 8.3% (5/60), and Slate G's bank-scaling
# plateau ruled out "more demos" as the fix.  The remaining
# hypothesis is that Square's raw obs vector doesn't separate
# its contact phases (the fingertip-on-peg state and the
# fingertip-just-released state look nearly identical in 46-d
# raw obs but require very different actions).  Training a
# contrastive encoder via InfoNCE on temporally-adjacent obs
# pairs inside each demo should recover phase identity and let
# the retrieval prior actually distinguish which trajectory to
# follow.
#
# Infrastructure already in place
# -------------------------------
#   - train script:   experiments/15_unified_hybrid/train_contrastive_encoder.py
#   - prior wiring:   priors_zerotrain.demo_1nn_contrastive (calls
#                     a saved encoder at retrieval time, projects
#                     the whole bank through it once at startup).
#   - run_can flag:   --prior demo_1nn_contrastive
#                     --contrastive-encoder-path artifacts/...pt
# This slate only chains those together; no new code in the
# planner / prior / training pipeline.
#
# Stages
# ------
#   stage 0  prereq check -- robomimic Square dataset present.
#   stage 1  train encoder on Square at default hparams (60 epochs,
#            batch 512, embed_dim=32, hidden=128, k_max=4, tau=0.1,
#            seed=0).  ~10-15 min on MPS, ~25-40 min on CPU.
#   stage 2  scored hybrid evaluation on Square (60 rollouts, 3
#            lambdas) with the new contrastive prior.
#   stage 3  comparison reference: demo_1nn (raw obs retrieval) on
#            Square at the same operating point.  This is the
#            apples-to-apples sanity that the gain comes from the
#            encoder, not from a protocol drift.
#
# Acceptance gates
# ----------------
#   STRONG  : contrastive succ rate > 15% at any of the three
#             lambdas.  Confirms the bottleneck WAS retrieval
#             representation; encoder unlocks Square.
#   WEAK    : contrastive succ rate in [10%, 15%] at best lambda.
#             Encoder helps but more demos / a better encoder are
#             needed.  Worth a follow-on with action-history
#             conditioning (A1 + C1) or a bigger embed_dim.
#   FAIL    : contrastive succ rate <= 10% at all lambdas, OR <= 2
#             pp above the demo_1nn raw-obs reference.  The
#             representation is not the bottleneck after all;
#             bridge / cost function need to be revisited.
#
# Wall-clock
# ----------
#   stage 1  ~15 - 40 min  (one CPU/MPS train run)
#   stage 2  ~2 - 3 h      (3 lambdas, ~30 - 60 min each at H=400)
#   stage 3  ~1 h          (1 lambda reference)
#   Total: ~3 - 5 h.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ARTIFACTS="${ARTIFACTS:-$REPO/artifacts}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_contrastive_square_2026_06_01"
ENCODER_CKPT="$ARTIFACTS/contrastive_encoder_square.pt"
N_WORKERS="${N_WORKERS:-10}"
ENCODER_DEVICE="${ENCODER_DEVICE:-cpu}"  # set to mps on macOS for ~3x faster train
mkdir -p "$OUTDIR" "$ARTIFACTS"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-CONTRASTIVE-SQUARE: learned-embedding retrieval for Square" | tee -a "$LOG"
echo "ENCODER_CKPT    = $ENCODER_CKPT" | tee -a "$LOG"
echo "ENCODER_DEVICE  = $ENCODER_DEVICE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Stage 0: prereq check
# ============================================================
SQUARE_BANK="$REPO/experiments/15_unified_hybrid/data/square_ph.hdf5"
if [ ! -f "$SQUARE_BANK" ]; then
  # Common fallback location.
  SQUARE_BANK_ALT="$(find "$REPO" -name 'square*.hdf5' -path '*ph*' 2>/dev/null | head -1)"
  if [ -z "$SQUARE_BANK_ALT" ]; then
    echo "FATAL: Square /ph hdf5 not found.  Sync the robomimic" | tee -a "$LOG"
    echo "       Square dataset before running this slate." | tee -a "$LOG"
    exit 2
  fi
  echo "[stage 0] Using fallback Square bank: $SQUARE_BANK_ALT" | tee -a "$LOG"
  SQUARE_BANK="$SQUARE_BANK_ALT"
fi
echo "[stage 0] Square bank: $SQUARE_BANK" | tee -a "$LOG"

# ============================================================
# Stage 1: train contrastive encoder on Square
# ============================================================
if [ ! -f "$ENCODER_CKPT" ]; then
  echo "" | tee -a "$LOG"
  echo "############## STAGE 1: train contrastive encoder ##############" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  encoder train starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/train_contrastive_encoder.py" \
    --task square \
    --embed-dim 32 --hidden-dim 128 \
    --epochs 60 --batch-size 512 \
    --k-max 4 --tau 0.1 \
    --lr 1e-3 --weight-decay 1e-5 \
    --seed 0 \
    --device "$ENCODER_DEVICE" \
    --output-path "$ENCODER_CKPT" \
    --action-history-len 0 \
    --verbose 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  encoder train done ===" | tee -a "$LOG"
else
  echo "[stage 1] encoder ckpt already present at $ENCODER_CKPT; skipping train." | tee -a "$LOG"
fi

# ============================================================
# Stage 2: scored hybrid with contrastive prior
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE 2: hybrid + contrastive prior (scored) ##############" | tee -a "$LOG"

# Operating point matches the Square headline in slate_g/h (the
# Slate F1 sweet spot): demo_1nn-style retrieval, lambda in {0, 1, 2},
# K=10, sigma=0.25, sigma-min=0.05, H=400, n_demos=10 x n_rollouts=6
# = 60 rollouts.
SQUARE_COMMON=(
  --task square
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 400
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 6
  --n-workers "$N_WORKERS"
  --seed 0
  --output-dir "$OUTDIR"
)

run_cell () {
  local PRIOR="$1"
  local TAG="$2"
  shift 2
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[stage 2] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${SQUARE_COMMON[@]}" \
    --prior "$PRIOR" \
    "$@" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# 2a. Contrastive prior at three lambdas (single invocation).
run_cell demo_1nn_contrastive "S_contrastive_lamsweep" \
  --contrastive-encoder-path "$ENCODER_CKPT" \
  --lambda-prior 0 1 2

# ============================================================
# Stage 3: comparison reference (raw-obs retrieval)
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE 3: raw-obs reference ##############" | tee -a "$LOG"

# 3a. Raw demo_1nn baseline (only re-run if not already present
#     elsewhere; if you have slate_g_2026_05_21 results, the
#     numbers are directly comparable -- this stage is for
#     same-protocol apples-to-apples).
run_cell demo_1nn "S_raw_obs_lamsweep" --lambda-prior 0 1 2

# ============================================================
# Stage 4: final summary
# ============================================================
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-CONTRASTIVE-SQUARE  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

out = Path("$OUTDIR")
print()
print(f"{'prior':<25}  {'lambda':>6}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 60)
rows = []
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    p = d.get("prior", "?")
    lam = d.get("lambda_prior", "?")
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    print(f"{p:<25}  {lam:>6}  {ns:>7}  {n:>4}  {sr:>6.1f}%")
    rows.append((p, lam, sr))

# Gate hint.
best_contrastive = max((sr for p, _, sr in rows if p == "demo_1nn_contrastive"), default=0)
best_raw = max((sr for p, _, sr in rows if p == "demo_1nn"), default=0)
print()
print(f"best contrastive: {best_contrastive:.1f}%   best raw-obs: {best_raw:.1f}%")
gain = best_contrastive - best_raw
print(f"gain over raw-obs: {gain:+.1f} pp")
if best_contrastive > 15.0 and gain > 2.0:
    print("VERDICT: STRONG  -- representation was the bottleneck; encoder unlocks Square.")
elif best_contrastive > 10.0:
    print("VERDICT: WEAK    -- encoder helps; more capacity / data may help further.")
else:
    print("VERDICT: FAIL    -- representation is not the bottleneck; revisit bridge / cost.")
PY

echo "" | tee -a "$LOG"
echo "Next:" | tee -a "$LOG"
echo "  - On STRONG, update Slide 5 Square cell + Slide 8 'Square is" | tee -a "$LOG"
echo "    representation-limited' lesson with the new ceiling." | tee -a "$LOG"
echo "  - On FAIL, the structural fix is elsewhere; the contrastive" | tee -a "$LOG"
echo "    branch can be archived." | tee -a "$LOG"
