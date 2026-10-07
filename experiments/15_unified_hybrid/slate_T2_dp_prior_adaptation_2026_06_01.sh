#!/usr/bin/env bash
# Slate T2-DP-PRIOR -- does the 78x adaptation story generalise
# beyond a retrieval prior?
#
# Future-work item 3 from the Slide 9 conclusion deck (2026-06-01).
# The headline adaptation result on Slide 7 reads: "78x speedup over
# a retrained baseline (1 s vs 78 s, 5 demos)".  The 1 s number is a
# *retrieval* story -- adding 5 demos to a bank is essentially free.
# The audience-facing question is whether the same speedup holds when
# the prior is a trained neural-network policy (DP) rather than the
# retrieval lookup -- which is the more common modern baseline.
#
# Two interpretations of "adapt with a DP prior" worth distinguishing
# -----------------------------------------------------------------
#   I1.  Re-train DP on the new (heavier-can) demos.  Costs ~minutes
#        (much faster than BC because DP starts from a strong base),
#        but obviously not 1 s.
#   I2.  Keep DP frozen, rely on the analytical cost gradient inside
#        the inner loop to handle the dynamics shift WITHOUT touching
#        the prior weights.
#
# This slate runs I2 -- the cheaper of the two, and the one that maps
# directly to the "1 s adaptation" framing.  If I2 holds up at
# mass=1.3 with a frozen DP prior + the analytical cost gradient,
# the 78x story is generic (cost gradient does the adaptation work).
# If I2 collapses at mass=1.3, the 78x story is retrieval-specific
# and I1 needs to be the follow-on.
#
# Cells
# -----
# Six cells = (canonical mass=1.0 + perturbed mass=1.3) x (3 lambdas).
# Score-mode = pull throughout (the freeze-anchor fix from Slide 8;
# WITHOUT it the trained DP prior loses 33 pp at K=10).
#
#   D1 (mass=1.0, lambda in {0.5, 1.0, 2.0}, ~3 h)
#     Baseline reference.  Expect:
#       - lambda=0.5 ~ Phase F GREEN cell (73% on Can/ph).
#       - lambda=2.0 ~ Phase F lift to 87%.
#       - lambda=0.0 ~ pure DP prior baseline.
#     If any of these are below the Phase F numbers by > 5 pp, the
#     scoring protocol drifted and we abort before running D2.
#
#   D2 (mass=1.3, lambda in {0.5, 1.0, 2.0}, ~3 h)
#     The adaptation test.  Expect:
#       - if lambda=2.0 mass=1.3 stays within 10 pp of lambda=2.0
#         mass=1.0, the cost gradient is adapting -- 78x story IS
#         generic.
#       - if lambda=2.0 mass=1.3 drops below 20%, the cost gradient
#         is not enough; DP + cost cannot self-adapt -- 78x story
#         IS retrieval-specific.
#
# Notes
# -----
# - DP checkpoint: v_theta_dp_tobs2_h16_v3b_ep0149.pt -- this is the
#   checkpoint Phase F's 73.3% headline (K=10 lam=0.5, 44/60) was
#   measured on: the epoch-curve PEAK per
#   docs/experiments/08_flow_matching_prior.md.  An earlier version of
#   this slate defaulted to ep0199 and CLAIMED it was the same prior
#   used in Phase F -- that was WRONG.  ep0199 is 50 epochs past the
#   hybrid peak; its 80% STANDALONE number does not transfer to the
#   K=10 lam=0.5 hybrid, where ep0199 lands at 60% (36/60) and trips
#   the PHASE_F_REF=0.733 gate.  See the 2026-06-03 root-cause writeup.
#   Override via DP_CKPT env.
# - Score-mode = pull is REQUIRED for trained priors.  Without it,
#   K=10 actively *destroys* the DP prior (Slide 8 lesson).
# - Operating point exactly matches Phase F (H=16, mpc-replans=49,
#   mpc-chunk-steps=8, n-steps=100, sigma=0.25) so D1 cells are
#   directly comparable to Phase F numbers.
# - 10 train + 50 test rollouts (n-demos=10, n-test-rollouts=50)
#   per cell = 60 rollouts at the DP apples-to-apples sample budget.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DP_CKPT="${DP_CKPT:-$REPO/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0149.pt}"
RUN_D2="${RUN_D2:-1}"   # set RUN_D2=0 to stop after the D1 ep0149 validation
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_T2_dp_prior_adaptation_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate T2-DP-PRIOR: does the 78x story generalise to a trained DP prior?" | tee -a "$LOG"
echo "DP_CKPT  = $DP_CKPT" | tee -a "$LOG"
echo "DEVICE   = $DEVICE" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ----- prereq: DP checkpoint must exist (~250 MB) ---------------------
if [ ! -f "$DP_CKPT" ] && [ ! -L "$DP_CKPT" ]; then
  echo "FATAL: DP checkpoint missing at $DP_CKPT" | tee -a "$LOG"
  echo "  Sync from Drive via" | tee -a "$LOG"
  echo "    python $REPO/experiments/08_flow_matching_prior/sync_ckpts_from_drive.py \\" | tee -a "$LOG"
  echo "      --pattern v3b_ep0149" | tee -a "$LOG"
  exit 2
fi
REAL_CKPT="$(readlink -f "$DP_CKPT" 2>/dev/null || python -c "import os; print(os.path.realpath('$DP_CKPT'))")"
echo "DP ckpt real path: $REAL_CKPT" | tee -a "$LOG"

# ----- shared flag bundle (Phase F operating point) -------------------
COMMON=(
  --task can --prior flow
  --prior-checkpoint "$DP_CKPT"
  --score-mode pull
  --H 16
  --mpc-replans 49 --mpc-chunk-steps 8
  --K 10 --M 16
  --n-demos 10 --n-rollouts 1
  --n-test-rollouts 50 --test-start-seed 100000
  --max-env-steps 400
  --prior-n-steps 100
  --sigma-max 0.25 --sigma-min 0.05
  --n-workers "$N_WORKERS"
  --device "$DEVICE"
  --seed 0
  --output-dir "$OUTDIR"
)

run_cell () {
  local MASS="$1"
  local LAM="$2"
  local MASS_TAG="${MASS//./p}"
  local LAM_TAG="${LAM//./p}"
  local TAG="D_dp_pull_mass${MASS_TAG}_lam${LAM_TAG}"

  if ls "$OUTDIR"/unified_flow_pull_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[T2-DP] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi

  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --mass-scale "$MASS" \
    --lambda-prior "$LAM" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# ----- D1: canonical (mass=1.0) reference ----------------------------
echo "##### D1: canonical reference, mass=1.0 #####" | tee -a "$LOG"
for LAM in 0.5 1.0 2.0; do
  run_cell 1.0 "$LAM"
done

# Sanity gate.  If D1 lambda=0.5 came in > 5 pp below the Phase F
# 73% reference, abort: the protocol drifted and D2 numbers are
# not interpretable.
python - <<PY 2>&1 | tee -a "$LOG"
import json, glob, sys
from pathlib import Path
out = Path("$OUTDIR")
ref_files = list(out.glob("unified_flow_pull_*_D_dp_pull_mass1p0_lam0p5.json"))
if not ref_files:
    print("[T2-DP gate] D1 lam=0.5 result missing -- cannot gate.")
    sys.exit(0)        # let D2 proceed; partial run.
d = json.load(open(ref_files[0]))
test_sr = d.get("test_succ_rate", float("nan"))
print(f"[T2-DP gate] D1 mass=1.0 lam=0.5 test_succ_rate = {test_sr*100:.1f}%")
PHASE_F_REF = 0.733
DELTA_PP = 0.05
if test_sr < PHASE_F_REF - DELTA_PP:
    print(f"[T2-DP gate] FAIL: > {DELTA_PP*100:.0f} pp below Phase F headline "
          f"({PHASE_F_REF*100:.1f}%).  Aborting before D2.")
    sys.exit(2)
print("[T2-DP gate] OK: within Phase F band.")
PY

# ----- D2: perturbed (mass=1.3), DP prior frozen ---------------------
# Gated behind RUN_D2 so the ep0149 protocol-validation pass can run D1
# only (the 2026-06-03 follow-up batch validates D1 first; D2 is the
# separate adaptation test, queued once D1 confirms ~73.3%).
if [ "$RUN_D2" = "1" ]; then
  echo "" | tee -a "$LOG"
  echo "##### D2: perturbed, mass=1.3, DP prior FROZEN #####" | tee -a "$LOG"
  for LAM in 0.5 1.0 2.0; do
    run_cell 1.3 "$LAM"
  done
else
  echo "" | tee -a "$LOG"
  echo "##### D2 SKIPPED (RUN_D2=0): D1-only ep0149 validation pass #####" | tee -a "$LOG"
fi

# ----- final summary -------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate T2-DP-PRIOR  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUTDIR")
print()
print(f"{'mass':>6}  {'lambda':>6}  {'test_n_succ':>11}  {'test_n':>6}  {'test_sr':>8}")
print("-" * 50)
for js in sorted(out.glob("unified_flow_pull_*_D_dp_pull_*.json")):
    d = json.load(open(js))
    p = d.get("params", {})
    mass = d.get("mass_scale", "?")
    lam = d.get("lambda_prior", "?")
    print(f"{mass:>6}  {lam:>6}  "
          f"{d.get('test_n_succ', 0):>11}  "
          f"{d.get('test_n', 0):>6}  "
          f"{d.get('test_succ_rate', 0)*100:>7.1f}%")

print()
print("Interpretation:")
print("  - If test_sr @ mass=1.3 stays within 10 pp of mass=1.0 at the")
print("    same lambda, the cost gradient is adapting -- 78x story IS")
print("    generic (does NOT depend on retrieval).")
print("  - If test_sr @ mass=1.3 drops below 20%, the cost gradient")
print("    is not enough; DP + cost cannot self-adapt -- 78x story")
print("    IS retrieval-specific (follow-on: I1 = fine-tune DP)." )
PY

echo "" | tee -a "$LOG"
echo "Next step depending on D2 outcome:" | tee -a "$LOG"
echo "  - Generic   -> update Slide 7 right panel to mention 'holds" | tee -a "$LOG"
echo "                 under DP prior too'." | tee -a "$LOG"
echo "  - Retrieval-specific -> queue follow-on slate that does I1" | tee -a "$LOG"
echo "                          (fine-tune DP on 5 mass=1.3 demos)" | tee -a "$LOG"
echo "                          and compare its wall-clock vs the 78s BC." | tee -a "$LOG"
