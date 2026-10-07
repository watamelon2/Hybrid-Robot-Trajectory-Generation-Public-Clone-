#!/usr/bin/env bash
# Slate A2 -- V_phi value-baseline headline eval (Branch A Task A2).
#
# Follow-up B1 from the 2026-06-01 backlog research.  Closes the only
# remaining Branch A item: tests whether pure-MBD's 0/60 is a
# *variance* problem (H2) or a *capacity* problem (H1).
#
# Hypothesis (docs/branch_plan.md sec A2)
# ---------------------------------------
#   H1 (capacity): the MBD denoiser cannot represent the policy
#                  manifold without a BC warm-start.  V_phi has no
#                  measurable effect.
#   H2 (variance): the per-step cost is too high-variance for MBD's
#                  softmax-over-cost weighting to make progress
#                  without an anchor.  Subtracting V_phi(s_t)
#                  reduces variance without changing expectation;
#                  succ rate climbs above noise.
#
# Test
# ----
# Pure MBD (lambda=0, no BC anchor) on can/ph with V_phi subtracted
# from per-step cost.  Operating point matches the canonical 0/60
# regime so the comparison is apples-to-apples.
#
# Cells
# -----
#   A2.1   pure MBD, lambda=0, sigma=0.05, M=64, K=10, no warm-start,
#          --value-baseline-ckpt experiments/16_value_baseline/results/V_phi.pt
#          60 rollouts (20 demos x 3 rollouts).
#
# Acceptance gates (per branch_plan.md sec A2)
# --------------------------------------------
#   H2 SUPPORTED (success): >= 5/60 = 8.3%.  V_phi is the variance
#                           bottleneck; strengthens §7.1's Pan-et-al.
#                           SNIS-variance argument quantitatively.
#   H1 SUPPORTED (failure): <= 3/60 = 5.0%.  V_phi has no effect;
#                           document as a negative result and keep
#                           the "capacity is the bottleneck for
#                           pure MBD on can/ph" framing.
#
# Stages
# ------
#   stage 0  prereq check (V_phi checkpoint present, or train it).
#   stage 1  scored single-cell evaluation.
#   stage 2  acceptance-gate verdict + integration hints.
#
# Wall
# ----
#   stage 0  ~5 min (data collect + ~30 s train) if V_phi.pt missing
#   stage 1  ~2 h on n-workers=2 (per 16_value_baseline/README.md)
#   stage 2  trivial

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/16_value_baseline/results"
N_WORKERS="${N_WORKERS:-2}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/slate_A2_headline_eval_2026_06_01.log"
date | tee "$LOG"
echo "Slate A2: V_phi value-baseline headline eval" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Stage 0: prereq -- V_phi checkpoint must exist
# ============================================================
V_PHI_CKPT="$OUTDIR/V_phi.pt"
BC_CKPT="$REPO/experiments/03_bc_baseline/results/policy.pt"

if [ ! -f "$BC_CKPT" ]; then
  echo "FATAL: BC checkpoint missing at $BC_CKPT" | tee -a "$LOG"
  echo "       Required for V_phi training data collection." | tee -a "$LOG"
  exit 2
fi

if [ ! -f "$V_PHI_CKPT" ]; then
  echo "[stage 0] V_phi.pt missing; training (~3 min data + ~30 s train)..." | tee -a "$LOG"
  python "$REPO/experiments/16_value_baseline/train.py" \
    --bc-checkpoint "$BC_CKPT" \
    --output "$V_PHI_CKPT" 2>&1 | tee -a "$LOG"
else
  echo "[stage 0] V_phi.pt present at $V_PHI_CKPT" | tee -a "$LOG"
fi

# ============================================================
# Stage 1: scored single-cell eval (A2.1)
# ============================================================
TAG="A2_1_pure_mbd_with_vphi"

if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
  echo "[stage 1] $TAG already done; skipping eval." | tee -a "$LOG"
else
  echo "" | tee -a "$LOG"
  echo "############## STAGE 1: pure MBD + V_phi single-cell eval ##############" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"

  # The cell is taken verbatim from experiments/16_value_baseline/README.md
  # "How to evaluate" section; see branch_plan.md sec A2 for the rationale
  # behind each flag.
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task can \
    --K 10 --mpc-replans 0 --lambda-prior 0 \
    --sigma-max 0.05 --M 64 \
    --prior bc \
    --no-warm-start \
    --value-baseline-ckpt "$V_PHI_CKPT" \
    --n-demos 20 --n-rollouts 3 \
    --n-workers "$N_WORKERS" \
    --seed 0 \
    --output-dir "$OUTDIR" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
fi

# ============================================================
# Stage 2: acceptance-gate verdict
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE 2: verdict ##############" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json, glob
from pathlib import Path

out = Path("$OUTDIR")
candidates = list(out.glob("unified_*_A2_1_pure_mbd_with_vphi.json"))
if not candidates:
    print("[verdict] no result JSON found; eval did not produce output.")
    raise SystemExit(2)
js = candidates[0]
d = json.load(open(js))
n = d.get("n_total", 0)
ns = d.get("n_success", 0)
sr = (ns / n * 100) if n else 0.0
print(f"A2.1 cell: {ns}/{n} = {sr:.1f}%")

# Wilson 95% CI (helper).
def wilson(s, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z / denom) * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
    return (max(0, centre - half), min(1, centre + half))

lo, hi = wilson(ns, n)
print(f"Wilson 95% CI: [{lo*100:.1f}, {hi*100:.1f}] %")

print()
if ns >= 5:
    print("VERDICT: H2 SUPPORTED -- V_phi as control variate helps.")
    print("         Pure MBD is variance-limited; subtracting V_phi(s_t)")
    print("         reduces softmax variance without changing the mean.")
    print("         Strengthens brief §7.1 SNIS-variance argument.")
elif ns <= 3:
    print("VERDICT: H1 SUPPORTED -- V_phi has no measurable effect.")
    print("         Pure MBD is capacity-limited, not variance-limited.")
    print("         Document as negative; keep the 'capacity bottleneck'")
    print("         framing.  Possible follow-ups: scale V_phi training")
    print("         to 2000 trajectories, add --action-noise 0.1 for")
    print("         better state-action coverage (per 16/README.md")
    print("         'Open questions' section).")
else:
    print("VERDICT: AMBIGUOUS -- in the 4/60 band.  Wilson CI overlaps")
    print("         both gates.  Recommend a 4-seed re-run before the")
    print("         result is cited.")
PY

echo "" | tee -a "$LOG"
echo "Next:" | tee -a "$LOG"
echo "  - H2 supported -> update docs/results.md §7.1 with the bound +" | tee -a "$LOG"
echo "    cell number; strengthens the brief's SNIS argument." | tee -a "$LOG"
echo "  - H1 supported -> document as a negative in" | tee -a "$LOG"
echo "    docs/experiments/16_value_baseline.md; close branch_plan A2." | tee -a "$LOG"
