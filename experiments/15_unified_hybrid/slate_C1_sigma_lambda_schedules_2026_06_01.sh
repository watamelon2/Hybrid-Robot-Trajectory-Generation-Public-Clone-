#!/usr/bin/env bash
# Slate C1 -- sigma-indexed lambda schedules.
#
# Follow-up C1 from the 2026-06-01 backlog research.  The k-indexed
# lambda schedules (linear / inverse_linear / cosine) were measured
# in Slate G (2026-05-21) at lambda=1 on Can and all underperformed
# 'constant' by 12-57 pp.  This slate tests three sigma-INDEXED
# variants (newly wired on this branch into run_can.py) at the
# current headline operating point (lambda=2.0, the Slate 6 GREEN
# cell).
#
# Why sigma-indexed not k-indexed
# -------------------------------
# k-indexed shapes use the iteration index k = 0, ..., K-1 directly.
# But the noise level sigma_k follows a cosine schedule from
# sigma_max to sigma_min, NOT a linear schedule in k.  So a
# k-indexed linear schedule and a sigma-indexed linear schedule
# differ in WHERE the bridge fades in/out relative to the actual
# noise.  Hypothesis: at the headline sigma=0.25 -> 0.05 schedule,
# the meaningful axis is sigma, not k.
#
# Cells
# -----
#   five cells = four lambda-schedule shapes x one operating point
#   plus a 'constant' control at the same op-point.
#
#     C1.1  constant          (control)
#     C1.2  sigma_linear      (strong at high noise, weak at low)
#     C1.3  sigma_inv_linear  (weak at high noise, strong at low)
#     C1.4  sigma_step        (binary on/off at sigma midpoint)
#     C1.5  cosine            (k-indexed, for direct comparison with
#                              the new sigma variants)
#
# Operating point: matches the Phase F GREEN cell at lambda=2.0:
#   prior=demo_1nn, lambda=2.0, K=10, M=16, sigma=[0.05, 0.25],
#   mpc-replans=1, mpc-chunk-steps=60, H=120, n_demos=10, n_rollouts=3.
#
# Acceptance gates
# ----------------
#   STRONG : any sigma-indexed schedule beats 'constant' by >= 5 pp
#            at the lambda=2.0 op-point.  Falsifies the Slate G
#            conclusion that "all schedules hurt" -- at least one
#            sigma-aware shape helps.
#   WEAK   : sigma-indexed schedules within +/- 3 pp of 'constant'.
#            They're not actively destructive (the k-indexed
#            inverse_linear was -57 pp on Can), so retained as
#            tunable hyperparameters but not promoted.
#   FAIL   : all sigma-indexed below 'constant' by >= 5 pp.
#            Confirms Slate G's conclusion robustly: the bridge
#            should NOT be reweighted within the inner loop.
#
# Wall
# ----
#   5 cells x ~12 min/cell = ~1 h on a 10-worker Mac at K=10 H=120.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_C1_sigma_lambda_schedules_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate C1: sigma-indexed lambda schedules at headline op-point" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ----- shared flag bundle -------------------------------------------
COMMON=(
  --task can
  --prior demo_1nn --lambda-prior 2.0
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers "$N_WORKERS"
  --seed 0
  --output-dir "$OUTDIR"
)

run_cell () {
  local SHAPE="$1"
  local TAG="C1_$SHAPE"
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    "${COMMON[@]}" \
    --lambda-schedule "$SHAPE" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for SHAPE in constant sigma_linear sigma_inv_linear sigma_step cosine; do
  run_cell "$SHAPE"
done

# ----- final summary + verdict --------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate C1  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

out = Path("$OUTDIR")
print()
print(f"{'shape':<18}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 45)
rates = {}
for js in sorted(out.glob("unified_*_C1_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    shape = js.stem.split("_C1_", 1)[1] if "_C1_" in js.stem else "?"
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    rates[shape] = sr
    print(f"{shape:<18}  {ns:>7}  {n:>4}  {sr:>6.1f}%")

print()
const = rates.get("constant", None)
if const is None:
    print("VERDICT: cannot compare (constant control missing).")
else:
    sigma_best = max(
        (rates.get(s, 0.0) for s in ("sigma_linear", "sigma_inv_linear", "sigma_step")),
        default=0.0,
    )
    delta = sigma_best - const
    print(f"constant control: {const:.1f}%   best sigma-shape: {sigma_best:.1f}%   delta: {delta:+.1f} pp")
    if delta >= 5.0:
        print("VERDICT: STRONG  -- sigma-indexed schedules help; falsifies Slate G's conclusion.")
    elif abs(delta) < 3.0:
        print("VERDICT: WEAK    -- within noise; sigma-indexed schedules neither help nor hurt.")
    elif delta <= -5.0:
        print("VERDICT: FAIL    -- sigma-indexed schedules hurt; confirms Slate G robustly.")
    else:
        print("VERDICT: AMBIGUOUS in [3, 5] pp band; recommend a 4-seed re-run.")
PY

echo "" | tee -a "$LOG"
echo "Code change shipped on this branch:" | tee -a "$LOG"
echo "  run_can.py --lambda-schedule choices extended with" | tee -a "$LOG"
echo "    {sigma_linear, sigma_inv_linear, sigma_step}." | tee -a "$LOG"
echo "  Schedule construction reads the cosine sigma_schedule and" | tee -a "$LOG"
echo "  builds the lambda multiplier as a function of sigma_k." | tee -a "$LOG"
