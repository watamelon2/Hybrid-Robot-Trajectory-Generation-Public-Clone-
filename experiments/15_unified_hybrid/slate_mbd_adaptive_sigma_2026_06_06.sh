#!/usr/bin/env bash
# Slate AD (follow-up #2) -- adaptive sigma_demo for off-distribution
# rescue (targets the NEGATIVE §6c #1/#8 result).
#
# WHY: at a calibrated static sigma_demo the gate does NOT rescue
# mass-2.5 / friction-3.0 (both product and max-gate ~0%, gate_frac
# stuck ~0.3).  A large static sigma would engage the gate but wreck
# on-distribution.  The new --demo-sigma-mode=adaptive shrinks the demo
# Gaussian (-> gate fires) exactly when the model's BEST rollout is worse
# than the demo (off-dist) and widens it (-> trust model) on-dist.
#
# Test: does adaptive (a) preserve on-dist (canonical Can) AND (b) lift
# the off-dist cells above the static baseline?  Compares static vs
# adaptive for product & max-gate on canonical + mass2.5 + fric3.0, plus
# a beta sensitivity probe.  Calibrated sigma_demo (auto).  10 cells.
#
#   screen -dmS mbd_ad bash experiments/15_unified_hybrid/slate_mbd_adaptive_sigma_2026_06_06.sh

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_adaptive_sigma_2026_06_06"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
if [ -z "${SIGMA_CAL:-}" ]; then
  SIGMA_CAL="$(python "$REPO/experiments/15_unified_hybrid/pick_calibrated_sigma.py" \
    --results-dir "$GATED_DIR" --fallback 1 2>>"$LOG")"
fi
echo "Slate AD  adaptive sigma_demo  SIGMA_CAL=$SIGMA_CAL" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3 --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn --combine-mode tempered --lambda-prior 1
  --sigma-demo "$SIGMA_CAL" --log-gate-diagnostic --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[ad] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# --- on-distribution (canonical Can): adaptive must NOT hurt ---
run_cell "AD_canon_prod_static" --temper-p 0   --demo-sigma-mode const
run_cell "AD_canon_prod_adapt"  --temper-p 0   --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0

# --- off-distribution mass 2.5: does adaptive rescue? ---
run_cell "AD_mass_prod_static"  --temper-p 0   --demo-sigma-mode const    --mass-scale 2.5
run_cell "AD_mass_prod_adapt"   --temper-p 0   --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0 --mass-scale 2.5
run_cell "AD_mass_prod_adapt_b2" --temper-p 0  --demo-sigma-mode adaptive --demo-sigma-adapt-beta 2.0 --mass-scale 2.5
run_cell "AD_mass_gate_adapt"   --temper-p inf --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0 --mass-scale 2.5

# --- off-distribution friction 3.0 (orthogonal axis) ---
run_cell "AD_fric_prod_static"  --temper-p 0   --demo-sigma-mode const    --friction-scale 3.0
run_cell "AD_fric_prod_adapt"   --temper-p 0   --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0 --friction-scale 3.0
run_cell "AD_fric_gate_adapt"   --temper-p inf --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0 --friction-scale 3.0
run_cell "AD_canon_gate_adapt"  --temper-p inf --demo-sigma-mode adaptive --demo-sigma-adapt-beta 1.0

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate AD  ALL DONE ===" | tee -a "$LOG"
