#!/usr/bin/env bash
# Slate CALIBRATED-A/C -- closes the sigma_demo loop that made the first
# slate's Section A/C uninterpretable.
#
# WHY: in slate_mbd_gated_tempered, Sections A & C ran at the DEFAULT
# sigma_demo=0.25.  Section D then showed that at 0.25 the demo Gaussian
# dwarfs the cost term (ldem-lmod = -720, gate_frac=0) so EVERY tempered
# cell degenerated -- tempered product = 0%, max-gate = 0%, while the
# legacy score-product got 66.7%.  Section D's calibrated point (where
# ldem-lmod crosses ~0 and the gate is alive at gate_frac~0.4) is
# sigma_demo=1.  This slate RE-RUNS A & C there so the product<->max-gate
# comparison is finally apples-to-apples.
#
# sigma_demo is auto-derived from the completed Section D via
# pick_calibrated_sigma.py (override with SIGMA_CAL=... ); a second
# gate-dominant point (SIGMA_HI, default 4) brackets the sensitivity.
#
# STAGED.  Run AFTER slate_mbd_gated_tempered (needs its Section D):
#   screen -dmS mbd_cal bash experiments/15_unified_hybrid/slate_mbd_calibrated_AC_2026_06_05.sh
#
# Cells: A' {p=0,1,2,5,inf} @ cal + A'' {p=0,inf} @ hi + C' {mass,fric}x
# {product,maxgate} @ cal = 5 + 2 + 4 = 11.  ~12 min/cell -> ~2.2 h.

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GATED_DIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_calibrated_AC_2026_06_05"
N_WORKERS="${N_WORKERS:-10}"
SIGMA_HI="${SIGMA_HI:-4}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"

# Auto-derive the calibrated sigma_demo from Section D (override w/ env).
if [ -z "${SIGMA_CAL:-}" ]; then
  SIGMA_CAL="$(python "$REPO/experiments/15_unified_hybrid/pick_calibrated_sigma.py" \
    --results-dir "$GATED_DIR" --fallback 1 2>>"$LOG")"
fi
echo "Slate CALIBRATED-A/C  SIGMA_CAL=$SIGMA_CAL  SIGMA_HI=$SIGMA_HI" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --combine-mode tempered --lambda-prior 1
  --log-gate-diagnostic
  --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[cal] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# --- A' : tempered-p sweep at the CALIBRATED sigma_demo ---
for P in 0 1 2 5 inf; do
  PT=$(echo "$P" | tr '.' 'p')
  run_cell "A2_cal_p${PT}" --temper-p "$P" --sigma-demo "$SIGMA_CAL"
done

# --- A'' : bracket at the gate-dominant sigma_demo (product vs maxgate) ---
run_cell "A2_hi_p0"   --temper-p 0   --sigma-demo "$SIGMA_HI"
run_cell "A2_hi_pinf" --temper-p inf --sigma-demo "$SIGMA_HI"

# --- C' : off-distribution product vs max-gate at the calibrated sigma ---
run_cell "C2_mass2p5_product" --mass-scale 2.5 --temper-p 0   --sigma-demo "$SIGMA_CAL"
run_cell "C2_mass2p5_maxgate" --mass-scale 2.5 --temper-p inf --sigma-demo "$SIGMA_CAL"
run_cell "C2_fric3p0_product" --friction-scale 3.0 --temper-p 0   --sigma-demo "$SIGMA_CAL"
run_cell "C2_fric3p0_maxgate" --friction-scale 3.0 --temper-p inf --sigma-demo "$SIGMA_CAL"

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate CALIBRATED-A/C  ALL DONE ===" | tee -a "$LOG"

python "$REPO/experiments/15_unified_hybrid/analyze_tempered_slate.py" \
  --results-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || true
