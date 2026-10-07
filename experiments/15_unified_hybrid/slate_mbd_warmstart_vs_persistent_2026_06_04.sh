#!/usr/bin/env bash
# Slate WARMSTART-VS-PERSISTENT -- follow-up #5 from
# docs/theory/09_mbd_data_guided_comparison.md §6b (tests §4.5b: the
# claim "MBD-Demo ~= warm-start").
#
# Mechanism test, not a tuning sweep: does a prior that is active only at
# the high-noise steps (warm-start-like) land NEAR MBD-Demo's
# equivalence point and BELOW the persistent product (prior on every
# step)?  If so, that is direct empirical support for the §4.5b
# max-gate<->warm-start equivalence.
#
# Reuses the existing per-iteration prior-weight schedule
# (--lambda-schedule) -- NO new code:
#   persistent      = --lambda-schedule constant  (prior every step)
#   warmstart_only  = --lambda-schedule linear    (prior strong at k=0
#                     high-sigma, annealed to 0 at the low-sigma tail)
#   anneal_in       = --lambda-schedule inverse_linear (the opposite;
#                     control that rules out "any schedule helps")
# plus two references:
#   pure_warmstart  = --lambda-prior 0            (warm-start, no pull)
#   product_ref     = --lambda-schedule constant (== persistent)
#
# --log-gate-diagnostic rides along (#3, §6.3) so every cell also logs
# gate_frac + ldem_minus_lmod for ~free.
#
# STAGED, not auto-launched.  Run AFTER slate_mbd_gated_tempered finishes
# (shares the machine):
#   screen -dmS mbd_ws bash experiments/15_unified_hybrid/slate_mbd_warmstart_vs_persistent_2026_06_04.sh
#
# Operating point = canonical Can (K=10 M=16 H=120 R=1, 10 demos x 3).
# 5 cells x ~12 min ~= 1 h.

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_warmstart_vs_persistent_2026_06_04"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate WARMSTART-VS-PERSISTENT (#5, tests §4.5b)" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --log-gate-diagnostic
  --output-dir "$OUTDIR"
)

run_cell () {
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

run_cell "WS_pure_warmstart"  --lambda-prior 0
run_cell "WS_persistent"      --lambda-prior 1 --lambda-schedule constant
run_cell "WS_warmstart_only"  --lambda-prior 1 --lambda-schedule linear
run_cell "WS_anneal_in"       --lambda-prior 1 --lambda-schedule inverse_linear
run_cell "WS_cosine_out"      --lambda-prior 1 --lambda-schedule cosine

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate WARMSTART-VS-PERSISTENT  ALL DONE ===" | tee -a "$LOG"

python "$REPO/experiments/15_unified_hybrid/analyze_tempered_slate.py" \
  --results-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || true

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUTDIR")
print()
print(f"{'cell':<22}{'succ %':>9}")
print("-" * 31)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name: continue
    d = json.load(open(js))
    tag = js.stem.split("_R-1_")[-1] if "_R-1_" in js.stem else js.stem
    n = d.get("n_total", 0); ns = d.get("n_success", 0)
    print(f"{tag:<22}{(ns/n*100 if n else 0):>8.1f}%")
print()
print("#5 prediction: warmstart_only ~ pure_warmstart (the MBD-Demo")
print("equivalence point) and BELOW persistent (the persistent product).")
PY
