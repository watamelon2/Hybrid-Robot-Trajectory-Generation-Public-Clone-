#!/usr/bin/env bash
# Slate MBD-DEMO-QUALITY -- follow-up #6 from
# docs/theory/09_mbd_data_guided_comparison.md §6b (tests §4.5b-2: the
# gate-ceiling claim).
#
# Prediction: as demo quality degrades (optimal -> +13% -> +27% cost),
# the HARD-GATE (§6.1, --temper-p inf) success PLATEAUS AT / TRACKS the
# demo's own cost, while the persistent PRODUCT (the validated score-
# bridge, best lambda) EXCEEDS it and degrades more gracefully -- the
# cleanest single plot for "persistent fusion beats warm-start".
#
# Demo pools are the length-graded banks from build_graded_demo_banks.py
# (cost surrogate = trajectory length; optimal/mid/worst terciles of the
# 200 canonical Can demos, ~ +0% / +13% / +27% median cost).
#
# STAGED, not auto-launched.  Run AFTER the other slates free the
# machine:
#   screen -dmS mbd_dq bash experiments/15_unified_hybrid/slate_mbd_demo_quality_2026_06_05.sh
#
# IMPORTANT: set SIGMA_DEMO from Section D of slate_mbd_gated_tempered
# (the calibrated scale where the gate actually engages).  Default 4 is
# a placeholder mid-sweep value.
#
# Cells: 3 bands x {gate p=inf, product score} = 6.  ~12 min/cell -> ~1.2 h.

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_demo_quality_2026_06_05"
DATADIR="$REPO/experiments/15_unified_hybrid/data"
N_WORKERS="${N_WORKERS:-10}"
SIGMA_DEMO="${SIGMA_DEMO:-4}"
LAMBDA_PRODUCT="${LAMBDA_PRODUCT:-1}"   # canonical best-lambda product
BANDS="${BANDS:-optimal mid worst}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate MBD-DEMO-QUALITY (#6, gate-ceiling) SIGMA_DEMO=$SIGMA_DEMO" | tee -a "$LOG"

# 0. build graded banks if missing.
if [ ! -f "$DATADIR/can_quality_worst.hdf5" ]; then
  echo "[dq] building graded demo banks..." | tee -a "$LOG"
  python "$REPO/experiments/15_unified_hybrid/build_graded_demo_banks.py" \
    --task can --out-dir "$DATADIR" --n-bands 3 2>&1 | tee -a "$LOG"
fi

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
    echo "[dq] $TAG already done; skipping." | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for BAND in $BANDS; do
  BANK="$DATADIR/can_quality_${BAND}.hdf5"
  # Gate (max-gate, p=inf): success should plateau near the band's cost.
  run_cell "DQ_${BAND}_gate" \
    --prior-bank-dataset "$BANK" --prior-bank-task can \
    --combine-mode tempered --temper-p inf --lambda-prior 1 \
    --sigma-demo "$SIGMA_DEMO"
  # Product (validated score-bridge, best lambda): should exceed the gate.
  run_cell "DQ_${BAND}_product" \
    --prior-bank-dataset "$BANK" --prior-bank-task can \
    --combine-mode score --lambda-prior "$LAMBDA_PRODUCT"
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate MBD-DEMO-QUALITY  ALL DONE ===" | tee -a "$LOG"

# Gate-ceiling table: success vs band cost for gate vs product.
python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUTDIR")
man = Path("$DATADIR/can_quality_manifest.json")
cost = {}
if man.exists():
    for b in json.load(open(man))["bands"]:
        cost[b["label"]] = b["cost_pct_vs_best"]
rows = {}
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name: continue
    d = json.load(open(js))
    tag = js.stem.split("_R-1_")[-1] if "_R-1_" in js.stem else js.stem
    n = d.get("n_total", 0); ns = d.get("n_success", 0)
    rows[tag] = (ns / n * 100) if n else 0.0
print()
print(f"{'band':<10}{'+cost%':>8}{'gate %':>9}{'product %':>11}")
print("-" * 38)
for band in ("optimal", "mid", "worst"):
    g = rows.get(f"DQ_{band}_gate"); p = rows.get(f"DQ_{band}_product")
    c = cost.get(band, float('nan'))
    gs = f"{g:.1f}" if g is not None else "--"
    ps = f"{p:.1f}" if p is not None else "--"
    print(f"{band:<10}{c:>7.1f}%{gs:>9}{ps:>11}")
floor = rows.get("DQ_pure_mbd_floor")
if floor is not None:
    print(f"\npure-MBD floor (no demo): {floor:.1f}%")
print("\n#6 prediction: product >= gate at every band, and the gap WIDENS")
print("as cost% grows (gate plateaus at demo cost; product exceeds it).")
PY
