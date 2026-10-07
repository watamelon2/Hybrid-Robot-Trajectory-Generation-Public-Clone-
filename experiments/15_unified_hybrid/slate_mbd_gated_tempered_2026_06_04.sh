#!/usr/bin/env bash
# Slate MBD-GATED-TEMPERED -- empirically test the product<->max-gate
# combination-rule family from
# docs/theory/09_mbd_data_guided_comparison.md §6.1/§6.2 + follow-ups
# #4/#7/#8.  Implements the new --combine-mode / --temper-p /
# --sigma-schedule flags (mbd.tempered_combine + denoise_loop).
#
# This slate is STAGED, not auto-launched.  Run it explicitly:
#     screen -dmS mbd_temper bash experiments/15_unified_hybrid/slate_mbd_gated_tempered_2026_06_04.sh
# Select a subset of sections with the SECTIONS env var (default all):
#     SECTIONS="A" bash .../slate_mbd_gated_tempered_2026_06_04.sh
#
# Sections
# --------
#   A (#7 tempered-p rule selection, canonical Can):
#       --combine-mode tempered with p in {0, 1, 2, 5, inf} (+ a legacy
#       --combine-mode score product reference).  Goal: show a FINITE p
#       Pareto-dominates both fixed rules.  p=0 == product (our rule);
#       p=inf == MBD-Demo max-gate (Eq. 13).
#   B (#4 sigma-mode ablation, canonical Can, legacy score path):
#       demo_1nn lambda=1 with --sigma-schedule {decay, const}.  Tests
#       §8.3: is the sigma_t-annealed pull what buys our edge?
#   C (#8 / §6.1 off-distribution, perturbed dynamics, canonical prior):
#       for mass=2.5 and friction=3.0, product (p=0) vs max-gate (p=inf).
#       Prediction: product COLLAPSES, max-gate RESCUES (the §5 robustness
#       gap).  Also logs gate_frac + ldem_minus_lmod (§6.3 diagnostic).
#
# Operating point matches the canonical Slate-E/T2 cells (K=10 M=16
# H=120 R=1, n-demos=10 x 3 rollouts = 30 train rollouts/cell).
# Wall-clock: ~12 min/cell.  A=6, B=2, C=4 -> ~12 cells -> ~2.4 h total.
#
# Hardening (per the v2 launcher pattern): no `set -e`, skip-existing
# JSONs, SIGTERM trap so a parallel agent's pkill can't wipe partials.

set -uo pipefail
trap 'echo "[slate] SIGTERM/SIGINT received; exiting cleanly" | tee -a "$LOG"; exit 143' TERM INT

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_mbd_gated_tempered_2026_06_04"
N_WORKERS="${N_WORKERS:-10}"
SECTIONS="${SECTIONS:-D A B C}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate MBD-GATED-TEMPERED: product<->max-gate family (§6.1/§6.2)" | tee -a "$LOG"
echo "SECTIONS = $SECTIONS   N_WORKERS = $N_WORKERS" | tee -a "$LOG"

RUNPY="$REPO/experiments/15_unified_hybrid/run_can.py"
COMMON=(
  --task can --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --n-demos 10 --n-rollouts 3
  --n-workers "$N_WORKERS" --seed 0
  --warm-start --prior demo_1nn
  --output-dir "$OUTDIR"
)

run_cell () {
  # $1 = tag, rest = extra args
  local TAG="$1"; shift
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"
  python "$RUNPY" "${COMMON[@]}" --output-name-suffix "_$TAG" "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# ---- Section D: sigma_demo calibration (RUN FIRST) -----------------------
# The demo Gaussian over H*action_dim dims makes ||tau-demo||^2/(2 sigma^2)
# dwarf the cost term -J/lambda at small sigma_demo, collapsing the
# max-gate to pure model (gate_frac->0).  Sweep sigma_demo to find the
# scale where the two branches are comparable (gate_frac transitions
# 0->1) BEFORE interpreting Sections A/C.  Diagnostic: ldem_minus_lmod
# should cross ~0 near the calibrated sigma_demo.
if [[ " $SECTIONS " == *" D "* ]]; then
  echo "" | tee -a "$LOG"; echo "######## SECTION D (calibration) sigma_demo sweep, max-gate ########" | tee -a "$LOG"
  for SD in 0.25 1 4 16 64; do
    run_cell "D_maxgate_sigmademo_${SD//./p}" \
      --combine-mode tempered --temper-p inf --lambda-prior 1 --sigma-demo "$SD"
  done
fi

# ---- Section A: tempered-p rule selection (canonical) --------------------
if [[ " $SECTIONS " == *" A "* ]]; then
  echo "" | tee -a "$LOG"; echo "######## SECTION A (#7) tempered-p sweep ########" | tee -a "$LOG"
  run_cell "A_score_product_ref" --combine-mode score --lambda-prior 1
  run_cell "A_temper_p0_product"  --combine-mode tempered --temper-p 0   --lambda-prior 1
  run_cell "A_temper_p1"          --combine-mode tempered --temper-p 1   --lambda-prior 1
  run_cell "A_temper_p2"          --combine-mode tempered --temper-p 2   --lambda-prior 1
  run_cell "A_temper_p5"          --combine-mode tempered --temper-p 5   --lambda-prior 1
  run_cell "A_temper_pinf_maxgate" --combine-mode tempered --temper-p inf --lambda-prior 1
fi

# ---- Section B: sigma-mode ablation (canonical, score path) --------------
if [[ " $SECTIONS " == *" B "* ]]; then
  echo "" | tee -a "$LOG"; echo "######## SECTION B (#4) sigma-mode ablation ########" | tee -a "$LOG"
  run_cell "B_sigma_decay" --combine-mode score --lambda-prior 1 --sigma-schedule decay
  run_cell "B_sigma_const" --combine-mode score --lambda-prior 1 --sigma-schedule const
fi

# ---- Section C: off-distribution product vs max-gate (#8 / §6.1) ---------
if [[ " $SECTIONS " == *" C "* ]]; then
  echo "" | tee -a "$LOG"; echo "######## SECTION C (#8) off-distribution rescue ########" | tee -a "$LOG"
  # mass=2.5: product collapses (per slate_T2_mass_extremes); does the
  # max-gate rescue it?  Canonical prior under perturbed dynamics.
  run_cell "C_mass2p5_product" --mass-scale 2.5 --combine-mode tempered --temper-p 0   --lambda-prior 1
  run_cell "C_mass2p5_maxgate" --mass-scale 2.5 --combine-mode tempered --temper-p inf --lambda-prior 1
  run_cell "C_fric3p0_product" --friction-scale 3.0 --combine-mode tempered --temper-p 0   --lambda-prior 1
  run_cell "C_fric3p0_maxgate" --friction-scale 3.0 --combine-mode tempered --temper-p inf --lambda-prior 1
fi

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate MBD-GATED-TEMPERED  ALL DONE ===" | tee -a "$LOG"

# ---- Aggregate: success + mean gate_frac + mean ldem-lmod per cell -------
python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
import numpy as np
out = Path("$OUTDIR")
print()
print(f"{'cell':<34}{'succ %':>8}{'n':>5}{'gate_frac':>11}{'dl_dem-mod':>12}")
print("-" * 70)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    tag = js.stem.split("_R-1_")[-1] if "_R-1_" in js.stem else js.stem
    n = d.get("n_total", 0); ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    gfs, dls = [], []
    for r in d.get("per_rollout", []):
        for ch in r.get("chunks", []):
            for it in ch.get("iter_curve", []):
                gf = it.get("gate_frac"); dl = it.get("ldem_minus_lmod")
                if gf is not None and np.isfinite(gf): gfs.append(gf)
                if dl is not None and np.isfinite(dl): dls.append(dl)
    gf = f"{np.mean(gfs):.3f}" if gfs else "   nan"
    dl = f"{np.mean(dls):+.2f}" if dls else "   nan"
    print(f"{tag:<34}{sr:>7.1f}%{n:>5}{gf:>11}{dl:>12}")
print()
print("A: look for a finite p with succ >= max(product, max-gate).")
print("C: product should drop while max-gate holds (the §5 robustness gap),")
print("   and gate_frac->1 / ldem-lmod>0 where the prior is off-distribution.")
PY
