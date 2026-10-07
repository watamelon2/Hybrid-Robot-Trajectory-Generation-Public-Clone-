#!/usr/bin/env bash
# Slate C6 -- per-step action (bias, variance) decomposition.
#
# Follow-up C6 from the 2026-06-01 backlog research.  Promotes the
# scaffolded ``measure_bias_variance.py`` to a runnable pipeline by
# combining:
#
#   1. The new ``--save-executed-actions`` hook in ``run_can.py``
#      (this branch, in unified_loop.py + run_can.py).
#   2. The rewritten ``measure_bias_variance_v2.py`` (this branch,
#      a pure JSON consumer that reads the executed-action arrays
#      and computes the (bias, variance) decomposition).
#
# What the decomposition tests
# ----------------------------
# The Pareto-bias-variance theorem
# (docs/theory/08_asymmetric_robustness_model.md) predicts T2.1
# (hybrid + 5-demo retrieval) strictly Pareto-dominates T2.3 (BC
# retrained on the same 5 demos) in (bias^2, variance) space.  The
# cell-level proxy decomposition already landed via
# feat/adapt-pareto-analysis (commit aaefba2) and confirmed
# dominance at the success-rate level.  This slate verifies the
# theorem at the action level -- which the proxy could not measure.
#
# Stages
# ------
#
#   stage 0  prereq check -- canonical Can /ph dataset + 205-demo
#            mass=1.3 bank + BC retrained on 205 mass=1.3 demos.
#
#   stage 1  generate executed-action data:
#              - hybrid at canonical mass=1.0  (reference a_ref)
#              - hybrid + 5-demo adapted bank at mass=1.3   (T2.1)
#              - BC retrained on 205 mass=1.3 demos at mass=1.3 (T2.3)
#            Each method x S seeds (default 3) -- 9 invocations total.
#            Saves rollout JSONs with per-step executed_actions.
#
#   stage 2  run measure_bias_variance_v2.py to read the JSONs and
#            produce bias_variance.{json,csv,md}.
#
# Default S = 3 seeds, N = 10 init states, H = 120 = 60 rollouts per
# (method, seed); compute per invocation ~12 min @ K=10 H=120 on a
# 10-worker Mac.  Total ~2 h.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARED_DATA="${SHARED_DATA:-/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_C6_bias_var_capture_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
N_SEEDS="${N_SEEDS:-3}"
mkdir -p "$OUTDIR" "$SHARED_DATA"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate C6: per-step (bias, variance) decomposition (T2.1 vs T2.3)" | tee -a "$LOG"
echo "SHARED_DATA  = $SHARED_DATA" | tee -a "$LOG"
echo "N_SEEDS      = $N_SEEDS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Stage 0: prereq check
# ============================================================
B205="$SHARED_DATA/can_mass1p3_205demos.hdf5"
BC_CKPT="$REPO/experiments/03_bc_baseline/results/policy_mass1p3_205.pt"

if [ ! -f "$B205" ]; then
  echo "FATAL: T2.1 bank missing at $B205" | tee -a "$LOG"
  echo "       Build via slate_incremental_collect_2026_05_26.sh first." | tee -a "$LOG"
  exit 2
fi
if [ ! -f "$BC_CKPT" ]; then
  echo "FATAL: T2.3 BC checkpoint missing at $BC_CKPT" | tee -a "$LOG"
  echo "       Build via slate_incremental_T2_bc_retrain_2026_05_26.sh first." | tee -a "$LOG"
  exit 2
fi

# ============================================================
# Stage 1: generate data (S seeds x 3 methods)
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE 1: data generation ##############" | tee -a "$LOG"

COMMON=(
  --task can
  --K 10 --M 16 --H 120
  --sigma-max 0.25 --sigma-min 0.05
  --mpc-replans 1 --mpc-chunk-steps 60
  --n-demos 10 --n-rollouts 1
  --n-workers "$N_WORKERS"
  --save-executed-actions
  --output-dir "$OUTDIR"
)

run_seed () {
  local SEED="$1"
  local KIND="$2"

  local TAG="${KIND}_seed${SEED}"
  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[stage 1] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"

  case "$KIND" in
    hybrid_canonical)
      python "$REPO/experiments/15_unified_hybrid/run_can.py" \
        "${COMMON[@]}" \
        --prior demo_1nn --lambda-prior 1 \
        --mass-scale 1.0 \
        --seed "$SEED" \
        --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
      ;;
    hybrid_t2_1)
      python "$REPO/experiments/15_unified_hybrid/run_can.py" \
        "${COMMON[@]}" \
        --prior demo_1nn --lambda-prior 1 \
        --mass-scale 1.3 \
        --prior-bank-dataset "$B205" --prior-bank-task can \
        --seed "$SEED" \
        --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
      ;;
    bc_t2_3)
      python "$REPO/experiments/15_unified_hybrid/run_can.py" \
        "${COMMON[@]}" \
        --prior bc --lambda-prior 5 \
        --bc-checkpoint "$BC_CKPT" \
        --mass-scale 1.3 \
        --seed "$SEED" \
        --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"
      ;;
    *)
      echo "FATAL: unknown KIND=$KIND" | tee -a "$LOG"
      return 2
      ;;
  esac

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

for KIND in hybrid_canonical hybrid_t2_1 bc_t2_3; do
  for SEED in $(seq 0 $((N_SEEDS - 1))); do
    run_seed "$SEED" "$KIND"
  done
done

# ============================================================
# Stage 2: bias-variance decomposition
# ============================================================
echo "" | tee -a "$LOG"
echo "############## STAGE 2: (bias, variance) decomposition ##############" | tee -a "$LOG"

python "$REPO/experiments/15_unified_hybrid/measure_bias_variance_v2.py" \
  --results-dir "$OUTDIR" \
  --method-prefix "hybrid_canonical=ref" \
  --method-prefix "hybrid_t2_1=t2_1_hybrid" \
  --method-prefix "bc_t2_3=t2_3_bc" \
  --output-dir "$OUTDIR/pareto" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate C6  ALL DONE ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Verdict + numbers:" | tee -a "$LOG"
echo "  $OUTDIR/pareto/bias_variance.md" | tee -a "$LOG"
echo "  $OUTDIR/pareto/bias_variance.{json,csv}" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next:" | tee -a "$LOG"
echo "  - If t2_1_hybrid Pareto-dominates t2_3_bc at the action level:" | tee -a "$LOG"
echo "    update docs/results_writeups/pareto_analysis.md 'Deferred'" | tee -a "$LOG"
echo "    subsection -- the journal-version follow-up has landed." | tee -a "$LOG"
echo "  - If Pareto-incomparable: tighten S (N_SEEDS=5) and retry;" | tee -a "$LOG"
echo "    if still incomparable, the proxy-level result holds at the" | tee -a "$LOG"
echo "    cell level but not at the action level (worth documenting)." | tee -a "$LOG"
