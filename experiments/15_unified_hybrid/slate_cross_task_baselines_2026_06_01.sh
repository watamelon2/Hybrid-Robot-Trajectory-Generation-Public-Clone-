#!/usr/bin/env bash
# Slate CROSS-TASK-BASELINES -- fill the "--" entries on Slide 5.
#
# Follow-up B4 from the 2026-06-01 backlog research.  The §7 cross-
# task table in the brief / Slide 5 currently lists BC and
# MBD-no-prior numbers only for Can; Lift / Square / tool_hang have
# "--" placeholders.  This slate measures them so the 3-way head-to-
# head (BC vs MBD-no-prior vs hybrid) is honest across all four
# tasks.  Per brief §8.2 #4: 2.5 h total estimated.
#
# What's covered here
# -------------------
# 6 cells = 3 tasks (lift, square, tool_hang) x 2 baselines (BC,
# MBD-no-prior).
#
# Transport is intentionally OMITTED because:
#   1. Transport BC has never been trained in this repo (no
#      14-d dual-arm BC checkpoint; 03_bc_baseline supports
#      single-arm tasks only).
#   2. Transport hybrid is itself queued separately on
#      feat/dense-reward-slate-K; comparing against an untrained
#      BC would be apples-to-oranges.
# A follow-up could land transport once a dual-arm BC variant
# exists; out of scope here.
#
# Per-task BC checkpoints
# -----------------------
# - Lift     -> 03_bc_baseline/results/policy_lift.pt (multi-task
#               BC obs schema; produced by feat/adapt-bc-multitask-obs)
# - Square   -> trained inline by this slate if missing
#               (uses --task square, multi-task obs schema)
# - tool_hang-> trained inline by this slate if missing
#               (uses --task tool_hang, multi-task obs schema)
# If any of the three BC checkpoints already exist, training is
# skipped.
#
# Operating points
# ----------------
# BC arm:        lambda=5, K=10, sigma=0.25, sigma-min=0.05, H tuned
#                per task (Lift 120, Square 400, tool_hang 300).
# MBD-no-prior:  lambda=0, K=10, sigma=0.05, M=64, --no-warm-start,
#                same H per task.  This matches the canonical "pure
#                MBD" cell that's 0/60 on Can; the question is
#                whether it's also 0/60 elsewhere.
#
# Acceptance gate (per cell): 60 rollouts (10 demos x 6 rollouts for
# Square / tool_hang where H is large; 20 demos x 3 rollouts for
# Lift where H=120 keeps wall low).
#
# Wall
# ----
#   ~25 min per cell x 6 cells = ~2.5 h.
#   Plus 2 BC trains (~80 s each) for Square / tool_hang if missing.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_cross_task_baselines_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate CROSS-TASK-BASELINES: fill BC + MBD-no-prior for Lift/Square/tool_hang" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ============================================================
# Stage 0: locate / train per-task BC checkpoints
# ============================================================

# Helper: find the canonical dataset for a task.
canonical_dataset () {
  local TASK="$1"
  local CANDIDATE="$REPO/experiments/15_unified_hybrid/data/${TASK}_ph.hdf5"
  if [ -f "$CANDIDATE" ]; then
    echo "$CANDIDATE"
    return 0
  fi
  local ALT
  ALT="$(find "$REPO" -name "${TASK}*.hdf5" -path '*ph*' 2>/dev/null | head -1)"
  if [ -z "$ALT" ]; then
    echo "FATAL: canonical ${TASK}/ph dataset not found." >&2
    return 2
  fi
  echo "$ALT"
}

# Helper: train BC for a task if its checkpoint is missing.
ensure_bc_ckpt () {
  local TASK="$1"
  local CKPT="$REPO/experiments/03_bc_baseline/results/policy_${TASK}.pt"
  if [ -f "$CKPT" ]; then
    echo "$CKPT"
    return 0
  fi
  local DATASET
  DATASET="$(canonical_dataset "$TASK")"
  echo "[BC train] $TASK: training on $DATASET (~80 s cpu)..." | tee -a "$LOG" >&2
  python "$REPO/experiments/03_bc_baseline/train.py" \
    --dataset "$DATASET" --task "$TASK" \
    --output "$CKPT" \
    --log "$REPO/experiments/03_bc_baseline/results/training_log_${TASK}.json" \
    --epochs 500 --batch-size 512 --lr 1e-4 --hidden 256 \
    --seed 0 --device cpu 2>&1 | tee -a "$LOG" >&2
  echo "$CKPT"
}

# ============================================================
# Per-cell runner
# ============================================================
# Per-task H tuning + sample budget (60 rollouts/cell).
declare -A TASK_H=([lift]=120 [square]=400 [tool_hang]=300)
declare -A TASK_NDEMOS=([lift]=20 [square]=10 [tool_hang]=10)
declare -A TASK_NROLLOUTS=([lift]=3 [square]=6 [tool_hang]=6)

run_cell () {
  local TASK="$1"
  local KIND="$2"  # bc | mbd_no_prior
  local TAG="cross_${TASK}_${KIND}"
  local H="${TASK_H[$TASK]}"
  local NDEMOS="${TASK_NDEMOS[$TASK]}"
  local NROLLOUTS="${TASK_NROLLOUTS[$TASK]}"

  if ls "$OUTDIR"/unified_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate] $TAG already done; skipping." | tee -a "$LOG"
    return 0
  fi

  local COMMON=(
    --task "$TASK"
    --mpc-replans 1 --mpc-chunk-steps 60
    --K 10 --M 16 --H "$H"
    --sigma-min 0.05
    --n-demos "$NDEMOS" --n-rollouts "$NROLLOUTS"
    --n-workers "$N_WORKERS"
    --seed 0
    --output-dir "$OUTDIR"
    --output-name-suffix "_$TAG"
  )

  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"

  if [ "$KIND" = "bc" ]; then
    local CKPT
    CKPT="$(ensure_bc_ckpt "$TASK")"
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --prior bc --bc-checkpoint "$CKPT" \
      --lambda-prior 5 \
      --sigma-max 0.25 2>&1 | tee -a "$LOG"
  elif [ "$KIND" = "mbd_no_prior" ]; then
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --prior bc --lambda-prior 0 \
      --no-warm-start \
      --sigma-max 0.05 --M 64 2>&1 | tee -a "$LOG"
  else
    echo "FATAL: unknown KIND=$KIND" | tee -a "$LOG"
    return 2
  fi

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# ============================================================
# Stage 1: run all 6 cells
# ============================================================
for TASK in lift square tool_hang; do
  echo "" | tee -a "$LOG"
  echo "##### task = $TASK #####" | tee -a "$LOG"
  run_cell "$TASK" bc
  run_cell "$TASK" mbd_no_prior
done

# ============================================================
# Stage 2: final summary table
# ============================================================
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate CROSS-TASK-BASELINES  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

out = Path("$OUTDIR")
print()
print(f"{'task':<10}  {'baseline':<14}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 50)
results = {}
for js in sorted(out.glob("unified_*_cross_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    # tag format: ..._cross_<task>_<kind>.json
    parts = js.stem.split("_cross_", 1)
    if len(parts) < 2:
        continue
    rest = parts[1]
    # rest = "<task>_<kind>"
    if rest.endswith("_mbd_no_prior"):
        task = rest[: -len("_mbd_no_prior")]
        kind = "mbd_no_prior"
    elif rest.endswith("_bc"):
        task = rest[: -len("_bc")]
        kind = "bc"
    else:
        task, kind = "?", rest
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    results[(task, kind)] = (ns, n, sr)
    print(f"{task:<10}  {kind:<14}  {ns:>7}  {n:>4}  {sr:>6.1f}%")

print()
print("Fills the '--' entries in:")
print("  - presentation/SLIDE_SPEC.md  Slide 5 cross-task table")
print("  - docs/results.md             §7 cross-task table")
PY

echo "" | tee -a "$LOG"
echo "Next: update Slide 5 cross-task table + docs/results.md §7." | tee -a "$LOG"
