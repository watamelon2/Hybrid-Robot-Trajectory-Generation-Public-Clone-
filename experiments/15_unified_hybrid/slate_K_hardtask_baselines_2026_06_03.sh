#!/usr/bin/env bash
# Slate K-BASELINES -- decompose the tool_hang / transport 0% result.
#
# Follow-up #3 to the 2026-06-03 future-directions review.  Slate K
# scored 0/60 on BOTH tool_hang and transport at every lambda in
# {0,1,2} (hybrid, demo_1nn, warm-start).  The per-rollout MBD
# telemetry showed best_cost == mean_cost == -0.0 across every
# denoising iteration with ESS pinned at M (=16): the planner gets a
# FLAT-ZERO cost landscape and never reweights -- it just emits the
# prior warm-start.  Reward scaled with lambda (prior-driven), not MBD.
#
# This slate triangulates the 0% with two baselines per task, at the
# EXACT Slate K operating point (H=300, replans=1, chunk=60, K=10,
# M=16, sigma=[0.01,0.25], 30 demos x 2 rollouts = 60/cell), so the
# numbers are directly comparable to Slate K's hybrid cells.
#
#   pure-MBD   : --prior demo_1nn --lambda-prior 0 --no-warm-start
#                (no prior at all: no warm-start, no per-step guidance)
#                EXPECT ~0% -- confirms the flat cost is fundamental to
#                the task/horizon, not an artifact of the prior.
#
#   prior-only : --prior demo_1nn --lambda-prior inf
#                (execute the retrieval prior directly, MBD off)
#                Isolates what the prior alone achieves.  Stands in for
#                the BC-only baseline (no task-specific BC checkpoint
#                exists for tool_hang/transport).
#
# Reading:
#   - hybrid (Slate K) == prior-only  -> MBD contributes nothing here
#     (consistent with the flat-cost finding); the task is prior-limited.
#   - pure-MBD ~ 0 AND prior-only ~ 0 -> the task is out of reach for
#     this whole approach at this operating point (boundary result).
#
# Runs on feat/dense-reward-slate-K's descendant so the dense-reward
# env wrapper auto-applies exactly as it did for Slate K.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_K_hardtask_baselines_2026_06_03"
N_WORKERS="${N_WORKERS:-10}"
TASKS="${TASKS:-tool_hang transport}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate K-BASELINES: pure-MBD + prior-only on tool_hang / transport" | tee -a "$LOG"
echo "TASKS = $TASKS  N_WORKERS = $N_WORKERS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_task () {
  local TASK="$1"
  echo "" | tee -a "$LOG"
  echo "================ TASK $TASK ================" | tee -a "$LOG"

  # Operating point identical to Slate K.
  local COMMON=(
    --task "$TASK"
    --mpc-replans 1 --mpc-chunk-steps 60
    --K 10 --M 16 --H 300
    --sigma-max 0.25 --sigma-min 0.01
    --n-demos 30 --n-rollouts 2
    --n-workers "$N_WORKERS" --seed 0
    --prior demo_1nn
    --output-dir "$OUTDIR"
  )

  local CELL_TAGS=(
    "baseline_purembd"
    "baseline_prioronly"
  )
  local -a CELL_0_ARGS=(--lambda-prior 0 --no-warm-start)
  local -a CELL_1_ARGS=(--lambda-prior inf)

  local i
  for i in 0 1; do
    local CELL_TAG="${CELL_TAGS[$i]}"
    local TAG_FULL="${CELL_TAG}_${TASK}"
    if ls "$OUTDIR"/unified_*"_${TAG_FULL}".json >/dev/null 2>&1; then
      echo "[$TASK] $CELL_TAG already done; skipping." | tee -a "$LOG"
      continue
    fi
    local -a CELL_ARGS=()
    case "$i" in
      0) CELL_ARGS=("${CELL_0_ARGS[@]}") ;;
      1) CELL_ARGS=("${CELL_1_ARGS[@]}") ;;
    esac
    echo "" | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  $TASK  $CELL_TAG  starting ===" | tee -a "$LOG"
    python "$REPO/experiments/15_unified_hybrid/run_can.py" \
      "${COMMON[@]}" \
      --output-name-suffix "_$TAG_FULL" \
      "${CELL_ARGS[@]}" 2>&1 | tee -a "$LOG"
    echo "=== $(date '+%H:%M:%S')  $TASK  $CELL_TAG  done ===" | tee -a "$LOG"
  done
}

for TASK in $TASKS; do
  run_task "$TASK"
done

echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate K-BASELINES  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUTDIR")
print()
print(f"{'task':>10}  {'baseline':<20}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}")
print("-" * 60)
for js in sorted(out.glob("unified_*.json")):
    if "summary" in js.name:
        continue
    d = json.load(open(js))
    n = d.get("n_total", 0)
    ns = d.get("n_success", 0)
    sr = (ns / n * 100) if n else 0.0
    name = js.stem
    tag = "purembd" if "purembd" in name else ("prioronly" if "prioronly" in name else "?")
    task = "tool_hang" if "tool_hang" in name else ("transport" if "transport" in name else "?")
    print(f"{task:>10}  {tag:<20}  {ns:>7}  {n:>4}  {sr:>6.1f}%")
print()
print("Compare against Slate K hybrid (demo_1nn warm-start): tool_hang")
print("0/60 and transport 0/60 at every lambda.")
PY
