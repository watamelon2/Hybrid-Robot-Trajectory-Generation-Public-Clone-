#!/usr/bin/env bash
# Slate K -- tool_hang + transport SCORED runs (post dense-reward fix).
#
# Future-work item 1 from the Slide 9 conclusion deck (2026-06-01).
# Closes the last two missing tasks from the cross-task headline panel.
#
# Background
# ----------
# Slate J (2026-05-22) was the smoke / acceptance probe for
# experiments/15_unified_hybrid/env_dense_rewards.py -- the wrapper
# that monkey-patches env.step on robosuite-1.5's ToolHang and
# TwoArmTransport (both of which ship with sparse-only rewards and
# literal `# TODO!` comments where the dense branch would live).
# Slate J ran at n_demos=20, n_rollouts=2 (40 rollouts/cell), which
# was enough to verify the wrapper produces a positive-mean reward
# trajectory but is NOT enough to put a number on the cross-task
# panel next to Lift (60/60), Can (177/240), and Square (5/60).
#
# Slate K bumps both tasks to the *scored* sample budget that
# matches the other three: 30 demos x 2 rollouts = 60 rollouts per
# cell, sweeping lambda over the same {0, 1, 2} grid the GREEN-cell
# Phase F results used.  This produces the first three honest
# numbers we can put on Slide 5 next to the existing headline.
#
# What the slate does NOT do
#   - It does not re-train any prior (retrieval = demo_1nn over the
#     stock robomimic banks).
#   - It does not run K > 10 -- the headline operating point sits
#     at K=10 / sigma=0.25 / sigma-min=0.01 / H=300.
#   - It does not chunk the lambda axis into separate invocations;
#     run_can.py natively accepts space-separated `--lambda-prior 0
#     1 2` and reuses the env + bank per lambda.
#   - It does not run transport on more than 30 demos because the
#     transport bank only contains 200 demos at ~700 steps each
#     and 30 of them already saturate the retrieval bank's
#     coverage of the start-state distribution.
#
# Cells (6 total = 3 lambdas x 2 tasks)
#   K1: tool_hang  demo_1nn  lambda in {0, 1, 2}  H=300  K=10
#   K2: transport  demo_1nn  lambda in {0, 1, 2}  H=300  K=10
#
# Acceptance gates (post-hoc, compared to Slate J smoke)
#   STRONG : any of the 6 cells reports a non-zero success count
#            and mean reward >= 5.  Lands the two missing headline
#            numbers in the {>= weak success} regime.
#   WEAK   : non-zero mean reward >= 3 in at least one cell per task
#            but no successes; the dense shaping is firing but H=300
#            still doesn't reach the goal state.  Recommend a follow-
#            up slate at H=500.
#   FAIL   : both tasks at mean reward < 3 across all three lambdas.
#            Indicates the shaping function needs re-calibration
#            (probably the discrete-subgoal weights are too small).
#
# Wall-clock
#   ~7 - 9 h total sequential on a 10-worker Mac (toolhang ~3 - 4 h,
#   transport ~4 - 5 h).  Fire on an unattended evening.
#
# Engineering posture
#   - Two separate run_can invocations (one per task) so a partial
#     failure on tool_hang does not skip transport.
#   - Lambda sweep is a single invocation per task so the bank + env
#     loading cost amortises across all three lambdas.
#   - Idempotent: if the output JSON for a given task already exists,
#     the cell is skipped.  Resume by re-running the slate.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/slate_K_toolhang_transport_scored_2026_06_01"
N_WORKERS="${N_WORKERS:-10}"
mkdir -p "$OUTDIR"

source /Users/szh/anaconda3/etc/profile.d/conda.sh
conda activate hybrid-traj
cd "$REPO"

LOG="$OUTDIR/run.log"
date | tee "$LOG"
echo "Slate K: tool_hang + transport scored runs (post dense-reward fix)" | tee -a "$LOG"
echo "Operating point: demo_1nn, lambda in {0, 1, 2}, H=300, K=10," | tee -a "$LOG"
echo "                 sigma=[0.01, 0.25], 30 demos x 2 rollouts = 60 rollouts/cell." | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Safety: refuse to start if other slates are still hot (10-worker
# slates conflict on a 16-core Mac).
for SCREEN in slate_h slate_i slate_j slate_incremental; do
  if screen -ls 2>/dev/null | grep -qE "\.${SCREEN}[[:space:]]"; then
    echo "FATAL: $SCREEN screen still alive.  Wait for it to finish." | tee -a "$LOG"
    exit 2
  fi
done
echo "No competing slates; OK to start." | tee -a "$LOG"

# ----- shared flag bundle ----------------------------------------------
COMMON=(
  --mpc-replans 1 --mpc-chunk-steps 60
  --K 10 --M 16 --H 300
  --sigma-max 0.25 --sigma-min 0.01
  --n-demos 30 --n-rollouts 2
  --n-workers "$N_WORKERS"
  --output-dir "$OUTDIR"
  --prior demo_1nn
  --lambda-prior 0 1 2
  --seed 0
)

run_task () {
  local TASK="$1"
  local TAG="K_${TASK}_1nn_lamsweep_dense_scored"
  echo "" | tee -a "$LOG"
  echo "=== $(date '+%H:%M:%S')  $TAG  starting ===" | tee -a "$LOG"

  # Idempotency: skip the task if a summary JSON already exists.
  if ls "$OUTDIR"/unified_summary_demo_1nn_*"_${TAG}".json >/dev/null 2>&1; then
    echo "[slate K] $TASK summary already present; skipping." | tee -a "$LOG"
    return 0
  fi

  python "$REPO/experiments/15_unified_hybrid/run_can.py" \
    --task "$TASK" \
    "${COMMON[@]}" \
    --output-name-suffix "_$TAG" 2>&1 | tee -a "$LOG"

  echo "=== $(date '+%H:%M:%S')  $TAG  done ===" | tee -a "$LOG"
}

# K1: tool_hang scored cells.
run_task tool_hang

# K2: transport scored cells.
run_task transport

# ----- final summary table ---------------------------------------------
echo "" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S')  Slate K  ALL DONE ===" | tee -a "$LOG"

python - <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

outroot = Path("$OUTDIR")
print()
print(f"{'task':<10}  {'lambda':>6}  {'n_succ':>7}  {'n':>4}  {'succ %':>7}  {'mean R':>7}")
print("-" * 50)
for summary in sorted(outroot.glob("unified_summary_demo_1nn_*.json")):
    s = json.load(open(summary))
    task = s.get("task", "?")
    for cell in s.get("cells", []):
        lam = cell.get("lambda_prior", 0)
        n = cell.get("n_total", 0)
        ns = cell.get("n_success", 0)
        sr = (ns / n * 100) if n else 0.0
        mr = cell.get("mean_reward", float("nan"))
        print(f"{task:<10}  {lam:>6.1f}  {ns:>7d}  {n:>4d}  {sr:>6.1f}%  {mr:>7.2f}")
PY

echo "" | tee -a "$LOG"
echo "Next: copy any GREEN cells (succ > 0, mean R >= 5) into" | tee -a "$LOG"
echo "  presentation/SLIDE_SPEC.md  Slide 5 cross-task panel" | tee -a "$LOG"
echo "  presentation/DECK_GUIDE.md  Slide 5 callout table" | tee -a "$LOG"
echo "  presentation/figures/headline_bar_crosstask.pdf (regen via" | tee -a "$LOG"
echo "     make_headline_figures.py --plot crosstask)" | tee -a "$LOG"
