#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUTDIR="$REPO/experiments/15_unified_hybrid/results/tool_hang_subgoal_scaffold_2026_06_09"
N_WORKERS="${N_WORKERS:-2}"
RUN="${RUN:-0}"
if [ "$RUN" = "1" ]; then
  mkdir -p "$OUTDIR"
fi

cmd=(
  "$PYTHON" "$REPO/experiments/15_unified_hybrid/run_can.py"
  --task tool_hang
  --prior demo_1nn --lambda-prior 1
  --combine-mode score
  --H 120 --mpc-replans 3 --mpc-chunk-steps 40 --max-env-steps 120
  --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05
  --n-demos 1 --n-rollouts 1
  --n-workers "$N_WORKERS" --seed 0
  --save-step-trace --log-gate-diagnostic
  --output-dir "$OUTDIR" --output-name-suffix "_tool_hang_key_discovery"
)

echo "Manual-only tool_hang subgoal/key-discovery scaffold."
echo "Default RUN=0 prints the command only. Set RUN=1 to execute one tiny probe."
echo "${cmd[*]}"

if [ "$RUN" = "1" ]; then
  "${cmd[@]}"
fi
