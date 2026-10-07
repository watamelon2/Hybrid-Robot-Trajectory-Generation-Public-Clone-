#!/usr/bin/env bash
cd /workspace/hrtg-cloud/HRTG-square-dp-followup/experiments/15_unified_hybrid
export HRTG_REMOTE_PYTHON=/workspace/hrtg-venv/bin/python
export HRTG_REMOTE_REPO=/workspace/hrtg-cloud/HRTG-square-dp-followup
export HRTG_PHASE4_TASKS=can
export HRTG_NWFIX_WORKERS=8
export HRTG_DP_MAX_TOTAL=24
export HRTG_DP_MAX_PER_GPU=12
exec /workspace/hrtg-venv/bin/python queue_dp_polish_phase4_2026_06_13.py --run > results/vast_safe_dp_nwfix_2026_06_12/orchestrator_phase4.log 2>&1
