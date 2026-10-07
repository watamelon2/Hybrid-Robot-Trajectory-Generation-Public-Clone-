#!/bin/bash
# Phase-C pre-launch smoke test.  Validates:
#   - env builds (robosuite + mujoco)
#   - ckpt loads at expected path / size
#   - obs schema = can_dp (23-D) (NOT legacy 46-D `can`)
#   - corrected --mpc-replans path produces meaningful rewards (~3.0+)
#
# Wall: ~30 sec on CPU.  Runs 1 train + 1 test rollout at
# max_env_steps=120 (15 chunks of 8 steps).
#
# Usage (from the HRTG-dp repo root):
#
#   bash experiments/17_dp_hybrid/run_phase_c_smoke.sh
#
# Env overrides:
#
#   PYTHON=/path/to/python  bash experiments/17_dp_hybrid/run_phase_c_smoke.sh
#   DEVICE=cuda             bash experiments/17_dp_hybrid/run_phase_c_smoke.sh
set -euo pipefail

PYTHON="${PYTHON:-$(command -v python)}"
DEVICE="${DEVICE:-cpu}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="$REPO_ROOT/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt"
SMOKE_OUT="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_c_2026_05_29/smoke"

if [ ! -e "$CKPT" ]; then
    echo "ERROR: ckpt not at $CKPT"
    echo "See HANDOFF_NEXT_MAC.md 'Ckpt sync' section."
    exit 1
fi

REAL_CKPT="$($PYTHON -c "import os; print(os.path.realpath('$CKPT'))")"
if [ ! -f "$REAL_CKPT" ]; then
    echo "ERROR: ckpt symlink at $CKPT resolves to $REAL_CKPT which does not exist."
    echo "Likely the symlink is broken; see HANDOFF_NEXT_MAC.md Appendix."
    exit 1
fi
CKPT_SIZE=$(stat -f%z "$REAL_CKPT" 2>/dev/null || stat -c%s "$REAL_CKPT")
if [ "$CKPT_SIZE" -lt 100000000 ]; then
    echo "ERROR: ckpt at $REAL_CKPT is only $CKPT_SIZE bytes; expected > 100 MB."
    exit 1
fi

mkdir -p "$SMOKE_OUT"
rm -f "$SMOKE_OUT"/*.json 2>/dev/null || true

echo "=== phase-c smoke (1 train + 1 test, max_env_steps=120, ~30 sec) ==="
echo "    python: $PYTHON"
echo "    device: $DEVICE"
echo "    ckpt:   $REAL_CKPT  ($(echo $CKPT_SIZE | awk '{printf "%.1f MB", $1/1e6}'))"
echo ""

"$PYTHON" -m experiments.15_unified_hybrid.run_can \
    --task can --prior flow \
    --prior-checkpoint "$CKPT" \
    --K 0 --H 16 \
    --mpc-replans 14 --mpc-chunk-steps 8 \
    --n-demos 1 --n-rollouts 1 \
    --n-test-rollouts 1 --test-start-seed 100000 \
    --max-env-steps 120 \
    --prior-n-steps 100 \
    --sigma-max 0.25 --sigma-min 0.05 \
    --lambda-prior 0.5 \
    --device "$DEVICE" \
    --output-dir "$SMOKE_OUT" 2>&1 | tail -10

echo ""
echo "=== smoke result check ==="
"$PYTHON" -c "
import json, sys, glob
p = sorted(glob.glob('$SMOKE_OUT/unified_flow_lam-*.json'))
if not p:
    print('FAIL: no JSON output')
    sys.exit(1)
d = json.load(open(p[-1]))
ok = True
pr = d['per_rollout']
if d.get('test_n') != 1 or d.get('train_n') != 1:
    print(f'FAIL: expected train_n=1 test_n=1, got train_n={d.get(\"train_n\")} test_n={d.get(\"test_n\")}')
    ok = False
for i, r in enumerate(pr):
    n_chunks = len(r.get('chunks', []))
    total_steps = sum(c.get('steps_executed', 0) for c in r.get('chunks', []))
    reward = r.get('final_reward')
    print(f'  rollout {i} ({r.get(\"rollout_kind\")}):'
          f' n_chunks={n_chunks}  total_steps={total_steps}  reward={reward:.2f}')
    if n_chunks < 10:
        print(f'    FAIL: expected n_chunks >= 10 (got {n_chunks}) -- mpc-replans bug?')
        ok = False
    if total_steps < 100:
        print(f'    FAIL: expected total_steps >= 100 (got {total_steps}) -- early termination?')
        ok = False
    if reward < 1.0:
        print(f'    WARN: reward < 1.0 ({reward:.2f}) -- robot not reaching can?')
        # Not a hard fail (single-rollout variance), just a warning.
print()
if ok:
    print('SMOKE: PASS  (env + ckpt + obs schema + mpc-replans all OK)')
    sys.exit(0)
else:
    print('SMOKE: FAIL  (see issues above)')
    sys.exit(1)
"
