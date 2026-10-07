#!/bin/bash
# Queue 1: chain the over-trained ckpt sweep after the current epoch-curve
# sweep finishes.  Polls every 60 s for the `phase_d_epoch_curve` screen
# session; once it's gone, launches the over-trained K=10 sweep.
#
# Story: the v3b training run produced 7 ckpts.  ep0049-ep0199 are in
# the current sweep; ep0249/0299/0349 are over-trained ckpts where
# standalone DP regressed from 80% (peak at ep0199) back to 68%.
# If MBD also recovers these over-trained ckpts to ~80%, the story
# upgrades to "MBD masks both undertraining AND overtraining
# pathologies" -- a textbook U-shape recovery plot.
#
# Usage (run in a detached screen so it survives terminal close):
#
#   screen -dmS phase_d_overtrained_chain \
#       bash scripts/local/chain_phase_d_overtrained.sh
#
# Idempotency: the underlying launcher already skips ckpts whose result
# JSON exists in the output dir, so re-launching is safe.

set -euo pipefail

# What screen session to wait for.
WAIT_FOR="${WAIT_FOR:-phase_d_epoch_curve}"
# What ckpts to run after the wait.
CKPT_TAGS="${CKPT_TAGS:-ep0249 ep0299 ep0349}"
# Poll interval in seconds.
POLL_SEC="${POLL_SEC:-60}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE_LOG="$REPO_ROOT/experiments/17_dp_hybrid/results/phase_d_2026_05_30/epoch_curve_local/queue_overtrained.log"

mkdir -p "$(dirname "$QUEUE_LOG")"

{
    echo "=========================================================="
    echo "[chain] starting: $(date)"
    echo "[chain] waiting for screen: '$WAIT_FOR'"
    echo "[chain] will fire on ckpts: $CKPT_TAGS"
    echo "[chain] poll every: ${POLL_SEC}s"
    echo "=========================================================="
} | tee -a "$QUEUE_LOG"

# Poll loop.
#
# CAREFUL: `screen -ls` exits with code 1 when ANY screens exist (success
# is exit 0, meaning "no sessions").  Under `set -o pipefail`, this turns
# the pipe `screen -ls | grep -q ...` into "always returns the leftmost
# non-zero", which fools any `if`/`while` test that uses the pipe directly.
# Workaround: capture screen's output to a variable (then grep on a string
# from `echo`, where pipefail can't bite us).
while true; do
    # `|| true` so set -e doesn't abort on screen's non-zero exit.
    screen_out=$(screen -ls 2>/dev/null || true)
    # `echo | grep` -- echo always returns 0, so pipefail is harmless here.
    if echo "$screen_out" | grep -q "$WAIT_FOR"; then
        echo "[chain] $(date '+%Y-%m-%d %H:%M:%S')  '$WAIT_FOR' still running, sleeping ${POLL_SEC}s" >> "$QUEUE_LOG"
        sleep "$POLL_SEC"
    else
        # Extra confirmation: poll one more time after 5 s in case screen -ls
        # was running concurrently with a transient session-state change.
        sleep 5
        screen_out=$(screen -ls 2>/dev/null || true)
        if echo "$screen_out" | grep -q "$WAIT_FOR"; then
            continue
        fi
        echo "[chain] $(date '+%Y-%m-%d %H:%M:%S')  '$WAIT_FOR' confirmed gone" >> "$QUEUE_LOG"
        break
    fi
done

{
    echo ""
    echo "=========================================================="
    echo "[chain] '$WAIT_FOR' is done at $(date)"
    echo "[chain] launching over-trained sweep ..."
    echo "=========================================================="
} | tee -a "$QUEUE_LOG"

# Fire the over-trained sweep.  Reuses the same launcher with different
# CKPT_TAGS.  We pass-through the same PYTHON and DEVICE.
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}" \
    CKPT_TAGS="$CKPT_TAGS" \
    bash "$REPO_ROOT/scripts/local/run_phase_d_epoch_curve.sh" 2>&1 | tee -a "$QUEUE_LOG"

rc=${PIPESTATUS[0]}

{
    echo ""
    echo "=========================================================="
    echo "[chain] over-trained sweep done at $(date), rc=$rc"
    echo "=========================================================="
} | tee -a "$QUEUE_LOG"

exit "$rc"
