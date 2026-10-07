#!/usr/bin/env bash
# tail -f the run.log of whichever stage is *currently active* in the
# overnight queue. Picks the newest run.log under
# results/overnight_2026_05_24/{standalone_*,mbd_*}/ that is still being
# written to (i.e. mtime within the last 60s).
#
# Falls back to tailing queue.log if no active stage is detected.
#
# Usage: bash experiments/08_flow_matching_prior/overnight_tail.sh
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="$REPO/experiments/08_flow_matching_prior/results/overnight_2026_05_24"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "no run dir: $RUN_DIR" >&2
    exit 1
fi

# Find the most recently modified run.log
latest=$(find "$RUN_DIR" -name "run.log" -type f -mmin -2 -print 2>/dev/null \
         | xargs -I{} stat -f "%m %N" {} 2>/dev/null \
         | sort -rn | head -1 | cut -d' ' -f2-)

if [[ -z "$latest" ]]; then
    echo "no active stage run.log; tailing queue.log instead"
    echo "------"
    tail -f "$RUN_DIR/queue.log"
else
    echo "tailing active stage: $latest"
    echo "------"
    tail -f "$latest"
fi
