#!/usr/bin/env bash
# Pull the GCE-completed pareto K{K}_R{R}.json files into the same
# local dir the local runner writes to. After this script, the
# experiments/10_kxreplans_grid/results/ directory contains all
# (up to) 9 cells -- 6 from GCE + 3 from the local runner -- and
# downstream analysis sees a flat directory.
#
# Idempotent: SKIPS any local file that already exists and is
# non-empty. So you can re-run safely after the GCE side finishes
# more cells.
#
# Why this exists:
#   GCS layout is gs://BUCKET/runs/<vm>/<utc>/results/K{K}_R{R}.json
#   but locally we want experiments/10_kxreplans_grid/results/K{K}_R{R}.json
#   This script bridges that.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BUCKET="${BUCKET:-gs://hybrid-traj-robin-2026-results}"
PREFIX="${PREFIX:-ht-r46394-}"    # the overnight run ID
OUT_DIR="${OUT_DIR:-experiments/10_kxreplans_grid/results}"

mkdir -p "$OUT_DIR"

echo "fetching pareto K*_R*.json from $BUCKET/runs/${PREFIX}* ..."
echo "into $OUT_DIR/"

# All VM dirs whose name starts with our run prefix:
vm_uris="$(gsutil ls "$BUCKET/runs/" 2>/dev/null | grep -E "/${PREFIX}k[0-9]+-r[0-9]+-[0-9]+/" || true)"
if [[ -z "$vm_uris" ]]; then
    echo "  no completed GCE VMs found under prefix $PREFIX"
    exit 0
fi

n_downloaded=0
n_skipped=0
while IFS= read -r vm_uri; do
    # Find the K{K}_R{R}.json under this VM's results tree. There
    # is exactly one per VM (the launcher submits one slug per VM).
    json_uri="$(gsutil ls -r "$vm_uri" 2>/dev/null \
                | grep -E '/K[0-9]+_R[0-9]+\.json$' \
                | head -1)"
    if [[ -z "$json_uri" ]]; then
        echo "  [skip] no K*_R*.json in $vm_uri"
        continue
    fi
    fname="$(basename "$json_uri")"
    dest="$OUT_DIR/$fname"
    if [[ -s "$dest" ]]; then
        echo "  [skip] $fname (already present locally, $(wc -c < "$dest" | tr -d ' ')B)"
        n_skipped=$((n_skipped+1))
        continue
    fi
    gsutil cp "$json_uri" "$dest" 2>&1 | grep -v 'Copying' | sed 's/^/  /'
    echo "  [ok]   $fname  <-  $json_uri"
    n_downloaded=$((n_downloaded+1))
done <<< "$vm_uris"

echo ""
echo "downloaded: $n_downloaded   already-present skipped: $n_skipped"
echo ""
echo "current state of $OUT_DIR:"
ls -la "$OUT_DIR"/K*_R*.json 2>/dev/null | awk '{ printf "  %s  %s\n", $5, $NF }' \
    || echo "  (empty)"
