#!/usr/bin/env bash
# Re-pull the remote result roots that were documented (aggregate numbers) but never
# synced to the local project. See docs/results_writeups/vast_remote_offload_2026_06_11.md
# ("Data sync gaps"). Safe: rsync PULLS only, never deletes local files.
#
# The Vast SSH host/port are NOT recorded anywhere locally (only instance id 40441157).
# Supply them yourself. Get them from the Vast web UI (Instances -> the instance ->
# the "Direct ssh" connect string), or via the CLI:
#     pip install vastai && vastai set api-key <KEY> && vastai ssh-url 40441157
# which prints  ssh://root@<HOST>:<PORT>.
#
# Usage:
#     VAST_SSH_HOST=ssh5.vast.ai VAST_SSH_PORT=12345 \
#     [VAST_SSH_KEY=~/.ssh/id_ed25519] \
#     bash experiments/15_unified_hybrid/repull_vast_missing_roots_2026_06_11.sh
set -euo pipefail

: "${VAST_SSH_HOST:?set VAST_SSH_HOST (e.g. ssh5.vast.ai)}"
: "${VAST_SSH_PORT:?set VAST_SSH_PORT (e.g. 12345)}"
VAST_USER="${VAST_USER:-root}"
SSH_KEY_OPT=""
[ -n "${VAST_SSH_KEY:-}" ] && SSH_KEY_OPT="-i ${VAST_SSH_KEY}"

REMOTE_ROOT="/workspace/hrtg-cloud/HRTG-square-dp-followup"
# Local repo root = two levels up from this script's dir (experiments/15_unified_hybrid).
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL="experiments/15_unified_hybrid/results"

ROOTS=(
  "$REL/vast_safe_dp_2026_06_10"
  "$REL/tail_boost_2026_06_10"
  "$REL/tail_boost2_static_2026_06_10"
  "$REL/offload_chain_2026_06_10"
  "$REL/research_queue_remote_remainder_2026_06_10"
)

SSH_CMD="ssh -p ${VAST_SSH_PORT} ${SSH_KEY_OPT} -o StrictHostKeyChecking=accept-new"
echo "Remote : ${VAST_USER}@${VAST_SSH_HOST}:${VAST_SSH_PORT}:${REMOTE_ROOT}"
echo "Local  : ${LOCAL_ROOT}"
echo

# Quick connectivity + remote existence probe before pulling.
echo "=== probing remote roots (counts) ==="
$SSH_CMD "${VAST_USER}@${VAST_SSH_HOST}" "for d in ${ROOTS[*]}; do n=\$(find ${REMOTE_ROOT}/\$d -name '*.json' 2>/dev/null | wc -l); echo \"  \$d: \$n json\"; done" || {
  echo "!! cannot reach remote (instance terminated or wrong host/port). Aborting."; exit 4; }

echo; read -r -p "Pull the above into ${LOCAL_ROOT}? [y/N] " ans
[ "${ans:-N}" = "y" ] || { echo "aborted by user"; exit 0; }

for d in "${ROOTS[@]}"; do
  echo "=== rsync $d ==="
  mkdir -p "${LOCAL_ROOT}/${d}"
  rsync -avz --info=progress2 -e "$SSH_CMD" \
    "${VAST_USER}@${VAST_SSH_HOST}:${REMOTE_ROOT}/${d}/" \
    "${LOCAL_ROOT}/${d}/"
done

echo; echo "=== post-pull summary counts ==="
for d in "${ROOTS[@]}"; do
  printf "  %-50s summaries=%s\n" "$d" "$(find "${LOCAL_ROOT}/${d}" -name '*summary*.json' 2>/dev/null | wc -l | tr -d ' ')"
done
echo "DONE. Re-run the offload summarizer to regenerate per-shard DP numbers."
