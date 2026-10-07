#!/usr/bin/env bash
set -uo pipefail

REPO="/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup"
PYTHON="${PYTHON:-/Users/szh/anaconda3/envs/hybrid-traj/bin/python}"
OUT="$REPO/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07"
LAUNCHER="$REPO/experiments/08_flow_matching_prior/slate_square_dp_prior_2026_06_07.sh"
CKPT="$OUT/v_theta_dp_square_tobs2_h16_ep0199.pt"
CHAIN_LOG="$OUT/chain_after_train.log"
STANDALONE_JSON="$OUT/standalone_square_dp_test30.json"
HYBRID_SUMMARY="$OUT/unified_summary_flow_pull_K-10_R-49_square_dp_pull_test30.json"

echo "=== square-dp chain start $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
echo "waiting for final checkpoint: $CKPT" | tee -a "$CHAIN_LOG"

while true; do
  if [[ -f "$CKPT" ]]; then
    if "$PYTHON" - "$CKPT" >> "$CHAIN_LOG" 2>&1 <<'PY'
import sys, torch
path = sys.argv[1]
blob = torch.load(path, map_location="cpu", weights_only=False)
print({k: blob.get(k) for k in ("task", "arch", "format_version", "horizon")})
assert blob.get("task") == "square_dp", blob.get("task")
assert blob.get("arch") == "dp_unet1d", blob.get("arch")
assert int(blob.get("config", {}).get("obs_dim", -1)) == 23
PY
    then
      echo "checkpoint verified at $(date '+%F %T %Z')" | tee -a "$CHAIN_LOG"
      break
    fi
    echo "checkpoint exists but is not loadable yet; waiting" | tee -a "$CHAIN_LOG"
  fi

  if ! pgrep -af "train_dp.py" >/dev/null 2>&1; then
    echo "training process not detected; still waiting for final checkpoint" | tee -a "$CHAIN_LOG"
  fi
  sleep 300
done

if [[ -f "$STANDALONE_JSON" ]]; then
  echo "standalone eval already exists: $STANDALONE_JSON" | tee -a "$CHAIN_LOG"
else
  echo "=== standalone eval start $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
  MODE=eval DEVICE=cpu bash "$LAUNCHER" 2>&1 | tee -a "$CHAIN_LOG"
  rc=${PIPESTATUS[0]}
  echo "=== standalone eval done rc=$rc $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
fi

if [[ -f "$STANDALONE_JSON" ]]; then
  "$PYTHON" - "$STANDALONE_JSON" >> "$CHAIN_LOG" 2>&1 <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
print("standalone_summary", {
    "test_n_succ": j.get("test_n_succ"),
    "test_n": j.get("test_n"),
    "test_succ_rate": j.get("test_succ_rate"),
})
PY
fi

if [[ -f "$HYBRID_SUMMARY" ]]; then
  echo "hybrid eval already exists: $HYBRID_SUMMARY" | tee -a "$CHAIN_LOG"
else
  echo "=== hybrid eval start $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
  MODE=hybrid DEVICE=cpu N_WORKERS="${N_WORKERS:-4}" bash "$LAUNCHER" 2>&1 | tee -a "$CHAIN_LOG"
  rc=${PIPESTATUS[0]}
  echo "=== hybrid eval done rc=$rc $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
  if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
  fi
fi

if [[ -f "$HYBRID_SUMMARY" ]]; then
  "$PYTHON" - "$HYBRID_SUMMARY" >> "$CHAIN_LOG" 2>&1 <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
print("hybrid_summary_keys", sorted(j.keys()))
print("hybrid_summary", j)
PY
fi

echo "=== square-dp chain complete $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
