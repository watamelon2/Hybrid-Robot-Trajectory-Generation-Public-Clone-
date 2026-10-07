#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/offload_chain_2026_06_10/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/queue_offload_chain_2026_06_10.py' --offload-chain-run
exit $?
