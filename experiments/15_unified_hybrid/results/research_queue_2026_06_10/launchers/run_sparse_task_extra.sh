#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane sparse_task_extra] start '$(date)
echo '[lane sparse_task_extra] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane sparse_task_extra] worker_budget=1'

echo '[lane sparse_task_extra] cell th_lam_ablate_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task tool_hang --prior demo_1nn --combine-mode score --H 300 --mpc-replans 3 --mpc-chunk-steps 75 --max-env-steps 300 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 5 --n-rollouts 1 --save-step-trace --save-executed-actions --log-gate-diagnostic --n-workers 1 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra' --output-name-suffix _th_lam_ablate_trace --lambda-prior 0 1 inf
rc=$?
echo '[lane sparse_task_extra] cell th_lam_ablate_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane sparse_task_extra] cell th_raw_sparse_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task tool_hang --prior demo_1nn --combine-mode score --H 300 --mpc-replans 3 --mpc-chunk-steps 75 --max-env-steps 300 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 5 --n-rollouts 1 --save-step-trace --save-executed-actions --log-gate-diagnostic --n-workers 1 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra' --output-name-suffix _th_raw_sparse_trace --lambda-prior 1 --raw-sparse-reward
rc=$?
echo '[lane sparse_task_extra] cell th_raw_sparse_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane sparse_task_extra] cell th_demo_phase_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task tool_hang --prior demo_1nn --combine-mode score --H 300 --mpc-replans 3 --mpc-chunk-steps 75 --max-env-steps 300 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 5 --n-rollouts 1 --save-step-trace --save-executed-actions --log-gate-diagnostic --n-workers 1 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra' --output-name-suffix _th_demo_phase_trace --lambda-prior 1 --demo-phase-reward --demo-phase-scale 1000
rc=$?
echo '[lane sparse_task_extra] cell th_demo_phase_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane sparse_task_extra] cell tr_lam_ablate_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task transport --prior demo_1nn --combine-mode score --H 300 --mpc-replans 3 --mpc-chunk-steps 75 --max-env-steps 300 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 5 --n-rollouts 1 --save-step-trace --save-executed-actions --log-gate-diagnostic --n-workers 1 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra' --output-name-suffix _tr_lam_ablate_trace --lambda-prior 0 1 inf
rc=$?
echo '[lane sparse_task_extra] cell tr_lam_ablate_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane sparse_task_extra] cell tr_demo_phase_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task transport --prior demo_1nn --combine-mode score --H 300 --mpc-replans 3 --mpc-chunk-steps 75 --max-env-steps 300 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 5 --n-rollouts 1 --save-step-trace --save-executed-actions --log-gate-diagnostic --n-workers 1 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/sparse_task_extra' --output-name-suffix _tr_demo_phase_trace --lambda-prior 1 --demo-phase-reward --demo-phase-scale 1000
rc=$?
echo '[lane sparse_task_extra] cell tr_demo_phase_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane sparse_task_extra] all done '$(date)
