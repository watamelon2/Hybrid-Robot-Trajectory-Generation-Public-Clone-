#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane coverage_fresh] start '$(date)
echo '[lane coverage_fresh] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane coverage_fresh] worker_budget=2'

echo '[lane coverage_fresh] cell can_fresh_bank1_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank1_seed100000 --prior-bank-size 1
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank1_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank1_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank1_seed200000 --prior-bank-size 1
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank1_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank2_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank2_seed100000 --prior-bank-size 2
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank2_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank2_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank2_seed200000 --prior-bank-size 2
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank2_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank5_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank5_seed100000 --prior-bank-size 5
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank5_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank5_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank5_seed200000 --prior-bank-size 5
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank5_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank10_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank10_seed100000 --prior-bank-size 10
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank10_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank10_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank10_seed200000 --prior-bank-size 10
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank10_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank20_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank20_seed100000 --prior-bank-size 20
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank20_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank20_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank20_seed200000 --prior-bank-size 20
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank20_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank50_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank50_seed100000 --prior-bank-size 50
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank50_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank50_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank50_seed200000 --prior-bank-size 50
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank50_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank100_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank100_seed100000 --prior-bank-size 100
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank100_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank100_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank100_seed200000 --prior-bank-size 100
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank100_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank200_seed100000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank200_seed100000 --prior-bank-size 200
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank200_seed100000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] cell can_fresh_bank200_seed200000 start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 200000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/coverage_fresh' --output-name-suffix _can_fresh_bank200_seed200000 --prior-bank-size 200
rc=$?
echo '[lane coverage_fresh] cell can_fresh_bank200_seed200000 done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane coverage_fresh] all done '$(date)
