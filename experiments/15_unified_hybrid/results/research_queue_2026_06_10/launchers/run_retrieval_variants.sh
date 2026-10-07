#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane retrieval_variants] start '$(date)
echo '[lane retrieval_variants] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane retrieval_variants] worker_budget=2'

echo '[lane retrieval_variants] cell can_demo_1nn_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_bank20_fresh --prior-bank-size 20
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_1nn_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_bank200_fresh --prior-bank-size 200
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_knn_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_knn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_knn_bank20_fresh --prior-bank-size 20 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_knn_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_knn_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_knn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_knn_bank200_fresh --prior-bank-size 200 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_knn_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_lwr_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_lwr --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_lwr_bank20_fresh --prior-bank-size 20 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_lwr_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_lwr_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_lwr --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_lwr_bank200_fresh --prior-bank-size 200 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_lwr_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_topk_random_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_topk_random --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_topk_random_bank20_fresh --prior-bank-size 20 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_topk_random_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_topk_random_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_topk_random --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_topk_random_bank200_fresh --prior-bank-size 200 --knn-k 5
rc=$?
echo '[lane retrieval_variants] cell can_demo_topk_random_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_softnn_beta0p5_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_softnn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_softnn_beta0p5_bank20_fresh --prior-bank-size 20 --knn-k 5 --retrieval-beta 0.5
rc=$?
echo '[lane retrieval_variants] cell can_demo_softnn_beta0p5_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_softnn_beta0p5_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_softnn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_softnn_beta0p5_bank200_fresh --prior-bank-size 200 --knn-k 5 --retrieval-beta 0.5
rc=$?
echo '[lane retrieval_variants] cell can_demo_softnn_beta0p5_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_softnn_beta2p0_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_softnn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_softnn_beta2p0_bank20_fresh --prior-bank-size 20 --knn-k 5 --retrieval-beta 2.0
rc=$?
echo '[lane retrieval_variants] cell can_demo_softnn_beta2p0_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_softnn_beta2p0_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_softnn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_softnn_beta2p0_bank200_fresh --prior-bank-size 200 --knn-k 5 --retrieval-beta 2.0
rc=$?
echo '[lane retrieval_variants] cell can_demo_softnn_beta2p0_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_1nn_proprio_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn_proprio --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_proprio_bank20_fresh --prior-bank-size 20
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_proprio_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_1nn_proprio_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn_proprio --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_proprio_bank200_fresh --prior-bank-size 200
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_proprio_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_1nn_object_bank20_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn_object --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_object_bank20_fresh --prior-bank-size 20
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_object_bank20_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] cell can_demo_1nn_object_bank200_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn_object --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/retrieval_variants' --output-name-suffix _can_demo_1nn_object_bank200_fresh --prior-bank-size 200
rc=$?
echo '[lane retrieval_variants] cell can_demo_1nn_object_bank200_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane retrieval_variants] all done '$(date)
