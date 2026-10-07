#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/square_dp_checkpoint_calibration/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane square_dp_checkpoint_calibration] start '$(date)
echo '[lane square_dp_checkpoint_calibration] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane square_dp_checkpoint_calibration] worker_budget=2'

echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0049_lamgrid_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task square --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07/v_theta_dp_square_tobs2_h16_ep0199_ep0049.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 --K 10 --M 16 --lambda-prior 0 0.5 1 2 --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --save-step-trace --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/square_dp_checkpoint_calibration' --output-name-suffix _square_dp_ep0049_lamgrid_trace
rc=$?
echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0049_lamgrid_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0099_lamgrid_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task square --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07/v_theta_dp_square_tobs2_h16_ep0199_ep0099.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 --K 10 --M 16 --lambda-prior 0 0.5 1 2 --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --save-step-trace --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/square_dp_checkpoint_calibration' --output-name-suffix _square_dp_ep0099_lamgrid_trace
rc=$?
echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0099_lamgrid_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0149_lamgrid_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task square --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07/v_theta_dp_square_tobs2_h16_ep0199_ep0149.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 --K 10 --M 16 --lambda-prior 0 0.5 1 2 --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --save-step-trace --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/square_dp_checkpoint_calibration' --output-name-suffix _square_dp_ep0149_lamgrid_trace
rc=$?
echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0149_lamgrid_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0199_lamgrid_trace start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task square --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/square_dp_prior_2026_06_07/v_theta_dp_square_tobs2_h16_ep0199.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 49 --max-env-steps 400 --K 10 --M 16 --lambda-prior 0 0.5 1 2 --sigma-max 0.10 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --save-step-trace --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/square_dp_checkpoint_calibration' --output-name-suffix _square_dp_ep0199_lamgrid_trace
rc=$?
echo '[lane square_dp_checkpoint_calibration] cell square_dp_ep0199_lamgrid_trace done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane square_dp_checkpoint_calibration] all done '$(date)
