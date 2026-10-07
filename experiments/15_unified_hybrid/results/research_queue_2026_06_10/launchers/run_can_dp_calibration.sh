#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/can_dp_calibration/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane can_dp_calibration] start '$(date)
echo '[lane can_dp_calibration] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane can_dp_calibration] worker_budget=2'

echo '[lane can_dp_calibration] cell can_dp_ep0199_smax0p1_lamgrid start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 29 --max-env-steps 240 --K 10 --M 16 --lambda-prior 0 0.5 1 2 5 10 --sigma-max 0.1 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/can_dp_calibration' --output-name-suffix _can_dp_ep0199_smax0p1_lamgrid
rc=$?
echo '[lane can_dp_calibration] cell can_dp_ep0199_smax0p1_lamgrid done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane can_dp_calibration] cell can_dp_ep0199_smax0p25_lamgrid start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior flow --score-mode pull --prior-checkpoint '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt' --H 16 --mpc-chunk-steps 8 --mpc-replans 29 --max-env-steps 240 --K 10 --M 16 --lambda-prior 0 0.5 1 2 5 10 --sigma-max 0.25 --sigma-min 0.05 --prior-n-steps 100 --deterministic-prior-warmstart --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --device cpu --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/can_dp_calibration' --output-name-suffix _can_dp_ep0199_smax0p25_lamgrid
rc=$?
echo '[lane can_dp_calibration] cell can_dp_ep0199_smax0p25_lamgrid done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane can_dp_calibration] all done '$(date)
