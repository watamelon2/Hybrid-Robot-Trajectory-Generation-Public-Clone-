#!/usr/bin/env bash
set -uo pipefail
LOG='/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction/queue.log'
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo '[lane adaptation_mass_friction] start '$(date)
echo '[lane adaptation_mass_friction] repo=/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup'
echo '[lane adaptation_mass_friction] worker_budget=2'

echo '[lane adaptation_mass_friction] cell can_mass0p5_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass0p5_canonicalbank_fresh --mass-scale 0.5
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass0p5_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass0p5_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass0p5_bank205_fresh --mass-scale 0.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass0p5_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass0p5_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass0p5_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass0p5_bank220_fresh --mass-scale 0.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass0p5_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass0p5_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p3_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p3_canonicalbank_fresh --mass-scale 1.3
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p3_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p3_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p3_bank205_fresh --mass-scale 1.3 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass1p3_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p3_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p3_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p3_bank220_fresh --mass-scale 1.3 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass1p3_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p3_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p6_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p6_canonicalbank_fresh --mass-scale 1.6
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p6_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p6_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p6_bank205_fresh --mass-scale 1.6 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass1p6_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p6_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass1p6_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass1p6_bank220_fresh --mass-scale 1.6 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass1p6_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass1p6_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass2p5_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass2p5_canonicalbank_fresh --mass-scale 2.5
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass2p5_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass2p5_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass2p5_bank205_fresh --mass-scale 2.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass2p5_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass2p5_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_mass2p5_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_mass2p5_bank220_fresh --mass-scale 2.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass2p5_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_mass2p5_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p3_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p3_canonicalbank_fresh --friction-scale 0.3
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p3_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p3_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p3_bank205_fresh --friction-scale 0.3 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p3_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p3_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p3_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p3_bank220_fresh --friction-scale 0.3 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p3_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p3_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p5_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p5_canonicalbank_fresh --friction-scale 0.5
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p5_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p5_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p5_bank205_fresh --friction-scale 0.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p5_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p5_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p5_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p5_bank220_fresh --friction-scale 0.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p5_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p5_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p7_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p7_canonicalbank_fresh --friction-scale 0.7
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p7_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p7_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p7_bank205_fresh --friction-scale 0.7 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p7_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p7_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric0p7_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric0p7_bank220_fresh --friction-scale 0.7 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric0p7_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric0p7_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric1p5_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric1p5_canonicalbank_fresh --friction-scale 1.5
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric1p5_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric1p5_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric1p5_bank205_fresh --friction-scale 1.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric1p5_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric1p5_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric1p5_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric1p5_bank220_fresh --friction-scale 1.5 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric1p5_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric1p5_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric2p0_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric2p0_canonicalbank_fresh --friction-scale 2.0
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric2p0_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric2p0_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric2p0_bank205_fresh --friction-scale 2.0 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric2p0_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric2p0_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric2p0_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric2p0_bank220_fresh --friction-scale 2.0 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric2p0_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric2p0_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric3p0_canonicalbank_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric3p0_canonicalbank_fresh --friction-scale 3.0
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric3p0_canonicalbank_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric3p0_bank205_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric3p0_bank205_fresh --friction-scale 3.0 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric3p0_205demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric3p0_bank205_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] cell can_fric3p0_bank220_fresh start '$(date)
/Users/szh/anaconda3/envs/hybrid-traj/bin/python '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/run_can.py' --task can --prior demo_1nn --lambda-prior 1 --combine-mode score --H 120 --mpc-replans 1 --mpc-chunk-steps 60 --max-env-steps 120 --K 10 --M 16 --sigma-max 0.25 --sigma-min 0.05 --n-demos 0 --n-rollouts 0 --n-test-rollouts 30 --test-start-seed 100000 --n-workers 2 --seed 0 --output-dir '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-square-dp-followup/experiments/15_unified_hybrid/results/research_queue_2026_06_10/adaptation_mass_friction' --output-name-suffix _can_fric3p0_bank220_fresh --friction-scale 3.0 --prior-bank-dataset '/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_fric3p0_220demos.hdf5' --prior-bank-task can
rc=$?
echo '[lane adaptation_mass_friction] cell can_fric3p0_bank220_fresh done rc='$rc' '$(date)
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

echo '[lane adaptation_mass_friction] all done '$(date)
