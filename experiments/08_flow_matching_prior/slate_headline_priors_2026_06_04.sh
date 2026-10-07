#!/usr/bin/env bash
# Headline methods-comparison re-run (2026-06-04).
#
# Purpose: put all THREE trained priors on ONE identical protocol so the
# deck's "methods comparison" figure is apples-to-apples and uses the
# *selected* DP checkpoint (ep0199, not ep0149) and the *real* CFM (v6),
# which the current figure omits entirely.
#
# Protocol: IDENTICAL EVAL SET (50 fresh-seed test rollouts, same seeds,
# max_env_steps 400), but EACH prior at its OWN headline operating point
# -- forcing one sampler-K / n_action_steps across architectures is wrong
# (DP is H=16 closed-loop with DDPM; v6/v3 are H=120 near-open-loop Heun).
# This is the standard "each method's best config on the same test set".
#   - 50 fresh-seed TEST rollouts  (--n-test-rollouts 50 --test-start-seed 100000)
#   - no train-init rollouts       (--n-demos 0)
#   per-prior sampler steps (n_steps) and chunk size (n_action_steps):
#     v6 CFM : n_steps=18 Heun,  n_action_steps=60  (its 83.3% protocol)
#     DP     : n_steps=100 DDPM, n_action_steps=8   (its Phase-B 80%-test protocol)
#     v3 EDM : n_steps=18 Heun,  n_action_steps=60  (H=120 family, like v6)
#
# Priors (standalone column = standalone_dp.py, no MBD):
#   - v6 CFM            : v_theta_v6_unet_xl.pt        (arch cfm_v6)  -- NEW on figure
#   - DP (selected)     : v_theta_dp_tobs2_h16_v3b_ep0199.pt (dp_unet1d, 80% test winner)
#   - v3 EDM (sigma-spec): v_theta_v3.pt               (karras_edm_v3) -- the slide's "flow-matching"
#
# This is the STANDALONE phase (Phase 1).  The +MBD-bridge column is a
# separate, per-prior phase (different H / replan / lambda per prior) and
# is intentionally NOT in this script.
#
# MUST be run in the `hybrid-traj` conda env (torch lives there).
# DO NOT launch while T4 is running -- Mac A is CPU-saturated.

set -uo pipefail

cd "/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation" || exit 1

OUT="experiments/08_flow_matching_prior/results/headline_priors_2026_06_04"
mkdir -p "$OUT"
LOG="$OUT/slate.log"
echo "=== headline priors slate start $(date '+%F %T %Z') ===" | tee -a "$LOG"

STANDALONE="experiments/08_flow_matching_prior/standalone_dp.py"

# tag|checkpoint|n_steps|n_action_steps
CELLS=(
  "v6_cfm|experiments/08_flow_matching_prior/results/v_theta_v6_unet_xl.pt|18|60"
  "dp_ep0199|experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0199.pt|100|8"
  "v3_edm|experiments/08_flow_matching_prior/results/v_theta_v3.pt|18|60"
)

for entry in "${CELLS[@]}"; do
  IFS='|' read -r tag ckpt nsteps nact <<< "$entry"
  out_name="standalone_${tag}_test50.json"
  if [[ -f "$OUT/$out_name" ]]; then
    echo "=== $(date '+%T')  $tag  already done, skipping ===" | tee -a "$LOG"
    continue
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "=== $(date '+%T')  $tag  MISSING ckpt $ckpt -- skipping ===" | tee -a "$LOG"
    continue
  fi
  echo "=== $(date '+%T')  $tag  starting (n_steps=$nsteps, n_action_steps=$nact, 50 test seeds) ===" | tee -a "$LOG"
  PYTHONHASHSEED=0 OMP_NUM_THREADS=2 \
  python "$STANDALONE" \
      --task can \
      --prior-checkpoint "$ckpt" \
      --n-demos 0 --n-rollouts 0 \
      --n-test-rollouts 50 --test-start-seed 100000 \
      --n-action-steps "$nact" --max-env-steps 400 \
      --n-steps "$nsteps" \
      --device cpu \
      --output-dir "$OUT" --output-name "$out_name" \
      2>&1 | tee -a "$LOG"
  echo "=== $(date '+%T')  $tag  rc=${PIPESTATUS[0]} ===" | tee -a "$LOG"
done

echo "=== summary $(date '+%T') ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os
d="experiments/08_flow_matching_prior/results/headline_priors_2026_06_04"
print(f"{'cell':<14} {'arch':<14} test_succ  n_test")
for f in sorted(glob.glob(d+"/standalone_*_test50.json")):
    j=json.load(open(f))
    print(f"{os.path.basename(f).replace('standalone_','').replace('_test50.json',''):<14} "
          f"{str(j.get('prior_arch')):<14} {j.get('test_succ_rate'):.3f}     {j.get('test_n')}")
PY
echo "=== headline priors slate done $(date '+%F %T %Z') ===" | tee -a "$LOG"
