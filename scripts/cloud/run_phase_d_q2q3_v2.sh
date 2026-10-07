#!/bin/bash
# Phase-D Q2 + Q3 (V2: deadline-cut for 2026-05-31 Monday presentation).
#
# Differences vs v1:
#   * DROP q3_K40 (the long pole; 60 rollouts @ K=40 = ~5 days on c3-22)
#   * q2_lam* and q3_K5 use N_TEST_ROLLOUTS=20 (was 50)  -> 30 rollouts/run
#   * q3_K20 uses N_TEST_ROLLOUTS=10                       -> 20 rollouts/run
#
# Net work: 4*30 + 30 + 20 = 170 rollouts (was 420).
#
# Usage:
#
#   bash scripts/cloud/run_phase_d_q2q3_v2.sh \
#       --ckpt    ~/v_theta_dp_tobs2_h16_v3b_ep0049.pt \
#       --dataset ~/low_dim_v141.hdf5

set -euo pipefail

CKPT=""
DATASET=""
OUTPUT_DIR="$HOME/phase_d_q2q3_results"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[ -f "$CKPT" ]    || { echo "ERROR: ckpt missing: $CKPT" >&2; exit 1; }
[ -f "$DATASET" ] || { echo "ERROR: dataset missing: $DATASET" >&2; exit 1; }

MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-hybrid-traj}"
# shellcheck disable=SC1091
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
PYTHON="$(which python)"
echo "[run] python : $PYTHON ($("$PYTHON" --version 2>&1))"

REPO_DIR="${REPO_DIR:-$HOME/HRTG-dp}"
[ -d "$REPO_DIR" ] || { echo "ERROR: repo missing: $REPO_DIR" >&2; exit 1; }
cd "$REPO_DIR"

# Ensure dataset symlink robomimic expects.
ROBO_DATASETS=$("$PYTHON" -c "import robomimic, os; print(os.path.dirname(os.path.dirname(robomimic.__file__)) + '/datasets/can/ph')")
mkdir -p "$ROBO_DATASETS"
ln -sf "$DATASET" "$ROBO_DATASETS/low_dim_v141.hdf5"
echo "[run] dataset: $DATASET -> $ROBO_DATASETS/low_dim_v141.hdf5"

mkdir -p "$OUTPUT_DIR"
echo "[run] output : $OUTPUT_DIR"
echo ""

# Shared eval knobs.
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
TEST_START_SEED=100000
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100
N_WORKERS=2

# Per-run config: "tag  K  lambda  n_test"
RUNS=(
    "q2_lam0p1     10  0.1   20"
    "q2_lam0p25    10  0.25  20"
    "q2_lam1p0     10  1.0   20"
    "q2_lam2p0     10  2.0   20"
    "q3_K5          5  0.5   20"
    "q3_K20        20  0.5   10"
)

echo "[run] planned runs (${#RUNS[@]}):"
for r in "${RUNS[@]}"; do echo "  $r"; done
echo ""

if [ "$DRY_RUN" = 1 ]; then
    echo "[run] --dry-run set; exiting."
    exit 0
fi

PIDS=()
TAGS=()
for r in "${RUNS[@]}"; do
    read -r tag K_val lam n_test <<< "$r"
    out_dir="$OUTPUT_DIR/$tag"
    mkdir -p "$out_dir"

    expected_json="$out_dir/unified_flow_lam-${lam}_K-${K_val}_R-${MPC_REPLANS}.json"
    if [ -f "$expected_json" ] && [ "$(stat -c%s "$expected_json" 2>/dev/null || stat -f%z "$expected_json")" -gt 200 ]; then
        echo "[skip] $tag  (cached: $expected_json)"
        continue
    fi

    run_log="$out_dir/run.log"
    echo "[fire] $tag  K=$K_val  lambda=$lam  n_test=$n_test  -> $out_dir"

    (
        nice -n 10 "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
            --task can \
            --prior flow \
            --H "$H" \
            --mpc-replans "$MPC_REPLANS" \
            --mpc-chunk-steps "$MPC_CHUNK_STEPS" \
            --n-demos "$N_DEMOS" \
            --n-rollouts "$N_ROLLOUTS" \
            --n-test-rollouts "$n_test" \
            --test-start-seed "$TEST_START_SEED" \
            --max-env-steps "$MAX_ENV_STEPS" \
            --prior-n-steps "$PRIOR_N_STEPS" \
            --sigma-max "$SIGMA_MAX" \
            --sigma-min "$SIGMA_MIN" \
            --lambda-prior "$lam" \
            --device cpu \
            --seed 0 \
            --K "$K_val" \
            --n-workers "$N_WORKERS" \
            --prior-checkpoint "$CKPT" \
            --output-dir "$out_dir" \
            2>&1 | sed "s|^|[$tag] |"
    ) > "$run_log" 2>&1 &
    PIDS+=("$!")
    TAGS+=("$tag")
done

echo ""
echo "[run] launched ${#PIDS[@]} parallel run(s)."
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[FAIL] ${TAGS[$i]} returned non-zero" | tee -a "$OUTPUT_DIR/master.log"
        FAIL=1
    else
        echo "[ OK ] ${TAGS[$i]} done" | tee -a "$OUTPUT_DIR/master.log"
    fi
done

echo ""
echo "[run] all parallel runs finished at $(date)"

echo ""
echo "============================================================"
echo "Phase-D Q2+Q3 (V2) results"
echo "============================================================"
"$PYTHON" - <<PY
import json, glob, os
out = "$OUTPUT_DIR"
rows = []
for d in sorted(os.listdir(out)):
    sub = os.path.join(out, d)
    if not os.path.isdir(sub):
        continue
    matches = sorted(glob.glob(f"{sub}/unified_flow_lam-*_K-*_R-*.json"))
    if not matches:
        rows.append((d, None))
        continue
    js = json.load(open(matches[0]))
    rows.append((d, js))

print(f"{'tag':<14} {'lambda':>6} {'K':>3}  {'train':>10}  {'test':>10}  {'total':>10}")
print("-" * 64)
for tag, js in rows:
    if js is None:
        print(f"  {tag:<12}     ---   --   ---    ---    ---")
        continue
    lam = js.get("lambda_prior", -1)
    K_val = js.get("K", -1) or js.get("params", {}).get("K", -1)
    train_n = js.get("train_n", 0); train_s = js.get("train_n_succ", 0)
    test_n  = js.get("test_n",  0); test_s  = js.get("test_n_succ", 0)
    tot_n   = js.get("n_total", 0); tot_s   = js.get("n_success", 0)
    train_p = train_s / max(1, train_n) * 100
    test_p  = test_s  / max(1, test_n)  * 100
    tot_p   = tot_s   / max(1, tot_n)   * 100
    print(f"  {tag:<12}  {lam:>5.2f}  {K_val:>3}  {train_s}/{train_n}={train_p:>4.0f}%  "
          f"{test_s}/{test_n}={test_p:>4.0f}%  {tot_s}/{tot_n}={tot_p:>4.0f}%")
PY
echo "============================================================"

exit "$FAIL"
