#!/bin/bash
# Phase-D Q2 + Q3 on a single GCP VM (e.g. c3-standard-22).
#
#   Q2  lambda-sweep at ep0049 (K=10 fixed)
#       lambda in {0.1, 0.25, 1.0, 2.0}
#
#   Q3  K-sweep at ep0049 (lambda=0.5 fixed)
#       K in {5, 20, 40}
#
# Fans out all 7 runs in PARALLEL on the VM (3 cores per run -> 21 cores;
# leave 1 core for the OS).  Each run writes to its own subdir; idempotent
# (skips runs whose result JSON already exists).
#
# Usage:
#
#   bash scripts/cloud/run_phase_d_q2q3.sh \
#       --ckpt    ~/v_theta_dp_tobs2_h16_v3b_ep0049.pt \
#       --dataset ~/low_dim_v141.hdf5
#
# Optional:
#   --output-dir DIR    (default: ~/phase_d_q2q3_results)
#   --auto-stop         stop the VM via metadata service when done
#                       (requires VM created with --scopes=compute-rw)
#   --runs-filter STR   substring match against run tags (e.g. "q3" or
#                       "lam0p1") to run only a subset
#   --dry-run           print what would run without launching anything

set -euo pipefail

# ---------- arg parsing ----------
CKPT=""
DATASET=""
OUTPUT_DIR="$HOME/phase_d_q2q3_results"
AUTO_STOP=0
RUNS_FILTER=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --auto-stop) AUTO_STOP=1; shift ;;
        --runs-filter) RUNS_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: --ckpt is required and the file must exist (got: '$CKPT')" >&2
    exit 1
fi
if [ -z "$DATASET" ] || [ ! -f "$DATASET" ]; then
    echo "ERROR: --dataset is required and the file must exist (got: '$DATASET')" >&2
    exit 1
fi

# ---------- conda env ----------
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-hybrid-traj}"
# shellcheck disable=SC1091
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PYTHON="$(which python)"
echo "[run] python  : $PYTHON ($("$PYTHON" --version 2>&1))"

# ---------- repo ----------
REPO_DIR="${REPO_DIR:-$HOME/HRTG-dp}"
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: repo not found at $REPO_DIR (run gcp_bootstrap.sh first)" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "[run] repo    : $REPO_DIR  (HEAD=$(git rev-parse --short HEAD))"

# ---------- dataset symlink ----------
# robomimic expects the dataset at $robomimic_pkg/datasets/can/ph/low_dim_v141.hdf5
ROBO_DATASETS=$("$PYTHON" -c "import robomimic, os; print(os.path.dirname(os.path.dirname(robomimic.__file__)) + '/datasets/can/ph')")
mkdir -p "$ROBO_DATASETS"
ln -sf "$DATASET" "$ROBO_DATASETS/low_dim_v141.hdf5"
echo "[run] dataset : $DATASET -> $ROBO_DATASETS/low_dim_v141.hdf5"

# ---------- output dir ----------
mkdir -p "$OUTPUT_DIR"
echo "[run] output  : $OUTPUT_DIR"
echo ""

# ---------- shared eval knobs ----------
H=16
MPC_REPLANS=49
MPC_CHUNK_STEPS=8
N_DEMOS=10
N_ROLLOUTS=1
N_TEST_ROLLOUTS=50
TEST_START_SEED=100000
MAX_ENV_STEPS=400
SIGMA_MAX=0.25
SIGMA_MIN=0.05
PRIOR_N_STEPS=100
N_WORKERS=2

# ---------- run definitions ----------
# Format: "tag K lambda"
RUNS=(
    # Queue 2: lambda-sweep at ep0049 (K=10 fixed)
    "q2_lam0p1     10  0.1"
    "q2_lam0p25    10  0.25"
    "q2_lam1p0     10  1.0"
    "q2_lam2p0     10  2.0"
    # Queue 3: K-sweep at ep0049 (lambda=0.5 fixed)
    "q3_K5          5  0.5"
    "q3_K20        20  0.5"
    "q3_K40        40  0.5"
)

# Filter runs.
if [ -n "$RUNS_FILTER" ]; then
    NEW_RUNS=()
    for r in "${RUNS[@]}"; do
        if echo "$r" | grep -q "$RUNS_FILTER"; then NEW_RUNS+=("$r"); fi
    done
    RUNS=("${NEW_RUNS[@]}")
fi
echo "[run] planned runs (${#RUNS[@]}):"
for r in "${RUNS[@]}"; do echo "  $r"; done
echo ""

if [ "$DRY_RUN" = 1 ]; then
    echo "[run] --dry-run set; exiting without launching."
    exit 0
fi

# ---------- launch ----------
# Fan-out: one background python per run.
PIDS=()
TAGS=()
for r in "${RUNS[@]}"; do
    read -r tag K_val lam <<< "$r"
    out_dir="$OUTPUT_DIR/$tag"
    mkdir -p "$out_dir"

    # Idempotency check.
    expected_json="$out_dir/unified_flow_lam-${lam}_K-${K_val}_R-${MPC_REPLANS}.json"
    if [ -f "$expected_json" ] && [ "$(stat -c%s "$expected_json" 2>/dev/null || stat -f%z "$expected_json")" -gt 200 ]; then
        echo "[skip] $tag  (cached: $expected_json)"
        continue
    fi

    run_log="$out_dir/run.log"
    echo "[fire] $tag  K=$K_val  lambda=$lam  -> $out_dir"

    (
        nice -n 10 "$PYTHON" -u -m experiments.15_unified_hybrid.run_can \
            --task can \
            --prior flow \
            --H "$H" \
            --mpc-replans "$MPC_REPLANS" \
            --mpc-chunk-steps "$MPC_CHUNK_STEPS" \
            --n-demos "$N_DEMOS" \
            --n-rollouts "$N_ROLLOUTS" \
            --n-test-rollouts "$N_TEST_ROLLOUTS" \
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
echo "[run] launched ${#PIDS[@]} parallel run(s).  tail per-run logs:"
for t in "${TAGS[@]}"; do echo "  tail -f $OUTPUT_DIR/$t/run.log"; done
echo ""
echo "[run] waiting for all to finish ..."

# Track first failure but always wait for all.
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

# ---------- aggregate ----------
echo ""
echo "============================================================"
echo "Phase-D Q2+Q3 results"
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

# ---------- auto-stop ----------
if [ "$AUTO_STOP" = 1 ]; then
    echo ""
    echo "[run] --auto-stop requested; stopping VM via metadata service in 60 s ..."
    echo "[run] (cancel with Ctrl-C or 'sudo shutdown -c')"
    sleep 60
    META="http://metadata.google.internal/computeMetadata/v1"
    H='-H Metadata-Flavor: Google'
    TOKEN=$(curl -sS -H "Metadata-Flavor: Google" "$META/instance/service-accounts/default/token" \
        | python -c "import sys, json; print(json.load(sys.stdin)['access_token'])")
    INSTANCE=$(curl -sS -H "Metadata-Flavor: Google" "$META/instance/name")
    ZONE=$(curl -sS -H "Metadata-Flavor: Google" "$META/instance/zone" | awk -F/ '{print $NF}')
    PROJECT=$(curl -sS -H "Metadata-Flavor: Google" "$META/project/project-id")

    echo "[run] stopping  project=$PROJECT  zone=$ZONE  instance=$INSTANCE"
    curl -sS -X POST -H "Authorization: Bearer $TOKEN" \
        "https://compute.googleapis.com/compute/v1/projects/$PROJECT/zones/$ZONE/instances/$INSTANCE/stop"
    echo ""
    echo "[run] stop API called.  VM should be STOPPED in ~30 s."
    echo "[run] charges only for boot-disk storage (~\$0.04/GB/month) while stopped."
    echo "[run] to fully clean up:  gcloud compute instances delete $INSTANCE --zone=$ZONE"
fi

exit "$FAIL"
