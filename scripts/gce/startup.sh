#!/usr/bin/env bash
# Compute Engine startup script: runs once at boot, installs deps,
# clones the repo, runs the experiment chosen via instance metadata,
# uploads result JSONs to GCS, then optionally shuts the VM down.
#
# Required instance metadata keys (-metadata flag of `gcloud compute
# instances create`):
#
#   gcs-bucket    : gs://your-bucket-name (no trailing slash)
#   experiment    : one of {seed-sweep, exp05, exp05b, exp04, exp04-mjx,
#                            taxonomy, sigma-sweep, kxreplans,
#                            lambda-sweep, train-vtheta, custom}
#   git-branch    : (optional) defaults to feat/further-experiments
#                   (use feat/mjx-rollouts for exp04-mjx)
#   shutdown-when-done : "1" to `shutdown -h` after upload (default 1)
#   custom-cmd    : when experiment=custom, the bash command to run
#   mjx-n-substeps : (exp04-mjx only) mjx.step per policy step, default 5
#   mjx-h          : (exp04-mjx only) horizon, default 120
#   mjx-m          : (exp04-mjx only) MBD perturbations, default 16
#   mjx-k          : (exp04-mjx only) MBD iterations, default 10
#
# All output (stdout/stderr) lands in /var/log/syslog and in
# gs://$BUCKET/runs/<instance>/run.log via gcloud logging.

set -euo pipefail

log() { echo "[startup] $*" >&2; }

META=http://metadata.google.internal/computeMetadata/v1
get_meta() {
    curl -sf -H "Metadata-Flavor: Google" "$META/instance/attributes/$1" || true
}

INSTANCE_NAME=$(curl -sf -H "Metadata-Flavor: Google" "$META/instance/name")
BUCKET=$(get_meta gcs-bucket)
EXPERIMENT=$(get_meta experiment)
BRANCH=$(get_meta git-branch); BRANCH=${BRANCH:-feat/further-experiments}
SHUTDOWN=$(get_meta shutdown-when-done); SHUTDOWN=${SHUTDOWN:-1}
CUSTOM_CMD=$(get_meta custom-cmd)
VTHETA_GCS=$(get_meta vtheta-gcs)
BC_GCS=$(get_meta bc-gcs)
N_DEMOS=$(get_meta n-demos); N_DEMOS=${N_DEMOS:-5}
N_ROLLOUTS=$(get_meta n-rollouts); N_ROLLOUTS=${N_ROLLOUTS:-1}
N_WORKERS=$(get_meta n-workers); N_WORKERS=${N_WORKERS:-4}
MJX_N_SUBSTEPS=$(get_meta mjx-n-substeps); MJX_N_SUBSTEPS=${MJX_N_SUBSTEPS:-5}
MJX_H=$(get_meta mjx-h); MJX_H=${MJX_H:-120}
MJX_M=$(get_meta mjx-m); MJX_M=${MJX_M:-16}
MJX_K=$(get_meta mjx-k); MJX_K=${MJX_K:-10}

if [[ -z "$BUCKET" || -z "$EXPERIMENT" ]]; then
    log "ERROR: gcs-bucket and experiment metadata are required"
    exit 2
fi

log "instance=$INSTANCE_NAME bucket=$BUCKET experiment=$EXPERIMENT branch=$BRANCH"

WORK_DIR=/opt/hybrid-traj
RESULTS_GCS="$BUCKET/runs/$INSTANCE_NAME/$(date -u +%Y%m%dT%H%M%S)"
REPO_TARBALL="$BUCKET/repo/repo.tar.gz"

# --- 1. system + fetch repo tarball -------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libegl1 libgl1 libglfw3 libosmesa6 ffmpeg patchelf \
    >/dev/null

# --- Ops Agent for memory/disk monitoring in Cloud Console -----------------
log "installing Ops Agent"
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
bash add-google-cloud-ops-agent-repo.sh --also-install >/dev/null 2>&1
rm -f add-google-cloud-ops-agent-repo.sh
log "Ops Agent installed"

mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
rm -rf repo  # always start fresh from the tarball

log "fetching repo tarball from $REPO_TARBALL"
if gsutil cp "$REPO_TARBALL" /tmp/repo.tar.gz 2>/dev/null; then
    tar xzf /tmp/repo.tar.gz
    rm -f /tmp/repo.tar.gz
    log "repo extracted from tarball"
else
    log "ERROR: failed to fetch repo tarball; run scripts/gce/upload_repo.sh first"
    exit 5
fi
cd repo

# --- 2. python deps -----------------------------------------------------
# For exp04-mjx we use the GPU-aware install which:
#   * verifies nvidia-smi works (fails fast if drivers missing)
#   * installs jax[cuda12] + mujoco-mjx (no robosuite/robomimic/mink)
#   * runs an end-of-install smoke that crashes the VM if jax can't
#     see a GPU device
# All other experiments keep the existing CPU install path.
python3 -m venv .venv
. .venv/bin/activate

if [[ "$EXPERIMENT" == "exp04-mjx" ]]; then
    log "MJX experiment detected -- using GPU install path"
    bash scripts/remote/install_linux_gpu.sh
    # Activate the venv that install_linux_gpu.sh created at .venv.
    # (install_linux_gpu.sh creates the same .venv path so we are
    # already in it, but re-source to be safe.)
    . .venv/bin/activate

    # Gate-4 fail-fast smoke: if this exits non-zero we will NOT
    # proceed to the full experiment. Saves ~$0.70 / hr for the
    # duration we would otherwise spend running on CPU fallback.
    log "running Gate-4 GPU smoke (test_mjx_gpu_smoke.py)"
    if ! python experiments/04_mbd_baseline/test_mjx_gpu_smoke.py; then
        log "ERROR: GPU smoke failed; aborting before full experiment"
        log "       (results dir will still be uploaded for triage)"
        # Continue to the upload stage so /tmp/install_linux_gpu.log
        # and any partial output gets pushed to GCS.
        SMOKE_FAILED=1
    else
        SMOKE_FAILED=0
    fi
else
    bash scripts/remote/install_linux.sh
    SMOKE_FAILED=0
fi

# --- 2b. optional v_theta.pt staging ------------------------------------
# If the launcher passed `vtheta-gcs=gs://.../v_theta.pt`, fetch it into
# the canonical location so exp 07's runner can find it. The lambda
# sweep + later seed sweeps need this.
if [[ -n "$VTHETA_GCS" ]]; then
    log "fetching v_theta.pt from $VTHETA_GCS"
    mkdir -p experiments/08_flow_matching_prior/results
    gsutil cp "$VTHETA_GCS" \
        experiments/08_flow_matching_prior/results/v_theta.pt || \
        log "WARN: failed to fetch v_theta.pt; runs that need it will fall back to BC prior"
fi

# --- 2c. optional BC policy.pt staging ---------------------------------
# If the launcher passed `bc-gcs=gs://.../policy.pt`, fetch it into the
# canonical location so exp 07's `--prior bc` mode (the lambda-sweep-bc
# pattern) can find it. policy.pt is gitignored, so it is NOT in the
# repo tarball and must be staged via this side channel.
if [[ -n "$BC_GCS" ]]; then
    log "fetching BC policy.pt from $BC_GCS"
    mkdir -p experiments/03_bc_baseline/results
    gsutil cp "$BC_GCS" \
        experiments/03_bc_baseline/results/policy.pt || \
        log "WARN: failed to fetch policy.pt; --prior bc runs will FileNotFoundError"
fi

# --- 3. dispatch the experiment ----------------------------------------
case "$EXPERIMENT" in
    taxonomy)
        python experiments/06_failure_taxonomy/run.py
        ;;
    train-vtheta)
        python experiments/08_flow_matching_prior/train.py \
            --task can --n-demos 20 --horizon 120 --epochs 2000
        ;;
    exp04)
        python experiments/04_mbd_baseline/run_can.py \
            --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" --seed 0 --no-live-view \
            --n-workers "$N_WORKERS" \
            --output experiments/04_mbd_baseline/results/can_ph.json \
            --video-dir ""
        ;;
    exp04-mjx)
        # Gate-4 GPU experiment: MJX rollout, no robosuite. Use
        # --no-live-view + --video-dir '' unconditionally because the
        # MJX path has no renderer wired up (run_can.py's video / live
        # preview rely on the robosuite env). Default H/M/K come from
        # the migration plan's "headline" config (H=120, M=16, K=10).
        if [[ "$SMOKE_FAILED" == "1" ]]; then
            log "skipping exp04-mjx run because GPU smoke failed"
        else
            python experiments/04_mbd_baseline/run_can.py --mjx \
                --H "$MJX_H" --M "$MJX_M" --K "$MJX_K" \
                --mjx-n-substeps "$MJX_N_SUBSTEPS" \
                --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" \
                --seed 0 --no-live-view --video-dir "" \
                --output experiments/04_mbd_baseline/results/can_mjx.json
        fi
        ;;
    exp05)
        python experiments/05_hybrid_method/run_can.py \
            --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" --seed 0 --no-live-view \
            --deterministic-bc-reset \
            --n-workers "$N_WORKERS" \
            --output experiments/05_hybrid_method/results/can_ph.json \
            --video-dir ""
        ;;
    exp05b)
        python experiments/05_hybrid_method/run_can.py \
            --n-demos "$N_DEMOS" --n-rollouts "$N_ROLLOUTS" --seed 0 --no-live-view \
            --mpc-replans 1 --mpc-chunk-steps 60 \
            --deterministic-bc-reset \
            --n-workers "$N_WORKERS" \
            --output experiments/05_hybrid_method/results/can_ph_mpc.json \
            --video-dir ""
        ;;
    sigma-sweep)
        bash experiments/09_sigma_sweep/run.sh
        ;;
    kxreplans)
        bash experiments/10_kxreplans_grid/run.sh
        ;;
    lambda-sweep)
        python experiments/07_lambda_sweep/run_can.py \
            --prior flow \
            --n-demos 20 --n-rollouts 3 --seed 0 --no-live-view \
            --lambda-prior 0 0.1 0.5 1.0 5.0 inf
        ;;
    seed-sweep)
        SEEDS="0 1 2" bash experiments/14_seed_sweep/run.sh
        ;;
    custom)
        if [[ -z "$CUSTOM_CMD" ]]; then
            log "ERROR: experiment=custom but custom-cmd metadata empty"
            exit 3
        fi
        log "running custom: $CUSTOM_CMD"
        bash -c "$CUSTOM_CMD"
        ;;
    *)
        log "ERROR: unknown experiment '$EXPERIMENT'"
        exit 4
        ;;
esac

# --- 3b. publish v_theta.pt to a stable GCS path -----------------------
# After train-vtheta, also drop v_theta.pt at $BUCKET/v_theta/v_theta.pt
# so subsequent fan-out VMs can fetch it via VTHETA_GCS without
# guessing the per-run instance path.
if [[ "$EXPERIMENT" == "train-vtheta" && -f experiments/08_flow_matching_prior/results/v_theta.pt ]]; then
    gsutil cp experiments/08_flow_matching_prior/results/v_theta.pt \
        "$BUCKET/v_theta/v_theta.pt" || \
        log "WARN: failed to publish v_theta.pt to $BUCKET/v_theta/v_theta.pt"
    log "v_theta.pt published at $BUCKET/v_theta/v_theta.pt"
fi

# --- 4. upload results to GCS ------------------------------------------
log "uploading results to $RESULTS_GCS"
# gsutil ships with the GCE base image (Debian/Ubuntu) by default.
gsutil -m cp -r experiments/*/results "$RESULTS_GCS/" || \
    log "WARN: gsutil cp failed; check IAM scopes (storage.objects.create)"

# Tarball + log too, for one-shot download.
tar czf /tmp/results.tar.gz experiments/*/results || true
gsutil cp /tmp/results.tar.gz "$RESULTS_GCS/results.tar.gz" || true

log "done; results at $RESULTS_GCS"

# --- 5. optional auto-shutdown -----------------------------------------
if [[ "$SHUTDOWN" == "1" ]]; then
    log "auto-shutdown in 60 s"
    sleep 60
    shutdown -h now
fi
