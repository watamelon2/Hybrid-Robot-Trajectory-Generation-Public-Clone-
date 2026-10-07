#!/usr/bin/env bash
# Launch a Compute Engine VM that runs one experiment overnight,
# uploads JSON results to GCS, and shuts itself down.
#
# Usage:
#   PROJECT=my-gcp-project BUCKET=gs://my-bucket \
#       scripts/gce/launch.sh seed-sweep
#
# Optional env vars:
#   ZONE         (default: us-central1-a)
#   MACHINE_TYPE (default: e2-standard-8 -- 8 vCPU, 32 GB RAM)
#   PREEMPTIBLE  (default: 1 -- spot/preemptible; cheaper, may be killed)
#   BRANCH       (default: feat/further-experiments)
#   NAME         (default: hybrid-traj-<experiment>-<timestamp>)
#   ACCELERATOR  (default: empty; set to "l4" to attach one NVIDIA L4
#                 GPU and switch to the Deep Learning VM image. Required
#                 for experiment=exp04-mjx.)
#   MJX_H, MJX_M, MJX_K, MJX_N_SUBSTEPS  (only forwarded to exp04-mjx)
#
# Cost estimate (us-central1, on-demand, late-2025):
#   e2-standard-8 spot       ~ $0.025/h   (CPU-only experiments)
#   g2-standard-4 + 1 L4     ~ $0.70 /h   (exp04-mjx)
#     seed-sweep   ~ 19 h => ~$0.50.
#   exp05b       ~  4 h => ~$0.10.
#   exp04-mjx M=16 K=10 H=120 n=20 ~ 0.5-1 h => ~$0.50 (est., to be
#     confirmed once L4 quota is approved).
#
# Watch the run:
#   gcloud compute instances tail-serial-port-output $NAME --zone $ZONE
#
# Pull results once done:
#   gsutil -m cp -r gs://$BUCKET/runs/$NAME/<timestamp>/ ./results/
#
# Cancel a run early:
#   gcloud compute instances delete $NAME --zone $ZONE --quiet

set -euo pipefail

EXPERIMENT="${1:-}"
if [[ -z "$EXPERIMENT" ]]; then
    cat <<'USAGE'
usage: launch.sh <experiment>

experiments:
  seed-sweep      3 seeds * (Exp 04 + 05 + 05b)  ~19 h
  exp04           Exp 04 single-shot 20x3        ~80 min
  exp04-mjx       Exp 04 MJX rollout (NEEDS GPU; ACCELERATOR=l4)  ~30 min on L4
  exp05           Exp 05 hybrid 20x3             ~80 min
  exp05b          Exp 05 + 1 MPC replan 20x3     ~3.7 h
  sigma-sweep     5-value sigma_max sweep        ~7 h
  kxreplans       3x3 K x replans grid           ~10 h
  lambda-sweep    6-value lambda sweep (prior)   ~8 h
  train-vtheta    flow-matching trainer          ~5 min on GPU, ~1 h on CPU
  taxonomy        post-hoc analysis              <1 s
  custom          set CUSTOM_CMD env var
USAGE
    exit 2
fi

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to gs://your-bucket-name (no trailing /)}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
# 2026-05-13: us-central1 region-wide n2-standard-2 stockout. Forcing
# e2-standard-2 (same 2 vCPU / 8 GB shape, different CPU family --
# capacity verified available). Revert this override once n2 capacity
# returns: just delete the next line.
[[ "$MACHINE_TYPE" == "n2-standard-2" ]] && MACHINE_TYPE=e2-standard-2
# Default to non-preemptible because new GCP projects ship with 0
# spot/preemptible CPU quota in us-central1. Override PREEMPTIBLE=1
# once you have a spot-quota bump approved (cuts cost ~4x).
PREEMPTIBLE="${PREEMPTIBLE:-0}"
BRANCH="${BRANCH:-feat/further-experiments}"
NAME="${NAME:-hybrid-traj-${EXPERIMENT}-$(date -u +%Y%m%dt%H%M%S)}"

# ---- GPU accelerator support (Gate 4 of docs/mjx_migration.md) ----------
# ACCELERATOR controls whether this VM gets a GPU attached. Supported
# values:
#   ""    -- no GPU (default). Uses MACHINE_TYPE as-is.
#   "l4"  -- one NVIDIA L4 attached to a g2-standard-4 (4 vCPU, 16 GB,
#            1 L4 24 GB). Required for `experiment=exp04-mjx`.
#
# Constraints when ACCELERATOR is non-empty:
#   * MACHINE_TYPE is forced to g2-standard-4 unless explicitly set.
#   * --maintenance-policy=TERMINATE (GPU VMs CANNOT live-migrate).
#   * Image switches to the Deep Learning VM family
#     `common-cu124-ubuntu-2204-py310` (in project
#     deeplearning-platform-release) which has the NVIDIA driver +
#     CUDA 12.4 preinstalled. install_linux_gpu.sh then only has to
#     add jax[cuda12] + mujoco-mjx.
#   * Boot disk bumped to 100 GB (DL VM image + CUDA + venv overhead
#     does not fit in 50 GB cleanly).
#
# Quota required (in the deploy project, in the deploy region):
#   * GPUS_ALL_REGIONS >= 1   (global, MUST be raised; new projects = 0)
#   * NVIDIA_L4_GPUS >= 1     (regional, e.g. us-central1)
#   * Disks_TOTAL_GB >= 100 free (regional)
ACCELERATOR="${ACCELERATOR:-}"
ACCEL_FLAG=""
case "$ACCELERATOR" in
    "" )  # CPU-only path (existing behavior, unchanged).
        ;;
    "l4")
        # Force compatible defaults if the caller didn't override.
        if [[ "$MACHINE_TYPE" == "e2-standard-4" || \
              "$MACHINE_TYPE" == "n2-standard-2" || \
              "$MACHINE_TYPE" == "e2-standard-2" ]]; then
            MACHINE_TYPE=g2-standard-4
        fi
        ACCEL_FLAG="--accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE"
        ;;
    *)
        echo "error: unknown ACCELERATOR='$ACCELERATOR' (supported: '' or 'l4')" >&2
        exit 3
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STARTUP_SCRIPT="$REPO_ROOT/scripts/gce/startup.sh"

PREEMPT_FLAG=""
if [[ "$PREEMPTIBLE" == "1" ]]; then
    # Standard spot VM. --provisioning-model=SPOT is the modern flag.
    PREEMPT_FLAG="--provisioning-model=SPOT --instance-termination-action=DELETE"
fi

META_KV="gcs-bucket=$BUCKET,experiment=$EXPERIMENT,git-branch=$BRANCH,shutdown-when-done=${SHUTDOWN_WHEN_DONE:-1}"
META_KV+=",n-demos=${N_DEMOS:-5},n-rollouts=${N_ROLLOUTS:-1},n-workers=${N_WORKERS:-4}"
if [[ -n "${VTHETA_GCS:-}" ]]; then
    META_KV+=",vtheta-gcs=$VTHETA_GCS"
fi
if [[ -n "${BC_GCS:-}" ]]; then
    META_KV+=",bc-gcs=$BC_GCS"
fi
# Forward MJX-specific knobs when set; startup.sh reads them only for
# experiment=exp04-mjx and ignores them otherwise.
[[ -n "${MJX_N_SUBSTEPS:-}" ]] && META_KV+=",mjx-n-substeps=$MJX_N_SUBSTEPS"
[[ -n "${MJX_H:-}"          ]] && META_KV+=",mjx-h=$MJX_H"
[[ -n "${MJX_M:-}"          ]] && META_KV+=",mjx-m=$MJX_M"
[[ -n "${MJX_K:-}"          ]] && META_KV+=",mjx-k=$MJX_K"

META_FROM_FILE="startup-script=$STARTUP_SCRIPT"
if [[ -n "${CUSTOM_CMD:-}" ]]; then
    # Avoid metadata-value length limits / quoting hell by stuffing the
    # custom command into a temp file and using --metadata-from-file.
    CUSTOM_CMD_FILE=$(mktemp)
    printf '%s\n' "$CUSTOM_CMD" > "$CUSTOM_CMD_FILE"
    trap 'rm -f $CUSTOM_CMD_FILE' EXIT
    META_FROM_FILE+=",custom-cmd=$CUSTOM_CMD_FILE"
fi

META_ARGS=(
    --metadata-from-file "$META_FROM_FILE"
    --metadata           "$META_KV"
)

# Image + boot disk sized to the accelerator setting.
if [[ -n "$ACCELERATOR" ]]; then
    # Deep Learning VM image: NVIDIA driver + CUDA 12.4 + cuDNN
    # preinstalled on Ubuntu 22.04 + Python 3.10. The exact family
    # slug is curated by Google; if it gets renamed, list candidates
    # with:
    #   gcloud compute images list --project=deeplearning-platform-release \
    #          --filter='family ~ common-cu' --format='value(family)'
    IMAGE_FAMILY="common-cu124-ubuntu-2204-py310"
    IMAGE_PROJECT="deeplearning-platform-release"
    BOOT_DISK_SIZE="100GB"
else
    IMAGE_FAMILY="ubuntu-2404-lts-amd64"
    IMAGE_PROJECT="ubuntu-os-cloud"
    BOOT_DISK_SIZE="50GB"
fi

set -x
# shellcheck disable=SC2086  # we WANT word-splitting on ACCEL_FLAG / PREEMPT_FLAG
gcloud compute instances create "$NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-standard \
    --scopes=storage-rw,logging-write,monitoring-write \
    $ACCEL_FLAG \
    $PREEMPT_FLAG \
    "${META_ARGS[@]}"
set +x

cat <<EOF

[launch] instance:    $NAME
[launch] zone:        $ZONE
[launch] experiment:  $EXPERIMENT
[launch] results -> gs://$BUCKET/runs/$NAME/<timestamp>/

watch:
  gcloud compute instances tail-serial-port-output $NAME --zone $ZONE

cancel early:
  gcloud compute instances delete $NAME --zone $ZONE --quiet
EOF
