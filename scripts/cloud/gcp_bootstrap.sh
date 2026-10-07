#!/bin/bash
# GCP VM bootstrap for Phase-D Q2 + Q3 (lambda-sweep + K-sweep at ep0049).
#
# Usage on a fresh Debian-12 (or Ubuntu-22.04) c3-standard-22 VM:
#
#   ssh into the VM, then:
#
#     curl -sSL https://raw.githubusercontent.com/watamelon2/Hybrid-Robot-Trajectory-Generation/feat/dp-phase-c/scripts/cloud/gcp_bootstrap.sh | bash
#
#   OR scp this file to the VM and run:
#
#     bash gcp_bootstrap.sh
#
# Idempotent: re-running detects existing miniconda / env / repo and
# skips already-installed pieces.  Safe to run multiple times.

set -euo pipefail

# ---------- knobs ----------
REPO_URL="${REPO_URL:-https://github.com/watamelon2/Hybrid-Robot-Trajectory-Generation.git}"
REPO_BRANCH="${REPO_BRANCH:-feat/dp-phase-c}"
REPO_DIR="${REPO_DIR:-$HOME/HRTG-dp}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-hybrid-traj}"
PY_VERSION="${PY_VERSION:-3.11}"

log() { echo "[bootstrap $(date '+%H:%M:%S')]  $*"; }

# ---------- system packages ----------
log "installing system packages (apt) ..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y -qq
    sudo apt-get install -y -qq \
        git curl wget bzip2 ca-certificates \
        build-essential pkg-config \
        libegl1 libglu1-mesa libgl1-mesa-glx libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        screen tmux htop
else
    log "WARN: no apt-get; assuming deps are present"
fi

# ---------- miniconda ----------
if [ ! -d "$MINICONDA_DIR" ]; then
    log "installing miniconda to $MINICONDA_DIR ..."
    arch=$(uname -m)
    case "$arch" in
        x86_64) mc_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
        aarch64) mc_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
        *) echo "unsupported arch: $arch" >&2; exit 1 ;;
    esac
    curl -sSL "$mc_url" -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
fi

# Source conda for this shell.
# shellcheck disable=SC1091
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# ---------- conda env ----------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "conda env '$ENV_NAME' exists; skipping create"
else
    log "creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

conda activate "$ENV_NAME"
log "active env: $CONDA_DEFAULT_ENV ($(python --version))"

# ---------- pip deps ----------
# CPU-only torch (no GPU needed; env stepping in MuJoCo dominates).
# Versions chosen to match the user's known-working local env:
#   torch >= 2.5  (CPU build), numpy >= 2.0 (robosuite needs >= 2.0,
#   any 2.x release works), robosuite 1.5.2, mujoco 3.8.1, robomimic 0.3.0
log "installing pip deps (~5-8 min) ..."

pip install --quiet --upgrade pip

# CPU torch from PyTorch's index (avoids pulling CUDA wheels).
pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.5,<3" "torchvision>=0.20,<1"

# Core scientific + RL stack.  numpy must be 2.x for robosuite 1.5.2;
# any modern 2.x works (locally we have 2.4.4 and it's fine).
pip install --quiet \
    "numpy>=2.0,<3" \
    "h5py>=3.10" \
    "matplotlib" \
    "tqdm" \
    "robomimic==0.3.0" \
    "robosuite==1.5.2" \
    "mujoco==3.8.1"

# Sanity prints.
log "deps installed; sanity-importing ..."
python - <<'PY'
import numpy, torch, robomimic, robosuite, mujoco, h5py
print(f"  numpy     : {numpy.__version__}")
print(f"  torch     : {torch.__version__}  (cuda: {torch.cuda.is_available()})")
print(f"  h5py      : {h5py.__version__}")
print(f"  robomimic : {robomimic.__version__}")
print(f"  robosuite : {robosuite.__version__}")
print(f"  mujoco    : {mujoco.__version__}")
PY

# ---------- repo ----------
# Three cases handled:
#   1. $REPO_DIR has a .git dir -> normal git update
#   2. $REPO_DIR exists but has no .git (SCPed snapshot) -> use as-is
#   3. $REPO_DIR doesn't exist -> clone fresh (requires public repo)
if [ -d "$REPO_DIR/.git" ]; then
    log "repo with .git exists at $REPO_DIR; fetching latest $REPO_BRANCH ..."
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$REPO_BRANCH"
    git -C "$REPO_DIR" log --oneline -3
elif [ -d "$REPO_DIR" ]; then
    log "repo at $REPO_DIR exists WITHOUT .git (SCPed snapshot); using as-is"
    log "  (no git update will be attempted; trust that contents are correct)"
else
    log "cloning $REPO_URL @ $REPO_BRANCH -> $REPO_DIR ..."
    git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
    git -C "$REPO_DIR" log --oneline -3
fi

# Verify the Phase-D fix is present (the chunk-boundary reset gate).
log "verifying Phase-D fix in unified_loop.py ..."
grep -n 'params.K' "$REPO_DIR/experiments/15_unified_hybrid/unified_loop.py" \
    | grep -i '> 0' || { echo "PHASE-D FIX MISSING in $REPO_BRANCH"; exit 1; }
log "  fix present."

# ---------- shell init ----------
# Add conda activation to ~/.bashrc so future SSH sessions are ready.
if ! grep -q "conda activate $ENV_NAME" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

# --- added by gcp_bootstrap.sh ---
source $MINICONDA_DIR/etc/profile.d/conda.sh
conda activate $ENV_NAME
EOF
    log "added conda activation to ~/.bashrc"
fi

log ""
log "=========================================================="
log "BOOTSTRAP DONE"
log "=========================================================="
log "next steps:"
log "  1. scp the ckpt + dataset to this VM:"
log "       gcloud compute scp v_theta_dp_tobs2_h16_v3b_ep0049.pt VM:~/"
log "       gcloud compute scp low_dim_v141.hdf5 VM:~/"
log "  2. run the Q2+Q3 evaluator:"
log "       cd $REPO_DIR"
log "       bash scripts/cloud/run_phase_d_q2q3.sh \\"
log "         --ckpt \$HOME/v_theta_dp_tobs2_h16_v3b_ep0049.pt \\"
log "         --dataset \$HOME/low_dim_v141.hdf5"
log "=========================================================="
