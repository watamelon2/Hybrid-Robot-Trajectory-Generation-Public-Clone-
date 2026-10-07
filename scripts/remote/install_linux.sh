#!/usr/bin/env bash
# Install the project's runtime dependencies on a fresh Linux box
# (Colab, a GCE VM, or any Ubuntu 22.04 / 24.04 environment).
#
# Strips macOS-only deps (pyobjc, appnope) from env/requirements.txt.
# Pins numpy < 2 and jax < 0.5 per env/requirements-explicit.txt's
# constraints.
#
# Usage:
#   bash scripts/remote/install_linux.sh             # full env
#   SKIP_DATASET=1 bash scripts/remote/install_linux.sh   # skip robomimic dl
#   TASK=square bash scripts/remote/install_linux.sh      # download Square instead of Can

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[install] python: $(python3 --version)"
echo "[install] cwd:    $REPO_ROOT"

# --- 1. system libs needed by mujoco's offscreen renderer + ffmpeg --------
if command -v apt-get >/dev/null 2>&1; then
    echo "[install] apt-get: system libs + build deps ..."
    sudo -n apt-get update -y >/dev/null 2>&1 || sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake \
        libegl1 libegl1-mesa-dev libgl1 libglfw3 libosmesa6 \
        libevdev-dev python3-dev \
        ffmpeg patchelf >/dev/null
fi

# --- 2. pip ---------------------------------------------------------------
python3 -m pip install --upgrade pip wheel setuptools

# --- 3. project deps ------------------------------------------------------
# Use the requirements-explicit list (lean) plus torch + matplotlib + numpy
# pinned for the mink/jax/numpy<2 corner. Strip pyobjc/appnope (macOS).
TMP_REQ=$(mktemp)
trap 'rm -f $TMP_REQ' EXIT

cat <<'EOF' > "$TMP_REQ"
# Core sim + dataset
mujoco==3.8.0
robosuite==1.5.2
robomimic==0.3.0
mink==0.0.5

# Numerics: pinned by mink + jax compatibility (numpy < 2, jax < 0.5).
numpy==1.26.4
scipy==1.13.1
jax==0.4.30
jaxlib==0.4.30

# Project-side
torch==2.4.1            # newer than env/requirements.txt; CPU/GPU agnostic
h5py==3.16.0
matplotlib==3.10.9
mediapy==1.2.6
imageio==2.37.3
imageio-ffmpeg==0.6.0

# Robosuite transitive deps that need system headers to build
egl_probe==1.0.2
pynput==1.8.1

# Utility
tqdm==4.67.3
EOF

echo "[install] pip install (this may take 5-10 min on first run) ..."
python3 -m pip install -r "$TMP_REQ"

# --- 4. dataset download (optional) ---------------------------------------
if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
    TASK="${TASK:-can}"
    echo "[install] downloading robomimic ${TASK}/ph low_dim dataset ..."
    python3 -m robomimic.scripts.download_datasets \
        --tasks "$TASK" --hdf5_types low_dim || \
        echo "[install] dataset download failed; rerun manually"
fi

# --- 5. smoke test --------------------------------------------------------
python3 - <<'PY'
import sys
print("[smoke] python:", sys.version.split()[0])
import numpy as np; print("[smoke] numpy", np.__version__)
import torch; print("[smoke] torch", torch.__version__, "cuda:", torch.cuda.is_available())
import mujoco; print("[smoke] mujoco", mujoco.__version__)
import robosuite; print("[smoke] robosuite", robosuite.__version__)
import robomimic; print("[smoke] robomimic", robomimic.__version__)
PY

echo "[install] done."
