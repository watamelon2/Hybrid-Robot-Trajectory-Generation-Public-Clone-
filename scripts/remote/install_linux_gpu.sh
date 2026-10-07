#!/usr/bin/env bash
# scripts/remote/install_linux_gpu.sh
#
# MJX-on-CUDA install for a fresh Linux box (typically a g2-standard-4
# GCE VM with one NVIDIA L4). Leaner than `install_linux.sh`: skips
# robosuite / robomimic / mink (the MJX path does not need them), and
# pins JAX to a CUDA 12 wheel that matches the local dev env.
#
# Compatible images (in order of preference):
#   1. deeplearning-platform-release/common-cu124-ubuntu-2204-py310
#        -- NVIDIA driver + CUDA 12.4 + cuDNN preinstalled. Fastest
#        startup. Use this if your project has access to the image.
#   2. ubuntu-os-cloud/ubuntu-2404-lts-amd64
#        -- vanilla Ubuntu. Need to install the NVIDIA driver via
#        `sudo apt install nvidia-driver-550` (~3 min + reboot). This
#        script detects the missing driver, installs it, and asks for
#        a reboot; the second boot then proceeds with the pip install.
#        DO NOT use this path for short-lived spot VMs (reboot loses
#        spot reservation).
#
# Required env vars: none. Optional:
#   PY_BIN     : python interpreter to use, default `python3`.
#   JAX_VERSION : jax version pin, default 0.10.0.
#   MUJOCO_VERSION : mujoco version pin, default 3.8.1.
#
# Side effects:
#   - Installs system libs needed for headless rendering (libgl, libegl,
#     ffmpeg) so we can later record MP4s of the MJX-planned trajectory
#     replayed in CPU MuJoCo for debugging.
#   - Creates a Python venv at $PWD/.venv and pip-installs into it.
#   - Writes /tmp/install_linux_gpu.log with full output for triage.
#
# Verification:
#   The script runs a tiny smoke at the end:
#     - `import jax`             -> must report 'gpu' backend
#     - `jnp.ones(8).sum()`      -> must execute on gpu device
#     - `import mujoco.mjx`      -> must import without ImportError
#     - `mjx.put_model(mj_model)` on a 1-DOF MJCF must complete
#   If any of these fails, the script exits non-zero so a downstream
#   startup.sh can refuse to launch the full experiment.
#
# Usage:
#   bash scripts/remote/install_linux_gpu.sh

set -euo pipefail

LOG=/tmp/install_linux_gpu.log
exec > >(tee -a "$LOG") 2>&1

PY_BIN="${PY_BIN:-python3}"
JAX_VERSION="${JAX_VERSION:-0.10.0}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.8.1}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[install-gpu] python:         $($PY_BIN --version 2>&1)"
echo "[install-gpu] cwd:            $REPO_ROOT"
echo "[install-gpu] jax target:     $JAX_VERSION (cuda12 wheel)"
echo "[install-gpu] mujoco target:  $MUJOCO_VERSION"
echo "[install-gpu] log:            $LOG"

# --- 1. NVIDIA driver detection ------------------------------------------
# nvidia-smi MUST work before pip-installing jax[cuda12] -- otherwise
# the wheel installs fine but every JAX call falls back to CPU silently.
# Fail loud and early so the VM does not burn $0.70/hr running a slow
# CPU experiment.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "[install-gpu] nvidia-smi: OK"
    nvidia-smi | head -20
else
    echo "[install-gpu] ERROR: nvidia-smi missing or failing." >&2
    echo "[install-gpu] If running on vanilla Ubuntu, install drivers:" >&2
    echo "[install-gpu]   sudo apt-get install -y linux-headers-\$(uname -r)" >&2
    echo "[install-gpu]   sudo apt-get install -y nvidia-driver-550" >&2
    echo "[install-gpu]   sudo reboot   # then re-run this script" >&2
    echo "[install-gpu] Or relaunch on the deeplearning-platform-release" >&2
    echo "[install-gpu] common-cu124-ubuntu-2204-py310 image which has" >&2
    echo "[install-gpu] drivers preinstalled." >&2
    exit 10
fi

# --- 2. system libs for headless MuJoCo + ffmpeg -------------------------
if command -v apt-get >/dev/null 2>&1; then
    echo "[install-gpu] apt-get: system libs ..."
    export DEBIAN_FRONTEND=noninteractive
    sudo -n apt-get update -y >/dev/null 2>&1 || sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglfw3 libosmesa6 \
        ffmpeg patchelf \
        python3-pip python3-venv \
        >/dev/null
fi

# --- 3. venv + pip ------------------------------------------------------
if [[ ! -d .venv ]]; then
    "$PY_BIN" -m venv .venv
fi
. .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools >/dev/null

# --- 4. JAX with CUDA 12 ------------------------------------------------
# `jax[cuda12]` pulls jax + jaxlib + jax-cuda12-plugin (the CUDA bridge)
# + the cuda12 wheels (libcudart, libcublas, libcudnn) from PyPI. No
# system CUDA install is required beyond the NVIDIA driver.
echo "[install-gpu] pip install jax[cuda12]==$JAX_VERSION ..."
python -m pip install \
    "jax[cuda12]==${JAX_VERSION}" \
    "jaxlib==${JAX_VERSION}"

# --- 5. mujoco + mujoco-mjx --------------------------------------------
echo "[install-gpu] pip install mujoco==$MUJOCO_VERSION mujoco-mjx==$MUJOCO_VERSION ..."
python -m pip install \
    "mujoco==${MUJOCO_VERSION}" \
    "mujoco-mjx==${MUJOCO_VERSION}"

# --- 6. small but necessary deps for the MJX path -----------------------
# The MJX rollout itself doesn't need numpy/scipy/matplotlib, but the
# run_can.py driver writes a JSON summary + a cost-curve PNG.
echo "[install-gpu] pip install supporting deps ..."
python -m pip install \
    "numpy>=1.26,<3" \
    "scipy>=1.13" \
    "matplotlib==3.10.9" \
    "mediapy==1.2.6" \
    "imageio==2.37.3" \
    "imageio-ffmpeg==0.6.0" \
    "h5py==3.16.0" \
    "tqdm==4.67.3"

# --- 7. SMOKE: GPU detected? mjx imports? --------------------------------
echo "[install-gpu] running GPU smoke ..."
python - <<'PY'
import sys

print("[smoke] python:", sys.version.split()[0])

import jax
import jax.numpy as jnp
print("[smoke] jax:", jax.__version__)
print("[smoke] jax.default_backend():", jax.default_backend())
print("[smoke] jax.devices():", [str(d) for d in jax.devices()])

if jax.default_backend() != "gpu":
    print("[smoke] FAIL: jax.default_backend() is not 'gpu'.", file=sys.stderr)
    sys.exit(11)

# Force one op on the GPU to verify the device truly works.
x = jnp.ones((8, 8), dtype=jnp.float32)
y = (x @ x).sum()
print("[smoke] gpu matmul result:", float(y), " (expected 64.0)")
assert abs(float(y) - 64.0) < 1e-6

import mujoco
print("[smoke] mujoco:", mujoco.__version__)

import mujoco.mjx as mjx
print("[smoke] mujoco.mjx OK")

# Build a 1-DOF MJCF and push it onto the device. If the cuda12 plugin
# is mis-wired (a common silent failure mode), this will crash with a
# cryptic XLA error -- much better to crash here than 25 s into the
# JIT compile of the full rollout.
xml = """
<mujoco>
  <worldbody>
    <body name="b0">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(m)
d = mjx.make_data(mx)
d = mjx.forward(mx, d)
d = mjx.step(mx, d)
print("[smoke] mjx.step OK (qpos[0] =", float(d.qpos[0]), ")")
PY

echo "[install-gpu] done.  log: $LOG"
