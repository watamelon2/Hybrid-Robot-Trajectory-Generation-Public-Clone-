#!/usr/bin/env bash
# Build (or live-rebuild) the official MuJoCo Sphinx docs locally.
#
# Setup (already done once):
#   git clone --depth 1 --filter=blob:none --sparse \
#     https://github.com/google-deepmind/mujoco.git third_party/mujoco-doc-src
#   git -C third_party/mujoco-doc-src sparse-checkout set doc
#   conda create -n mujoco-docs python=3.11 pip -y
#   conda activate mujoco-docs
#   python -m pip install -r third_party/mujoco-doc-src/doc/requirements.txt
#
# Usage:
#   scripts/build_mujoco_docs.sh           # one-shot build, then open
#   scripts/build_mujoco_docs.sh --watch   # auto-rebuild on save (sphinx-autobuild)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/third_party/mujoco-doc-src/doc"
OUT="$REPO_ROOT/docs/_build/mujoco-docs"

# Activate env (idempotent). Works inside or outside an already-active shell.
# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate mujoco-docs

mkdir -p "$OUT"

if [[ "${1:-}" == "--watch" ]]; then
  # Live-reload preview at http://127.0.0.1:8000
  if ! python -c 'import sphinx_autobuild' 2>/dev/null; then
    echo "Installing sphinx-autobuild into mujoco-docs env..."
    python -m pip install sphinx-autobuild
  fi
  cd "$SRC"
  exec sphinx-autobuild -j auto --open-browser . "$OUT"
fi

cd "$SRC"
sphinx-build -b html -j auto . "$OUT"
echo
echo "Built: $OUT/index.html"
[[ "${2:-${1:-}}" != "--no-open" ]] && open "$OUT/index.html"
