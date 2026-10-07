# Official MuJoCo Documentation

We do **not** vendor the MuJoCo documentation inside this repo. The upstream
Sphinx source lives under a sparse clone at
`third_party/mujoco-doc-src/` (gitignored) and is rendered to HTML on
demand. This keeps the repo small and avoids drift from upstream.

## Read online

For browsing, the canonical source is always the live site:

- **Full docs:** <https://mujoco.readthedocs.io/en/stable/>
- **Overview:** <https://mujoco.readthedocs.io/en/stable/overview.html>
- **XML reference (MJCF):** <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- **Modeling:** <https://mujoco.readthedocs.io/en/stable/modeling.html>
- **Python bindings:** <https://mujoco.readthedocs.io/en/stable/python.html>
- **MJX (MuJoCo XLA / JAX):** <https://mujoco.readthedocs.io/en/stable/mjx.html>
- **Programming guide:** <https://mujoco.readthedocs.io/en/stable/programming/index.html>
- **Changelog:** <https://mujoco.readthedocs.io/en/stable/changelog.html>
- **Upstream source:** <https://github.com/google-deepmind/mujoco>

## Read offline (build locally)

If you want to grep / browse the docs without internet access, build them
locally with the helper script:

```bash
scripts/build_mujoco_docs.sh            # one-shot build, opens in browser
scripts/build_mujoco_docs.sh --watch    # live-reload preview on :8000
```

Output goes to `docs/_build/mujoco-docs/` (also gitignored). The script
assumes you have done the one-time setup once:

```bash
# 1. Sparse clone of upstream docs.
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco.git third_party/mujoco-doc-src
git -C third_party/mujoco-doc-src sparse-checkout set doc

# 2. Dedicated conda env with Sphinx + MuJoCo's doc requirements.
conda create -n mujoco-docs python=3.11 pip -y
conda activate mujoco-docs
python -m pip install -r third_party/mujoco-doc-src/doc/requirements.txt
```

After that, `scripts/build_mujoco_docs.sh` is idempotent and fast
(Sphinx only rebuilds pages whose source changed).

## Notebooks

The official tutorial notebooks (`python/tutorial.ipynb`,
`python/LQR.ipynb`, `python/least_squares.ipynb`, `python/rollout.ipynb`,
`python/mjspec.ipynb`, `mjx/tutorial.ipynb`, `mjx/training_apg.ipynb`)
are checked in under `notebooks/mujoco_official/` so that teammates can
run them directly; see `notebooks/mujoco_official/` for the full list.
