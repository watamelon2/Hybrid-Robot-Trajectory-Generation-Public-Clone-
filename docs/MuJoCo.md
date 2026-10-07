# MuJoCo for the Hybrid-Robot-Trajectory-Generation project

This document is a focused reference for using **MuJoCo** in this
project (data-driven flow-matching prior + online model-based score
correction, with batched physics rollouts on a Franka Panda). It
combines a general MuJoCo quickstart with the project-specific bits
needed to run the simulator, the model-based diffusion (MBD) baseline,
and our hybrid method.

Links to upstream libraries and models are consolidated in
[`docs/references/README.md`](references/README.md). The official
MuJoCo Sphinx docs are not vendored — see
[`docs/references/mujoco-official/README.md`](references/mujoco-official/README.md)
for how to build them locally or read them on
[mujoco.readthedocs.io](https://mujoco.readthedocs.io/en/stable/).

---

## 1. What is MuJoCo?

MuJoCo (*Multi-Joint dynamics with Contact*) is a fast, differentiable-
friendly rigid-body simulator originally developed by Emo Todorov and
now maintained by Google DeepMind. Models are described in **MJCF**
(an XML dialect; URDF is also accepted), and the simulator supports
contact, constraints, tendons, actuators, sensors, and a built-in
OpenGL viewer.

Two flavours we use:

- **`mujoco`** — the canonical Python bindings (CPU, single-process).
  Best for prototyping, demonstration replay, and the macOS dev loop.
- **`mujoco-mjx`** — a JAX re-implementation of the MuJoCo dynamics on
  GPU/TPU. We use this for the *batched rollouts* in the online MBD
  score, where we need $M\!\in\![64,512]$ trajectories per denoising
  step.

Useful upstream links (mirrored locally where possible):

- Official docs (HTML): <https://mujoco.readthedocs.io/en/stable/>
- Python bindings: <https://mujoco.readthedocs.io/en/stable/python.html>
- MJCF reference: <https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- MJX: <https://mujoco.readthedocs.io/en/stable/mjx.html>
- Source: <https://github.com/google-deepmind/mujoco>
- Robot models (Menagerie): <https://github.com/google-deepmind/mujoco_menagerie>

---

## 2. Environment setup

### 2.1 macOS (development machine)

We use **conda** (Miniconda or Miniforge). Install Miniforge once if
you don't have it:

```bash
# Miniforge (community, conda-forge default channel) — recommended on macOS
curl -LsSf -o /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Create the project env (run from `Hybrid-Robot-Trajectory-Generation/`):

```bash
# Note: include `pip` explicitly — recent conda does not add it by default.
conda create -n hybrid-traj python=3.11 pip -y
conda activate hybrid-traj

# core simulator + viewer + numpy/scipy/torch
# (mujoco, robosuite, robomimic are pip-only, so we install them via pip
# inside the conda env — this is the standard pattern)
# Using `python -m pip` avoids PATH ambiguity if `pip` shadowing exists.
python -m pip install \
  "mujoco>=3.2" \
  "numpy<2" "scipy" \
  "torch" "torchvision" \
  "matplotlib" "tqdm" "imageio[ffmpeg]"
```

For reproducibility, freeze the env with
`conda env export --from-history > environment.yml`.

Verify:

```bash
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -m mujoco.viewer   # opens an empty viewer window
```

> **Apple Silicon note.** MuJoCo ships universal2 wheels; no extra
> steps are needed. The viewer uses native Cocoa.

### 2.2 Linux + GPU (training / experiments)

Same as macOS for the CPU stack; for **MJX** add a JAX build matching
your CUDA driver. CUDA 12 example:

```bash
conda create -n hybrid-traj python=3.11 pip -y
conda activate hybrid-traj

python -m pip install "mujoco>=3.2" "mujoco-mjx>=3.2"
python -m pip install --upgrade "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> **Tip.** Install CUDA-aware PyTorch via the official conda channel
> if you also want torch on GPU:
> `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`.

Verify GPU is visible:

```bash
python -c "import jax; print(jax.devices())"   # should list GPUs
python -c "import mujoco, mujoco.mjx as mjx; print(mjx.__name__)"
```

For a *headless* server, set:

```bash
export MUJOCO_GL=egl     # offscreen rendering without X11
```

(`MUJOCO_GL=osmesa` is the software fallback.)

### 2.3 Robot models (Menagerie)

The Franka Panda MJCF lives in the Menagerie repo. Clone it once into
`third_party/` (gitignored):

```bash
git clone --depth 1 \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  third_party/mujoco_menagerie
```

The Franka description is at
`third_party/mujoco_menagerie/franka_emika_panda/`. Upstream refs:
[`panda.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml)
·
[subdir README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/README.md).

### 2.4 Demonstration data (Robomimic / Robosuite)

We train the offline flow-matching prior on Robomimic's
proficient-human splits. Robosuite is the underlying MuJoCo-based
benchmark.

```bash
conda activate hybrid-traj
python -m pip install robosuite robomimic h5py
# download proficient-human datasets for can / square
python -m robomimic.scripts.download_datasets \
  --tasks can square --dataset_types ph
```

Datasets land in `~/.robomimic/datasets/<task>/ph/low_dim_v15.hdf5`.
We then retarget them to torque control via inverse dynamics — see
§7 below.

### 2.5 Sanity check script

Save as `scripts/check_env.py`:

```python
import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print("ok, qpos =", np.round(data.qpos, 3))
```

```bash
python scripts/check_env.py
```

---

## 3. Quickstart: load, step, render

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

# Interactive viewer (blocks)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

For headless rendering use `mujoco.Renderer`:

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="closeup")
img = renderer.render()      # (H, W, 3) uint8 ndarray
```

### 3.1 Inline rendering inside VS Code / Jupyter

`mujoco.viewer.launch_passive` always opens an external OS window. To
keep everything **inside the VS Code editor** (no second window), use
`mujoco.Renderer` from a Jupyter kernel and display frames with
[`mediapy`](https://github.com/google/mediapy):

```bash
conda activate hybrid-traj
python -m pip install jupyter ipykernel mediapy
```

Then open a `.ipynb` (or use VS Code's `# %%` interactive window) and:

```python
import mujoco, mediapy as media
model    = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data     = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
media.show_image(renderer.render())                 # single frame, inline

frames = []
for _ in range(500):
    mujoco.mj_step(model, data)
    if _ % 10 == 0:
        renderer.update_scene(data)
        frames.append(renderer.render())
media.show_video(frames, fps=50)                    # inline HTML5 video
```

A ready-to-run version covering the Franka, a falling-box scene, and a
torque-driven rollout is checked in at
[`notebooks/01_inline_render.ipynb`](../notebooks/01_inline_render.ipynb).
Open it in VS Code, click *Select Kernel* → `hybrid-traj`, and *Run All*.

> **Tip.** `mediapy.write_video("out.mp4", frames, fps=50)` saves an MP4
> that VS Code can preview directly — useful on a headless server where
> you don't want to round-trip through a browser.

---

## 4. MJCF essentials

Minimal MJCF that the project uses for the LQ / sandbox tests:

```xml
<mujoco model="double_integrator_2d">
  <option timestep="0.02" integrator="RK4"/>
  <worldbody>
    <body name="point" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="x" gear="1" ctrlrange="-1 1"/>
    <motor joint="y" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

Key elements (full reference: <https://mujoco.readthedocs.io/en/stable/XMLreference.html>):

| Tag             | Purpose                                                |
|-----------------|--------------------------------------------------------|
| `<option>`      | Integrator, timestep, gravity, solver iterations.      |
| `<default>`     | Inheritance defaults; reused by every Menagerie model. |
| `<worldbody>`   | The kinematic tree. Bodies nest; joints connect them.  |
| `<joint>`       | Hinge, slide, ball, free.                              |
| `<geom>`        | Visual + collision geometry. `contype`/`conaffinity` mask collisions. |
| `<actuator>`    | `motor` (torque), `position`, `velocity`, `general`.   |
| `<sensor>`      | Force, accelerometer, jointpos, framepos, etc.         |
| `<contact>`     | `pair` to enable, `exclude` to disable.                |
| `<equality>`    | Holonomic constraints (welds, connect, joint).         |
| `<keyframe>`    | Named `qpos`/`qvel`/`ctrl` snapshots.                  |

Useful `option` flags for our setup:

- `integrator="implicitfast"` — semi-implicit, recommended in MuJoCo
  ≥ 3 for stiff manipulators; falls back to RK4 if absent.
- `timestep="0.002"` for the simulator, with control held constant
  for 10 sub-steps to give a 0.02 s control period (matches the
  $\Delta t$ in the report).
- `cone="elliptic"` for more accurate friction; `pyramidal` is faster.

---

## 5. Python API essentials

```python
mujoco.mj_step(model, data)            # full step (forward + integrate)
mujoco.mj_forward(model, data)         # forward dynamics, no integration
mujoco.mj_inverse(model, data)         # inverse dynamics: qfrc_inverse
mujoco.mj_jac(model, data, jacp, jacr, point, body)   # geometric Jacobian
mujoco.mj_resetData(model, data)
mujoco.mj_resetDataKeyframe(model, data, key_id)
```

Field cheatsheet on `MjData`:

| Field                 | Meaning                                |
|-----------------------|----------------------------------------|
| `qpos` / `qvel`       | Generalized position / velocity.       |
| `ctrl`                | Actuator inputs (length `model.nu`).   |
| `xpos`, `xmat`        | Cartesian body pose.                   |
| `site_xpos`           | Cartesian site position (use for FK of the EE). |
| `qfrc_bias`           | Coriolis + gravity bias $C\dot q + g$. |
| `qM`                  | Sparse mass matrix; `mj_fullM` to densify. |
| `contact`, `ncon`     | Active contact list.                   |

A typical "torque-controlled rollout" loop, used both for demonstrations
and for the online MBD score:

```python
def rollout(model, data, u_seq):
    """u_seq: (H, nu). Returns (H+1, nq+nv) state trajectory."""
    H = u_seq.shape[0]
    xs = np.zeros((H + 1, model.nq + model.nv))
    xs[0] = np.concatenate([data.qpos, data.qvel])
    for k in range(H):
        data.ctrl[:] = u_seq[k]
        mujoco.mj_step(model, data)
        xs[k + 1] = np.concatenate([data.qpos, data.qvel])
    return xs
```

---

## 6. Batched rollouts (MJX)

Online MBD evaluates
$\nabla\log p^{\text{model}}_t \approx
\mathbb{E}_{w(\tau)}[\tau-\tau_t]/\sigma_t^2$
which requires $M$ rollouts of horizon $H$ at every denoising step.
On CPU MuJoCo this is sequential and slow; **MJX** runs the same
dynamics on GPU and is `jax.vmap`-able over the batch axis.

```python
import jax, jax.numpy as jnp
import mujoco, mujoco.mjx as mjx

mj_model = mujoco.MjModel.from_xml_path(
    "third_party/mujoco_menagerie/franka_emika_panda/panda.xml")
mjx_model = mjx.put_model(mj_model)

def step(data, u):
    data = data.replace(ctrl=u)
    return mjx.step(mjx_model, data), data.qpos

@jax.jit
def rollout(qpos0, qvel0, u_seq):           # u_seq: (H, nu)
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=qpos0, qvel=qvel0)
    _, qs = jax.lax.scan(step, data, u_seq)
    return qs                                # (H, nq)

batched = jax.vmap(rollout, in_axes=(None, None, 0))   # over u_seq batch
qs = batched(qpos0, qvel0, u_batch)          # u_batch: (M, H, nu)
```

Caveats (from <https://mujoco.readthedocs.io/en/stable/mjx.html>):

- MJX supports a subset of MuJoCo: convex contacts only (no
  ellipsoid/cylinder pairs), no general equality constraints. The
  Franka Menagerie model works out of the box; verify with
  `mjx.validate_model(mj_model)`.
- Compilation is slow (tens of seconds) but happens once per shape
  signature; cache by keeping `H`, `M`, `nu` fixed.
- Use `jax.lax.scan`, not Python loops, for the time axis.

---

## 7. Project-specific setup

The deployed system is the 7-DoF **Franka Panda** doing tabletop
reaching, defined exactly as in the milestone (see
`Progress Report/week3.tex` §V):

```
state x = (q, q_dot) ∈ R^14
ctrl  u ∈ R^7,  saturated at ±87 Nm
dt    = 0.02 s
horizon H = 64  (≈1.3 s)
cost  ℓ = ‖FK(q) − p_goal‖_Q^2 + ‖u‖_R^2
goal  p_goal ~ Uniform(0.4 × 0.4 × 0.2 m box)
obstacles: 2–3 cylinders, randomized per episode
```

### 7.1 Adapting the Menagerie Franka MJCF

The stock `panda.xml` ships with `<general>` actuators that emulate
the on-robot joint impedance controller (large `gainprm` + negative
`biasprm` ≈ a stiff PD on `qpos`). Replace them with pure `<motor>`
actuators to get raw torque control in $\pm 87$ Nm:

```xml
<actuator>
  <motor name="actuator1" joint="joint1" ctrlrange="-87 87" gear="1"/>
  <motor name="actuator2" joint="joint2" ctrlrange="-87 87" gear="1"/>
  <!-- ... joints 3–7 ... -->
</actuator>
```

(See the upstream
[`panda.xml`](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml)
for the original.) Add a worktable plane and 2–3 cylinder obstacles
in `<worldbody>` at randomized poses; clamp the goal site to a
0.4×0.4×0.2 m box.

### 7.2 Retargeting Robomimic demos to torque control

Robomimic ships end-effector / joint-position action streams. Convert
to torques offline once via inverse dynamics:

```python
def retarget(model, qpos_seq, qvel_seq, qacc_seq):
    data = mujoco.MjData(model)
    u_seq = np.zeros((len(qpos_seq), model.nu))
    for k, (q, qd, qdd) in enumerate(zip(qpos_seq, qvel_seq, qacc_seq)):
        data.qpos[:] = q; data.qvel[:] = qd; data.qacc[:] = qdd
        mujoco.mj_inverse(model, data)
        u_seq[k] = data.qfrc_inverse[:model.nu]
    return u_seq
```

Save the retargeted `(q, q_dot, u)` triples as the `D` used to train
$v_\theta$.

### 7.3 LQ surrogate

The 2D double-integrator MJCF in §4 is the analytical companion. Use
`scipy.linalg.solve_discrete_are` to solve the DARE; the resulting
$-P_t x_t$ is the closed-form ground truth against which the
importance-sampling estimator is validated in isolation.

---

## 8. Troubleshooting

| Symptom                                              | Fix |
|------------------------------------------------------|-----|
| `RuntimeError: An OpenGL platform library has not been loaded` (Linux server) | `export MUJOCO_GL=egl` (or `osmesa`). |
| Viewer crashes on macOS after a few seconds         | Use `launch_passive` from a *main thread*; do not call from a subprocess. |
| Simulation explodes / NaNs after a few steps         | Reduce `timestep`, switch `integrator` to `implicitfast`, or check actuator `gear`/`ctrlrange`. |
| `mjx.validate_model` complains                       | The model uses unsupported features (e.g., cylinder geoms in collision). Replace with capsules/boxes or exclude the contact pair. |
| Robomimic dataset fails to load                      | Check h5py version; use `low_dim_v15.hdf5` for the proficient-human split. |
| Slow MJX compilation on every run                    | Pin shapes (`H`, `M`, `nu`) and use `jax.jit` with `static_argnames`. |

---

## 9. Hands-on learning path

This document is a *reference*; it is dense and assumes you already
know the framework. If you are coming to MuJoCo for the first time,
work through the resources below in order — each step is interactive
("read code, run code, modify code") rather than prose. Estimated
total time: **~3–4 h for the core path** (§9.1), plus **~2–3 h** for
the project-relevant extras (§9.2).

Before starting, complete the env setup in §2 and run
`scripts/check_env.py` so you have a working `hybrid-traj` env.

### 9.1 Core path — do these in order

1. **Official Python tutorial notebook** (~60 min) —
   <https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb>
   The canonical interactive walkthrough. Covers MJCF strings, stepping
   the simulator, kinematic queries, rendering, contacts, and the
   `mujoco.viewer`. The fastest single resource for "I can now use
   MuJoCo from Python." Run it locally:
   ```bash
   conda activate hybrid-traj
   python -m pip install jupyter
   curl -L -o /tmp/mj_tutorial.ipynb \
     https://raw.githubusercontent.com/google-deepmind/mujoco/main/python/tutorial.ipynb
   jupyter notebook /tmp/mj_tutorial.ipynb
   ```
   (Or click "Open in Colab" from the GitHub page if you prefer the
   hosted runtime.)

2. **MJCF modeling guide** (~60 min) —
   <https://mujoco.readthedocs.io/en/stable/modeling.html>,
   read in parallel with
   <https://mujoco.readthedocs.io/en/stable/XMLreference.html>.
   After finishing you should be able to open the Menagerie
   `panda.xml` and explain every element. **Exercise:** hand-edit the
   double-integrator MJCF in §4 — add a third slide joint, attach a
   `framepos` sensor on the body, then add a `<keyframe>` and reset to
   it from Python with `mj_resetDataKeyframe`.

3. **MJX tutorial notebook** (~45 min) —
   <https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb>
   The same workflow on GPU/TPU, with `jax.vmap` over a batch axis.
   Maps one-to-one onto our §6 batched-rollout code; do this **before**
   writing the online MBD score so you have intuition for compile times,
   shape signatures, and `jax.lax.scan`.

### 9.2 Project-relevant extras

4. **MuJoCo Menagerie tour** —
   [Menagerie README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md).
   After cloning the menagerie (§2.3), open the Franka model in the
   viewer:
   ```bash
   python -m mujoco.viewer \
     --mjcf=third_party/mujoco_menagerie/franka_emika_panda/panda.xml
   ```
   In the viewer, drag joint sliders (right panel) and watch
   `qpos`/`ctrl` update. Compare the stock `<general>` actuators with
   the pure `<motor>` rewrite in §7.1.

5. **Robosuite + Robomimic quickstarts** —
   <https://robosuite.ai/docs/quickstart.html> and
   <https://robomimic.github.io/docs/tutorials/getting_started.html>.
   Run the Robosuite `Lift` demo, then stream one Robomimic
   proficient-human episode (`can` task). This is the upstream of our
   demonstration data, so understanding the action/observation spaces
   is required before writing the retargeting in §7.2.

6. **`dm_control` suite tour** *(optional)* —
   [dm_control README](https://github.com/google-deepmind/dm_control/blob/main/README.md).
   DeepMind's higher-level RL wrapper. The `cartpole`, `reacher`, and
   `manipulator` tasks are useful sanity environments and a good place
   to test the LQ surrogate (§7.3) against an existing benchmark.

### 9.3 Community / background resources

- **Pranav Bhounsule's MuJoCo Bootcamp** —
  <https://pab47.github.io/mujoco.html>. Lecture videos plus code on
  dynamics, contacts, and control. Slower, classroom-style; useful if
  the official notebook moves too fast.
- **MuJoCo source `python/` examples** —
  <https://github.com/google-deepmind/mujoco/tree/main/python>.
  Smaller notebooks on rendering, LQR, and MPC beyond the main
  tutorial.
- **MuJoCo paper** (Todorov, Erez & Tassa, *MuJoCo: A physics engine
  for model-based control*, IROS 2012) — background on the soft
  contact solver and why MuJoCo's friction-cone model differs from
  Bullet / PhysX. Useful when debugging contact-rich rollouts.
- **MJX paper / blog** —
  <https://mujoco.readthedocs.io/en/stable/mjx.html#design>. Read
  the "design" subsection to understand which features are dropped on
  GPU and why `mjx.validate_model` is the gate for batched rollouts.

By the end of §9.1 you can read every line of the rollout loops in §5
and §6; by the end of §9.2 you can run the full demo→retarget→
flow-matching pipeline end-to-end. §9.3 is purely supplementary.

---

## 10. References

We do not vendor upstream docs. See
[`docs/references/README.md`](references/README.md) for the full set
of links (MuJoCo, `dm_control`, Menagerie incl. Franka MJCF,
robosuite, robomimic, MBD). For a local Sphinx build of the MuJoCo
docs see
[`docs/references/mujoco-official/README.md`](references/mujoco-official/README.md)
and `scripts/build_mujoco_docs.sh`. The seven official MuJoCo
tutorial notebooks are checked in under
[`notebooks/mujoco_official/`](../notebooks/mujoco_official/).

Online (canonical):

- MuJoCo docs: <https://mujoco.readthedocs.io/en/stable/>
- MuJoCo source: <https://github.com/google-deepmind/mujoco>
- MuJoCo Menagerie: <https://github.com/google-deepmind/mujoco_menagerie>
- MJX guide: <https://mujoco.readthedocs.io/en/stable/mjx.html>
- Python tutorial notebook: <https://github.com/google-deepmind/mujoco/blob/main/python/tutorial.ipynb>
- Robosuite: <https://robosuite.ai/>
- Robomimic: <https://robomimic.github.io/>
- Model-Based Diffusion: <https://github.com/LeCAR-Lab/model-based-diffusion>
