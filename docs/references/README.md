# External References

We do **not** vendor READMEs or reference models from upstream projects
in this repo. For quick access, here are the canonical upstream sources
for the libraries and models this project touches.

## Libraries

| Project | Description | Upstream |
|---|---|---|
| **MuJoCo** | Physics engine (Google DeepMind). | [github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) · [docs](https://mujoco.readthedocs.io/en/stable/) |
| **`dm_control`** | DeepMind's Python RL-environment stack on top of MuJoCo. | [README](https://github.com/google-deepmind/dm_control/blob/main/README.md) · [tutorial notebook](https://colab.research.google.com/github/google-deepmind/dm_control/blob/main/tutorial.ipynb) |
| **MuJoCo Menagerie** | Curated collection of high-quality MJCF robot models. | [README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/README.md) |
| **robosuite** | Modular robot-learning simulation framework (ARISE). | [README](https://github.com/ARISE-Initiative/robosuite/blob/master/README.md) · [docs](https://robosuite.ai/docs/overview.html) |
| **robomimic** | Imitation-learning benchmark / dataset tools (ARISE). | [README](https://github.com/ARISE-Initiative/robomimic/blob/master/README.md) · [docs](https://robomimic.github.io/) |
| **model-based-diffusion (MBD)** | Diffusion-based trajectory optimization (LeCAR Lab). | [README](https://github.com/LeCAR-Lab/model-based-diffusion/blob/main/README.md) · [website](https://lecar-lab.github.io/mbd/) |

## Models

| Model | Upstream |
|---|---|
| Franka Emika Panda (Menagerie) | [`franka_emika_panda/`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) · [panda.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml) · [subdir README](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/README.md) |

## Offline / local copies

The Menagerie checkout is vendored under `mujoco_menagerie/` (gitignored)
so Franka and other MJCF models are available without network access.
The MuJoCo Sphinx source is similarly available as a sparse clone under
`third_party/mujoco-doc-src/` — see
[`mujoco-official/README.md`](mujoco-official/README.md) for the build
workflow.
