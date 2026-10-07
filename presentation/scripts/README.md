# `presentation/scripts/` — README

Five build scripts (and two defunct ones kept for code-archaeology),
all runnable as `python -m presentation.scripts.<name>` from the
repo root with the `hybrid-traj` conda env active.

> **You do not normally need to run any of these.**  All generated
> artefacts (PDFs, PNG previews, MP4s) are committed.  These
> scripts exist so you can regenerate after a number changes.

## Active scripts (5)

### `make_headline_figures.py`

Generates the six headline PDFs used on slides 4 / 5 / 5b / 6 / 7.

```bash
cd "$(git rev-parse --show-toplevel)"
python -m presentation.scripts.make_headline_figures --plot all
```

Available `--plot` values:

| Value | Output | Slide |
|---|---|---|
| `crosstask`            | `headline_bar_crosstask.pdf` | 5 |
| `green_slices`         | `green_slices.pdf`           | 6 |
| `heatmap_ckpt_K`       | `heatmap_ckpt_K.pdf`         | 6 (inset) |
| `scaling_law`          | `scaling_law.pdf` (copied)   | 7 left panel |
| `t2_pareto`            | `t2_pareto.pdf` (copied)     | 7 right panel |
| `t2_magnitude`         | `t2_magnitude.pdf`           | 7 sub-panel + 9 thumbnail |
| `methods_comparison`   | `methods_comparison.pdf`     | 5b |
| `method_architecture`  | `method_architecture.pdf`    | 4 |
| `all`                  | all of the above             | — |

Add `--ext png` to write the preview thumbnail under
`figures/preview/` instead of the high-res PDF.

Numbers are hard-coded in dict literals at the top of the file
(`SRC_CROSSTASK`, `SRC_LAMBDA_SLICE`, `SRC_K_SLICE`,
`SRC_METHODS`).  Update the dict to refresh.  The
ckpt-×-K heatmap reads `q2q3_AGGREGATE.json` directly — no manual
copy needed.

### `render_trajectory_on_env.py`

Per-task end-effector ensemble overlaid on the robosuite env
(playhead-revealed MP4 + static PDF).

```bash
# One task
python -m presentation.scripts.render_trajectory_on_env --task can

# All five at once
python -m presentation.scripts.render_trajectory_on_env --task all
```

Knobs: `--n-candidates 24`, `--cluster-radius-xy 0.04`,
`--camera agentview`, `--duration-s 5`, `--fps 30`,
`--static-only`, `--no-task-filter`.  See `--help`.

### `render_best_rollout.py`

Per-task best-demo replay MP4 (the robot actually doing the task).

```bash
# Short tasks: action-replay (honest physics)
python -m presentation.scripts.render_best_rollout --task lift
python -m presentation.scripts.render_best_rollout --task can
python -m presentation.scripts.render_best_rollout --task square

# Long-horizon tasks: state-replay (pixel-perfect, guarantees
# success at the end; action-replay diverges over T>300 steps)
python -m presentation.scripts.render_best_rollout --task tool_hang --mode state
python -m presentation.scripts.render_best_rollout --task transport --mode state
```

### `render_trajectory_ensemble.py`

Generic 3-D end-effector ensemble (no env background).  Produces a
single static PDF + animated MP4.  Currently unused in the deck
(the env-overlay version `render_trajectory_on_env.py` superseded
it) but kept as a fallback if the env render path breaks.

```bash
python -m presentation.scripts.render_trajectory_ensemble \
    --n-candidates 24 --duration-s 5 --fps 30
```

### `make_previews.sh` (none — see below)

PNG previews under `figures/preview/` are rendered by calling the
plot functions in `make_headline_figures.py` with a `.png` output
path instead of `.pdf`.  There is no standalone script; see the
inline `figures/preview/` README block or the build commands in
`presentation/DECK_GUIDE.md`.

## Defunct scripts (kept for archaeology)

### `render_rollouts.py`  *(do not invoke)*

Original BC / MBD / hybrid comparison MP4 generator.  Imports
`robomimic.utils.env_utils`, which transitively pulls
`mujoco_py` → `ModuleNotFoundError` in the current `hybrid-traj`
conda env (we use the newer `mujoco` bindings).  Functionality
superseded by `render_best_rollout.py`, which talks to
`robosuite.make` directly.

### `render_videos.py`  *(do not invoke)*

Same `mujoco_py` issue as `render_rollouts.py`.  Superseded by
`render_best_rollout.py`.

## Dependencies (all in the `hybrid-traj` conda env)

- `numpy`, `matplotlib`, `h5py`
- `imageio` + `imageio-ffmpeg`
- `robosuite` (offscreen renderer), `mujoco`
- `torch` for reading the DP-v3b checkpoint's `wall_s` field

The conda env is locked in [`env/environment-from-history.yml`](../../env/environment-from-history.yml).
