# Presentation Asset Index

Every figure, video, and script in `presentation/`, with
its absolute path, source script, source data, and the slide(s)
it appears on.  Use this as the single source of truth for
"where is X stored?".

Last refreshed: 2026-06-07.

---

## Directory tree

```
presentation/
├── INDEX.md              ← this file
├── README.md             ← folder overview + headline narrative
├── OUTLINE.md            ← verbatim speaking script (long form)
├── SLIDE_SPEC.md         ← build-ready slide spec (this is the
│                           document deck-builders should follow)
├── ASSETS.md             ← legacy inventory (superseded by INDEX.md)
├── figures/              ← 13 PDFs, ~2.1 MB total
├── videos/               ← 11 MP4s, ~11 MB total
└── scripts/              ← 8 build scripts + README.md
```

---

## Figures (13 PDFs in `presentation/figures/`)

Absolute path prefix for every entry below:
`/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation/presentation/figures/`

### Headline charts (slide-level visuals)

| File | Size | Used on | Source script | Source data |
|---|---:|---|---|---|
| `headline_bar_crosstask.pdf`   | 18 KB | **slide 5** main visual          | `make_headline_figures.py --plot crosstask`     | hard-coded `SRC_CROSSTASK` from per-task AGGREGATE.json |
| `green_slices.pdf`             | 21 KB | **slide 6** main visual          | `make_headline_figures.py --plot green_slices`  | hard-coded `SRC_LAMBDA_SLICE` + `SRC_K_SLICE` from `phase_d_2026_05_30/q2q3_AGGREGATE.json` |
| `heatmap_ckpt_K.pdf`           | 21 KB | **slide 6** inset (optional)     | `make_headline_figures.py --plot heatmap_ckpt_K` | `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json` (read directly) |
| `scaling_law.pdf`              | 25 KB | **slide 7** left panel (coverage axis)  | `make_headline_figures.py --plot scaling_law`   | June-7 bank-size data hard-coded in `SRC_BANK_SIZE_SCALING` |
| `t2_pareto.pdf`                | ~25 KB| **slide 7** right panel (dynamics axis) | `make_headline_figures.py --plot t2_pareto`     | drawn from `experiments/15_unified_hybrid/results/slate_incremental_aggregate/T2_pareto_data.csv` with audience-facing labels |
| `t2_magnitude.pdf`             | ~24 KB| **slide 7** sub-panel | `make_headline_figures.py --plot t2_magnitude` | drawn from `experiments/15_unified_hybrid/results/slate_incremental_T2_magnitude_aggregate/summary.json` |
| `methods_comparison.pdf`       | 29 KB | **slide 5b** (optional) main     | `make_headline_figures.py --plot methods_comparison` | hard-coded `SRC_METHODS`: BC `train_log.json`, DP-v3b ckpt `wall_s`, Phase-D success rates |
| `method_architecture.pdf`      | 32 KB | **slide 4** schematic            | `make_headline_figures.py --plot method_architecture` | pure matplotlib boxes-and-arrows, no external data |

### Trajectory overlays (5 per-task + 1 generic)

| File | Size | Used on | Source script | Demo |
|---|---:|---|---|---|
| `trajectory_on_env_lift.pdf`      | 265 KB | **slide 5** grid                  | `render_trajectory_on_env.py --task lift`      | demo_73,  T=36  |
| `trajectory_on_env_can.pdf`       | 344 KB | **slide 5** grid                  | `render_trajectory_on_env.py --task can`       | demo_116, T=82  |
| `trajectory_on_env_square.pdf`    | 290 KB | **slide 5** grid                  | `render_trajectory_on_env.py --task square`    | demo_91,  T=107 |
| `trajectory_on_env_tool_hang.pdf` | 345 KB | *archived, not in deck*           | `render_trajectory_on_env.py --task tool_hang` | demo_134, T=332 |
| `trajectory_on_env_transport.pdf` | 357 KB | *archived, not in deck*           | `render_trajectory_on_env.py --task transport` | demo_193, T=373 |
| `trajectory_ensemble.pdf`         |  98 KB | unused (kept for reference)       | `render_trajectory_ensemble.py`                | 3-D plot, no env overlay |

All overlays use 24 candidates by default, clustered to within
4 cm of the optimal demo's can xy (so the cloud shows the *same*
task, not 24 different tasks).

---

## Videos (11 MP4s in `presentation/videos/`)

Absolute path prefix for every entry below:
`/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation/presentation/videos/`

### Best-rollout replays — one per task (the "expert success" videos)

| File | Size | Duration | Used on | Source script | Mode |
|---|---:|---:|---|---|---|
| `best_rollout_lift.mp4`      | 288 KB |  ~2 s | **slide 2** + slide 5 | `render_best_rollout.py --task lift`      | action-replay |
| `best_rollout_can.mp4`       | 435 KB |  ~4 s | **slide 1/2** hero    | `render_best_rollout.py --task can`       | action-replay |
| `best_rollout_square.mp4`    | 675 KB |  ~5 s | slide 5               | `render_best_rollout.py --task square`    | action-replay |
| `best_rollout_tool_hang.mp4` | 2.5 MB | ~13 s | *archived, not in deck* | `render_best_rollout.py --task tool_hang --mode state` | state-replay¹ |
| `best_rollout_transport.mp4` | 2.4 MB | ~14 s | *archived, not in deck* | `render_best_rollout.py --task transport --mode state` | state-replay¹ |

¹ State-replay (`set_sim_state(env, states[t])` per step) needed
because open-loop action-replay drifts on the long-horizon tasks.
Visually identical, but guarantees `success=True` at the end.

### Trajectory-ensemble animations (one per task + 1 generic)

| File | Size | Used on | Source script |
|---|---:|---|---|
| `trajectory_on_env_lift.mp4`      | 368 KB | slide 2 motivation panel | `render_trajectory_on_env.py --task lift`      |
| `trajectory_on_env_can.mp4`       | 478 KB | **slide 2** motivation panel | `render_trajectory_on_env.py --task can`       |
| `trajectory_on_env_square.mp4`    | 539 KB | slide 5 grid             | `render_trajectory_on_env.py --task square`    |
| `trajectory_on_env_tool_hang.mp4` | 637 KB | *archived, not in deck*  | `render_trajectory_on_env.py --task tool_hang` |
| `trajectory_on_env_transport.mp4` | 687 KB | *archived, not in deck*  | `render_trajectory_on_env.py --task transport` |
| `trajectory_ensemble.mp4`         | 2.0 MB | unused (kept for reference) | `render_trajectory_ensemble.py`             |

All are 5 s @ 30 fps (150 frames) with a playhead reveal +
gentle 3D spin from −60° → −35° azimuth.

---

## Scripts (8 files in `presentation/scripts/`)

Absolute path prefix for every entry below:
`/Users/szh/Documents/SZH/Courseworks/ECE M237/Project/Hybrid-Robot-Trajectory-Generation/presentation/scripts/`

### Active scripts

| File | What it produces | Invoke as |
|---|---|---|
| `make_headline_figures.py`     | 8 PDFs (headline bar, green slices, heatmap, scaling-law, methods-comparison, method-architecture, adaptation Pareto, magnitude sweep) | `python -m presentation.scripts.make_headline_figures --plot {all,crosstask,green_slices,heatmap_ckpt_K,scaling_law,methods_comparison,method_architecture,t2_pareto,t2_magnitude} [--ext {pdf,png}]` |
| `render_trajectory_on_env.py`  | per-task ensemble PDF + MP4 (env overlay)         | `python -m presentation.scripts.render_trajectory_on_env --task {lift,can,square,tool_hang,transport,all}` |
| `render_best_rollout.py`       | per-task expert-demo replay MP4                   | `python -m presentation.scripts.render_best_rollout --task {…} --mode {action,state}` |
| `render_trajectory_ensemble.py`| generic 3-D ensemble (no env)                     | `python -m presentation.scripts.render_trajectory_ensemble --n-candidates 24 --duration-s 5` |
| `README.md`                    | run instructions                                  | — |

### Defunct (do not invoke; kept for code archaeology)

| File | Why defunct |
|---|---|
| `render_rollouts.py` | Imports `robomimic.utils.env_utils`, which transitively pulls `mujoco_py` → ModuleNotFoundError in the current `hybrid-traj` conda env.  Functionality superseded by `render_best_rollout.py` (uses `robosuite.make` directly). |
| `render_videos.py`   | Same `mujoco_py` issue.  Superseded by `render_best_rollout.py`. |

---

## Source-data anchors

The figure scripts read data from these canonical locations.  If
re-aggregation changes the numbers, edit the dict literal at the
top of `make_headline_figures.py` (or re-run the script with the
new path).

| Data | Path |
|---|---|
| Phase-D Q1 epoch curve            | `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q1_AGGREGATE.json` |
| Phase-D Q2 / Q3 / E / F (2-D grid)| `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json` |
| Bank-size scaling law             | `docs/results_writeups/bank_size_scaling.md` + `presentation/scripts/make_headline_figures.py::SRC_BANK_SIZE_SCALING` |
| Adaptation Pareto (Can mass=1.3)  | `experiments/15_unified_hybrid/results/slate_incremental_aggregate/T2_pareto_data.csv` (re-drawn at build time with clean labels) |
| Magnitude sweep (Can 1.1-1.5)     | `experiments/15_unified_hybrid/results/slate_incremental_T2_magnitude_aggregate/summary.json` (drawn at build time) |
| BC training time                  | `experiments/03_bc_baseline/results/training_log.json` (`total_wall_s` field) |
| DP-v3b training time              | `experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0149.pt` (`wall_s` field in ckpt blob) |
| Cross-task retrieval / BC / MBD   | `experiments/15_unified_hybrid/results/slate_*/AGGREGATE.json` |
| HDF5 demo banks (5 tasks)         | `~/anaconda3/envs/hybrid-traj/lib/python3.11/site-packages/datasets/{lift,can,square,tool_hang,transport}/ph/low_dim_v141.hdf5` |
| Custom can mass-1.3 bank          | `~/Documents/SZH/Courseworks/ECE M237/Project/HRTG-shared-data/can_mass1p3_220demos.hdf5` |

---

## Build everything from a clean state

```bash
cd "$(git rev-parse --show-toplevel)"   # the Hybrid-Robot-Trajectory-Generation repo
PY="$HOME/anaconda3/envs/hybrid-traj/bin/python"

# All 6 headline PDFs
$PY -m presentation.scripts.make_headline_figures --plot all

# 5 ensemble-overlay PDFs + 5 MP4s
$PY -m presentation.scripts.render_trajectory_on_env --task all

# 5 best-rollout MP4s (3 action-replay + 2 state-replay)
$PY -m presentation.scripts.render_best_rollout --task lift
$PY -m presentation.scripts.render_best_rollout --task can
$PY -m presentation.scripts.render_best_rollout --task square
$PY -m presentation.scripts.render_best_rollout --task tool_hang --mode state
$PY -m presentation.scripts.render_best_rollout --task transport --mode state
```

Approximate total build time on the local Mac: **~2 min** for
all figures + ensembles, **~1 min** for all rollouts, **~3 min
total**.

---

## Slide-asset cross-reference (which file is on which slide)

| Slide | Title | Primary visual | Secondary visuals |
|---|---|---|---|
| 1 | Title                       | `videos/best_rollout_can.mp4` (looping, faded) | — |
| 2 | Motivation                  | 3-up video panel: `videos/best_rollout_lift.mp4`, `videos/trajectory_on_env_can.mp4`, `videos/best_rollout_can.mp4` | — |
| 3 | Problem formulation         | LaTeX equation (Beamer or LaTeXiT) — *no* figure file | — |
| 4 | Method + baselines          | `figures/method_architecture.pdf` | 6-row comparison table (markdown copy in SLIDE_SPEC.md) |
| 5 | Cross-task results          | `figures/headline_bar_crosstask.pdf` (top-left) | `figures/trajectory_on_env_{lift,can,square}.pdf` (other three quadrants) |
| 5b | Pareto *(optional)*        | `figures/methods_comparison.pdf` | — |
| 6 | GREEN cells                 | `figures/green_slices.pdf` | `figures/heatmap_ckpt_K.pdf` (inset, optional) |
| 7 | Adaptation (two axes)       | `figures/scaling_law.pdf` (coverage) + `figures/t2_pareto.pdf` (dynamics) | `figures/t2_magnitude.pdf` (sub-panel, mass=1.1-1.5 curves) |
| 8 | Lessons + negative results  | text-only (two-column wins / didn't-work bullets) | — |
| 9 | Conclusions + Future work   | text-only (two-column: 3 headline results / 4 next steps) | — |

---

## Quick FAQ

**Q: Where are the diagrams stored?**
A: `presentation/figures/` for static PDFs, `presentation/videos/` for animations.  Full paths above.

**Q: How do I rebuild a single figure?**
A: `python -m presentation.scripts.make_headline_figures --plot <name>`.
Available `<name>`: `crosstask`, `green_slices`, `heatmap_ckpt_K`,
`scaling_law`, `methods_comparison`, `method_architecture`,
`t2_pareto`, `t2_magnitude`, `all`.  Add `--ext png` to render the
preview thumbnail under `figures/preview/` instead of the PDF.

**Q: How do I change which demo is used as the "optimal" reference?**
A: Pass `--optimal {shortest,first}` to either ensemble script,
or pass `--data <path>` to point at a different HDF5.

**Q: How do I render the videos at a different camera angle?**
A: Pass `--camera {agentview,frontview,sideview,birdview}` to
the trajectory or rollout scripts.  All `ph` tasks ship with
those four cameras.

**Q: How do I get a slower (or faster) playback for tool_hang?**
A: `python -m presentation.scripts.render_best_rollout
--task tool_hang --mode state --slow-factor 0.5` halves the
playback rate.  Or use `--max-steps` to truncate.
