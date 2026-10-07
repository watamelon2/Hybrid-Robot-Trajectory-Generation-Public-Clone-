# Asset inventory — Week-8 presentation

Every figure / video referenced by `OUTLINE.md` and where it lives.

## Figures (`presentation/figures/`)

| File | Slide | Source script | Source data |
|---|---|---|---|
| `headline_bar_crosstask.pdf`   | 5 | `scripts/make_headline_figures.py --plot crosstask`     | `experiments/15_unified_hybrid/results/slate_*/AGGREGATE.json`, `experiments/17_dp_hybrid/results/phase_d_2026_05_30/q1_AGGREGATE.json` |
| `green_slices.pdf`             | 6 | `scripts/make_headline_figures.py --plot green_slices`  | Phase E + F numbers in `phase_d_2026_05_30/q2q3_AGGREGATE.json` |
| `heatmap_ckpt_K.pdf` (opt.)    | 6 | `scripts/make_headline_figures.py --plot heatmap_ckpt_K` | same |
| `scaling_law.pdf`              | 7 | `scripts/make_headline_figures.py --plot scaling_law`   | June-7 bank-size data hard-coded in `SRC_BANK_SIZE_SCALING` |
| `hero_can_grasp.png` (opt.)    | 1 | `scripts/render_rollouts.py --preset hybrid_success --save-still` | regenerated rollout |

Build them all with:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m presentation.scripts.make_headline_figures --plot all
```

## Videos (`presentation/videos/`)

| File | Slide | How to (re-)render |
|---|---|---|
| `bc_failure.mp4`     | 2 | `python -m presentation.scripts.render_rollouts --preset bc_failure --out presentation/videos/` |
| `mbd_failure.mp4`    | 2 | `python -m presentation.scripts.render_rollouts --preset mbd_failure --out presentation/videos/` |
| `hybrid_success.mp4` | 2 | `python -m presentation.scripts.render_rollouts --preset hybrid_success --out presentation/videos/` |

Or render the full trio in one shot:

```bash
python -m presentation.scripts.render_rollouts \
    --preset comparison-trio \
    --out presentation/videos/ \
    --save-still
```

## Equations (Slide 3)

Composite-score equation, in LaTeX (paste into Beamer / Keynote
LaTeXiT):

```latex
\nabla \log p_t(\tau)
  \;\approx\;
  \underbrace{\nabla \log p^{\mathrm{data}}_t(\tau)}_{\text{learned prior}}
  \;+\;
  \lambda(\sigma_t)\,
  \underbrace{\nabla \log p^{\mathrm{model}}_t(\tau)}_{\text{MBD cost}}
\,,\quad
p^{\mathrm{model}}(\tau)
  \propto
  \exp\!\bigl(-J(\tau)/\beta\bigr).
```

Prior-pull (Method, Slide 4), LaTeX:

```latex
\Delta \tau_t^{(m)}
  \;=\;
  -\,\lambda(\sigma_t)\,
  \frac{\tau_t^{(m)} - \tau^{\mathrm{prior}}}{\sigma_t^2}
```

## Color palette

Use these consistently across all figures:

| Role | Hex | Used for |
|---|---|---|
| Neutral grey | `#888888` | BC bars |
| Red    | `#c23333` | MBD-alone bars |
| Blue   | `#3399aa` | Retrieval bars |
| Green  | `#1a7f3c` | Trained-DP hybrid / GREEN cells |
| Pale grey dotted | `#bbb` | reference / baseline lines |

## Pre-flight checklist

Before the live talk:

- [ ] All three videos render and play in the slide tool.
- [ ] All four PDFs open and are vector (no rasterisation).
- [ ] Run-through with stopwatch — target ≤ 6:20 to leave 40 s
      of slack before the hard stop at 7:00.
- [ ] Backup PDF of every slide on a thumb drive in case the
      computer dies.
- [ ] First sentence rehearsed without notes.
