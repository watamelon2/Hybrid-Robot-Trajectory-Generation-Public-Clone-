---
description: Rebuild all presentation deck assets (figures, PNG previews, videos) that feed the manually-assembled PowerPoint
---

# Skill: refresh-deck-assets

Regenerate every generated asset the Week-8 deck depends on. The PowerPoint
itself is assembled MANUALLY from `presentation/DECK_GUIDE.md` + these
assets; this skill only rebuilds the assets (it does not emit a .pptx).

## When to use
- Numbers changed (re-aggregation, checkpoint correction) and a chart,
  preview, or rollout video needs refreshing before updating the deck.

## Facts about this repo
- All assets are already committed; rebuild only when inputs change.
- Run from repo root with the conda env python:
  `PY="$HOME/anaconda3/envs/hybrid-traj/bin/python"`.
- Scripts are modules: `python -m presentation.scripts.<name>`.

## Steps
1. Headline figures (PDF) + architecture schematic:
   ```bash
   $PY -m presentation.scripts.make_headline_figures --plot all
   ```
2. PNG previews the deck embeds (same generator, png output):
   ```bash
   $PY presentation/scripts/make_headline_figures.py --ext png
   ```
3. Trajectory-on-env ensembles (5 PDFs + 5 MP4s):
   ```bash
   $PY -m presentation.scripts.render_trajectory_on_env --task all
   ```
4. Best-rollout MP4s (action-replay for short tasks; state-replay for the
   long-horizon ones that diverge over T>300):
   ```bash
   $PY -m presentation.scripts.render_best_rollout --task lift
   $PY -m presentation.scripts.render_best_rollout --task can
   $PY -m presentation.scripts.render_best_rollout --task square
   $PY -m presentation.scripts.render_best_rollout --task tool_hang --mode state
   $PY -m presentation.scripts.render_best_rollout --task transport --mode state
   ```
5. Update the matching numbers/footnotes in `presentation/DECK_GUIDE.md`
   (provenance: n, seed, checkpoint, lambda), then commit `presentation/`
   via the commit-and-push skill.

## Verify
- `git status --short presentation/` shows only intended changes (figures
  re-render byte-identically unless their data changed).
- Total rebuild wall is ~3 min on a recent Mac with `hybrid-traj`.

## Note
- This repo has NO programmatic .pptx generator; the deck is hand-built.
  If you ever want a scripted .pptx, that is new work (python-pptx) and a
  separate decision — flag it, don't assume it.
