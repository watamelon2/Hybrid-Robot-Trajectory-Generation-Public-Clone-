---
description: Regenerate the presentation headline figures (PDF + preview PNG) and refresh deck stats
---

# Skill: regen-presentation-figures

Use after any results change that affects the deck (new success rates,
new prior, checkpoint correction).

## When to use
- Editing `SRC_METHODS` / `SRC_CROSSTASK` numbers in the figure script.
- Updating `presentation/DECK_GUIDE.md` stats tables.

## Key facts
- Figure generator: `presentation/scripts/make_headline_figures.py`.
- Hardcoded data blocks at the top: `SRC_CROSSTASK` (bar chart) and
  `SRC_METHODS` (standalone-vs-hybrid bars + Pareto scatter).
- Needs torch/matplotlib -> run with the conda env's python:
  `/Users/szh/anaconda3/envs/hybrid-traj/bin/python`.
- Every number on the deck must be footnoted with provenance
  (n, seed, checkpoint, lambda). Keep standalone-vs-hybrid bars on the
  SAME checkpoint; do not mix ep0149 hybrid with ep0199 standalone on one
  bar (note the other as a footnote).

## Steps
1. Edit the data block(s) in `make_headline_figures.py` and the matching
   table/narrative in `DECK_GUIDE.md`.
2. Regenerate high-res PDFs:
   ```bash
   /Users/szh/anaconda3/envs/hybrid-traj/bin/python \
     presentation/scripts/make_headline_figures.py
   ```
3. Regenerate the preview PNGs the deck embeds:
   ```bash
   /Users/szh/anaconda3/envs/hybrid-traj/bin/python \
     presentation/scripts/make_headline_figures.py --ext png
   ```
4. Visually verify the changed figure (e.g. `read_file` on
   `presentation/figures/preview/methods_comparison.png`).
5. Commit with the commit-and-push skill (`git add presentation/`).

## Verify
- `git status --short presentation/figures/` shows only the figures you
  intended to change (others re-render byte-identically).
