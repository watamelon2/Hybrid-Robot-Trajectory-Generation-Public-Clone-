# Week-8 Presentation

6-minute talk + 3-minute Q&A (hard stop at 7 min, peer-evaluated).

**Open [`DECK_GUIDE.md`](./DECK_GUIDE.md) first** — it is the
single self-contained document with inline figure previews, copy-
paste-ready slide content, and per-slide speaker scripts.

## Folder layout

| Path | What it is |
|---|---|
| [`DECK_GUIDE.md`](./DECK_GUIDE.md)         | **Open this first.** Per-slide spec with inline PNG previews + relative links to PDFs/MP4s + ≤50-word scripts.  Self-contained. |
| [`SLIDE_SPEC.md`](./SLIDE_SPEC.md)         | Same content as `DECK_GUIDE.md` but without inline previews and with more verbose commentary.  Backup reference. |
| [`OUTLINE.md`](./OUTLINE.md)               | Verbatim full-prose speaking script (longer than `DECK_GUIDE.md`'s scripts). |
| [`INDEX.md`](./INDEX.md)                   | Asset inventory: every figure/video/script with its absolute path, source script, source data, and slide assignment. |
| [`ASSETS.md`](./ASSETS.md)                 | Earlier inventory, superseded by `INDEX.md`.  Kept for git-archaeology. |
| [`figures/`](./figures/)                   | 12 PDFs (headline charts + per-task trajectory overlays) + 11 PNG previews under `figures/preview/`. |
| [`videos/`](./videos/)                     | 11 MP4s: 5 best-rollout replays (one per task) + 5 trajectory-ensemble animations (one per task) + 1 generic 3-D ensemble. |
| [`scripts/`](./scripts/)                   | Build scripts.  See [`scripts/README.md`](./scripts/README.md) for invocation details. |

## Headline narrative (3 sentences)

1. **Composite-score hybrid** of a learned demonstration prior and
   the analytical score of model-based diffusion (MBD), sharing one
   denoising loop.
2. **Best result: 86.7 %** (26 / 30) at `(ep0149, K=10, λ=2.0)` on
   `PickPlaceCan/ph` — a **+13.4 pp** lift over the same prior at
   the canonical `λ=0.5` baseline, at the *same* compute.  A
   second GREEN cell at `(ep0149, K=20, λ=0.5)` reaches **83.3 %**
   (25 / 30) along an independent knob.
3. **Mechanism:** MBD helps iff (a) the prior's score is informative
   at the operating noise level **and** (b) the prior has not
   saturated the task.  Demonstrated by sweeping `ckpt × K × λ` —
   the lift lives only at the moderately-trained checkpoint
   (`ep0149`); the weak (`ep0049`) and over-trained (`ep0199+`)
   priors stay flat.

## Build everything from a clean state (rarely needed)

All figures, videos, and PNG previews are **already committed** —
you do not need to rebuild anything to view the deck.  These
commands exist only if you change numbers (e.g. re-aggregate
Phase-D) and want to refresh a chart.

```bash
cd "$(git rev-parse --show-toplevel)"
PY="$HOME/anaconda3/envs/hybrid-traj/bin/python"

# 6 headline PDFs + 1 method-architecture schematic
$PY -m presentation.scripts.make_headline_figures --plot all

# 5 trajectory-on-env ensemble PDFs + 5 MP4s
$PY -m presentation.scripts.render_trajectory_on_env --task all

# 5 best-rollout MP4s (action-replay for short tasks,
# state-replay for tool_hang / transport)
$PY -m presentation.scripts.render_best_rollout --task lift
$PY -m presentation.scripts.render_best_rollout --task can
$PY -m presentation.scripts.render_best_rollout --task square
$PY -m presentation.scripts.render_best_rollout --task tool_hang --mode state
$PY -m presentation.scripts.render_best_rollout --task transport --mode state
```

Total wall ≈ 3 min on a recent Mac with the `hybrid-traj` conda env.

## Source-data dependencies

| Number | Where it comes from |
|---|---|
| Phase-D `ckpt × K × λ` grid (GREEN cells) | [`experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json`](../experiments/17_dp_hybrid/results/phase_d_2026_05_30/q2q3_AGGREGATE.json) |
| Phase-D Q1 epoch curve            | [`experiments/17_dp_hybrid/results/phase_d_2026_05_30/q1_AGGREGATE.json`](../experiments/17_dp_hybrid/results/phase_d_2026_05_30/q1_AGGREGATE.json) |
| Bank-size scaling law             | [`docs/results_writeups/bank_size_scaling.md`](../docs/results_writeups/bank_size_scaling.md) and `presentation/scripts/make_headline_figures.py::SRC_BANK_SIZE_SCALING` |
| BC training time                  | [`experiments/03_bc_baseline/results/training_log.json`](../experiments/03_bc_baseline/results/training_log.json) (`total_wall_s`) |
| DP-v3b training time              | [`experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0149.pt`](../experiments/08_flow_matching_prior/results/v_theta_dp_tobs2_h16_v3b_ep0149.pt) (`wall_s` field) |

See `INDEX.md` for the complete source-data map.

## Branch status

This work landed on `main` in commit `bb942d4` (merge of
`feat/presentation-week8`).  The feature branch is preserved in
the local repo for archaeology; it can safely be deleted whenever
convenient with `git branch -d feat/presentation-week8`.
