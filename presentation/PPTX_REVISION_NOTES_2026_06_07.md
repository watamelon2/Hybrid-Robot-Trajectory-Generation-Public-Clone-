# PPTX revision notes — 2026-06-07

Use this for revising `/Users/szh/Downloads/M237 Final Presentation.pptx` without rebuilding the deck from scratch.  Keep the existing slide order and visual style; only patch stale claims and follow the scripts in `DECK_GUIDE.md`, `SLIDE_SPEC.md`, and `OUTLINE.md`.

## Minimal edits

### Slide 5 — Cross-task results

Add the `Trained-DP + MBD` series if the chart has room.  Frame it as
a trained-prior reference, not as the main claim:

- `Trained-DP + MBD`: Lift `60/60 = 100%`, Can `26/30 = 86.7%`,
  Square `5/60 = 8.3%`.
- Keep the headline on zero-training retrieval + MBD:
  Lift `60/60`, Can `177/240 = 73.8%`, Square `15/30 = 50.0%`
  at `H=300, λ=2`.

Suggested slide title:

> Cross-task headline: zero-training retrieval, with trained-DP reference

### Slide 6 — GREEN Cells

Keep the table and main story.  Make only these small text fixes:

- Replace the vague “investigate higher λ” sentence with:
  - `λ sensitivity is prior/task dependent; Square H=300, λ=2 is the key follow-up.`
- Replace the ungrammatical DP-prior footnote with:
  - `All trials here are run using the DP prior.`

Speaker-line to follow:

> The important result is not “larger λ always wins.”  It is that the bridge lifts only when the prior is coherent and still has headroom.  The GREEN cell is `λ=2` at the moderate prior, and Square now has a separate `H=300, λ=2` follow-up to replicate.

### Slide 7 — Adaptation Scaling

Remove the two stale callouts that frame Square as an old ceiling
and the 200-demo point as a single-seed outlier.

Replace with:

> Bank-size sweep on Can: `23.3 → 35.0 → 65.0 → 68.3 → 70.0 → 68.3 %` for `N_b = 5, 10, 20, 50, 100, 200`.
>
> Takeaway: coverage saturates by `N_b = 20-50`; the full 200-demo bank is not a hidden memorization requirement.

Speaker-line to follow:

> We swept the retrieval bank from five to two hundred demos.  The curve has under-coverage below twenty demos, a knee at twenty demos, and a flat plateau from fifty to two hundred.  That supports finite mode coverage, not full-bank memorization.

### Slide 10 — Future Work and Conclusion

Replace the stale old-Square conclusion with:

- `Square is no longer a pure floor: H=300, λ=2 reaches 15/30 = 50.0%.`

Replace the future-work bullets with:

- `Replicate and localize Square: rerun H=300, λ=2; sweep nearby λ; log failure phase.`
- `Tool_hang / transport need subgoals or value signals; more product/gate tuning is deprioritized.`
- `Repeat adaptation under a trained-policy prior.`
- `Turn bank-size saturation into a finite-coverage theory story.`

Speaker-line to follow:

> The updated closing is: Lift is solved, Can is strong at 73.8%, and Square is now partially unlocked at 50% when the horizon and bridge are strong enough.  The next step is replication and failure localization, not another large product/gate sweep.

## Figure checks

Use these updated assets from the repository:

| Slide | Asset |
|---|---|
| Slide 5 | `presentation/figures/headline_bar_crosstask.pdf` |
| Slide 7 | `presentation/figures/scaling_law.pdf` |
| Slide 7 preview | `presentation/figures/preview/scaling_law.png` |

## Do not change tonight

- Do not add raw result tables beyond the single bank-size sequence.
- Do not add tool_hang / transport headline success numbers.
- Do not rewrite the method slides.
- Do not add the optional Pareto slide unless rehearsal is comfortably under time.
