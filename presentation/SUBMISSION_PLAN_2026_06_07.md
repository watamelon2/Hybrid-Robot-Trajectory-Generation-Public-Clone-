# Tonight presentation submission plan — 2026-06-07

## Priority order

1. **Build the deck from `presentation/DECK_GUIDE.md`.**  It is the first-open file and now carries the June-7 Square, bank-size, and SOTA updates.
2. **Use the default 9-slide deck.**  Do not include optional slide 5b unless rehearsal is comfortably under time.
3. **Make slide 5 visually clean first.**  Use `headline_bar_crosstask.pdf` plus the three task trajectory overlays.  The key update is Square `15/30 = 50.0%` at `H=300, λ=2`.
4. **Make slide 7 clean second.**  Use the regenerated `scaling_law.pdf` with the June-7 bank-size curve and `t2_pareto.pdf` for dynamics adaptation.
5. **Keep slide 9 short.**  Conclusions first, then four future-work bullets.  If time is tight, drop the bank-size theory bullet.

## Must-use files

| Purpose | File |
|---|---|
| Build guide with previews | `presentation/DECK_GUIDE.md` |
| Minimal PPTX patch notes | `presentation/PPTX_REVISION_NOTES_2026_06_07.md` |
| Backup canonical spec | `presentation/SLIDE_SPEC.md` |
| Full speaking script | `presentation/OUTLINE.md` |
| Asset/source map | `presentation/INDEX.md` |
| Main cross-task figure | `presentation/figures/headline_bar_crosstask.pdf` |
| Bank-size figure | `presentation/figures/scaling_law.pdf` |
| Preview images | `presentation/figures/preview/headline_bar_crosstask.png`, `presentation/figures/preview/scaling_law.png` |

## Tonight checklist

- [ ] Open `presentation/DECK_GUIDE.md` and copy the nine non-optional slides.
- [ ] Confirm slide 5 says: Lift 100%, Can 73.8%, Square 50.0% at `H=300, λ=2`.
- [ ] Confirm slide 5 includes SOTA tick explanation or a small footer: Lift 99.3%, Can 92.3%, Square 74.7%.
- [ ] Confirm slide 7 uses the June-7 bank-size curve: 23.3, 35.0, 65.0, 68.3, 70.0, 68.3%.
- [ ] Omit tool_hang and transport headline numbers; answer Q&A with “no honest post-fix number yet.”
- [ ] Rehearse once without slide 5b.
- [ ] If over 7 minutes: trim slide 4 by 15 s, slide 6 by 20 s, slide 8 by 10 s.
- [ ] Export a PDF backup of the deck before submission.

## 30-second fallback story

> The hybrid method combines a demonstration prior with MBD’s analytical cost gradient.  Zero-training retrieval plus the bridge solves Lift, reaches 73.8% on Can, and now partially unlocks Square at 50% with longer horizon and stronger guidance.  The bridge also lifts a moderately trained DP prior to 86.7%, but only when the prior is coherent and has headroom.  The bank-size curve shows the retrieval result is finite mode coverage, not full-bank memorization, and the next research step is Square replication plus subgoal/value signals for harder long-horizon tasks.
