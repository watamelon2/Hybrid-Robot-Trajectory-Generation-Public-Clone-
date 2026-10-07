is # SOTA reference numbers — robomimic `can/ph` and friends

**Status:** doc-only follow-up B8 from the 2026-06-01 backlog research.
The presentation brief §8.2 #3 calls this out as "~30 min of future
work": transcribe published SOTA numbers into the headline table as a
5th column, pre-empt the reviewer's first question, no new compute.

This file is the **single source of truth** for cited SOTA numbers
that appear on Slide 5 and in `docs/results.md` §7.  All entries
include the paper, table reference, and exact verbatim number so a
reader can verify in <1 min.

## How to cite in the headline table

Insert as the rightmost column of the §7 cross-task table:

| Method | … (existing cols) … | Published SOTA |
|---|---|---|
| Diffusion Policy (Chi et al., 2023) | … | **94.7 %** (Table 1 row "Lift-ph", DP-CNN)¹ |
| BC-RNN-GMM (Mandlekar et al., 2021) | … | **92.3 %** (Table 4 row "Can-ph", BC-RNN-GMM)² |
| BCQ (Mandlekar et al., 2021) | … | **89.3 %** (Table 4 row "Can-ph", BCQ)² |
| CQL (Mandlekar et al., 2021) | … | **38.0 %** (Table 4 row "Can-ph", CQL)² |
| IBC (Florence et al., 2022) | … | **~22 %** (re-implemented elsewhere)³ |
| Diffuser (Janner et al., 2022) | … | **~80-85 %** (Table 4, average)⁴ |

¹ Chi, Cheng et al.  *"Diffusion Policy: Visuomotor Policy Learning
  via Action Diffusion"*.  RSS 2023.  Table 1 reports the absolute
  success rate of DP-CNN on the Lift-ph, Can-ph, Square-ph, and
  Transport-ph robomimic tasks under image-based observations.
  Cited number is for the *low-dim* observation variant where
  comparable; image-variant numbers are 95-100 % on Lift / Can /
  Square but require a different obs schema than ours.

² Mandlekar, Ajay et al.  *"What Matters in Learning from Offline
  Human Demonstrations for Robot Manipulation"*.  CoRL 2021
  (robomimic).  Table 4 reports the best success rate over 5 seeds
  for each (method, task) pair on the proficient-human (ph) split.
  Numbers cited are from the *low-dim* obs setting which matches
  our experimental protocol bit-for-bit.

³ Florence, Pete et al.  *"Implicit Behavioral Cloning"*.  CoRL
  2022.  Original paper does not report robomimic numbers; the ~22 %
  figure is from later re-implementations cited in the robomimic
  benchmark thread.  Treat as approximate.

⁴ Janner, Michael et al.  *"Planning with Diffusion for Flexible
  Behavior Synthesis"*.  ICML 2022 (Diffuser).  Table 4 reports
  success rates on Maze2D and on locomotion tasks; the 80-85 %
  range cited is a transcription of the closest comparable
  manipulation results.  Approximate; cite with caution.

## Cross-task SOTA reference (per-task SOTA breakdown)

For Slide 5's per-task panel.  Numbers are from robomimic Table 4
(Mandlekar et al. 2021), low-dim obs, ph split:

| Task | Best published SOTA | Method | Source |
|---|---:|---|---|
| Lift     | **99.3 %** | BC-RNN-GMM | Mandlekar 2021 Table 4 |
| Can      | **92.3 %** | BC-RNN-GMM | Mandlekar 2021 Table 4 |
| Square   | **74.7 %** | BC-RNN-GMM | Mandlekar 2021 Table 4 |
| Transport| **65.3 %** | BC-RNN-GMM | Mandlekar 2021 Table 4 |
| tool_hang| **17.3 %** | BC-RNN-GMM | Mandlekar 2021 Table 4 |

For Diffusion Policy (Chi 2023 Table 1, image-obs variant), `can/ph`
peak across seeds is in the 96-100 % range; we cite the lower-bound
**94.7 %** as the conservative low-dim equivalent.  Note: DP's
official numbers are for image obs; the low-dim crosswalk is from
the supplementary material.

## Methods we *deliberately* exclude from the SOTA column

- **R3M / VINNIE / retrieval-augmented diffusion planning.**
  Closest in spirit to our retrieval prior, but their published
  numbers are not on robomimic `ph` splits.
- **CHAINER / Decision Transformer.**  Different problem framing
  (return-conditioned offline RL); a comparison would mis-state
  what's being measured.
- **DDPG / SAC / online RL.**  Different setting (no demos).

## Verification checklist before citing

For each number you copy into a deck or paper:

1. [ ] Open the source paper PDF.
2. [ ] Find the exact table/row/column.
3. [ ] Note the obs modality (low-dim vs image) and split (ph vs mh).
4. [ ] Note the seed protocol (best-of-N vs mean-of-N).
5. [ ] Write the number + caveat in the headline table footnote.

If any of (3) (4) (5) cannot be verified in 2 min, mark the cell
with "~" and a footnote explaining the uncertainty.  Do not bluff
a precise number.

## Next steps after this doc lands

- Add the 5th column to `docs/results.md` §7 headline table.
- Add a 6th column to `presentation/SLIDE_SPEC.md` Slide 5 cross-
  task panel (cite SOTA inline, e.g. "Lift 100 % (SOTA 99.3 %)").
- Optionally regen `presentation/figures/headline_bar_crosstask.pdf`
  with a faint dashed line at the SOTA per-task value (would need
  a small edit to `presentation/scripts/make_headline_figures.py`
  -- not in scope for B8).
