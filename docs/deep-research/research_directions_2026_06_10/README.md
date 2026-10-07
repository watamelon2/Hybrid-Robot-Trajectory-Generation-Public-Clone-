# Deep research pass — research directions, 2026-06-10

This directory contains a focused literature-search and research-direction synthesis for the Hybrid Robot Trajectory Generation project.

## Primary output

Read first:

- `phase6_report/report.md`

## Supporting outputs

- `phase1_frontier/frontier.md` — frontier trends and project fit.
- `phase2_survey/survey.md` — broad landscape survey.
- `phase3_deep_dive/selection.md` — selected papers and rationale.
- `phase3_deep_dive/deep_dive.md` — structured notes for selected papers.
- `phase4_code/code_repos.md` — relevant implementation ecosystem.
- `phase5_synthesis/synthesis.md` — cross-paper/project synthesis.
- `phase5_synthesis/gaps.md` — open gaps and hypotheses.
- `phase6_report/references.bib` — generated BibTeX.

## Databases

- `paper_db_raw_supplemented.jsonl` — 295 unique raw records after API search and merge.
- `paper_db.jsonl` — ranked top-80 project-affinity records.
- `paper_db_curated.jsonl` — 45 manually curated project-relevant records.
- `search_stats.json` — search/ranking summary.

## Headline recommendation

Prioritize:

1. phase-conditioned MBD for `tool_hang` / long-horizon sparse tasks;
2. retrieval prior coverage theory and diagnostics;
3. Square/contact-rich residual failure diagnosis;
4. calibration rules for learned-prior/MBD coupling;
5. fast adaptation through bank editing.

The Square `λ=0.5` trace completed during this research pass at 19/30 =
63.3%, with 4 `align_or_insert` and 7 `post_contact_unclear` failures.
