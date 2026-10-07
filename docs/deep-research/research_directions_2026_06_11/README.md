# research_directions_2026_06_11 — deep-research refresh

Full 6-phase deep-research pass to decide the **single next experiment** that finishes
the HRTG final-report story, refreshing `../research_directions_2026_06_10/` with live
2025-2026 searches and the now-finished remote offload.

## Decision

**Calibration of test-time guidance for robot priors** — predict when/how-strongly to
guide a frozen prior and explain why product/gate over-optimize. Low compute (reuses
data on disk + light logging). Retrieval prior = flagship case study; anytime-valid
testing = evaluation method. See `phase6_report/report.md`.

## Layout

- `phase1_frontier/frontier.md` — frontier scan; dominant trend = test-time guidance of pretrained policies.
- `phase2_survey/survey.md` — 6-theme landscape over `paper_db.jsonl` (70 papers).
- `phase3_deep_dive/{selection,deep_dive}.md` — 12 papers, methodology + experiments.
- `phase4_code/code_repos.md` — code ecosystem + HRTG platform + tooling gaps.
- `phase5_synthesis/{synthesis,gaps}.md` — ranked directions + 6 gaps/hypotheses.
- `phase6_report/{report.md,references.bib}` — the decision + bibliography.
- `paper_db_raw.jsonl` (167) / `paper_db.jsonl` (70) — databases.

## Provenance

Searches: `search_arxiv.py` (14 queries × 25) + `search_semantic_scholar.py` (partial,
rate-limited). Rank/merge: `/tmp/dr_rank_2026_06_11.py`. Bib: `/tmp/dr_make_bib_2026_06_11.py`.
Planning-grade; verify venues/numbers from PDFs before camera-ready.
