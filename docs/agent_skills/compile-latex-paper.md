---
description: Compile a LaTeX document in this repo (progress reports, literature review, topic intro) with bib resolution
---

# Skill: compile-latex-paper

Clean, dependency-light LaTeX build for this repo's `.tex` deliverables.
Authored for this project (no third-party skill vendored).

## When to use
- Building any `.tex` under `docs/progress_reports/`,
  `docs/literature_review/`, or `docs/topic_introduction/`.

## Facts about this repo
- `papers/references.bib` is the single canonical bibliography, symlinked
  into the directories that cite it; entries carry `eprint`/`archiveprefix`.
- Built PDFs are gitignored (`docs/**/*.pdf`) — they are artifacts, do not
  commit them.
- TeX is NOT part of the conda env; this needs a system TeX distribution
  (MacTeX / TeX Live) providing `latexmk` + `biber`.

## Steps
1. Pre-flight (fail clearly if TeX is missing):
   ```bash
   command -v latexmk >/dev/null || { echo "Install MacTeX/TeX Live (need latexmk + biber)"; exit 1; }
   ```
2. Build from the document's OWN directory so the symlinked
   `references.bib` resolves:
   ```bash
   # example: literature review
   ( cd docs/literature_review && \
     latexmk -pdf -interaction=nonstopmode -halt-on-error literature_review_accessible.tex )
   ```
   `latexmk` auto-detects biblatex/biber vs natbib/bibtex and runs the
   right number of passes — do NOT hand-run pdflatex/biber.
3. On a bib/citation change, force a clean bib pass:
   ```bash
   ( cd <dir> && latexmk -pdf -g <file>.tex )
   ```
4. Clean aux clutter when done (keeps the PDF):
   ```bash
   ( cd <dir> && latexmk -c <file>.tex )
   ```

## Verify
- `<file>.pdf` is produced and page count looks right.
- The `.log` has no `LaTeX Error` / `Undefined control sequence`, and no
  `Citation 'x' undefined` (those mean the bib pass didn't run).

## Gotchas
- Long multiline shell: write to a temp script and run it (user rule).
- Do not `git add` the resulting PDF or the `*.aux/*.bbl/*.bcf` files —
  they are gitignored on purpose.
