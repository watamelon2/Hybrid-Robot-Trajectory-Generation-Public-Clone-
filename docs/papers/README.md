# Papers & Reference Management

## Directory Structure

```
docs/
├── papers/
│   ├── references.bib       ← CANONICAL single source of truth (tracked in git)
│   ├── references_old.bib   ← Earlier snapshot, kept for reference
│   ├── paper_briefings.md   ← Paper summaries
│   ├── download_missing.sh  ← Fetch arXiv PDFs whose entries have `eprint`
│   ├── README.md            ← This file
│   ├── *.pdf                ← Paper PDFs (local only, gitignored)
│   └── fix_zotero_keys.js   ← Local Zotero key-fix helper (gitignored)
├── literature_review/references.bib  → symlink to ../papers/references.bib
├── topic_introduction/references.bib → symlink to ../papers/references.bib
└── progress_reports/references.bib   → symlink to ../papers/references.bib
```

**All `.tex` files in `docs/` share the same `references.bib`** via symlinks.
Edit only `docs/papers/references.bib` — changes propagate everywhere automatically.

PDFs are intentionally **not** committed to the repo (see `.gitignore`). Keep them
locally in `docs/papers/` or in your Zotero library.

---

## Zotero Setup (Recommended)

### One-Time Setup

1. **Install Better BibTeX (BBT)** — the key plugin for auto-sync:
   - Download from: https://retorque.re/zotero-better-bibtex/
   - In Zotero → Tools → Add-ons → gear icon → Install from file

2. **Import the bib into Zotero:**
   - File → Import → select `docs/papers/references.bib`
   - BBT will preserve the citation keys (e.g., `ho2020ddpm`)
   - Create a collection called **"ECE M237 Project"**

3. **Attach PDFs:**
   - For each entry, right-click → Add Attachment → Attach Link to File
   - Point to the corresponding PDF in `docs/papers/`
   - Or: drag-and-drop PDFs onto their entries

4. **Set up auto-export (keeps bib in sync):**
   - Right-click the "ECE M237 Project" collection
   - Export Collection → format: **Better BibTeX**
   - Check **"Keep updated"**
   - Save to: `docs/papers/references.bib`
   - Now whenever you add/edit entries in Zotero, the .bib auto-updates

### Daily Workflow

| Task | How |
|------|-----|
| **Add a new paper** | Add to Zotero (drag PDF, use DOI, or browser connector) → auto-exported to bib → available in all .tex files |
| **Download a paper** | Zotero → right-click → Find Available PDF → move PDF to `docs/papers/` (kept local, gitignored) |
| **Cite in LaTeX** | Use the BibTeX key shown in Zotero (e.g., `\cite{ho2020ddpm}`) |
| **Change a cite key** | In Zotero, click the entry → Extra field → add `Citation Key: mykey` |

### BBT Settings (Preferences → Better BibTeX)

- **Citation key formula:** `auth.lower + year + shorttitle(1,1).lower`
  (or keep "preserve imported" to maintain existing keys)
- **Export → Fields → File field:** set to `off` if you don't want file paths in the bib
- **Automatic export:** On Change (default)

---

## Quick Reference: Paper Status

### Downloaded (27 PDFs)
All ML/robotics papers plus: PPO, SAC, DQN, OT-CFM, Kappen, Hyvärinen,
Vincent, DIAL-MPC.

### Bulk download from arXiv

Run `bash docs/papers/download_missing.sh` from the repo root. It parses
`references.bib`, fetches any entry that has an `eprint = {NNNN.NNNNN}`
field from `https://arxiv.org/pdf/<eprint>.pdf`, and reports which
entries still need manual attention (no `eprint` in bib, or pay-walled).

### Needs Manual Download (books & paywalled)
Use UCLA library proxy or Zotero's "Find Available PDF" for paywalled
articles. Once you have the PDF (and the arXiv id, if any), update the
entry in `references.bib` so the next `download_missing.sh` run can
pick it up automatically. Books generally need purchase or library
access.
