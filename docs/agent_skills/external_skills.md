# External open-source skills — vetted registry

A curated, **license-aware** pointer list of open-source agent skills
relevant to this project. We deliberately do **not** vendor third-party
skill files into this repo unless their license clearly permits it (see
"Licensing" below). Instead, this registry says what exists, whether it's
safe/relevant, and how to install it on demand.

> Vetting date: 2026-06-08. Re-check licenses before adopting — several
> upstreams had no SPDX license at the time of writing.

## Licensing caution
- **`anthropics/skills`** had **no detected LICENSE file** (GitHub license
  API returned 404); usage terms are only in its README/disclaimer. Treat
  as "all rights reserved unless the README says otherwise" — do NOT copy
  its files into this repo without confirming terms. Using them through
  Claude Code / Claude.ai (where Anthropic ships them) is the intended path.
- Community "awesome-*" repos are mostly **link directories**, not skill
  content. Licenses vary (MIT / Apache-2.0 / NOASSERTION / none). Only
  adopt individual skills whose own license is explicit and compatible.
- Skill files can contain instructions that run shell commands. **Read any
  skill before enabling it**; never auto-run an unvetted skill's steps.

## Tier 1 — directly useful for THIS project (document/deliverable skills)
From `github.com/anthropics/skills/tree/main/skills` (license-gated — use
via Claude Code, or re-author a clean adaptation; see `regen-*`/author note):

| Skill | Project use |
|---|---|
| `pptx` | Build/QA the presentation deck (`presentation/`). Has a structured create→QA→export-to-images loop. |
| `pdf` | Parse / assemble the Progress Report PDFs and the SOTA reference PDFs in `docs/papers/`. |
| `docx` | Only if a Word deliverable is needed; the papers here are LaTeX, so lower value. |
| `xlsx` | Tabulating sweep results / aggregates if you want spreadsheets. |
| `skill-creator` | Meta-skill: scaffolds new SKILL.md files in the official format — useful for authoring more of our own. |
| `doc-coauthoring` | Long-form collaborative editing of the writeups in `docs/`. |

## Tier 2 — community catalogs (browse for more; verify license per-skill)
| Repo | License | Notes |
|---|---|---|
| `ComposioHQ/awesome-claude-skills` | none stated | Large curated list; pointers only. |
| `hesreallyhim/awesome-claude-code` | NOASSERTION | Skills + hooks + slash-commands index. |
| `heilcheng/awesome-agent-skills` | MIT | Tutorials + skill directory (license clean). |
| `rohitg00/awesome-claude-code-toolkit` | Apache-2.0 | Agents/skills/commands toolkit (license clean). |

> Skepticism note: ignore obviously off-topic high-star search hits (e.g. a
> Rust WiFi-sensing repo tagged "claude") — star counts on some of these are
> not a reliability signal.

## Installed community skills (vendored into `.windsurf/workflows/`, local-only)
| Skill | Source | License | Upstream SHA | Vetting |
|---|---|---|---|---|
| `python-best-practices` | `rohitg00/awesome-claude-code-toolkit` `skills/python-best-practices/SKILL.md` | Apache-2.0 | `79573df` | Read in full; Python type-hints/dataclasses/async/packaging guidance; self-contained single file; no shell commands. Installed unmodified. |
| `deep-research` bundle (`deep-research`, `literature-search`, `literature-review`, `citation-management`, `novelty-assessment`) | `lingzhi227/agent-research-skills` `skills/<name>/` | none stated (personal use OK'd by user) | `9e6c085` | 40 scripts safety-scanned: only `urllib` GETs to arXiv/Semantic Scholar/OpenAlex/CrossRef + list-form `subprocess` wrappers around git/gh/latexmk; no `os.system`/`shell=True`/`rm -rf`/`eval`/sockets/POST-exfil. Vendored as **full folders** to `.windsurf/skills/<name>/` (scripts intact) + flat launchers in `.windsurf/workflows/`. |

Considered and **rejected**:
- `testing-strategies` (same repo): examples are TypeScript / web-service
  (Pact, testcontainers) — not Python/pytest; not relevant here.
- `git-advanced` (same repo): would conflict with this repo's strict
  commit/merge conventions (`commit-and-push`, `merge-to-main`).
- `heilcheng/awesome-agent-skills` (MIT): website/catalog only, no skill
  content to vendor.

> Caveat (`deep-research` bundle): the upstream `SKILL.md`s contain the
> author's hardcoded absolute paths (`/Users/lingzhi/...`) and reference an
> external `paper_finder` tool not in the repo. The self-contained `urllib`
> search scripts under each `.windsurf/skills/<name>/scripts/` work as-is;
> ignore the absolute paths and call scripts by their vendored location.

> Apache-2.0 note: the file is used locally (`.windsurf/` is gitignored,
> not redistributed). Source + license recorded here for attribution; do
> not relicense. If ever committed/redistributed, include the upstream
> NOTICE and mark modifications.

## Research-specific skills
There is **no polished open-source catalog of ML/robotics research skills**
(eval orchestration, checkpoint selection, experiment tracking). For those,
prefer authoring project-specific skills (see the rest of `docs/agent_skills/`)
modeled on the official `skill-creator` format.

## How to install one (when you choose to)
```bash
# Inspect first (never blind-run):
#   browse github.com/<owner>/<repo>/tree/main/skills/<name>
# For Claude Code, skills live under ~/.claude/skills/<name>/SKILL.md
# or per-project .claude/skills/. For Cascade, copy a SKILL.md's steps
# into .windsurf/workflows/<name>.md (note: .windsurf/ is gitignored here).
```
