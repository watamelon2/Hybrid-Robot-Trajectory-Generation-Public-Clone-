# Agent Skills

Reusable, project-specific procedures ("skills") for AI agents working in
this repo. Each file captures a battle-tested workflow with concrete
commands, gotchas, and verification steps, so an agent can execute it
reliably without rediscovering the conventions.

## Why these live here
`.windsurf/` is gitignored, so workflows placed there are not
version-controlled or shareable. These tracked Markdown skills are the
source of truth. To use one as a Cascade slash-command, copy/symlink it
into `.windsurf/workflows/`; the YAML frontmatter (`description:`) is
already in the workflow format.

## Index

| Skill | When to use |
|---|---|
| [`commit-and-push.md`](./commit-and-push.md) | Any commit/push — `/tmp/commit_msg.txt` message convention + `--no-verify` LFS-hook workaround. |
| [`merge-to-main.md`](./merge-to-main.md) | Promoting a `feat/*` branch — `--no-ff`, cleanup-on-branch, SAFETY-tag reshape. |
| [`launch-slate.md`](./launch-slate.md) | Starting a long-running eval slate — core sizing, MPS rule, v2 launcher hardening. |
| [`resume-slate.md`](./resume-slate.md) | Waking a paused (SIGSTOP'd) slate — `pkill -CONT` parent + workers. |
| [`regen-presentation-figures.md`](./regen-presentation-figures.md) | Refreshing deck figures + stats after a results change. |
| [`refresh-deck-assets.md`](./refresh-deck-assets.md) | Full deck-asset rebuild (figures + previews + videos) feeding the manual PowerPoint. |
| [`compile-latex-paper.md`](./compile-latex-paper.md) | Build a `.tex` deliverable (progress report / lit review / topic intro) with bib resolution via `latexmk`. |
| [`external_skills.md`](./external_skills.md) | Vetted, license-aware registry of open-source skills (Anthropic + community) relevant to this project. |

## Conventions referenced
- Python: conda env `hybrid-traj`; run via
  `/Users/szh/anaconda3/envs/hybrid-traj/bin/python` or after
  `conda activate hybrid-traj`.
- Commits: write message to `/tmp/commit_msg.txt`, then
  `git commit -F /tmp/commit_msg.txt` (never `-m`, never heredoc).
- Push: `git push --no-verify` (git-lfs not on PATH on the main Mac).
- Long/multiline shell: write to a temp script and run it.
