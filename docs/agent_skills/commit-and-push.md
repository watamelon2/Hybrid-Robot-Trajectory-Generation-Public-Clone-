---
description: Commit and push changes using this repo's commit-message + LFS-hook conventions
---

# Skill: commit-and-push

Use this whenever you are about to commit and/or push in this repo. It
encodes two hard-won project conventions (see MEMORY + `docs/`).

## When to use
- Any commit on any branch (including `--amend`, `rebase --continue`).
- Any `git push` from the user's main Mac.

## Steps

1. Stage exactly the intended files. Be explicit; never blanket-`git add -A`
   without checking `git status --short` first. In particular, NEVER stage
   the large `*.pt` checkpoints (they are gitignored / LFS and the root
   `v_theta_*_ep0199.pt` is a self-referential symlink).

2. Write the full commit message to a temp file, then commit from it:
   ```bash
   # write_to_file -> /tmp/commit_msg.txt   (use the write_to_file tool)
   git commit -F /tmp/commit_msg.txt
   ```
   - Do NOT use `git commit -m "..."` (multi-line / shell-quoting breaks).
   - Do NOT use heredoc `git commit -F - <<EOF ... EOF` (the user's shell
     cancels these).

3. Push. The repo has Git-LFS hooks via `core.hookspath` but `git-lfs` is
   NOT on PATH on the user's main Mac, so the `pre-push` hook errors out:
   ```bash
   git push --no-verify origin <branch>
   ```
   - `--no-verify` is SAFE only when the pushed commits contain no
     LFS-tracked binary changes (the common case: docs / results JSON /
     code). If you are pushing new/changed LFS binaries, stop and resolve
     `git-lfs` on PATH instead.

## Verify
- `git log --oneline -2 <branch>` shows your commit at HEAD.
- `git status --short` is clean (or only intentionally-unstaged files
  like the `.pt` type-change remain).

## Gotchas
- Long/multiline shell commands: write them to a temp script and run that
  (user global rule), don't inline giant pipes/heredocs.
