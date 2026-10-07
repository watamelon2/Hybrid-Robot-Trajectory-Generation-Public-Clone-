---
description: Merge a feature branch into main with this repo's strict branch-hygiene convention
---

# Skill: merge-to-main

Use when integrating a feature branch into `main`. Encodes the user's
strict convention: **cleanup/scope commits live ON THE BRANCH before
merging — never as a direct-to-main commit after the merge.**

## When to use
- Promoting any `feat/*` branch into `main`.
- Reshaping a merge where another agent landed cleanup directly on `main`.

## Pre-flight
1. `git fetch origin` and inspect what will merge:
   ```bash
   git diff --stat main...origin/<branch>
   git merge-base --is-ancestor main origin/<branch> && echo "branch up to date with main"
   ```
2. If the branch needs a scope-aligned cleanup commit, commit it **on the
   branch first** (use the commit-and-push skill), then merge.

## Standard merge
1. Make sure any in-tree, unrelated working changes (e.g. the `.pt`
   type-change) stay UNSTAGED so they don't enter the merge commit.
2. ```bash
   git checkout main
   git pull --ff-only origin main
   git merge --no-ff origin/<branch>     # always --no-ff
   ```
3. Resolve conflicts. For auto-generated artifacts (logs, FAILURE_NOTES):
   - take the branch's result log: `git checkout --theirs <path>`
   - keep main's newer notes: `git checkout --ours <path>`
   then `git add <paths>`.
4. Commit the merge with `git commit -F /tmp/commit_msg.txt` (no `-m`).
5. Verify the `.pt`/binaries did NOT enter:
   `git show --stat HEAD | grep -E 'v_theta|\.pt' || echo clean`
6. Push: `git push --no-verify origin main`.

## Reshape (cleanup landed on main by mistake)
1. Backup tag both heads:
   `git tag SAFETY-pre-reshape-main-$(date +%F)` and same for the branch.
2. `git reset --hard <pre-bad-merge-sha>` on main.
3. In the branch worktree: reset branch to pre-cleanup, cherry-pick the
   cleanup commit, then cherry-pick any commits that followed it (use the
   SAFETY tag as the upper bound).
4. From the canonical clone: `git merge --no-ff <cleanup-sha>` (target the
   cleanup commit, not the branch tip).
5. `git push --force-with-lease` both refs.

## Verify
- Tree identity after reshape: `git diff --stat <old-main-tip> main` is
  EMPTY (only the graph changed, file contents identical).
