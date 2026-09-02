---
name: commit
description: Stage the relevant changes and create a git commit with a well-written message derived from the actual diff. Use when the user asks to commit, save work, or write a commit message.
version: 1.0.0
author: Xerxes Agent
license: MIT
metadata:
  xerxes:
    tags: [Git, Commit, Version-Control]
    related_skills: [github-pr-workflow]
---

# Commit

Create a git commit from the current working tree. The message must describe the
**diff**, not the conversation.

## Procedure

1. Run these in parallel to see the full picture:
   - `git status --porcelain` — what is untracked/modified
   - `git diff` — unstaged changes
   - `git diff --cached` — already-staged changes
   - `git log -5 --oneline` — the repo's message style (follow it)
2. Decide what belongs in THIS commit. Never stage:
   - files that look like they contain secrets (`.env`, `*token*`, `*credential*`, keys)
   - OS/tooling noise (`.DS_Store`, editor swap files) not covered by `.gitignore`
   - changes unrelated to the task the user asked about — leave them and say so
3. Stage deliberately (`git add <paths>` — never blanket `git add -A` when the tree
   is mixed).
4. Write the message: imperative subject ≤72 chars, matching the repo's recent style
   (Conventional Commits if `git log` shows them). Body only when the *why* is not
   obvious from the diff. No "Generated with" footers unless the repo already has
   that convention.
5. `git commit -m "…"` and report the short hash + subject. If a pre-commit hook
   fails, show the failure and stop — never `--no-verify` unless the user asks.

If the tree is clean, say so and stop. If nothing the user asked about is present
in the diff, say that instead of committing unrelated leftovers.
