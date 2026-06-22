---
name: prepare-commit-pr
description: "Prepare a commit or pull request safely: staging, checks, branch hygiene, PR summaries, release notes, and avoiding unrelated user changes."
tags: [software-development, git, workflow]
---

# Skill: Prepare Commit Or PR

Use this for commit and PR hygiene. For correctness review, load `review-pr`; for test selection, load `test-workspace`.

## First Reads

- `git status --short`
- `git diff --stat`
- Project root `AGENTS.md` and `.agents/AGENTS.md` when present.
- `.pre-commit-config.yaml`, CI files, and release scripts when relevant.

## Worktree Discipline

- Do not stage unrelated user changes.
- If a file has user edits plus your edits, inspect the diff carefully and stage only intended hunks.
- Never use destructive reset or checkout commands unless the user explicitly asks for that operation.
- Do not create tags, releases, external issues, or PRs unless the user requested that workflow.

## Checks

Run focused tests selected through `test-workspace` for the touched area. Also run project lint/format/pre-commit commands when the project defines them and the scope warrants it.

Pre-commit hooks may auto-fix and report failure because files changed. When that happens, inspect the diff, restage intended files, and rerun.

## Commit And PR Text

Keep claims aligned with verification. PR summaries should cover:

- Changed surfaces.
- Behavioral changes.
- Tests run.
- Skipped hardware, external service, or end-to-end checks.
- Release or migration notes when relevant.

Do not add self-credit trailers or generated-by lines unless the user explicitly asked for them.
