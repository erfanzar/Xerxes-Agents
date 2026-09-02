---
name: code-review
description: Review the current changes (working tree, staged, or a branch vs its base) for correctness, bugs, and maintainability. Use when the user asks for a review before committing, pushing, or opening a PR.
version: 1.0.0
author: Xerxes Agent
license: MIT
metadata:
  xerxes:
    tags: [Code-Review, Quality, Git]
    related_skills: [github-code-review, security-review]
---

# Code Review

Review changes like a strong senior engineer: findings first, ranked by severity,
each with file:line and a concrete fix. No summary padding, no compliments.

## Scope resolution

1. If the user named a target (branch, commit range, PR), review that.
2. Otherwise review the current work: `git status --porcelain`, then
   `git diff HEAD` (fall back to `git diff` vs `main...HEAD` if HEAD is clean).
3. Empty scope → say there is nothing to review and stop.

## Review pass

Read every changed file **in full context** (open the file, not just the diff —
bugs hide in the unchanged lines around the edit). Check, in priority order:

1. **Correctness**: logic errors, off-by-one, wrong operator, swapped args,
   unreachable branches, error paths that swallow or misreport failures.
2. **State & concurrency**: races, stale closures, un-awaited promises,
   shared mutable state, resource leaks (unclosed handles, missing cleanup).
3. **Contract drift**: changed signatures without updated callers, wire-format
   or persisted-shape changes without migration notes, test expectations
   edited to match a regression.
4. **Security-adjacent**: injection sinks, unvalidated external input at
   boundaries, secrets in code (deep-dive belongs to `security-review`).
5. **Consistency**: violations of patterns the surrounding code clearly follows;
   dead code left behind by the change.

## Output format

```
## Findings
- [blocker] src/loop.ts:142 — retries increment before the guard, so maxRetries
  is effectively +1. Move the increment after the guard.
- [warning] …
- [nit] …

## What's solid
(one or two lines, only if true)
```

Verdict at the end: **merge-ready**, **fix-then-merge**, or **do-not-merge** with
the blocking reasons. For deep PR-level review with inline comments, chain into
the `github-code-review` skill.
