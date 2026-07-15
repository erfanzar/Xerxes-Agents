---
name: review-pr
description: High-signal correctness review for pull requests, branch diffs, or uncommitted changes. Use when asked to review code for bugs, contract breaks, missing tests, or release/commit hygiene.
tags: [software-development, review, workflow]
---

# Skill: Review PR

Provide a high-signal review. Findings must be bugs, broken contracts, security issues, behavioral regressions, or concrete testing-policy violations. Avoid style nits and speculative concerns.

## Required Context

1. Identify changed files with `git diff --name-only`, `git show --name-only`, or a PR diff command.
2. Read project instructions: root `AGENTS.md` plus `.agents/AGENTS.md`, `.agents/SKILL_MAP.md`, and `.agents/ops/OPS.md` when present.
3. Read relevant package/module config and docs for touched areas.
4. For testing-policy questions, load `test-workspace` or the project-local `.agents/skills/test-workspace/SKILL.md`.

## Review Passes

When subagents are available, split review into independent passes:

- Gate: summarize the branch/PR intent and detect whether review should stop.
- Context: collect only relevant project rules, package config, docs, and test policy.
- Compliance: check changed files against project boundaries and release rules.
- Bugs: inspect the diff for compile failures, missing imports, incorrect logic, bad state assumptions, data-shape mismatches, or runtime regressions.
- Validation: independently confirm every proposed finding before reporting it.

If subagents are unavailable, run the same passes manually and keep separate notes.

## What To Flag

Flag only issues grounded in code or docs:

- Public contract or API breaks.
- Missing or wrong migration, persistence, schema, or protocol handling.
- Security or sandbox bypasses.
- Tests that assert implementation details instead of behavior.
- Release, version, or commit text that violates project policy.
- Performance claims without a relevant measured path when the change is performance-sensitive.

Do not flag broad missing coverage unless a specific changed behavior lacks any observable check.

## Output

Lead with findings ordered by severity. Each finding needs a file/line reference, why it is a real bug or contract violation, the relevant rule or code path, and the smallest credible fix direction. If there are no findings, say that clearly and name residual risk.
