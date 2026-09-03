---
name: simplify-code
description: Parallel 4-reviewer cleanup of recent code changes.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [code-review, cleanup, refactor, parallel, simplify]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/simplify-code/SKILL.md
---

# Simplify Code — Parallel Review & Cleanup

Review recent code changes with four focused reviewers running in parallel,
aggregate their findings, and apply the fixes worth applying.

**This is a cleanup pass, not a bug hunt.** You are improving code that
already works: removing duplication, flattening needless complexity, cutting
waste, deepening band-aid fixes. Correctness review is a different pass —
use the `code-review` or `requesting-code-review` skill for that.

**Core principle:** four narrow reviewers beat one broad reviewer. Each
deeply searches for a single class of problem. They run concurrently, so
you pay the latency of one review, not four.

## When to Use

- The user says "simplify", "clean up my changes", or "review my recent changes"
- Optional modifiers: a focus (`reuse`, `quality`, `efficiency`, `altitude`),
  a dry run ("just report, change nothing"), or a scope ("the last commit",
  "staged", a specific file)

Do not auto-run after every edit. It costs several subagents' worth of
tokens; invoke it only when asked.

## The Process

### Phase 1 — Identify the changes

Capture the diff with the shell/terminal tool, in this default order:
`git diff` (uncommitted), then `git diff HEAD` (include staged), then scoped
variants such as `git diff --staged`, `git diff HEAD~1`,
`git diff main...HEAD`, or `git diff -- path/to/file`. If there are no
changed files and the user named none, say so and stop. Warn before
reviewing very large diffs (>2000 changed lines) and offer to scope down.

### Phase 2 — Launch four reviewers in parallel

Use the agent spawning tools with all four tasks in one batch so they run
concurrently. Give **every** reviewer the **complete diff** (cross-file
issues hide in the gaps) plus the repo path so they can search the wider
codebase with file read and search tools.

If delegation is unavailable, work through all four angles yourself,
sequentially, with the same standards — and say plainly in the summary that
this was a single-pass inline review.

Tell each reviewer to:
- Search the existing codebase for evidence; do not reason from the diff alone.
- Apply Chesterton's Fence: run `git blame` before flagging removals; if the
  original purpose is unclear, mark the finding `confidence: low`.
- Report structured findings:
  `file:line → problem → cost → suggested fix | confidence | risk`
  where risk is SAFE (cannot affect behavior — auto-apply), CAREFUL
  (improves without changing semantics — apply with tests), or RISKY
  (may change behavior or break contracts — flag for the user).
- Skip nits and style-only churn.

**Reviewer 1 — Code Reuse.** Duplicate functionality that already exists in
the codebase: utility modules, shared helpers, ad-hoc re-implementations of
existing parsing, path, env, or type-guard logic. Name the existing thing to
use and where it lives.

**Reviewer 2 — Code Quality.** Redundant or derivable state, parameter
sprawl, copy-paste-with-variation, leaky abstractions, stringly-typed code
where a constant/registry exists, deeply nested conditionals, and
AI-generated slop (obvious restating comments, unnecessary defensive checks
on validated inputs, `as any` casts, patterns inconsistent with the file).

**Reviewer 3 — Efficiency.** Redundant computation, repeated file reads,
N+1 access, missed concurrency, hot-path bloat, TOCTOU pre-checks instead of
op-then-error-handling, unbounded growth and listener/handle leaks,
overly broad reads, and silent failures (empty catches, ignored error
returns) that should at minimum log before swallowing.

**Reviewer 4 — Altitude.** Band-aids on shared infrastructure: special cases
added to generic paths, symptoms patched at one call site while siblings
keep the flaw, workarounds stacked on workarounds, flags routing around a
broken default. Identify the underlying mechanism and describe the deeper
fix — or note honestly when it should be its own task. What looks like a
band-aid is sometimes a deliberate boundary (compat shims, staged
migrations); check `git blame` first.

### Phase 3 — Aggregate and apply

1. **Merge** findings, deduplicating overlaps.
2. **Discard false positives** silently — you have the most context.
3. **Resolve conflicts** with the order: correctness > the user's focus >
   readability/reuse > micro-perf. When both options are defensible, pick
   the one touching less code and note the alternative.
4. **Apply in risk-tier order:** SAFE first (run tests after), CAREFUL next
   one file at a time (run tests after each; revert any that break), RISKY
   last — presented, never auto-applied. Dry run: present all tiers, apply
   nothing.
5. **Verify:** run the project's targeted tests for touched files and any
   linter or type check the repo uses (for Xerxes: `bun run check` and the
   focused `bun test` file).
6. **Summarize** applied fixes grouped by reviewer category and risk tier,
   plus deliberate skips and why.

## Pitfalls

- **Do not fan out wider than 4.** More reviewers adds cost and conflicting
  suggestions, not coverage.
- **Give the whole diff to each reviewer.** Splitting defeats the design.
- **Reviewers search, they do not guess.** Drop findings without
  `file:line` evidence.
- **Apply ≠ rewrite.** Scope edits to the diff plus the minimal surrounding
  change a fix requires. Deeper altitude fixes get flagged, not rebuilt.
- **Do not drift into bug-hunting.** Report genuine correctness bugs
  prominently but separately from cleanup fixes.
- **Respect project conventions** (AGENTS.md, linter configs) in reviewer
  prompts so suggestions match house style.
- **Dead-code tools over-trust.** `knip`, `ts-prune`, `depcheck` flag exports
  used dynamically; always grep for the symbol before removing.
- **Renaming public contracts is RISKY.** Export names, API paths, config
  keys are contracts even when the name is bad.
- **Not every empty catch is dead code.** Some are intentional; flag rather
  than remove.

## Related

`subagent-driven-development` covers parallel review during implementation
per task; this is the standalone after-the-fact cleanup pass.
`requesting-code-review` is the pre-commit gate — the bug hunt.

---

Adapted from the `simplify-code` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent (inspired by Claude Code /simplify).
