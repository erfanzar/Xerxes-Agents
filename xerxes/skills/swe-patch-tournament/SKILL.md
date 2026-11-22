---
name: swe-patch-tournament
description: Solve hard or ambiguous software engineering bugs via a Best-of-N Git worktree tournament with competing root-cause hypotheses, independent regression verification, and minimal-diff selection.
version: 1.0.0
author: Xerxes-Agents
tags: [swe, bugfix, debugging, worktree, tournament, swarm]
---

# SWE Patch Tournament Guide

## Metadata

- License: Apache-2.0
- Related skills: immortals-coordinator, immortals-monitor, immortals-narrator
- Built-in: true

Activate this skill when tackling hard, ambiguous, or high-regression-risk software engineering bugs where multiple plausible root causes exist (for example async cancellation races, subtle state/cache invalidation bugs, cross-module pagination or schema edge cases, or previously failed fix attempts). Instead of committing early to a single fragile hypothesis in the main checkout, orchestrate a **Best-of-N Worktree Tournament** on Xerxes's native swarm.

## When to Run a Patch Tournament

- **Use a tournament (2–3 candidates)** when:
  - Multiple subsystems could own the invariant violation (such as producer vs consumer cleanup, parser coercion vs validator rejection, or graph ordering vs stale cache reuse).
  - A naive symptom patch could pass the immediate reproduction test while breaking callers or adjacent edge cases.
  - A single-agent attempt already regressed existing tests or stalled on conflicting hypotheses.
- **Skip a tournament (use 1 author + 1 verifier)** when:
  - The root cause and target lines are unambiguous (such as a deterministic typo, missing import, or single-site type mismatch).

## Best-of-N Tournament Architecture

### 1. Formulate Orthogonal Root-Cause Hypotheses

Before spawning workers, inspect the failure symptoms and identify **2 to 3 distinct root-cause hypotheses** — never spawn identical copies with vague briefs that converge on the same superficial patch:

- **Hypothesis A (`solver-1`)**: Fix the invariant at the upstream data/state source or lifecycle owner.
- **Hypothesis B (`solver-2`)**: Fix state transition, boundary normalization, or consumer cancellation handling across callers.
- **Hypothesis C (`solver-3`, optional for high ambiguity)**: Refactor or harden the shared contract/validator with explicit edge-case handling.

### 2. Spawn Isolated `workspace: "share"` Solvers and 1 Independent `verifier`

Use `swarm{action: "add"}` to give every candidate solver an isolated detached Git worktree (`workspace: "share"`) while keeping one read-only `verifier` (`workspace: "inherit"`):

1. **Candidate solvers (`solver-1`, `solver-2`, optional `solver-3` with `workspace: "share"`)**:
   - Assign each solver its specific root-cause hypothesis and relevant module entry points.
   - Require every solver to:
     1. Run or add a failing reproduction test in its own worktree before editing production code.
     2. Implement a clean, minimal root-cause fix — never hardcode test fixture literals, special-case test IDs, or swallow errors.
     3. Run targeted tests and typechecks inside its isolated worktree, leaving no conflict markers or unrelated formatting churn.
     4. Hand off via `post_message` to `@verifier` **only** (never `@coordinator`) with its hypothesis outcome, touched files, and test summary, then call `swarm_wait` in the same turn.
2. **Independent judge (`verifier` with `workspace: "inherit"`)**:
   - Parks immediately with `swarm_wait` until solvers report ready.
   - Never edits candidate or parent files; evaluates all candidate worktrees objectively.

### 3. Independent Reproduction and Holdout Regression Grading

When `@verifier` wakes after candidate handoffs, it inspects live `workspace_path` directories via `swarm{action: "status"}` or `read_board{status: true}` and audits each candidate worktree:

1. **Diff hygiene check**:
   - Inspect tracked and untracked changes relative to the worktree's base `HEAD` (`git -C <workspace_path> status --porcelain` and `git -C <workspace_path> diff --no-renames HEAD`).
   - Disqualify any candidate with empty diffs, unresolved merge/conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), or unauthorized edits outside scope.
2. **Reproduction + adversarial regression suite**:
   - Run the bug reproduction suite inside `<workspace_path>`.
   - Run broader module/package regression tests and boundary probes (such as empty inputs, concurrent aborts, boundary cursors, diamond dependencies, or malformed inputs) to catch symptom-patching that only satisfies the visible reproduction case.
3. **Structured scorecard to `@coordinator`**:
   - Report `[done]` to `@coordinator` listing each candidate's pass/fail status, `changedFiles` count, line delta (`insertions + deletions`), and root-cause soundness.

### 4. Winner Selection Criteria (Verification + Patch Minimality)

Rank candidates deterministically (matching `SwarmManager.compareCandidatePatches`):

1. **Verification gate**: Candidate passes both reproduction tests and broader regression checks (`verified: true`).
2. **Patch cleanliness**: Non-empty diff, zero conflict markers, and clean dry-run applicability against the parent workspace (`git apply --check`).
3. **File minimality**: Fewest `changedFiles` wins first — localized invariant fixes beat wide cross-file workarounds.
4. **Line minimality**: Smallest total line churn (`insertions + deletions`) breaks ties among equally localized passing patches.
5. **Architectural tie-breaker**: If still tied, prefer enforcing the invariant at the authoritative boundary over defensive patches in downstream callers.

If all candidates fail regression checks, synthesize what each candidate uncovered and wake the strongest solver with targeted feedback from `@verifier` rather than landing a broken diff.

### 5. Safe Landing Before Worktree Retirement

Remember that `swarm{action: "kill"}` permanently deletes a worker's `"share"` worktree or `"branch"` clone. Always land and verify the winning patch **before** killing candidate workers:

1. Export the winning candidate's unified diff from its `workspace_path` (including any newly created untracked files) or use `SwarmManager.applyWinnerPatchToWorkspace`.
2. Validate applicability in the parent workspace with `git apply --check --binary`, then apply cleanly with `git apply --binary` (never run `git commit` unless the user explicitly asks).
3. Re-run the reproduction, regression, and typecheck gates once in the parent workspace to confirm clean integration.
4. Retire `solver-*` and `verifier` with `swarm{action: "kill"}` so temporary worktrees are pruned cleanly.
