---
name: sdlc-review
description: Independently verify review-lane handoffs and route verdicts.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [review, quality, verification, sdlc, handoff]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/devops/sdlc-review/SKILL.md
---

# SDLC Review Skill

Independently verify work handed from an implementation run to the review
lane, then approve it, request changes, or escalate. This skill reviews the
deliverable and its evidence; it does not take over the implementer's work.

## When to Use

Use this skill when all of the following are true:

- a dispatcher spawned you to review a task claimed from a review lane;
- an implementer submitted a handoff claiming the work is done;
- the task needs an independent verdict before it can be completed.

Do not use it for a separate downstream review card. A downstream card is
ordinary implementation work with a review-oriented specification and
completes through its own lifecycle.

## Prerequisites

- A worker context with the current task and run identifiers.
- Workspace access through file read, search, and the shell/terminal tool
  when the deliverable is code.
- The task's original specification, acceptance criteria, handoff summary,
  and prior run history must be available from the dispatcher or task record.

## How to Run

Start from the task record before inspecting files or choosing a verdict.

1. Read the task specification and the latest handoff claim.
2. Inspect the actual deliverable and run relevant verification.
3. Choose exactly one verdict: approve, request changes, or escalate.
4. Record concrete evidence with the verdict returned to the dispatcher.

| Verdict | When | Outcome |
|---|---|---|
| Approve | Acceptance criteria and verification pass | Task completes |
| Request changes | Correctable implementation defects remain | Returned to the implementer with findings |
| Escalate | A human decision or external prerequisite is required | Task blocked pending input |

## Review Lenses

Vary the inspection on each round instead of repeating the same pass.
Determine the round by counting prior `changes_requested` entries in the
task history; round 1 has zero.

| Round | Lens | How to apply it |
|---|---|---|
| 1 | Artifact | Read the diff or deliverable cold, before the implementer's summary. Form an independent judgment, then compare it against the handoff narrative and investigate every mismatch. |
| 2 | Execution | Check out the work and actually run it via the shell/terminal tool: build, test, and exercise the reported behavior yourself. Verify each handoff claim empirically. |
| 3+ | Contract | Re-read the ORIGINAL task body and acceptance criteria, then audit the deliverable strictly against them. Confirm every item from every prior change request actually landed. |

The baseline duties below apply on every round; the lens sets which
inspection you lead with.

For ad-hoc review fan-outs with the agent spawning tools, give each parallel
reviewer a different lens — one diff-only brief, one full-context brief, one
checkout-and-run brief. Identical briefs produce correlated verdicts and
duplicate findings.

## Procedure

### 1. Orient from the task record

Identify the original task body and acceptance criteria, the latest
implementation summary, changed files and commit identifiers, test evidence,
and findings from prior review rounds. Treat the handoff as a claim to
verify, not as proof the work is correct.

### 2. Compare requested behavior with delivered behavior

Map every acceptance criterion to concrete evidence. Note omissions, changed
semantics, and unrelated scope before deciding on deeper checks.

For code work:

1. Inspect the changed paths and their callers with file read and search
   tools.
2. Use the shell/terminal tool to run the project's existing focused tests,
   lint, type checks, or build commands (for Xerxes: `bun run check`, the
   focused `bun test` file, `bun run build` when packaging changed).
3. Exercise the reported failure path and at least one ordinary control path
   when practical.
4. Check error handling, edge cases, concurrency boundaries, data
   preservation, and security boundaries relevant to the change.
5. Confirm tests assert behavior rather than snapshotting source text.

For non-code work, inspect the complete deliverable, check correctness,
completeness, formatting, and provenance, and validate referenced external
facts when they affect the verdict.

### 3. Choose one verdict

**Approve** only when the acceptance criteria are satisfied and the evidence
is sufficient. Include the exact checks that passed and any bounded caveat
that does not block acceptance.

**Request changes** for specific, correctable defects. State where the
defect is, how it reproduces, why it violates the task, and what minimum
outcome would resolve it. The task returns to its original implementer.

**Escalate** only when reviewer and implementer cannot resolve the problem
without a human decision or an external prerequisite. Explain the blocked
decision and the smallest information needed to continue.

### 4. Preserve role separation

Do not edit the implementation while acting as reviewer. Request changes and
let the implementer produce the next candidate; then verify that candidate
independently in the next review run.

## Pitfalls

- **Rubber-stamping:** a passing handoff summary is not independent evidence.
- **Reviewer implementation:** editing the deliverable hides ownership and
  weakens the re-review boundary.
- **Vague findings:** "needs work" gives no reproducible correction target.
- **Style-only blocking:** do not request changes for preference-level nits
  when behavior and repository standards are satisfied.
- **Skipping prior rounds:** re-review must confirm both the requested
  corrections and preservation of previously passing behavior.
- **Escalating ordinary rework:** correctable defects belong in a
  change request; reserve escalation for genuine external blockers.
- **Completing without evidence:** every approval must name the checks or
  artifacts actually inspected.

## Verification

- [ ] The task record and latest handoff were read first
- [ ] Every acceptance criterion was mapped to evidence
- [ ] The actual deliverable was inspected
- [ ] Relevant focused checks were run, or an explicit reason was recorded
      when execution was impossible
- [ ] Prior requested changes were re-tested on re-review
- [ ] Unrelated regressions and scope changes were considered
- [ ] The verdict uses exactly one terminal action
- [ ] The summary contains concrete, non-secret evidence
- [ ] No implementation files were edited by the reviewer

---

Adapted from the `sdlc-review` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Jakub Wolniewicz (@frizikk) + Hermes Agent.
