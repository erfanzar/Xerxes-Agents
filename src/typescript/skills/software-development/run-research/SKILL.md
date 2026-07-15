---
name: run-research
description: Base workflow for non-trivial research, debugging, optimization, or implementation tasks that require hypotheses, experiments, evidence, and a clean handoff.
tags: [software-development, research, workflow]
---

# Skill: Run Research

Use this as the base workflow for multi-step work in a real repository. Project-specific skills may build on this one; this file owns the shared cadence.

## How To Apply This Skill

1. Load this file first.
2. Check for project-local context in `.agents/AGENTS.md`, `.agents/SKILL_MAP.md`, and `.agents/ops/OPS.md`.
3. Load any more specific project skill under `.agents/skills/<skill-name>/SKILL.md` when the task maps to one.
4. Keep project-specific commands, symptoms, and recovery steps in project `.agents` files rather than baking them into generic answers.

## Grounding Rule

Do not cite a script, flag, entry point, environment variable, function, or path until you have opened it or found it with repository search. If a useful target should exist but does not, say that and create or update the routing doc only when the user asked for that kind of maintenance.

## Workflow

1. Inventory the real surface with `git status --short`, `git diff --stat`, and targeted `rg`.
2. State the current hypothesis and the exact evidence that would falsify it.
3. Make the smallest code or doc change that tests the hypothesis.
4. Run focused verification first. Use `test-workspace` or project `.agents/skills/test-workspace/SKILL.md` for test selection.
5. Keep only changes supported by verification.
6. Record durable design or research notes in `.agents/projects/` only when the task is too large to finish cleanly in the current pass. Operational recovery steps belong in `.agents/ops/OPS.md`.

## Project Notes

For long-running work, create a note under `.agents/projects/<topic>.md` with:

- Goal and stop condition.
- Baseline command and result.
- Hypotheses tested.
- Exact command/output summary for each meaningful attempt.
- Negative results worth preserving.
- Next action.

Do not create issues, PRs, tags, release artifacts, or external runs unless the user asks for that workflow or the task already depends on it.

## Reporting

Report:

- Files changed.
- Verification commands and outcomes.
- Remaining risk, especially skipped hardware, integration, or end-to-end checks.
- For performance work, direct baseline vs candidate numbers with hardware, shape/dtype, compile-including timing when relevant, and steady-state timing.
