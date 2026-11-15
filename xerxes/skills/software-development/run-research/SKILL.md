---
name: run-research
description: Default workflow for any non-trivial research, debugging, optimization, or implementation task. Plans first in plan mode, keeps a live plan artifact updated after every step, and hands off with evidence. Use it whenever a task needs more than a couple of tool calls.
tags: [software-development, research, workflow, planning, artifacts]
---

# Skill: Run Research

Use this as the base workflow for multi-step work in a real repository. Project-specific skills may build on this one; this file owns the shared cadence.

## When To Use It

Use it by default whenever the task is more than a small, obvious edit: new features, multi-file or architectural changes, debugging with an unknown cause, performance work, research questions, or anything where several approaches are valid. Skip it only for one-line fixes, greetings and quick factual answers.

## How To Apply This Skill

1. Load this file first.
2. Check for project-local context in `.agents/AGENTS.md`, `.agents/SKILL_MAP.md`, and `.agents/ops/OPS.md`.
3. Load any more specific project skill under `.agents/skills/<skill-name>/SKILL.md` when the task maps to one.
4. Keep project-specific commands, symptoms, and recovery steps in project `.agents` files rather than baking them into generic answers.

## Plan First (Plan Mode)

1. Enter plan mode yourself with `EnterPlanModeTool` when the task changes code in more than one place, the requirements are unclear, or there is a real choice between approaches. You do not need the user to ask. Do not enter plan mode in objective mode or for a small, clear fix.
2. In plan mode, only read and research: inventory the surface, find the files involved, state the hypothesis.
3. Write the plan as an artifact with `ArtifactTool` (action `create`, kind `markdown`): goal, steps as a checklist, files to touch, verification, risks and open questions.
4. Call `ExitPlanModeTool`. The user sees a "Plan ready" card with two choices: **Continue with the plan**, or a note that refines it. Do not ask "should I go ahead?" in prose; the card is the question.
5. If the exit is refused with feedback, stay in plan mode, revise the same artifact (action `update` with its id), and call `ExitPlanModeTool` again.
6. Once approved, execute the plan step by step.

## Artifact After Every Step

The plan artifact is the user's live view of the work. After every completed step:

- Update the artifact (action `update`, same id) and tick the step: `- [x]` done, `- [~]` in progress, `- [ ]` pending.
- Under the step, add one or two lines of evidence: the command run and its result, the file and line changed, or the finding.
- When the plan changes (a hypothesis was falsified, a new step is needed), edit the checklist and say why in a short "Changes to the plan" section.
- Keep the chat reply short and point at the artifact rather than repeating it.

## Grounding Rule

Do not cite a script, flag, entry point, environment variable, function, or path until you have opened it or found it with repository search. If a useful target should exist but does not, say that and create or update the routing doc only when the user asked for that kind of maintenance.

## Workflow

1. Inventory the real surface with `git status --short`, `git diff --stat`, and targeted search.
2. State the current hypothesis and the exact evidence that would falsify it.
3. Make the smallest code or doc change that tests the hypothesis.
4. Run focused verification first. Use `test-workspace` or project `.agents/skills/test-workspace/SKILL.md` for test selection.
5. Keep only changes supported by verification.
6. Update the plan artifact with the step's result before starting the next step.
7. Record durable design or research notes in `.agents/projects/` only when the task is too large to finish cleanly in the current pass. Operational recovery steps belong in `.agents/ops/OPS.md`.

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

Finish by updating the artifact one last time with a "Result" section, then report briefly in chat:

- Files changed.
- Verification commands and outcomes.
- Remaining risk, especially skipped hardware, integration, or end-to-end checks.
- For performance work, direct baseline vs candidate numbers with hardware, shape/dtype, compile-including timing when relevant, and steady-state timing.
