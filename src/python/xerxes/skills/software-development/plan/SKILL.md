---
name: plan
description: Plan mode for Xerxes — inspect context, write a markdown plan into `~/xerxes-agent/plans/`, and do not execute the work.
version: 1.0.0
author: Xerxes Agent
license: MIT
metadata:
  xerxes:
    tags: [planning, plan-mode, implementation, workflow]
    related_skills: [writing-plans, subagent-driven-development]
---

# Plan Mode

Use this skill when the user wants a plan instead of execution.

## Core behavior

For this turn, you are planning only.

- Do not implement code.
- Do not edit project files. The only file you may write is the plan markdown file under `~/xerxes-agent/plans/`.
- Do not run mutating terminal commands, commit, push, or perform external actions.
- You may inspect the repo or other context with read-only commands/tools when needed.
- Your deliverable is a markdown plan saved under the user's home `xerxes-agent` folder, not inside the current working directory or active project.

## Output requirements

Write a markdown plan that is concrete and actionable.

Include, when relevant:
- Goal
- Current context / assumptions
- Proposed approach
- Step-by-step plan
- Files likely to change
- Tests / validation
- Risks, tradeoffs, and open questions

If the task is code-related, include exact file paths, likely test targets, and verification steps.

## Save location

Save the plan with `write_file` under:
- `~/xerxes-agent/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

Treat that as an absolute home-relative path. Do not save plans under `.xerxes/`, `./plans/`, `tmp-files/`, or any other directory inside the active project.

If the runtime provides a specific target path, use that exact path.
If not, create a sensible timestamped filename yourself under `~/xerxes-agent/plans/`.

## Interaction style

- If the request is clear enough, write the plan directly.
- If no explicit instruction accompanies `/plan`, infer the task from the current conversation context.
- If it is genuinely underspecified, ask a brief clarifying question instead of guessing.
- After saving the plan, reply briefly with what you planned and the saved path.
