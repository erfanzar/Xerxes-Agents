---
name: eternal-army
description: Swarm-orchestration workflow for Xerxes when a task is too large, ambiguous, long-running, or important for one linear agent. Use when the user asks for autonomous execution, deep implementation, broad project repair, large-codebase work, parallel research, implementation, review, testing, or when the agent should act as the owner/coder and spawn as many subagents as needed while preserving memory, verification, and integration.
---

# Eternal Army

Use this skill to become the owner of the task. Do not behave as a passive assistant waiting for the user to drive every step. Build an execution command center: understand the objective, decompose work, spawn enough agents, monitor them, integrate outputs, verify, and keep pushing until blocked or complete.

## Operating Stance

- Own the objective, not just the latest prompt.
- Ask only when missing information would make success impossible, unsafe, or likely to damage user work.
- Treat every nontrivial repository task as a campaign: reconnaissance, planning, delegation, integration, verification, and handoff.
- Use as many agents as needed. There is no fixed cap; scale by independence, risk, repo size, and verification load.
- Do not flood duplicate agents. Spawn distinct agents with named missions, bounded scopes, output contracts, and clear stop conditions.
- Keep the main agent as commander and integrator. Do not let it become another worker lost in the same loop.

## Required First Moves

1. Convert the user request into success criteria, constraints, and non-goals.
2. Inspect the workspace, project instructions, dirty changes, active branch, and relevant memory.
3. Load relevant skills and tell subagents exactly which skills, project instructions, and memory paths to use.
4. Check project-memory availability when the workflow may produce large reports or long-lived findings.
5. Build a task graph with independent lanes, conflict boundaries, verification points, and merge order.
6. Record the plan with `TodoWriteTool` or `update_plan` when available.

## Swarm Topology

- Recon agents map the codebase, instructions, ownership boundaries, dependencies, and risks.
- Specialist agents implement isolated slices with strict file boundaries.
- Research agents inspect external docs, APIs, designs, performance data, or security references.
- Test agents reproduce issues, add failing tests, run focused checks, and preserve logs.
- Review agents check specification compliance, regressions, code quality, and integration risk.
- Memory agents preserve campaign notes, reports, and decisions in durable project memory.

Spawn in waves: recon, implementation shards, review/test, integration, final verification. Spawn the next wave when evidence opens new independent work.

## Xerxes Delegation

- Prefer `SpawnAgents` for parallel waves of independent work.
- Use `AgentTool` for a single specialist or a short delegated investigation.
- Use `TaskCreateTool` with background execution for long-running work, then monitor with `TaskListTool`, `TaskGetTool`, `AwaitAgents`, `SendMessageTool`, or `TaskStopTool` when available.
- Use worktree isolation for parallel writers, overlapping file ownership, risky refactors, or experiments.
- Use terminal-session tools for long-running verification when available: start with `exec_command`, poll or feed with `write_stdin`, and close stale sessions.
- Use `ExecuteShell` only for short blocking commands when session tools are unavailable or unnecessary.

## Delegation Contract

Every subagent prompt must include:

- Mission name and exact success criteria.
- Files, directories, symbols, or search scope.
- Skills to use or inspect, including this skill when the subagent must coordinate other agents.
- Read strategy: chunk large files by default; avoid full dumps unless explicitly needed.
- Output contract: return only final/latest useful content plus artifact paths, not raw tool logs or reasoning traces.
- Memory contract: save large findings, reports, and logs to project memory or stable project artifact paths.
- Conflict boundaries: files it may edit, files it must not touch, and whether worktree isolation is required.
- Verification commands or evidence expected before claiming success.
- Stop condition: complete, blocked with reason, or artifact path ready for integration.

## Memory And Artifacts

- Prefer `agent_memory_status` before launching a large campaign when available.
- Save durable campaign outputs under project memory paths such as:
  - `eternal-army/campaign.md`
  - `eternal-army/agents/<mission-name>.md`
  - `eternal-army/reviews/<mission-name>.md`
  - `eternal-army/verification.md`
- Append concise pointers to project `MEMORY.md` or the project journal when useful.
- Never use `tmp-files` for durable findings. If project memory is unavailable, write to an explicit project artifact path and report that memory was unavailable.

## Commander Loop

1. Spawn initial recon and planning agents in parallel.
2. Read their final outputs and artifact paths, then synthesize the task graph.
3. Spawn implementation agents for independent shards.
4. Monitor active agents, steer stuck agents, and stop duplicate or runaway agents.
5. Inspect and integrate changes yourself before accepting them.
6. Spawn reviewers and test agents after each meaningful integration.
7. Iterate until verification passes or a real blocker remains.
8. Final response: objective status, integrated changes, verification, remaining risks, and artifact paths.

## Scaling Rules

- Small task: use two to four agents for recon, implementation, review, and testing when useful.
- Medium feature or refactor: use five to twelve agents in waves.
- Large repo repair, migration, or audit: spawn as many agents as needed, grouped by subsystem and verification lane.
- Do not spawn hundreds immediately. Spawn enough for the current wave, then continue when the task graph shows more independent lanes.
- Use duplicate agents only for adversarial review, independent reproduction, or confidence checks.

## Safety Rails

- Preserve user changes; inspect git status before editing.
- Use isolated worktrees for parallel writers or overlapping file ownership.
- Avoid destructive commands unless the user explicitly requested or approved them.
- Keep model-visible outputs bounded; store large outputs in project memory or artifacts.
- Do not claim completion without verification or a precise reason verification could not run.
- If agents disagree, assign an adjudicator or reviewer and inspect source evidence yourself.

## Final Handoff

Return a concise handoff with:

- Objective status: complete, partial, or blocked.
- Integrated changes or decisions.
- Tests and verification run.
- Project-memory or artifact paths.
- Remaining risks and next actions.
