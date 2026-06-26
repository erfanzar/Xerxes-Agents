---
name: bug-bounty-hunter
description: Autonomous multi-iteration bug-hunting and repair campaign for large codebases. Use when the user asks to find bugs, fix regressions, audit an entire project, run many iterations such as 100 rounds, spawn large swarms of agents, create bug bounty reports, or repeatedly scan, patch, verify, and continue until clean or blocked.
version: "1.0.0"
tags: [bugs, audit, swarm, repair, verification, large-codebase]
required_tools:
  - SpawnAgents
  - AgentTool
  - TaskCreateTool
  - TaskGetTool
  - TaskListTool
  - TodoWriteTool
  - ReadFile
  - GrepTool
  - GlobTool
  - FileEditTool
  - WriteFile
  - exec_command
  - write_stdin
  - close_terminal_session
---

# Bug Bounty Hunter

Run this skill as an autonomous bug bounty campaign: discover defects, prove them, fix them, verify them, write a report, then continue to the next iteration. The agent is the campaign owner, not a passive assistant.

## Inputs

Parse inline configuration when present:

- `Iterations: N` or `Rounds: N`: maximum campaign rounds. Default to `100`.
- `Scope: ...`: directories, files, packages, or subsystems to include. Default to the whole project.
- `Exclude: ...`: generated files, vendored code, lockfiles, build outputs, or user-specified paths.
- `Fix: yes|no`: whether to patch confirmed bugs. Default to `yes`.
- `Risk: low|medium|high`: how aggressive to be. Default to `medium`.
- `Report: path`: project-memory path for the campaign report. Default to `bug-bounty/report.md`.

If the user gives only a broad request, proceed with the defaults. Ask only when missing scope or permissions would make the campaign unsafe.

## Non-Negotiables

- Preserve user changes. Inspect the worktree before editing.
- Do not use `tmp-files` or `/tmp` for durable findings. Use project memory first.
- Use chunked reads by default: `ReadFile(file_path=..., offset=..., limit=...)`. Use `limit=-1` only when a full-file read is intentional and justified.
- Do not return raw tool logs, full transcripts, or huge file dumps from subagents.
- Do not invent a numeric tool-call cap. Agents may use as many tool calls as needed, but they must compact findings into memory when context pressure rises.
- Every claimed bug needs evidence: file path, code location or search evidence, reproduction or reasoning, severity, and a fix or explicit reason it is deferred.
- Every fix needs verification: test, lint, type check, reproduction command, or a precise explanation for why verification could not run.

## Project Memory Contract

Before the first swarm, check project memory with `agent_memory_status()` when available.

Use these default project-memory paths:

- Campaign report: `bug-bounty/report.md`
- Iteration reports: `bug-bounty/iterations/round-<NNN>.md`
- Agent findings: `bug-bounty/agents/round-<NNN>/<agent-name>.md`
- Verification log: `bug-bounty/verification.md`
- Open findings ledger: `bug-bounty/open-findings.md`

Each iteration must write a concise report before moving to the next round. If project memory is unavailable, write to an explicit project artifact path under `.agents/projects/bug-bounty/` and report that memory was unavailable.

## Campaign Loop

For each round from `1` to `Iterations`:

1. Refresh context: worktree status, project instructions, existing campaign report, open findings ledger, and recent verification failures.
2. Choose a round theme based on evidence. Examples: import boundaries, async lifecycle, session persistence, tool schemas, path security, test isolation, provider quirks, TUI rendering, channel adapters, packaging, CI.
3. Spawn a swarm sized to the round. Use as many agents as useful for independent coverage, grouped by subsystem and bug class.
4. Require each subagent to write full findings to project memory and return only the compact output contract.
5. Triage findings yourself: reject weak findings, deduplicate, rank by severity and blast radius.
6. Fix confirmed bugs when `Fix` is enabled. Assign isolated fixes to subagents only when file ownership is clear.
7. Integrate patches yourself. Inspect diffs before accepting them.
8. Verify the round with targeted commands first, then broader checks when risk warrants it.
9. Write the round report and update the campaign report, verification log, and open findings ledger.
10. Continue unless all scoped areas are clean, the iteration budget is exhausted, or a real blocker remains.

## Swarm Design

Spawn in waves, not as an undirected blast:

- Recon agents map ownership, call graphs, configs, tests, and recent changes.
- Bug-class agents hunt for specific defect types such as race conditions, missing validation, stale state, incorrect schema, async blocking, resource leaks, path traversal, injection, flaky tests, serialization bugs, and UI state drift.
- Subsystem agents cover independent directories or packages.
- Reproduction agents build failing tests or minimal commands for suspected bugs.
- Fix agents patch confirmed defects inside strict file boundaries.
- Review agents inspect diffs for regressions and missing tests.
- Verification agents run focused checks, long-running tests, and terminal sessions.

Scale by independence. Large repos may need dozens or hundreds of agents across the full campaign, but each wave should have distinct missions and stop conditions.

## Subagent Prompt Contract

Every subagent prompt must include:

- Mission name, round number, scope, and exact bug class or subsystem.
- Read strategy: inventory first, then chunked reads; avoid generated files, lockfiles, and full dumps unless necessary.
- Evidence contract: every finding must include path, symbol or location, why it is a bug, impact, and how to verify.
- Memory contract: write full findings to `bug-bounty/agents/round-<NNN>/<agent-name>.md`.
- Return contract: return only `memory_path`, `status`, severity counts, 5-10 newest useful findings, verification commands, and blockers.
- Edit contract: no edits unless explicitly assigned as a fix agent; fix agents must stay inside assigned files.
- Stop condition: complete, no findings, blocked with reason, or memory path ready.

Example prompt:

```text
Round 007 mission: hunt async lifecycle bugs in src/python/xerxes/daemon and src/python/xerxes/streaming.
Use chunked reads by default. Do not read generated files or lockfiles. Write full findings to project memory at bug-bounty/agents/round-007/daemon-async.md.
Return only: memory_path, status, severity counts, 5-10 newest useful findings with evidence, verification commands, and blockers.
Do not edit files.
```

## Finding Format

Store each confirmed or suspected finding in this shape:

```markdown
## Finding: <short title>

- Severity: critical|high|medium|low
- Status: suspected|confirmed|fixed|deferred|rejected
- Evidence: <file path, symbol, line, command, or trace>
- Impact: <what breaks and who is affected>
- Root cause: <specific code-level cause>
- Fix plan: <patch strategy or deferral reason>
- Verification: <command/test/result or required command>
```

## Fix Policy

- Fix critical and high severity confirmed bugs immediately when inside scope.
- Fix medium bugs when the patch is localized and verification is practical.
- Defer low severity or speculative issues unless they are easy and risk-free.
- Add or update tests when the defect crosses a public behavior boundary.
- Prefer focused patches over broad rewrites.
- Do not accept a fix from a subagent until the commander inspects the diff and verifies the behavior.

## Verification Strategy

Use the cheapest meaningful verification first:

- Targeted unit tests for changed modules.
- Reproduction commands for the original failure.
- `uv run ruff check` on touched Python files.
- `uv run ruff format --check` on touched files.
- Type checks for typed surfaces when relevant.
- Integration tests for daemon, runtime, TUI, channels, tools, memory, and provider paths when touched.
- Terminal sessions for long-running commands: start with `exec_command`, poll with `write_stdin`, and close stale sessions.

If verification fails, treat that as the next round's highest-priority input.

## Report Format

The campaign report at `bug-bounty/report.md` should be concise and cumulative:

```markdown
# Bug Bounty Hunter Report

## Campaign State
- Iterations requested:
- Iterations completed:
- Scope:
- Fix mode:
- Current status:

## Summary
- Bugs confirmed:
- Bugs fixed:
- Bugs deferred:
- Tests added or changed:
- Highest-risk remaining areas:

## Latest Round
- Round:
- Theme:
- Agents spawned:
- Findings:
- Fixes:
- Verification:
- Next round recommendation:

## Fixed Findings
...

## Open Findings
...

## Verification Log
...

## Artifact Index
...
```

The final chat response should name the report path, summarize the latest round, list verification run, and state whether the campaign is complete, paused, or blocked. Do not paste the full report unless the user asks.

## Stop Conditions

Stop when one of these is true:

- The requested iteration count is complete.
- No new confirmed or plausible findings appear for several consecutive rounds and broad verification passes.
- A blocking dependency or user decision is required.
- Further work would require destructive changes, secrets, external credentials, or production access.

When stopping, leave a clean handoff: report path, open findings ledger, current worktree status, tests run, remaining risks, and the exact command or prompt to resume.
