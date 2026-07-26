# Intelligent Workflows ("Claude-workflow"-style) — Design Proposal

## Status

Proposed 2026-07-26. Not started. This document is a design sketch only; no
implementation work has been done and none is authorized by this document.

## Problem

Today a Xerxes turn is reactive: the main agent decomposes work, spawns
sub-agents, and waits. Nothing tracks a multi-step effort across turns as a
first-class object. As a result:

- Long efforts lose state between turns; the model re-derives the plan from
  transcript scrollback and often repeats or contradicts earlier steps.
- Sub-agents re-do work that is already done (stale file reads, duplicate
  edits, re-run tests) because nothing records what the codebase looked like
  when each step completed.
- There is no durable "where are we" artifact a user can inspect, resume, or
  checkpoint after a crash, compaction, or session resume.

## Goals

1. A durable, inspectable **workflow object**: ordered steps with status,
   ownership (main agent vs. named sub-agent), inputs, and produced artifacts
   (files touched, commands run, test results).
2. **Codebase-state awareness**: each step records the workspace fingerprint
   it ran against (git head, dirty-file set, relevant repo-map digest), so a
   later step can detect "the world moved" instead of blindly continuing.
3. **Duplicate/stale-work avoidance**: before executing a step, the runtime
   checks whether an equivalent step already completed against an unchanged
   fingerprint and offers a skip/reuse decision instead of re-running.
4. **Resumability**: a workflow survives compaction and session resume; the
   main agent re-enters at the first incomplete step with evidence, not from
   scratch.
5. **User visibility**: the TUI can render workflow progress from the same
   event vocabulary the daemon already emits (no parallel UI state).

## Non-goals

- Not a general DAG/pipeline engine or a YAML workflow DSL competing with CI.
- Not autonomous multi-turn operation without a user turn; workflows advance
  inside the existing turn/permission model.
- No new provider calls, background daemons, or hidden network activity.
- No migration of the v35 daemon wire protocol in phase 1; workflow state is
  exposed through existing session/transcript persistence first.
- Not a replacement for skills or interaction modes; those remain prompt- and
  policy-level concerns.

## Existing pieces to build on

- `xerxes/src/cortex/` (`orchestrator.ts`, `planner.ts`, `task.ts`,
  `dynamic.ts`): already models multi-agent topology, task decomposition, and
  per-agent assignment. A workflow is a *persisted* cortex plan with
  fingerprints and step results attached.
- `xerxes/src/runtime/executionRegistry.ts`: typed `RegistryEntry` /
  `ExecutionStatus` / `ExecutionResult` records with duration and error
  fields — the natural per-step outcome record; reuse its status vocabulary
  rather than inventing a parallel one.
- `xerxes/src/agents/subagentManager.ts` and
  `xerxes/src/daemon/subagentCoordinator.ts`: own spawned-agent lifecycle,
  titles, and result collection. Workflow steps that delegate simply reference
  the coordinator's agent ids, so cohort tracking stays in one place.
- `xerxes/src/session/` + `xerxes/src/runtime/transcript.ts`: durable session
  records, replay, and export. Workflow state rides session persistence so
  resume/replay/export pick it up without a new storage backend.
- `xerxes/src/context/repoMap.ts` and git probing in
  `runtime/promptContext.ts` (`gitContext`): existing sources for the
  workspace fingerprint (branch, dirty count, ranked file map).
- `xerxes/src/runtime/changeGuard.ts` and `workflowMemory.ts`: change-guard
  hooks and the existing explicit-workflow memory file (`WORKFLOW.md`) give
  policy enforcement and a user-facing narrative artifact to align with.
- `xerxes/src/runtime/interactionModes.ts` (`objective` mode): already has an
  acceptance-criteria loop; workflows generalize that loop's bookkeeping
  instead of duplicating it.

## Design sketch

Core types (new module, e.g. `xerxes/src/runtime/workflow.ts`):

- `WorkflowStep { id, title, status, owner: 'main' | agentId, fingerprint,
  artifacts: string[], result?: ExecutionResult, summary }` where `status`
  reuses `ExecutionStatus` values.
- `Workflow { id, goal, createdAt, steps: WorkflowStep[], cursor }` — `cursor`
  is the first non-complete step; advancing it is the only mutation that
  matters for resume.
- `WorkspaceFingerprint { gitHead, dirtyFilesHash, repoMapDigest }` — captured
  through existing `gitContext`/repo-map ports; all fields optional so
  non-git workspaces still work.

Behavior:

1. **Plan**: when the main agent (or cortex planner) decomposes a multi-step
   task, it may persist a `Workflow` alongside the session. Planning stays
   model-driven; the runtime only stores and validates the shape.
2. **Before each step**: compute the current fingerprint. If the step's
   recorded fingerprint matches and its status is complete, report
   "already done against unchanged state" and skip with evidence. If the
   fingerprint changed, mark the step `stale` and require an explicit
   re-verify rather than silently trusting the old result.
3. **Delegation**: steps owned by sub-agents reference coordinator agent ids;
   completion events from the coordinator update step status, keeping the
   await-all semantics added to the orchestration prompt intact.
4. **Resume**: on session load, the runtime injects a compact workflow summary
   (goal, per-step status, stale markers) into the prompt context as another
   profile-gated section, capped like memories so it cannot grow unbounded.
5. **Cancellation**: aborting a turn marks running steps `cancelled` via the
   existing `AbortSignal` path; nothing half-writes.

## Phased implementation

- **Phase 1 — state + persistence (read-only intelligence).**
  `runtime/workflow.ts` types, validation, and serialization into session
  records; fingerprint capture via existing ports; unit tests for
  skip/stale/resume decisions. No prompt or UI changes yet.
- **Phase 2 — prompt integration.**
  A profile-gated `[Workflow]` section in `PromptContextBuilder` (FULL only,
  capped), plus tool-level helpers so the main agent can create/advance/close
  a workflow explicitly. Contract tests that caps hold and NONE profile omits
  it.
- **Phase 3 — coordinator wiring.**
  Sub-agent completion events update owned steps; stale-step re-verification
  hooks into `changeGuard`. Daemon emits workflow status through the existing
  event vocabulary; contract tests on replay.
- **Phase 4 — TUI + export.**
  Render workflow progress in the agents rail/overlay from daemon events;
  include workflows in `xerxes export`. Narrow-terminal fallback preserved.
- **Phase 5 — heuristics (opt-in).**
  Suggest skip/reuse automatically when fingerprints match; suggest splitting
  oversized steps. Off by default, behind a feature flag in
  `runtime/features.ts`.

## Risks and open questions

- Fingerprint granularity: too coarse (git head only) misses dirty-tree edits;
  too fine (full tree hash) is expensive. Start with git head + dirty-file
  hash and measure.
- Prompt budget: workflow summaries must obey profile caps; a large workflow
  needs a ranked/trimmed rendering, not raw JSON injection.
- Model compliance: phases 1-3 work even if the model ignores workflows; the
  value of phase 5 depends on prompt guidance actually being followed — gate
  it behind real-use evaluation before enabling by default.
- Concurrency: two agents advancing the same workflow need a single-writer
  rule (main agent owns cursor mutations; sub-agents only report results).
