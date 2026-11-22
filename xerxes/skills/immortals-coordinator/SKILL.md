---
name: immortals-coordinator
description: Coordinate a multi-agent Xerxes Immortals force over a shared message board with isolated workspaces, sharded worker pools, channels, live overview/stall agents, and reactive wakeups.
version: 1.0.0
author: Xerxes-Agents
tags: [immortals, multi-agent, coordination, parallel, message-board]
---

# Xerxes Immortals Coordinator Guide

## Metadata

- License: Apache-2.0
- Related skills: immortals-monitor, immortals-narrator, swe-patch-tournament
- Built-in: true

Activate this skill when coordinating multi-agent work on Xerxes's native Immortals message board. As the **coordinator**, your job is to decompose the problem, delegate technical work, research, builds, tests, and verification to workers, resolve cross-cutting blockers, and synthesize the verified outcome for the user.

## Coordinator Role and Sizing

- **Delegate execution**: Do not write application code, run heavy test suites, or chase deep codebase investigations yourself when workers can execute in parallel. Keep your context lean so you can steer architecture, arbitrate interfaces, and review results.
- **Right-size the roster**: Spawn only the workers that have real, independent work right now. A 2-file bug fix needs 1 author and 1 verifier, not 8 agents. Every message on the board is read by waking agents, so a lean roster finishes faster and spends fewer tokens.
- **Strict non-overlapping ownership**: Assign disjoint directories, files, or subsystems in every worker's `brief`. Two authors editing the same file is the primary failure mode to prevent up front. State both what each worker owns and what it must not touch.
- **Pair authors with verifiers**: Add **1 read-only verifier per 2–4 authors** (`workspace: "inherit"`). Have authors hand off directly to `@verifier` when their slice compiles and passes targeted checks. The verifier inspects ownership, conventions, and behavioral tests; sends actionable fixes back to `@<author>`; and notifies `@coordinator` with `[done]` once slices pass.

## Standard Opening Wave (`title`, `narrator`, `monitor`, and Workers)

Before an Immortals run has workers, the coordinator session exposes only the `swarm` management tool. As soon as your first `swarm{action: "add", ...}` call succeeds and activates the force, Xerxes unlocks `post_message`, `read_board`, and `swarm_wait` on your very next turn.

In your opening turn, launch the complete opening wave together:

1. **Name the run**: Set a short descriptive title with `swarm{action: "title", title: "..."}` for the Desktop, VS Code, and TUI Immortals Board (`coordinator` only).
2. **Spawn `narrator` by default** *(when the `immortals-narrator` skill is available)*: Call `swarm{action: "add", name: "narrator", workspace: "inherit", brief: "..."}` instructing it to read and follow the `immortals-narrator` skill. It observes all channels (`*`) and user feedback, curates the live summary card via `swarm_overview`, and never posts messages to the board.
3. **Spawn `monitor` by default** *(when the `immortals-monitor` skill is available)*: Call `swarm{action: "add", name: "monitor", workspace: "inherit", brief: "..."}` instructing it to read and follow the `immortals-monitor` skill. It watches live roster state via `read_board{status: true}` (including `coordinator`'s live `working` / `waiting` state and `parked_since` cursor) on a bounded silent-stall timer and nudges the single responsible member with `[stall]` only if the whole Immortals force stalls with unfinished work.
4. **Spawn your initial workers and verifier(s)**: Give each worker strict file ownership, clear acceptance criteria, and direct peer-to-peer handoff instructions.

Once board tools unlock on your next turn, call `swarm_wait{since: <last_read_seq>}` rather than polling `status` in a loop.

## Managing Immortals (`swarm` Tool)

### 1. Spawning Workers and Sharded Pools (`action: "add"`)

Each new worker starts fresh unless `fork` is set. Its `brief` is posted publicly on the board as `@<name> <brief>`, which is everything a fresh worker knows about the assignment.

- **Single specialist**: Set `name` (`^[a-z0-9][a-z0-9_-]{0,23}$`; never `all`, `coordinator`, or `coordinator-*`) and `brief`.
- **Homogeneous pool (`n`)**: Pass `n: 3` to spawn `<name>-1`, `<name>-2`, `<name>-3` sharing the same role and `brief`.
- **Sharded tasks (`tasks`)**: Pass `tasks: ["src/auth/", "src/session/", "src/api/"]` to spawn `<name>-1`..`<name>-<k>`, combining each shard target with the shared `brief`. Never pass both `n > 1` and `tasks`.
- **Group addressing**: Mentioning `@<base>` wakes all living `<base>-1`..`<base>-<k>` members, and `kill{name: "<base>"}` terminates all living members in that group.
- **Warm branching (`fork`)**: Pass `fork: "coordinator"` or `fork: "<existing-member>"` when a new worker needs a cloned snapshot of that member's conversation history instead of starting cold.
- **Worker-spawned helpers**: Workers can also call `swarm` (`status`, `models`, `add`, `model`, `kill`, `channel`) to spawn scoped helpers for sub-tasks, which report back to `@<parent-worker>` and inherit ownership bounds from their creator.

### 2. Workspaces and Landing Work (`workspace`)

Choose `workspace` per worker based on write concurrency:

- `"inherit"` *(default)*: Runs inside the adder's workspace directory. Best for read-only explorers, verifiers, `narrator`, `monitor`, or authors working on strictly non-overlapping files that do not clash during builds.
- `"share"`: Allocates an isolated git worktree (`git worktree add --detach`) so concurrent authors can edit, compile, and run tests simultaneously without dirtying each other's working tree.
- `"branch"`: Creates a standalone local git clone under `.xerxes/swarm-workspaces/` for isolated branch experiments or destructive refactors.

**Critical rule before `kill`**: `swarm{action: "kill", name}` is **permanent** — it aborts any in-flight turn, marks the member `killed` (killed names can never be reused; use `worker-2`, `narrator-2`, etc. if replacing), and **deletes its `"share"` worktree or `"branch"` clone**. Always merge, cherry-pick, or apply a worker's verified changes into your main working tree before killing it, and push to a remote repository only when the user explicitly requests it. Do not kill idle workers just to pause them; a `waiting` worker consumes zero tokens.

#### Ambiguous or High-Difficulty SWE Bugfixes (`swe-patch-tournament`)

When diagnosing a hard or ambiguous software engineering bug with competing root-cause hypotheses (for example async cancellation races, subtle state/cache invalidation bugs, or cross-module boundary regressions), activate and follow the `swe-patch-tournament` skill:

- Spawn 2–3 candidate solvers (`solver-1`, `solver-2`, ...) with `workspace: "share"` and **distinct root-cause hypotheses**, plus 1 independent read-only `verifier` (`workspace: "inherit"`).
- Have `verifier` evaluate both the failing reproduction test and broader regression suites across each candidate's `workspace_path`, disqualifying symptom patches or conflict markers.
- Select the minimal clean passing candidate (`changedFiles.length`, then `insertions + deletions`), apply its unified diff cleanly to your parent workspace (`git apply --check` + `git apply`) **before** calling `swarm{action: "kill"}`, verify once in the parent workspace, and then retire the candidate worktrees.

### 3. Channels (`action: "channel"`)

Keep high-volume sub-team traffic out of `#main` when running larger Immortals forces:

- `op: "create"` — create a named channel with a `topic`.
- `op: "subscribe"` / `"unsubscribe"` — manage channel membership (coordinator can update any member; workers can update themselves and their own helpers).
- `op: "list"` — inspect channels, topics, archive state, and subscriber counts.
- `op: "archive"` — freeze a completed channel so new posts are rejected while history stays readable.

`#main` always exists and every member reads it. `coordinator`, `narrator*`, and `monitor*` automatically observe all channels (`*`). Use `@#<channel>` inside a channel post to wake all subscribers of that custom channel (`@#main` is rejected; use `@all` for whole-force broadcasts).

### 4. Automatic Worker Error & Quota Recovery (`models` & `model`)

- `swarm{action: "models"}` (pass `all: true` for every configured model) lists available models and live quota health. Leave `model` unset on `add` to use your current model unless the user requests another or quota requires routing elsewhere.
- Whenever a worker fails — including HTTP 429, rate-limit, quota exhaustion, capacity, or overloaded provider errors — Xerxes automatically records `member.error`, transitions the worker to `waiting`, and wakes `coordinator` immediately (returning `status: "worker_error"` from an active `swarm_wait`, or dispatching an automatic coordinator wake turn when idle).
- On a quota/capacity wakeup or user-requested model switch, **never `kill` and `add` a replacement**. Instead:
  1. Call `swarm{action: "models"}` (or `{action: "models", all: true}`) to choose a healthy model with available quota or fallback credits.
  2. Call `swarm{action: "model", name: "<worker>", model: "<model>", reason: "quota exhausted"}` to switch the worker **in place**. Xerxes preserves its full conversation history, explains the reason to the worker, and resumes its unfinished work automatically.

## Relaying User Feedback to `narrator`

When a `narrator` is on the roster and the user provides feedback, preferences, constraints, or direction — or when you apply a sensible default on the user's behalf:

- Relay verbatim user feedback with `post_message`:
  ```text
  @narrator user feedback: <verbatim user feedback>
  ```
- Relay coordinator-chosen defaults with `post_message`:
  ```text
  @narrator user feedback (default): <default choice and brief context>
  ```
- **Strip `@` before any member name, group prefix, `all`, or `#channel`** inside the relayed text (for example, write `author`, `verifier`, `all`, `#backend` instead of `@author`, `@verifier`, `@all`, `@#backend`) so the post wakes **only** `@narrator` and never triggers spurious worker turns.

## Board Discipline

- **`@name` is a wakeup doorbell, never a CC**: Every `@mention` wakes that sleeping agent and spends a model turn. On peer-to-peer handoffs (for example `@author` -> `@verifier` -> `@author`), mention only the peer who acts next. Tell workers to tag `@coordinator` only when final verified results are ready or when blocked on a cross-boundary decision.
- **Reserve `@all` for force-wide changes**: Use `@all` only for shared interface contracts, global rules, or plan changes — never for status updates or greetings.
- **Cite `#<seq>` instead of re-quoting**: Refer to earlier board messages by sequence number (`#7`, `#19`) and pull specific context with `read_board{seqs: [7, 19]}` when needed.
- **`[done]` completion prefix**: Require every worker to begin its final deliverable post with `[done]` immediately after the `@mention` (for example, `@coordinator [done] Verified auth refresh and 14 daemon tests pass`).
- **Pass exact `since` to `swarm_wait`**: Always pass the highest sequence number you have actually read (`last_read_seq`) as `since` — never guess or pass an unread head, or you can sleep past a message posted while you were working.

## Clean Finish and Reporting to the User

When all assigned workstreams are verified and complete:

1. **Land isolated work first**: Merge, cherry-pick, or cleanly apply any `share` worktree or `branch` clone changes into your working tree before calling `kill` (push to a remote only if the user explicitly asked).
2. **Retire finished workers, keep `monitor` and `narrator`**: Call `swarm{action: "kill", name: "<worker>"}` for completed task workers whose work is landed. **Leave `monitor` and `narrator` alive** unless the user explicitly asks to tear down the entire Immortals run so the overview stays live and follow-up waves can resume seamlessly.
3. **Post ONE closing `[done]` message with NO `@` mentions**: Publish a single concise summary on `#main` starting with `[done] ` and containing **zero `@` mentions** (for example, `[done] All 3 slices verified and landed; full check + test suite green.`). This cleanly quiets `monitor` (which cancels its silent-stall schedules and parks) and lets `narrator` publish its final settled `swarm_overview` without waking any worker.
4. **Include `open_board` on its own line**: At the end of launch confirmations, interim checkpoints, and final synthesized replies to the user, place the `open_board` link on its own line so Desktop, VS Code, and TUI can surface the Immortals Board cleanly:

[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)
