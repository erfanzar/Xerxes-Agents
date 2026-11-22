# Xerxes Immortals

Xerxes Immortals coordinates multiple agents inside a single session over a durable, append-only local message board with reactive zero-polling wakeups, topic channels, isolated workspaces, automatic error/quota recovery, live activity inspection, and TUI / Desktop / VS Code observability.

## Overview

Unlike fire-and-forget subagent calls where a parent blocks on one child at a time, Xerxes Immortals connects a **coordinator** and named **workers** (`author`, `verifier`, `narrator`, `monitor`, or sharded worker pools) around a shared sequence-numbered board (`#1`, `#2`, ...):

- **Durable per-session state**: Board messages, channels, overview snapshots, and roster metadata persist on disk per `(workspaceDir, sessionId)` under `.xerxes/swarms/` for backwards compatibility. If the daemon restarts, previously living workers reload in the `stopped` state and revive automatically the next time a board message names them.
- **Coordinator live roster tracking**: `coordinator` appears directly in `status().members` and `read_board{status: true}` with live `state` (`working` during an active turn, `waiting` when idle or parked inside `swarm_wait` with `parked_since` and `mentions_only`), `last_read_seq`, `last_post_seq`, `last_post_time`, and `conversation_id`. `SwarmStatusSnapshot` also includes `tail_from_seq: 1`.
- **Zero-polling reactive wakeups**: Sleeping workers (`waiting`) consume zero model tokens. Posting `@<name>`, `@<group>`, `@#<channel>`, or `@all` wakes only the targeted members and runs their next turn against Xerxes's native turn loop and tool registry.
- **Automatic error & quota recovery**: When a worker fails (including HTTP 429, quota exhaustion, rate-limit, capacity, or overloaded provider errors), `SwarmManager` records `member.error`, transitions the worker to `waiting`, and immediately wakes `coordinator` (unblocking an active `swarm_wait` with `status: "worker_error"` or firing `onCoordinatorWake` with `reason: "worker_error"` and `worker_error: { member_name, message, quota_exhausted }`).
- **Live activity & conversation inspector**: Every worker stream event (`thinking`, `text`, `tool_start`, `tool_end`, `turn_done`) updates `member.current_activity` and emits `member_activity` lifecycle events (`SwarmMemberActivitySnapshot`). Clients can inspect any member's bounded conversation transcript and running/recent tools via `SwarmManager.inspectMember(name, { limit })` / `swarm.inspect`.
- **Hierarchy & helper ownership**: Workers can inspect the Immortals roster and spawn their own helpers (`added_by` tracks lineage). A worker can switch models or kill only the helpers it added; the `coordinator` governs all members and owns the Immortals `title`.

## Using `/immortals` and `xerxes immortals`

Run `/immortals` (or the backwards-compatible alias `/swarm`) from the interactive TUI, Desktop app, VS Code extension, or daemon slash router, or run `xerxes immortals` (alias `xerxes swarm`) from the CLI:

| Command | Behavior |
| --- | --- |
| `/immortals` or `/immortals board` | Open the live **Immortals Board** UI (or render the current roster + recent board messages in text-only surfaces). |
| `/immortals <task>` | Activate the `immortals-coordinator` skill for the current session and start coordinating `<task>`. |
| `/immortals status` | Render the Immortals title, active/waiting/stopped/killed counts, narrator overview, and member roster. |
| `/immortals channels` | List `#main` and custom channels with topics, archive status, and subscriber counts. |
| `/immortals kill <name\|all>` | Permanently stop a worker, a sharded group prefix, or `'all'` living workers (never `coordinator`) and clean up isolated workspaces. |
| `/immortals stop` | Alias for killing all living Immortals workers in the active session. |

Assistant replies during an active Immortals run should end with `[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)` on its own line. Desktop and VS Code intercept both `sidecar://immortals/board` and legacy `sidecar://swarm/swarm_board` links to open the Immortals Board panel directly; TUI displays a keyboard-accessible `/immortals` hint and live status badge.

## Immortals Tools

For backwards compatibility with existing prompts and persisted histories, the underlying tool identifiers remain `swarm`, `post_message`, `read_board`, `swarm_wait`, and `swarm_overview`.

### Tool Visibility (One-Turn-Late Unlocking)

To keep ordinary single-agent turns uncluttered, a coordinator session initially exposes only `swarm`. As soon as the first `swarm{action: "add", ...}` creates a worker and activates the run (`isActive()` is true once at least one worker exists, excluding `coordinator`), Xerxes unlocks `post_message`, `read_board`, and `swarm_wait` for subsequent coordinator turns.

Workers always receive `swarm`, `post_message`, `read_board`, and `swarm_wait`. Members named `narrator` or `narrator-<N>` additionally receive `swarm_overview`.

### 1. `swarm`

Manages the Immortals roster, models, title, and channels:

- `status`: Returns the current `SwarmStatusSnapshot` (`active`, `session_id`, `title`, `board: "local"`, `open_board`, `head`, `tail_from_seq`, `overview`, `channels`, `members` — including live `coordinator` state, `conversation_id`, and `current_activity` on working members).
- `models`: Lists selectable models (`all: true` includes every configured catalog entry) with display names and quota availability from `modelInventory` and `profileQuota`. Quota-exhausted models are refused on `add` / `model` unless covered by available credits.
- `add`: Spawns one worker (`name`), a homogeneous pool (`n > 1` -> `<name>-1`..`<name>-<n>`), or a sharded pool (`tasks: string[]` -> `<name>-1`..`<name>-<k>`). Supports `model`, `fork` (`"coordinator"` or an existing member name to clone conversation history), `workspace` (`"inherit" | "share" | "branch"`), and initial `channels`. Posts `@<name> <brief>` on the board, immediately dispatches the worker's startup turn, and returns each added member (with `conversation_id`), `open_board`, and a `note` reminding coordinator to place `open_board` on its own line in user reports.
- `model`: Switches a `waiting` or `stopped` member (or the caller itself) to another model while preserving its conversation history, accepting an optional `reason` (for example `"quota exhausted"`) that is relayed to the resumed worker. Switching another member while it is `working` is refused.
- `kill`: Permanently terminates a worker, a sharded group (`<base>`), or (from host RPCs) `"all"` living workers, aborts any in-flight turn, and deletes its `"share"` git worktree or `"branch"` clone. `coordinator` cannot be killed, and killed names cannot be reused.
- `title`: Sets the human-readable Immortals title (`coordinator` only).
- `channel`: Manages channels via `op: "create" | "subscribe" | "unsubscribe" | "list" | "archive"`.

### 2. `post_message`

Posts `text` to `channel` (defaults to `"main"`), tagging the message with the caller's roster name and monotonic `seq`. Mentions are parsed automatically:

- `@<name>` wakes that specific member (and revives it if `stopped`).
- `@<group>` wakes every living `<group>-1`..`<group>-<k>` member.
- `@#<channel>` wakes every subscriber of `<channel>` (`@#main` is rejected; use `@all`).
- `@all` broadcasts a wakeup to every living member.
- Prefix worker completion posts right after the `@mention` with `[done]` (for example, `@coordinator [done] Slice verified`). When closing the run, `coordinator` posts one unaddressed `[done] <summary>` with zero `@` mentions to quiet `monitor` and let `narrator` settle its final overview.

### 3. `read_board`

Reads board messages oldest-first or inspects roster status:

- `since`: Return messages with `seq > since`.
- `limit`: Maximum messages to return (default `20`).
- `seqs`: Fetch exact sequence numbers (`[4, 12]`), ignoring `since` and `limit`.
- `channel`: Filter to a specific channel (`"main"`, a custom channel, or `"*"` for all visible channels).
- `status: true`: Return the live roster snapshot — including `coordinator`'s live `working` / `waiting` state, `parked_since`, and `mentions_only` — instead of message text.

### 4. `swarm_wait`

Suspends until a matching post after `since` arrives (or a worker fails with an error):

- `since` *(required)*: Highest sequence number the caller has already read. If a matching message from another member already exists after `since`, `swarm_wait` returns immediately without sleeping.
- `mentions_only` *(default `true`)*: Wake only on `@<self>`, `@<self-group>`, `@all`, or `@#<subscribed-channel>`. Set `false` to wake on any message from another member in visible channels.
- **Worker behavior**: Parks the worker in `waiting`, ends the current turn cleanly at zero token cost, and triggers a fresh worker turn as soon as a matching post lands.
- **Coordinator behavior**: Marks `coordinator` as `waiting` (`parked_since`, `mentions_only`) while awaiting matching board activity or a `worker_error` wakeup inside the active turn (restoring `working` on wakeup; interruptible via `AbortSignal` or user steer). If the coordinator is idle between turns when `@coordinator`, `@all`, or a worker error occurs, the daemon automatically schedules a coordinator wake turn.

### 5. `swarm_overview` (`narrator` only)

Updates `{ markdown, summary? }` on the Immortals live overview card without posting to the message board.

## Workspaces

| Mode | Isolation | Cleanup on `kill` |
| --- | --- | --- |
| `inherit` | Shares the creator's working directory. Ideal for readers, verifiers, `narrator`, `monitor`, and non-conflicting edits. | Workspace is preserved. |
| `share` | Dedicated git worktree (`git worktree add --detach`) sharing repository objects. | Worktree is removed on `kill` — land changes first. |
| `branch` | Standalone local clone under `.xerxes/swarm-workspaces/<session>/<member>` (`git clone --local` with copy fallback). | Clone directory is deleted on `kill` — land changes first. |

## Bundled Immortals Skills

Xerxes ships three coordinated Apache-2.0 skills under `xerxes/skills/` (legacy names `swarm-coordinator`, `swarm-monitor`, and `swarm-narrator` continue to resolve transparently through `SkillRegistry.get()`/`has()` without duplicating autocomplete listings):

- **`immortals-coordinator`**: Coordinator playbook covering opening-wave setup (`title`, default `narrator` and `monitor` alongside scoped workers), non-overlapping file ownership, 1-per-2–4-authors verifier pairing, `@narrator user feedback:` / `(default)` relay discipline, automatic worker error & quota model-switching (`models` -> `model` with `reason`), workspace landing before `kill`, leaving `monitor` & `narrator` alive with one unaddressed closing `[done]` post, and ending user replies with `open_board` on its own line.
- **`immortals-monitor`**: Dedicated stall detector for the `monitor` worker. Inspects `read_board{status: true}` (including `coordinator`'s live `working` / `waiting` state) on a bounded `schedule` silent-stall timer + `swarm_wait` loop, stays silent while any worker or `coordinator` is working, posts `@<member> [stall] ...` citing `#<seq>` only when all active participants are asleep with unfinished work, and cancels timers once `[done]` appears.
- **`immortals-narrator`**: Dedicated live overview curator for `narrator`. Never calls `post_message`, watches all channels plus `@narrator user feedback:` / `(default)` notes and `read_board{status: true}` live activity, and keeps `swarm_overview` up to date.

## Daemon v35 RPC & Event Wire Contract

All Immortals RPCs keep the backwards-compatible `swarm.*` method names, accept `session_id` (or `session_key`), and default to the active session when omitted:

- **`swarm.status`** `{ session_id? }` -> `SwarmStatusSnapshot` (`active`, `session_id`, `title`, `board: 'local'`, `open_board`, `head`, `tail_from_seq`, `overview`, `channels`, `members`).
- **`swarm.board`** `{ session_id?, since?, limit?, seqs?, channel? }` -> `{ head: number, messages: SwarmBoardMessage[] }`.
- **`swarm.inspect`** `{ session_id?, name: string, limit?: number }` -> `SwarmMemberInspectResult` (`session_id`, `member`, `conversation_id`, `activity`, `messages`, `total_messages`).
- **`swarm.post`** `{ session_id?, text: string, channel?: string, as_feedback?: boolean }` -> `{ status: 'posted', seq: number, channel: string }`. When `as_feedback: true`, `@` prefixes before member names in `text` are stripped so user UI feedback wakes only `@narrator` as `@narrator user feedback: <sanitized>`.
- **`swarm.kill`** `{ session_id?, name: string }` -> `{ status: 'killed', killed: string[] }` (`name` may be a worker name, group prefix, or `'all'`).
- **`swarm.model`** `{ session_id?, name?: string, model?: string, reason?: string }` -> `{ status: 'switched', name: string, model: string }`.
- **`swarm.activate`** `{ session_id?, enabled?: boolean }` -> `{ ok: true, active: boolean, prompt_overlay: string }`.

### Live Stream Event (`swarm_event`)

Whenever an Immortals title, member, live worker activity, channel, overview, or board message changes, the daemon emits a `'swarm_event'` over `EventSubscriptions` and session observers (`kind` includes `'member_activity'` alongside `'activated'`, `'titled'`, `'member_added'`, `'member_updated'`, `'member_killed'`, `'message_posted'`, `'channel_updated'`, and `'overview_updated'`):

```json
{
  "session_id": "sess_123",
  "kind": "member_activity",
  "head": 14,
  "status": { "...": "SwarmStatusSnapshot" },
  "member": { "name": "author", "state": "working", "current_activity": "Reading server.ts" },
  "activity": { "phase": "tool", "summary": "Reading server.ts", "active_tools": [], "recent_tools": [] }
}
```
