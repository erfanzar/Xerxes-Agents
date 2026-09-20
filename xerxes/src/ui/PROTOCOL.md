<!--
Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
Licensed under the Apache License, Version 2.0.
-->

# Xerxes TUI ⇄ Daemon Wire Protocol

`/plugin-creator` is a client session shortcut, like `/creator`: it uses the existing new-session flow with `agent_preset: "plugin-creator"`. It adds no RPC method or wire format. Busy-session switching remains guarded and prior sessions remain saved.

This is the frozen contract between the **TypeScript/OpenTUI frontend** (`xerxes/src/ui/`)
and the **Bun TypeScript daemon** (`xerxes/src/cli.ts daemon`). The
frontend owns the screen; the daemon owns sessions, tools, model calls, and
command logic. The TS client is a JSON-RPC peer over a Unix domain socket — it
does not import or launch a second runtime.

The Bun implementation sources for this document are:

- Transport + framing: `xerxes/src/daemon/server.ts`
- RPC dispatch + params: `DaemonServer.dispatch()` in that file
- Runtime/session lifecycle: `xerxes/src/daemon/runtime.ts`
- UI connection bootstrap: `xerxes/src/ui/gatewayClient.ts`

> **Stability:** the event-name map (snake_case ⇄ PascalCase) is asserted by a
> Bun contract test (`xerxes/test/daemonServer.test.ts`) and the TUI
> gateway tests in `xerxes/src/ui/__tests__/`. If you change an event name or
> add an event, update those tests **and** `gatewayTypes.ts`.

---

## 1. Transport

Newline-delimited JSON (NDJSON), one JSON-RPC 2.0 object per line, UTF-8, over a
**Unix domain socket**. There is no length prefix; `\n` terminates every frame.

### Connection bootstrap

Optional reconnect ownership: `initialize` may advertise
`connection_lease_supported: true`. Before submitting work, call
`connection.lease {}` to obtain an opaque `token` and `grace_ms` (30000).
Keep this credential in client memory only. After a transport drops, call
`connection.lease {token, project_dir}` on its replacement **before** resuming
the saved session with `initialize`. Only a detached connection in the same
workspace can be reclaimed; a connected owner cannot be taken over.
Initialization returns `pending_interactions` (type/payload pairs) for unanswered
permission/question requests owned by the connection. After a reclaimed lease,
`reconnect_events` contains ordered missed event frames: retain the current
session's rendered transcript, apply these frames once, then restore pending
interactions. The journal is bounded to 1 MiB; overflow expires the lease and
cancels the turn rather than silently omitting output. Reclaiming without
completing initialization does not extend the deadline.
RPC replies stay bound to the socket that issued them.

After expiry, `{ok:false, code:"lease_expired"}` permits normal saved-session
reopening and a new lease; cancelled work does not restart automatically.
Expiry and daemon shutdown retain normal cancellation and interaction cleanup.
Clients that do not opt in retain immediate disconnect cancellation. Older
daemons omit the capability and must not receive the optional method. This is
an additive v35 capability; existing requests and events are unchanged.

1. Resolve the project dir to the nearest Git root when available; otherwise
   use `realpath(cwd)` (falling back to the absolute path).
2. Compute the per-user global socket path through the native `daemonPaths()`
   helpers (default layout):

   ```md
   $XERXES_HOME/daemon/global-<sha256(resolved_xerxes_home)[:16]>.sock
   ```

   where `$XERXES_HOME` defaults to `~/.xerxes`. (`.pid` sits next to `.sock`.)
   `XERXES_DAEMON_SOCKET` can explicitly override the socket path; the gateway
   and daemon use the same deterministic home-path calculation.
3. Try to `net.connect({ path })`. A compatible daemon already listening on the
   socket is attached without replacing it.
4. If not reachable, spawn the daemon and poll `runtime.status` until the
   configured startup timeout expires (15 seconds by default):

   ```sh
   bun xerxes/src/cli.ts daemon \
     --project-dir <dir> \
     --socket <sock-path> \
     --pid-file <pid-path>
   ```

   The gateway resolves the command in this order:

   | Setting | Purpose |
   | --- | --- |
   | `XERXES_TUI_BUN` or `XERXES_BUN` | Bun executable (default: `bun`) |
   | `XERXES_TUI_BUN_DAEMON` or `XERXES_BUN_DAEMON` | Explicit TypeScript daemon CLI path; relative paths are resolved from the project root |
   | no daemon-path setting | A colocated `xerxes/src/cli.ts` or built `xerxes/dist/cli.js` |

   `GatewayClient` also accepts `bunBinary` and `bunDaemonPath` for embedders
   that should not depend on process environment.

   Spawn detached, ignore standard input/output, and capture `stderr` into an
   in-memory ring (≤200 lines) for diagnostics — never write child stderr to
   the terminal.
5. Each TS connection picks a connection-local session key: `tui:<uuid12>`.
   The daemon binds sessions per-connection, so concurrent clients don't cross
   the streams.

### Frame shapes

**Request** (client → daemon):

```json
{ "jsonrpc": "2.0", "id": 7, "method": "prompt", "params": { "user_input": "hi" } }
```

**Response** (daemon → client, echoes `id`):

```json
{ "jsonrpc": "2.0", "id": 7, "result": { "ok": true } }
```

**Error response**:

```json
{ "jsonrpc": "2.0", "id": 7, "error": { "code": -32000, "message": "..." } }
```

Codes: `-32700` invalid JSON, `-32000` handler raised. Unknown method →
`{ "result": { "ok": false, "error": "Unknown method: X" } }` (note: a *result*,
not a JSON-RPC error). Most handlers return `{ "ok": bool, ... }`; treat
`ok === false` as a soft failure to surface, not a transport error.

**Event** (daemon → client, no `id`, may be broadcast to all clients):

```json
{ "jsonrpc": "2.0", "method": "event",
  "params": { "type": "text_part", "payload": { "session_id": "…", "text": "hello" } } }
```

The client must demux by presence of `id`: a frame with `id` is a
response/error; a frame with `method === "event"` is a streaming event.

> **Casing — read this.** The **daemon socket emits `type` in snake_case**
> (`text_part`, `turn_begin`, `tool_call`, `question_request`, …).
> `DaemonServer` forwards those native names verbatim. The optional native
> bridge can expose Kimi-style PascalCase aliases for external clients; the TUI
> keys on snake_case and tolerates the alias through `normalizeEventType()`.

---

## 2. Requests (methods the TS client calls)

`params` shapes are handled by `DaemonServer.dispatch()`. Omitted optional
params fall back to per-connection defaults.

| Method                               | Params                                         | Result                                                         | Notes                                                                             |
| ------------------------------------ | ---------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `initialize`                         | `{}` (impl-specific)                           | `{ ok, ... }` + emits `InitDone`                               | Handshake; sets up the connection session.                                        |
| `prompt`                             | `{ user_input, mode?, plan_mode? }`            | `{ ok }`                                                       | Submit a user turn. Streams events back.                                          |
| `turn.submit`                        | `{ session_key?, text, mode?, plan_mode?, images? }` | `{ ok }`                                              | Lower-level submit; `prompt` wraps this. Optional `images`: `[{ media_type, data(base64) }]` — validated (strict base64, png/jpeg/gif/webp magic-byte sniff, 10MB/image, 20MB/turn) and sent to the provider as `image_url` data-URL content parts. |
| `turn.background`                    | `{ session_key?, text }`                       | `{ ok, task_id, session_key }`                               | Dispatch a new independent live session without changing the connection's attached session. Agent View uses this for its bottom prompt; `/background` itself is a local detach action. |
| `turn.steer` / `steer`               | `{ session_key?, content }`                    | `{ ok }`                                                       | Inject steer text into the active turn.                                           |
| `turn.cancel` / `cancel`             | `{ session_key? }`                             | `{ ok }`                                                       | Cancel this connection's turn.                                                    |
| `cancel_all`                         | `{}`                                           | `{ ok, cancelled }`                                            | Cancel every session.                                                             |
| `session.open`                       | `{ session_key?, agent_id?, project_dir? }`    | `{ ok, session }`                                              | Open/attach a session within the active or explicit project boundary. `agent_id` is the DSH-style agent preset; changing it is rejected after the first transcript message. Mid-turn, `session.inflight` additively carries `started_at` (epoch seconds), `thinking`, and `tools: [{ id?, name, arguments?, ok?, duration_ms?, error? }]` — the turn's work so far, since the transcript only covers completed turns. The session payload also carries cumulative `llm_duration_ms`, `llm_steps`, `tool_duration_ms`, `tool_steps`, `ttft_total_ms`, `ttft_samples`, and `ttft_avg_ms` when observed. Attaching also drains that session's pending background-completion notices as `notification` events (at-most-once; they accumulate while no client is attached). |
| `agentPreset.list`                   | `{}`                                           | `{ ok, presets, default_id, authorable, has_document }`         | Uncached roster of built-in, user, and project agent compositions; broken presets remain visible with a reason. |
| `agentPreset.select`                 | `{ agent_preset, session_key? }`                | `{ ok, agent_preset }`                                         | Recompose a blank session only. A started session returns `agent-preset-locked`. |
| `agentPreset.read`                   | `{ agent_preset }`                              | `{ ok, preset, content, guarded_write }`                       | Read the exact `agent.yaml` composition; advertises optimistic editing support. |
| `agentPreset.copy`                   | `{ from, agent_preset, name? }`                 | `{ ok, preset, path }`                                         | Duplicate a known-good preset into the user root; ids match `[a-z0-9][a-z0-9-]*`. |
| `agentPreset.write`                  | `{ agent_preset, content, expected_content? }`  | `{ ok, preset }`                                               | Replace one user composition atomically after strict version-1 validation. When provided, expected content must match the current file. |
| `agentPreset.setDefault`             | `{ agent_preset }`                              | `{ ok, preset, default_id }`                                   | Changes only the default for sessions created later. |
| `agentPreset.openDocument` / `agentPreset.remove` | `{ agent_preset }`                    | `{ ok, path? }`                                                | User presets only; running sessions are unaffected. |
| `session.active_list`                | `{}`                                           | `{ ok, sessions }`                                             | List live top-level and subagent sessions. Agent View filters subagents into their parent and polls this for live state. |
| `session.list`                       | `{}`                                           | `{ ok, sessions }`                                             | For `/resume` picker.                                                             |
| `goal.inspect`                       | `{}`                                           | `{ ok, session_id, goal, token_usage, continuation }`            | Reads the connection's active goal, evidence and durable continuation receipt. |
| `goal.decision`                      | `{ session_id, goal_id, revision, criterion_id, summary }` | `{ ok, session_id?, goal?, token_usage?, continuation?, error? }` | Records a user acceptance note against the displayed criterion; requires an idle session and matching revision. |
| `session.status`                     | `{ session_key? }`                             | `{ ok, session: { ..., profile_name } \| null }`                | `profile_name` is the exact matching stored profile or `null` for an overridden/unmatched runtime. |
| `runtime.status`                     | `{}`                                           | `{ ok, runtime_ready, pid, daemon_protocol, daemon_build_id, active_subagents?, channels, ... }` | Liveness probe; `runtime_ready` reports configured-provider readiness and `active_subagents` protects live child work during upgrades. |
| `runtime.reload`                     | `{ ... }`                                      | `{ ok, ... }`                                                  | Reload runtime config; re-emits status.                                           |
| `browser.manage`                     | `{ action?, cdp_url? }`                        | `{ ok, status?, pages?, error? }`                              | `connect` attaches to an explicitly supplied, already-running Chromium CDP endpoint; `disconnect` only detaches. |
| `slash`                              | `{ command }`                                  | `{ ok }`                                                       | Native daemon slash dispatch; unsupported commands report an explicit result.     |
| `set_plan_mode`                      | `{ enabled \| plan_mode, mode? }`              | `{ ok }`                                                       | Re-emits `StatusUpdate`.                                                          |
| `set_mode`                           | `{ mode }`                                     | `{ ok }`                                                       | Re-emits `StatusUpdate`.                                                          |
| `permission_response`                | `{ request_id, response }`                     | `{ ok }`                                                       | `response ∈ approve \| approve_for_session \| reject`. Answers `ApprovalRequest`. |
| `question_response`                  | `{ request_id, answers }`                      | `{ ok }`                                                       | `answers: {questionId: value}`. Answers `QuestionRequest`.                        |
| `channel.list`                       | `{}`                                           | `{ ok, channels }`                                             | Multi-platform channels.                                                          |
| `channel.enable` / `channel.disable` | `{ name }`                                     | `{ ok }`                                                       |                                                                                   |
| `fetch_models` / `provider_models`   | `{ profile_name }`                             | `{ ok, models, catalog?, source, warning? }`                    | Catalog rows may carry effective `context_limit`, `max_output_tokens`, provenance sources, and `overridden`; credentials never leave the daemon. |
| `provider_model_override`            | `{ profile_name, model, context_limit?, max_output_tokens? }` | `{ ok, model }`                          | Positive safe integers set per-model user overrides; `null` clears a field. Precedence: user override → provider metadata → generated Pi catalog → unknown. Explicit runtime/profile `max_tokens` still wins for requests. |
| `provider_list`                      | `{}`                                           | `{ ok, profiles }`                                             |                                                                                   |
| `provider_save`                      | `{ name, base_url, api_key, model, provider }` | `{ ok, profile }` + emits `InitDone`                           |                                                                                   |
| `provider_select`                    | `{ name }`                                     | `{ ok }` + emits `InitDone`                                    |                                                                                   |
| `provider_delete`                    | `{ name }`                                     | `{ ok }` + emits `InitDone`                                    |                                                                                   |
| `shutdown`                           | `{}`                                           | `{ ok }`                                                       | Stops the daemon.                                                                 |

**Migrated/removed** (return `{ ok: false, error: MIGRATED_ERROR }`): `task.submit`,
`task.cancel`, `task.list`, `task.status`, bare `submit` / `list` / `status`.

> Slash routing: the TS client handles presentation-only commands locally and
> sends daemon-owned commands to the Bun daemon. `/creator` visibly starts a
> fresh session initialized with the built-in `creator` preset; `/preset` lists
> and manages the live roster. The v35 daemon provides
> completion, session controls, approval/question replies, profile CRUD/model
> discovery, browser CDP attachment, plugins, skills, runtime controls, and
> cron/session workflows. Unknown or unavailable operations return an explicit
> native error; neither layer fabricates success.

The TUI compatibility method `session.peek` is intentionally not a daemon RPC.
`GatewayClient` maps it to a targeted, read-only `session.status` request so Agent
View can preview and reply to a live chat without calling `session.open` or changing
the connection's attached session.

---

## 3. Events (daemon → client)

Every event arrives as `params: { type, payload }`. The `type` is the
discriminator. Over the **daemon socket** transport `type` is **snake_case**
(see the casing note in §1). The optional native bridge can emit PascalCase
aliases for compatible external clients, so the table below lists both. **The
TS client keys on the snake_case column** and maps the PascalCase column to the
same handler as a tolerated alias.

Every event emitted by a submitted turn also carries the owning `session_id`
in its payload. A single TUI connection can keep several sessions live, so
clients must route late text, reasoning, tool, interaction, status, and
terminal turn events by that identity instead of the currently selected tab.

### Event name map

| PascalCase (bridge alias) | snake_case (daemon wire) | Payload fields                                                                          |
| ------------------ | --------------------- | --------------------------------------------------------------------------------------- |
| `InitDone`         | `init_done`           | `model, session_id, cwd, git_branch, context_limit, agent_name, skills[]`               |
| `TurnBegin`        | `turn_begin`          | `user_input: string \| Part[]`                                                          |
| `TurnEnd`          | `turn_end`            | optional `stop_reason`, `cancelled`, `unstarted`                                                                                       |
| `StepBegin`        | `step_begin`          | `n`                                                                                     |
| `StepEnd`          | `step_end`            | `n`                                                                                     |
| `StepInterrupted`  | `step_interrupted`    | —                                                                                       |
| `SteerInput`       | `steer_input`         | `content`                                                                               |
| `CompactionBegin`  | `compaction_begin`    | —                                                                                       |
| `CompactionEnd`    | `compaction_end`      | —                                                                                       |
| `HookTriggered`    | `hook_triggered`      | `hook_name, trigger_type`                                                               |
| `HookResolved`     | `hook_resolved`       | `hook_name`                                                                             |
| `MCPLoadingBegin`  | `mcp_loading_begin`   | `server_name`                                                                           |
| `MCPLoadingEnd`    | `mcp_loading_end`     | `server_name, success`                                                                  |
| `BtwBegin`         | `btw_begin`           | —                                                                                       |
| `BtwEnd`           | `btw_end`             | —                                                                                       |
| `TextPart`         | `text_part`           | `text` (assistant text delta)                                                           |
| `ThinkPart`        | `think_part`          | `think` (reasoning delta)                                                               |
| `ImageURLPart`     | `image_url_part`      | `url, alt?`                                                                             |
| `AudioURLPart`     | `audio_url_part`      | `url`                                                                                   |
| `VideoURLPart`     | `video_url_part`      | `url, alt?`                                                                             |
| `ToolCall`         | `tool_call`           | `id, name, arguments?: string`                                                          |
| `ToolCallPart`     | `tool_call_part`      | `arguments_part` (streamed args delta)                                                  |
| `ToolResult`       | `tool_result`         | `tool_call_id, return_value, duration_ms, display_blocks[]`                             |
| `ToolCallRequest`  | `tool_call_request`   | `id, tool_call_id, name, arguments: object`                                             |
| `ApprovalRequest`  | `approval_request`    | `id, tool_call_id, action, description`                                                 |
| `ApprovalResponse` | `approval_response`   | `request_id, response, feedback?`                                                       |
| `QuestionRequest`  | `question_request`    | `id, tool_call_id, questions: QuestionItem[]`                                           |
| `QuestionResponse` | `question_response`   | `id, answers: {string: string}`                                                         |
| `StatusUpdate`     | `status_update`       | `context_tokens, max_context, mcp_status, plan_mode, mode, reasoning_effort, llm_duration_ms?, ttft_ms?, tokens_per_second?, cache_hit_rate?` |
| `AgentPresetSelected` | `agent_preset_selected` | `session_id, agent_preset`                                                           |
| `Notification`     | `notification`        | `id, category, type, severity, title, body, payload`                                    |
| `PlanDisplay`      | `plan_display`        | `content, file_path?`                                                                   |
| `SubagentEvent`    | `subagent_event`      | `parent_tool_call_id?, agent_id?, subagent_type?, event: WireEvent` (nested, recursive) |

Context capacity fields are provider-reported metadata, never a model-name or
provider fallback. A missing field or numeric `0` means **unknown**; clients must
not render a percentage, infer a denominator, or trigger threshold compaction.

### Nested types

```ts
// TurnBegin.user_input items, ToolResult.display_blocks, etc.
type Part =
  | { type: 'text'; text: string }
  | { type: 'think'; think: string }
  | { type: 'image_url'; url: string; alt?: string | null }
  | { type: 'audio_url'; url: string }
  | { type: 'video_url'; url: string; alt?: string | null }

type QuestionItem = {
  id: string
  question: string
  options: string[]
  allow_free_form: boolean
}

type DisplayBlock =
  | { type: 'brief'; body: string }
  | { type: 'diff'; diff: string; language: string }
  | { type: 'todo'; items: object[] }
  | { type: 'background_task'; title: string; status: string }
  | { type: 'generic'; content: string }
```

`SubagentEvent.event` is itself a `{ type, payload }` (or a flat WireEvent) and
must be decoded recursively — that's how subagent activity is multiplexed into
the parent stream for the delegation/subagent tree.

---

## 4. Prompt flows (daemon pauses, asks the client)

The daemon can block a turn and request structured input. These map to in-tree
UI branches in the TS app, not separate screens:

- `ApprovalRequest` → approval prompt → reply via `permission_response`
  (`approve` / `approve_for_session` / `reject`).
- `QuestionRequest` → clarify/choice prompt → reply via `question_response`
  with `answers: { [questionItem.id]: value }`. `allow_free_form` enables an
  "Other" free-text entry.

Sudo/secret masked-input flows, if present, surface through `Notification` /
provider-flow `QuestionRequest`s; the masked editor is a client concern.

---

## 5. Client → daemon event echoes

`ApprovalResponse` / `QuestionResponse` are defined as wire events too (the
daemon may echo them), but the **client answers via the RPC methods**
(`permission_response` / `question_response`), not by emitting events.

## Durable run history (optional host capability)

Hosts configured with run history accept `run.list`, `run.inspect`, and
`run.acknowledge` without changing existing v35 methods. Unconfigured hosts
return `{ ok: false, error }`. Session resolution matches terminal methods.

- `run.list`: optional `unread_only: boolean`; returns `{ ok, runs }` with
  metadata only, newest first (maximum 100). Output is obtained through inspect.
- `run.inspect`: required `run_id`; returns `{ ok, run }` including the retained
  output tail and `outputTruncated` flag. Unknown or differently owned runs are
  not exposed.
- `run.acknowledge`: required `run_id` and positive integer `revision`; succeeds
  only for the revision the user actually saw. It does not cancel execution.

Records include `id`, `ownerSessionId`, `workspace`, `kind`, `sourceId`, `title`,
`state`, `startedAt`, `endedAt`, `error`, `revision`, and `unread`. States are
`running`, `succeeded`, `failed`, `cancelled`, and `interrupted`. A recovered
record whose process exited is interrupted, never a restored live process.

`/runs [list|unread|inspect <id>|ack <id> <revision>]` exposes the same history
through the native slash path. The initial adapter records cron turn execution;
delivery outcomes and other run kinds are separate integration work.

Terminal runs additionally expose `terminalKind` and `exitCode` in run records.
Existing `terminal.list`/`terminal.inspect` responses can contain archived rows
with an opaque `run:<uuid>` terminal ID. All archived controls are false;
`terminal.control` does not reinterpret that ID as a live process. Run completion
uses an existing `notification` event with `payload.run_id` and `revision`, sent
only to connections attached to the owning session. A notification does not
acknowledge the durable result.

Run RPCs optionally accept `scope: "workspace"`; omission keeps session scope.
Workspace list, inspect and acknowledge use the workspace resolved from the
session on the daemon, not a caller-supplied filesystem path. The UI defaults
to workspace scope and offers W to toggle. Agent runs use `kind: "agent"` and
`sourceId` identifies the delegated task; each attempt has a distinct run `id`.

`/monitors [list|stop <id>]` manages owner-scoped live terminal watches when the
host configures them. Matching output uses the existing notification event with
`payload.run_id`, `sequence`, and `session_id`. Match notifications are scoped
to attached owner sessions. Monitor runs appear with `kind: "monitor"`; the
source process remains controlled through terminal methods. No new background
model turn is implied by a monitor notification.

### Incremental run evidence

`run.events` is an additive read-only RPC for durable evidence on a run. It
accepts `run_id`, optional integer `after_sequence` (exclusive, default 0),
optional integer `limit` (1–50, default 20), and the same session/workspace
`scope` as `run.inspect`. Workspace identity is resolved by the server.
It returns `{ok, events: [{sequence, text, at}], next_cursor, has_more}`.
Reading does not acknowledge the run or consume evidence. Retain the last
cursor only after processing that page; repeating a cursor safely repeats the
same evidence. Unknown or inaccessible runs fail without exposing events.

In the TUI, `/runs` → V opens saved events for the selected run. N/P navigate
event pages; R refreshes the current page or retries the failed request without
skipping evidence; Escape returns to the selected result. Page content remains
visible on transport failure. Event text scrolls independently at narrow widths.

`/preset manage` opens full composition inspection and editing. Shipped presets
are read-only; copy one with `/preset copy <source> <new-id>` before editing.
`agentPreset.read` additively advertises `guarded_write: true`. Editors may send
`expected_content` to `agentPreset.write`; a mismatch returns
`code: "agent-preset-stale"` before writing. Omitting it preserves legacy callers.
The TUI requires that capability for writes, retains failed drafts, and keeps
drafts scoped to the session and preset when the editor closes. Ctrl+S saves,
Escape keeps the draft, and Ctrl+D explicitly discards it. O shows the folder
returned by `agentPreset.openDocument`; it does not claim to launch an editor.

Terminal monitor events are stored with contiguous per-run sequence numbers
and their displayed output in one transaction before live notification. A
conflicting duplicate or gap fails; an identical duplicate is idempotent.
Each match increments the run revision and makes it unread while still running.
The existing revision acknowledgement therefore cannot hide a later match.
Event text is bounded to 8,300 characters and a run supports at most 1,000
monitor events, matching the monitor admission limit. The full event history
survives loss of the shorter displayed tail and daemon reconnection.

This evidence cursor is not a model-delivery acknowledgement or a promise that
a model has acted. Durable wakeup claiming, cancellation generations, budgets,
and reaction outcomes are separate pending runtime work.

`run.inspect` additionally returns nullable `run.reaction_health` for a watch
with a reaction policy: `state`, `attempts`, `maxReactions`, `pendingEvents`,
`expiresAt`, `activeClaimId`, `lastOutcome`, and `lastError`. States are waiting,
queued, running, cancelling, awaiting-cleanup, cancelled, expired, or exhausted.
An expired deadline with an unresolved executor remains awaiting-cleanup, not
completed cancellation. Inspection and health resolution use the same run owner
and workspace authorization as output inspection. This read does not admit work.

Reaction health additionally includes `usage: {inputTokens, outputTokens,
complete}` for observed parent-turn tokens across claims. Missing reports,
legacy records and unsettled/failed attempts without a report make `complete`
false. These counters are not a spending cap and do not promise aggregate
child-agent or auxiliary-call accounting.

### Monitor inspector

`monitor.list` returns `{ok, monitors}` for the active authenticated session.
Each row contains watch metadata, reaction health and `eventCount`; event bodies
are omitted from list responses. `monitor.inspect` accepts `monitor_id` and
returns `{ok, monitor}` with the latest 20 events plus `omittedEvents` (including
older events removed from the live tail). Durable older evidence remains in
`run.events`. `monitor.stop` accepts `monitor_id` and returns the stopped watch;
it revokes that watch's reactions and does not stop its source command. Unknown
or other-session identities fail. No caller-supplied owner is trusted.
Rows expose `stopAction` as `stop-watch`, `cancel-reactions`, or null. The UI
labels and enables S from this capability. A completed entry may still have
queued/active reactions to cancel; a detached entry has no local stop action.
New watches persist configuration with their run. Lists include up to 100 stored
watch records in addition to attached in-memory watches. Stored rows report
`archived`, `interrupted` or `detached` state and optional `sourceStatus`
explaining the retained outcome and lack of attachment. They do not imply a
restored process handle. Stopping an archived/interrupted watch revokes its
remaining reaction grant; its stored execution outcome is unchanged. Detached
watches belonging to another daemon cannot be stopped through this host.
Pre-existing run records without watch configuration remain available in Runs.

`monitor.create` accepts `terminal_id`, literal `match`, `duration_seconds`
(integer 1–86400, default 3600), `react` (boolean, default false),
`max_reactions` (integer 1–10, default 3), and `reaction_timeout_seconds`
(integer 1–120, default 60). It returns `{ok, monitor}`. The source must be a
live terminal owned by the authenticated session. Invalid settings or an
unavailable source fail without creating a watch. Enabling reactions requires
a configured reaction host; there is no silent notification-only fallback.

File watches use `source_kind: "file"`, `file_path`, and `trigger: "change"`
instead of `terminal_id` and `match`. The path must resolve to an existing
regular file within the authenticated session's workspace; traversal and
symlink escapes are rejected. The optional summary `source` is
`{kind: "file", path, workspace}` with canonical paths. An allowed in-workspace
symlink is resolved once at attachment; the watch follows that target file,
not subsequent retargeting of the original symlink. `terminalId` is
empty. Legacy terminal summaries can omit `source`. File events describe
metadata changes, deletion and recreation; they never include file contents.
Rapid changes may coalesce. Event limits, expiry, ownership and reaction
budgets apply equally to file watches. Shutdown interrupts file watches;
stored inspection reports an observation gap and requires a new watch to
establish a baseline. Restart does not automatically reattach a file watch.

WebSocket watches use `source_kind: "websocket"`, `websocket_url`,
`trigger: "output"`, and a case-insensitive literal `match`, without
`terminal_id` or `file_path`. Summary `source` is `{kind: "websocket", url}`;
`terminalId` is empty. The endpoint must use `wss://`, or loopback `ws://`.
URL credentials, query parameters and fragments are rejected. This generic
adapter receives server-pushed text; it does not send authentication headers
or application subscription frames. Binary messages and text messages over
64 KiB fail the watch. Matching messages are deduplicated by content within
a rolling 256-entry set; retained evidence is limited to 8192 characters per
event and marks truncated messages.

After a connected socket drops, the source records an observation gap and
tries up to three lifetime reconnections, delayed 250 ms, 1 s and 4 s.
Connection attempts time out after five seconds. Gap events count toward the
event limit and may enter an explicitly granted reaction queue even when they
do not contain `match`. Generic feeds do not offer a replay cursor, so messages
lost during disconnection are not claimed as recovered. `sourceStatus` shows
connection health. Shutdown interrupts the watch, and later inspection reports
that missed messages were not observed; restart does not reconnect it.

Bare `/monitors` opens the inspector; explicit `/monitors list` and
`/monitors stop <id>` retain native text behavior. The inspector uses arrows to
select, S to stop, R to refresh, Page Up/Down to scroll evidence and Escape to
close. N opens creation: Tab/Shift-Tab changes fields, left/right selects a
source or reaction mode, Enter submits, and Escape returns to inspection.
Failed submission preserves the form. User-opened inspection survives turn
completion and restores the chat.

### Schedule management extension

`schedule.list` returns `{ok, jobs}` scoped to the connection's current project.
`schedule.inspect`, `schedule.pause`, `schedule.resume`, `schedule.cancel` and
`schedule.run` accept `schedule_id`. Unknown or other-project jobs return
`{ok:false,error}`. Legacy jobs with no project association are not included in
this workspace API; existing `/cron` management remains available.
Schedule RPCs optionally accept `expected_project_directory` as a caller-side
assertion. A mismatch with the connection's canonical project rejects the
request before reading or changing jobs; this field cannot select another
project. The CLI supplies it to detect a mismatched explicit socket.

Inspection and pause/resume return `{ok,job}`. Job fields include the existing
cron payload plus `project_root`, `metadata`, and `execution_state` (`idle`,
`running`, `cancelling`). Resume recalculates the next recurring fire; pause
prevents future scheduled admission but does not cancel an active run.
Cancel returns `{ok,requested,job}`: `requested:false` means no active local
execution. A true value acknowledges the cancellation request, not completed
cleanup. Admission remains held until the underlying runner settles.

Run uses the same lease/concurrency checks and archive/delivery path as
`/cron run`; it returns the normal cron run result after execution. It does not
silently dispatch to another daemon when this daemon lacks the scheduler lease.
These methods operate on the cron store; reconciliation of the separate durable
trigger store is still pending.

`schedule.remove` accepts `schedule_id` and uses the same workspace visibility
rules. It returns `{ok:true,schedule_id,removed:true}` only after removal. Running
or cancelling jobs and jobs with unreconciled execution receipts cannot be
removed. A revision check under the store writer lock rejects changes between
inspection and deletion. This does not remove archived run output or delivery
history. The native `xerxes schedule` CLI uses these project daemon RPCs and does
not create or execute legacy triggers.

Bare `/schedules` opens a workspace list/detail panel using schedule.list and
schedule action RPCs. The registered native `/schedules <arguments>` is an alias
of the existing `/cron` command. The panel preserves the transcript and draft,
survives turn completion, and blocks background shortcuts until Escape closes
it. Arrow keys select; P pauses/resumes; G runs; X cancels; R refreshes; Page
Up/Down scrolls details. Manual run requests do not block cancellation controls.

`schedule.create` takes `prompt` (1–32000 trimmed characters), required boolean
`paused`, and exactly one of `schedule` (five-field cron), `interval_seconds`, or
`at` (future ISO timestamp with explicit timezone). Optional `timezone` is an
IANA zone for recurring cron; create defaults to UTC and update preserves the
stored zone when omitted. Job responses include the canonical zone. Nonexistent
local times are skipped; repeated local times can fire twice. Interval and
one-shot timing remain elapsed seconds and absolute instants respectively.
It returns `{ok,job}` and assigns the
connection's project, ignoring caller-supplied ownership. Default delivery is
none. Invalid settings fail before persistence.

`schedule.preview` accepts the same timing fields (`schedule`, `interval_seconds`
or `at`, plus optional `timezone`) without a prompt or job ID. It returns
`{ok,next_run_at,timezone}` using the create/update timing validator and does not
persist or execute a job. The ISO instant is an estimate from the current daemon
clock; saving recomputes it. Paused jobs remain paused regardless of the preview.

`schedule.options` returns `{ok,destinations:[{name,enabled}]}` containing archive
only (`none`) and configured native channel names/status, without credentials.
Create/update accept `deliver` and `recipient`; omitted fields preserve existing
delivery settings on update. New external destinations must name a configured
adapter and a nonempty single-line recipient/room ID (maximum 512 characters).
Use `deliver:"none",recipient:""` to disable forwarding. Saving does not send
a notification or start an adapter. Actual execution archives output and uses
the existing durable delivery outbox; delivery failure does not rerun the model.

Create/update accept optional `missed_run_policy` (`coalesce` or `skip`) and
`misfire_grace_seconds` (integer 1–86400) default to `coalesce` and 300 on create;
omitted fields preserve the stored policy on update. Coalesce runs one occurrence
after downtime. Skip advances recurring jobs beyond the current clock when they
are later than the allowance; missed one-shots pause for review. Skips record
`metadata.last_missed_run` and never update `last_run_at`. Existing execution
receipts still require review before any missed-run handling. Job responses
expose these settings and `overlap_policy: "forbid"`; overlapping runs are never
admitted. Manual run is an explicit override of timing, subject to the same lease
and concurrency limits.

`max_model_calls` is optional on create/update: integer 1–10000 limits logical
model-call admissions per execution attempt, `null` removes the limit, and an
omitted update preserves it. Parent and native child streaming attempts and
auxiliary `completeLlm` requests share the counter. Provider-internal HTTP retries
are not individual admissions; this is not a token or billing cap. Required-call
exhaustion fails the run and prevents normal output delivery. Optional auto-title
generation skips when no admission remains. The job records
`metadata.model_call_usage:{used,maximum,exhausted}`. Closed scopes reject later
model calls. Each schedule retry starts a new execution attempt and counter.

`schedule.update` takes those same settings plus `schedule_id` and `revision`
from list/inspect. It rejects stale revisions and jobs running/cancelling in
this daemon. Updates preserve identity, project and delivery configuration.
Job responses now include an opaque revision of the persisted record. This
check protects edits within the daemon; cross-process write serialization is
still a separate prerequisite before concurrent-host editing is advertised.

The schedule panel now exposes N create and E edit. It snapshots the selected
job's revision when editing; background refresh cannot silently replace an
in-progress draft. The form uses Tab/Shift-Tab for fields, arrows for timing and
paused state, Enter to save, and Escape to return. A rejected request retains
its input. New jobs default to paused and use a 09:00 UTC cron expression until
changed. One-shot mode submits an explicit timezone-bearing timestamp instead.

Cron writes now acquire a shared SQLite writer lock alongside the JSON store;
the revision is checked again under that lock. Contention returns an error for
retry rather than blocking the event loop. The OS releases the lock when its
process exits. This coordinates updated clients; older executables that write
the JSON directly must be stopped before enabling concurrent management.

`run.list` accepts optional `kind` (schedule/terminal/agent/monitor) and
`source_id` filters. These apply within the authorized session/workspace scope,
before the existing 100-result limit. Unknown kinds are rejected. The schedule
panel's H action opens the existing Runs inspector with kind=schedule and the
selected job ID; Escape returns to schedules without closing that overlay.

`schedule.deliveries` accepts `schedule_id` and lists bounded delivery metadata
without output bodies. `schedule.delivery.inspect` additionally takes
`delivery_id` and returns `{ok,delivery}` including the retained payload.
All delivery methods enforce the same current-workspace job check as schedules.

`schedule.delivery.send` sends only a pending entry through the configured
native channel manager, without a model turn or a new archive. Missing channel
configuration leaves it pending; sending/uncertain entries cannot be retried
implicitly. Sent entries are idempotent while their receipt is retained.

`schedule.delivery.resolve` takes delivery_id, schedule_id, the observed
`attempts` integer and `decision: sent|retry`. Only uncertain entries at that
attempt count can change. `sent` records an operator-confirmed outcome; `retry`
makes it pending but does not send. Operators must check the destination before
choosing retry because an uncertain prior send may already have arrived. Active
or unresolved sending claims cannot be reset through this API.

The schedule panel's D action opens retained deliveries. It displays destination,
state, attempts and the inspected payload. S sends pending output; A/T propose
sent/retry reconciliation for uncertain outcomes and Enter confirms the selected
decision. Retry resolution never automatically invokes send. Sending entries
have no reset control. Arrow keys select, Page Up/Down scroll, R refreshes and
Escape returns. Action errors survive refresh.

New delivery claims persist an executor PID and a unique claim ID. Opening the
outbox converts sending entries with a confirmed-dead executor to uncertain,
with an unknown-outcome message. They still require explicit reconciliation.
Live/reused PIDs and legacy entries without executor identity remain fenced.
Settlement checks the claim ID, so a stale completion cannot settle a newer
attempt. Recovery does not establish whether the destination received output.

Schedule create/update optionally accept `timeout_seconds` (integer 1–3600)
and `max_retries` (integer 0–10, one-shot prompt failures only). Omission on
update preserves existing values. Job responses expose both fields, null for
inherited daemon defaults. Persisted jobs use additive timeout_ms/max_retries
fields; older records remain valid. The scheduler uses per-job values for both
manual timeout and automatic admission; cancellation retains cleanup ownership.

Schedule create/update accepts `interval_seconds` (integer 1–86400) as a third,
mutually exclusive timing mode alongside `schedule` and `at`. Responses expose
interval_seconds (null for other modes); persisted records add the same field.
Changing timing mode clears the previous interval. Resume starts the next
interval from the current time. The poll runner coalesces missed intervals and
retains the job after success. UI timing choices include Interval, and editing
preserves an interval job's mode.

Legacy schedule migration uses native slash commands through the existing
command surface: `/schedules legacy` returns `triggers` with `id`, `owner`,
`objective`, `enabled`, `destination`, `supported`, and `reason`, together with
`project_root`. `/schedules migrate <trigger-id>` returns the imported `job` in
the same shape as schedule management responses. Imported jobs are paused and
workspace-bound; unsupported conversions fail before disabling their source.
`/cron` accepts these same subcommands. No wire-version change is required.

`run.list` accepts optional `before_started_at` (nonnegative integer milliseconds)
and `before_id` (nonempty string) together. The pair is an exclusive cursor in
`startedAt DESC, id DESC` order. Scope, unread, source and kind filters apply
before paging. Responses contain up to 100 `runs` and additive `has_more`.
Use the last row's startedAt/id as the next cursor; no offset is needed and new
results do not shift the older-page boundary. Invalid or incomplete cursors fail.

`run.list` additionally accepts `state`: `running`, `succeeded`, `failed`,
`cancelled`, or `interrupted`. Omit it for all states. Unknown values fail.
The state filter combines with kind, scope, source, unread and page cursors and
is applied in the store before the page limit.

`run.inspect` includes `cancel_label` (string or null) for a live supported owner
control. `run.cancel` takes `run_id`, `revision` and the same session/workspace
scope as inspection. It rejects stale revisions, completed/unsupported rows and
foreign scope. A successful response `{ok:true,requested:true}` acknowledges the
request, not process termination. Terminal controls stop their owner process;
watch controls stop the watch without killing its source. Reaction cancellation
also revokes future reactions for that watch. Schedule cancellation targets only
the exact active execution and leaves future scheduling configuration unchanged.

Scheduled execution metadata may contain `execution_receipt` and
`execution_recovery_required`. An unfinished receipt prevents automatic replay
and pauses the job for review. `schedule.resume` and native `/cron resume`
acknowledge that receipt, retain `previous_execution_receipt`, and clear the
recovery fence only while the local execution is idle. The UI warns that an
overdue one-shot may repeat prior effects. Recurring resume selects a future
occurrence. These fields are additive; older jobs without receipts retain their
normal scheduling behavior.

Reaction usage now also includes observed counters for children whose
`subagent_event` turn begins during the reaction. Cumulative events for the same
agent/task are deduplicated. A reaction with children remains `complete: false`
until the child event contract can attest to complete usage. Failed/cancelled
executions preserve measured counters with `complete: false`; unknown usage
must not be interpreted as zero cost. Runs labels these measured reaction
counts rather than parent-only tokens.

`/hooks [list]` returns `{ok, inspection}` for the active session's loaded shell
hook runner. Inspection contains workspace, workspaceTrusted, loadedAt, sources,
errors, hooks (event, command, matcher, timeoutMs, blocking), and recent
execution results (event, one-based hookIndex, status, at, durationMs). Recent
results optionally include failureKind (`timeout`, `exit`, or `execution`) and
exitCode for a nonzero exit. These fields do not contain error output. Recent
results are bounded to 100 per cached workspace runner, are not persisted,
and contain no hook input/output. Unmatched hooks are omitted. It is read-only
and reflects cached configuration. Unsupported host runners return an explicit
unavailable result. The TUI command forwards through native slash without
replacing the transcript or changing workspace trust.

`agent.settings.options` accepts optional `provider_profile` and `model` and
returns `{ok, model, reasoning_efforts}` from catalog/provider capabilities,
including the authenticated Codex catalog with offline fallback. Results and
save validation use the selected profile; reasoning caches are scoped by
profile name, endpoint and model. This read-only lookup does not hold the
connection's session-mutation queue while waiting for discovery.
An omitted profile uses the active profile; an omitted model uses that profile's
model. Missing or unsupported profiles fail explicitly. It neither changes the
active profile nor returns credentials.

`/permissions` reports the current session's effective policy, falling back to
the daemon default only when that session has no explicit choice. A successful
`/permissions <mode>` saves the choice with the existing durable session before
acknowledging it. Failed saves leave the prior policy active and return an
actionable storage error. The TUI retains its transcript and ordinary command
completion; retry after resolving storage failure. Concurrent session writes
are serialized, and cancelling a turn does not publish a rejected choice.
As with other settings, empty sessions do not create a history entry until work
has been saved.

`agent.settings.get` returns `{ok, revision, settings, profiles}` without profile
credentials. `agent.settings.save` takes `revision` and a complete `settings`
object. Settings retain the existing default/light/balanced/smart keys. Tier
values may be legacy model strings or `{model, provider_profile?,
reasoning_effort?}`. Provider profiles and reasoning values are validated before
atomic persistence; stale revisions fail. The TUI `/config` overlay preserves
the transcript/draft and closes with Escape. `config runtime` retains the native
text configuration view. Newly spawned native children carry profile and effort
through their run configuration and retry snapshots.
The `subagent.retry` response also includes optional `provider_profile` and
`reasoning_effort` fields alongside `model`. Resetting a recovered child and
sending new input retains these settings; rejected empty input does not consume
the reset permission.
`subagent.interrupt {task?, session_key?}` stops live delegated work: a named
task narrows the cancel to that child (same ownership rules as
`subagent.retry`); unnamed, every live child of the session. Interrupted
handles stay inspectable and retryable. The response carries `ok`, `found`
(whether a stoppable child was targeted — clients gate the stop UI on it) and
`interrupted` (how many children were cancelled).
Model-tool `SendMessageTool` queues input for running native children and continues
finished or interrupted open children under the same identity and saved history.
Recovered children follow the same path with existing workspace, provider-route,
budget and policy validation. Concurrent id/name messages are admitted in order;
acceptance does not wait for completion. Explicitly closed children require
explicit resume, and policy-invalidated children require a new authorized agent.
Live subagent events and persisted snapshot rows carry optional
`provider_profile` and `reasoning_effort` alongside the assigned model. Clients
preserve these across partial progress updates. The agent inspector renders
explicit assignments; missing fields do not imply a fabricated provider or
reasoning setting.
Terminal `status_update` events from the native turn adapter include optional
`stop_reason` from the streaming loop. Scheduled execution treats explicit
reasons other than `completed` and `objective_verified` as failure, preserving
partial output in run history and skipping success delivery. Older/injected
runners that omit the field retain their existing completion behavior.
Monitor reactions likewise reject explicit non-success stop reasons. Their
mailbox outcome is failed, measured usage remains recorded (with completeness
false), and partial output remains available in run history. A failed reaction
does not reset the configured attempt limit.

### Todo state on session reattachment

Session payloads include additive `todos: [{id,content,status}]`, independent of
whether the turn is running. The daemon reads the latest successful canonical
TodoWriteTool result, preferring the current in-flight result over persisted tool
execution history. An explicit empty update returns `[]`. Activation/resume
restores this list after clearing the previous chat; starting another turn in the
same chat retains it until a replacement arrives. Existing transcripts need no
migration. Replayed and in-flight tool rows use the same semantic formatter as
live rows, including bounded legacy argument previews.

Schedule metadata includes additive `token_usage` for the last attempt: `input_tokens`, `output_tokens`, `measured_calls`, `settled_calls`, `pending_calls`, and `complete`. Counts are provider-reported, not billing estimates. Failed, cancelled, missing-usage, or still-pending calls prevent `complete: true`. `model_call_usage.maximum` is null when unlimited. Neither object is a token cap or an aggregate over all historical attempts.

Run records expose additive nullable `tokenUsage` with the same token/count/completeness fields as schedule metadata `token_usage`. The snapshot is persisted atomically with the terminal run outcome; older records return null. It is scoped to one attempt, remains available through `run.inspect`, and follows existing owner/workspace authorization. Repeated finalization cannot replace its original usage evidence.

For running scheduled attempts, `tokenUsage` is checkpointed before provider-call admission and after settlement. These checkpoints do not increment the acknowledgement revision or emit completion notifications. Recovery preserves the checkpoint when marking a dead owner's run interrupted. Pending calls remain unknown; completed run snapshots cannot be overwritten by late receipts.

Monitor reaction run records now include `tokenUsage` and live usage checkpoints from the shared provider-call scope. Their existing `reaction_health.usage` aggregate receives the same observed token totals at settlement (failed outcomes remain incomplete). Event-only injected runners may provide partial counters, but cannot set shared-call completeness. A watch record itself has no model-call usage; each reaction is a separate run.

Live `reaction_health.usage` includes absolute checkpoints from the active claim, always incomplete until settlement. Recovery retains these counters, sets completeness false, and preserves the existing interrupted/cancelled recovery fence. Claim checkpoints are owner/range/deadline scoped and monotonic. They are written before the run projection; cross-store atomicity is not implied.

`slash.exec` supports `hooks preview <event> [tool-name]`. It returns `{ok:true,preview:{event,toolName,executed:false,matched,hooks}}`; hooks include execution-order index, matching status, command, matcher, timeout and blocking capability. Selection uses loaded trusted configuration only. Invalid events or syntax return an error. No command execution, verdict prediction or trust mutation occurs. The normal notification carries the readable preview, preserving the transcript like hook inventory.

`slash.exec` supports `hooks failures [event]`, returning `{ok:true,failures:{event,retainedExecutions,failed,denied,results}}`. Event aliases normalize through the hook configuration resolver; absent filters return `event:null`. Results are newest-first failed/denied entries from the existing bounded in-memory history. Inspection never runs hooks. The notification includes retention/reset semantics and configuration errors; TUI help and native command forwarding expose the command.

### Retained workspaces slash inspection

`slash.exec` forwards `workspaces [list|after <cursor>|inspect <id>]` for the
active session's repository. The daemon returns `{ok:true,inventory:{records,next?}}`
or `{ok:true,review}` and emits readable slash output. Records contain allocation
`id`, `path`, `branch`, and either `taskId` or an `error`; review also contains
`base`, optional `snapshotTree`, `head`, `status` and `diff`. Invalid ownership,
invalid arguments and bounded-output failures return `{ok:false,error}`. This is
inspection only; mutation uses the separate checked apply RPC below. The TUI preserves the
transcript and uses the existing native slash output/error path.

`workspace.list` (`after?: string`) and `workspace.inspect` (`workspace_id: string`)
return the same inventory/review shapes without slash transcript notifications.
Both require an active session and use its repository; there is no fallback to
the daemon working directory. The TUI opens its workspace review overlay for bare
`/workspaces`; explicit slash subcommands retain the textual inspection path.

Workspace reviews include `reviewId`, a SHA-256 identity of the workspace ID,
starting baseline, source HEAD and full binary-capable patch. `workspace.checkApply`
requires `workspace_id` and `review_id` in an active session. It rechecks ownership
and the review identity, runs Git's apply check against that session's parent
repository and returns `{ok:true,check:{reviewId,destination,destinationHead,
checkedAt,canApply,destinationState?,error?}}`. Stale/invalid review identities return `{ok:false,error}`.
No files or indexes are applied. Check results describe only check-time state;
they are not reusable mutation authorization. Older daemons without reviewId
continue to support inspection but cannot offer a bound integration check.

`workspace.apply` requires an active session, `workspace_id`, `review_id`,
`destination_state` and `confirm:true`. The destination fingerprint binds the
canonical parent directory, HEAD and affected file contents/modes/link targets.
The backend revalidates the review and destination before applying, keeps the
index and source checkout intact, and persists original/expected states and
backups. Success returns `{ok:true,integration:{id,status:"applied",destination,
backupPath}}`; failures return `{ok:false,error}` including the backup location
when application began. Concurrent content is preserved on rollback. An unknown
RPC outcome must not trigger an automatic retry. The TUI enables A only after a successful fingerprinted check,
then requires Y confirmation (Esc cancels). Selection/refresh and every apply
attempt invalidate the check. Earlier daemons remain check/inspection-only.

`workspace.integrations` takes `after?: string` and returns
`{ok:true,inventory:{records,next?}}` for the active session repository. Records
include `id`, `backupPath`, and validated `destination`, `status`, `paths`, or an
`error` for an unavailable artifact. Pages contain at most 100 records. Inventory
does not read/verify backup contents and does not imply that an owner is dead.
`workspace.recover` requires `integration_id` and `confirm:true`. It validates all
backups, serializes recovery via an OS-released SQLite transaction lock, rejects
live/unverifiable owners, and re-reads state under ownership before restoring.
Completed applies only release leftover locks and preserve their files. Success returns `{ok:true,recovery:{id,
status,conflicts,backupPath}}`, with status `abandoned`, `applied`, `rolled-back` or `needs-recovery`.
Conflicts preserve newer content. Errors return `{ok:false,error}`. I opens
integration history in the workspace panel; B requests restoration, Y confirms,
Esc cancels or returns to review. Recovery completion refreshes the inventory.

Apply publishes a `preparing` record before atomically publishing its lock owner.
No destination writes occur before the durable `prepared` transition. Recovery
of `preparing` records returns `abandoned` without restoring files; apply checks
that preparation was not abandoned after acquiring ownership. Terminal
`applied`, `rolled-back` and `abandoned` records only release leftover locks,
preserving current files. Malformed artifacts remain unavailable rather than
being assumed safe to discard. A directory lost before its first record was
written has no published integration lock.

`workspace.integration.inspect` requires `integration_id` in an active session.
It validates the record and backup hashes and returns `{ok:true,inspection:{id,
destination,status,backupPath,files,checkedAt,headChanged}}`. File entries carry
`path`, `action` (`restore`, `unchanged`, `conflict`, `preserve`) and `reason`.
A changed HEAD or unreadable/different destination file is a conflict. Terminal
operations preserve files. Inspection is read-only, is not a reservation, and
recovery rechecks current state. The recovery panel loads this preview on
selection, discards stale replies, pages file rows with PgUp/PgDn and scrolls
long paths/reasons with Left/Right. Corrupt backups are reported as inspection
errors rather than assumed restorable.

Workspace reviews may include optional `setup: string`, a bounded presentation
of persisted setup status/error/stdout/stderr. The TUI displays it in workspace
inspection; older records without setup remain valid. Incomplete records do not
assert that setup is still alive. Setup runs via the native allocation host,
not through a UI-side process or new model-controlled setup argument.

`monitor.create` additionally accepts `trigger: "output" | "completion"`
(default `output`). `match` is required for output watches only. Completion
watches accept live or already-finished terminals owned by the session, store
one exit result, and use the existing reaction settings. Monitor summaries
include `trigger`; older summaries can omit it. No process input capability is
implied by either trigger.

### Incremental terminal output

`terminal.output` takes `terminal_id`, optional `cursor: {streamId, offset}`,
and `max_output_chars` (1–200000, default 20000). It returns
`{ok:true,page:{text,cursor,droppedChars,hasMore,running}}`. Pass the returned
cursor to continue; reads never consume another reader's output. Offsets count
UTF-16 code units. `droppedChars` reports the gap when retention overtakes a
cursor. A wrong stream identity or future offset is rejected, not silently reset.
Use `run:<run_id>` to read archived terminals with the same cursor. Legacy run
records without cursor metadata return an actionable error; `terminal.inspect`
continues to expose their saved tail. Live snapshots retain the registry's bound;
durable snapshots retain 64000 code units and checkpoint at most every 250 ms
while output is flowing, with a final checkpoint before completion.

Workspace-scoped `run.list` additionally returns `upcoming_total` and `upcoming`
(up to three records sorted by next due time, then ID). Each record includes
`id`, bounded `title`, `next_run_at`, `timezone` and `execution_state`. These are
unpaused schedules in the active workspace, independent of execution-history
filters. Session scope returns an empty upcoming list. Older daemons can omit
these additive fields; clients retain their history-only view.

`run.list` additionally returns `attention_total` and up to three `attention`
records `{id,kind,title}`, where kind is `approval` or `question`. These describe
live waits for the authenticated current session, including when listing
workspace history. Tool arguments are omitted. These fields are read-only and
confer no permission to answer; existing response ownership rules still apply.
Clients connected to older daemons may omit this section.

On hosts with durable run history, `terminal.output` commits the observed offset
before returning a live cursor. Persistence failure returns an error rather than
acknowledging an offset absent from recovery storage. Retention gaps and legacy
record limitations remain as above; hosts without run history provide live-only
cursors.

### Snapshot restore preview through slash

`slash` with command `/rollback diff <snapshot-id>` returns
`{ok:true,snapshot_id,revision,diff,truncated}` and a slash notification. Diff direction is
current captured files to the target snapshot; no restore occurs. `diff` is capped
at 100,000 characters. Git output above 2 MiB returns `{ok:false,error}` rather than
an incomplete successful patch. Ignored uncaptured files are excluded. Failure
uses the existing slash error result; no protocol version change is required.
The TUI forwards this via its native slash adapter and preserves the transcript.
Legacy `rollback.diff` is not a native RPC alias.

`/rollback apply <snapshot-id> <revision>` validates the SHA-256 preview revision
against the target commit and pre-restore captured tree before workspace writes.
A mismatch returns `{ok:false,error}` and a slash error notification. The existing
direct restore syntax remains available without this guard. Validation is not
a filesystem lock against external writers; ignored uncaptured data is excluded.

### Snapshot timeline reads

`snapshot.list` accepts the normal session key and returns `{ok:true,snapshots}`
using existing snapshot payloads (including optional session/turn coordinates).
`snapshot.preview` accepts `snapshot_id` and returns
`{ok:true,snapshot_id,revision,diff,truncated}` with the same limits and restore
revision as `/rollback diff`. Both resolve the session workspace and return
`{ok:false,error}` on unavailable session, invalid input or capture failure.
Neither emits slash transcript notifications. The TUI gateway maps its session ID
to the native session key. Restores still use the guarded native slash command
and retain its notification and error behavior.

`snapshot.preview` additionally returns `files`, a list of changed literal paths.
An optional `path` scopes its diff and revision to a single regular file or
symlink and returns `action: "restore" | "remove"`. A path absent from both trees,
a directory/submodule path, or traversal is rejected. Full preview revision
semantics are unchanged; selected-file revisions bind target, path, and current
captured entry (including mode), so unrelated edits do not invalidate them.

`snapshot.restoreFile` requires `snapshot_id`, literal `path`, and a file preview
`revision`. It checks the revision before writing, captures a backup and restores
that file, or removes the backed leaf when absent from the target. Returns
`{ok:true,path,previous,snapshot}` using existing snapshot records and emits a
slash completion notification. Errors return `{ok:false,error}`. The gateway
maps session IDs for this RPC as for snapshot reads. It does not accept an
unguarded deletion or infer permission to restore the whole workspace.

`snapshot.list` additionally returns `restore_attempts` from the bounded private
restore journal. Each entry contains `id`, `targetId`, `backupId`, `path` (null for
full restore), `phase` (`prepared`, `completed`, `failed`, `recovered`, `reverted`), `updatedAt`
and optional bounded `error`. Prepared/failed entries keep their backup snapshots
from ordinary pruning. Prepared is not proof of a live process. Reads and backup
selection do not trigger recovery; an explicit successful backup restore covering
the original scope marks that attempt recovered. Corrupt journals return an error
rather than silently hiding recovery evidence or allowing an unjournaled restore.

Recovery attempt history is not an unlimited audit log. Retrying the same backup
and scope replaces its previous unresolved retry record while retaining the
original unfinished recovery anchor. Completed-history retention shrinks as
needed to keep the private journal within its validated entry limit. The original
backup remains pinned until successful recovery covers that scope.

Caught restore failures attempt conditional reversal to the captured backup.
A fully verified reversal records `reverted` and still returns `ok:false` with
the original error and recovery outcome. Unknown concurrent content, path
obstructions, verification failures and unprocessed files retain phase `failed`
with bounded conflict details. No automatic reversal runs after process death.

### Runtime tool inventory

`workspace.filePreview` accepts `{session_key,path}` for an active session and returns
`{ok:true,path,content,truncated}`. Paths resolve inside that session's workspace;
canonical paths outside it (including symlinks) and non-regular files are rejected.
UTF-8 text is limited to 128 KiB and retains whitespace; binary or invalid UTF-8
files return errors. This read-only operation does not append messages, execute a
tool, or call a provider. Older daemons may report an unsupported method.

`capabilities.list` returns a read-only catalog for the attached session:
`{ok:true,skills,total_skills,tools,tools_source,usage_scope,model,provider_profile,reasoning_effort}`.
Skills contain name, description, tags, source, platform_supported and uses;
tools extend the tool inventory records with uses. `usage_scope` is
`retained session history`: tool execution records and canonical user skill
activation messages, not lifetime totals. At most 1000 skills are returned;
`total_skills` reports the full discovered count. Discovery preserves existing
skill trust rules. `capabilities.inspect` accepts `{name}` from that catalog
and returns `{ok:true,instructions,source,truncated}` with a 32000-character
preview. It does not expand commands, activate skills, or call a provider.
Missing sessions, unknown skills and discovery failures return `{ok:false,error}`.
Neither endpoint appends transcript messages. The TUI `/features` opens this
catalog; the daemon's existing text `/features` response remains compatible.

`tool.inventory` is a read-only session-scoped projection of the runtime tool
registry, with an optional embedding-host catalog taking precedence. It returns
`{ok:true,tools,source,execution_readiness}`. Sources are `runtime-registry`,
`host-catalog`, or `unavailable`; readiness is `not_checked` for an attached
inventory or `unknown` when absent. Runtime entries include name, optional
description, exposure (`loaded`, `deferred`, `filtered`, `unexposed`) and reason.
Exposure follows the current transcript's deferred loading and agent/mode filters.
It is not a permission or connectivity verdict. Invalid enforcement profiles
return `{ok:false,error}`. Reads invoke neither a tool nor a provider and produce
no transcript notification; `/tools` renders the same data. Gateway session IDs
are mapped to native session keys.

### MCP connection inventory

Additive read-only `mcp.status` returns `{ok: true, configured, servers}` without
transcript notifications. `configured` reports whether a native manager exists;
`servers` is a name-keyed map with `connected`, `tools`, `resources`, `prompts`,
optional `lastError`, and optional `state` (`disabled`, `failed`, `disconnected`).
Failed and disabled configurations remain visible with zero capabilities.
Configuration launch arguments, headers and environment values are not returned;
lifecycle error messages redact configured secrets.

Native `/mcp [status|reconnect <name>]` exposes this inventory or explicitly retries
one enabled registration. Reconnect returns `{ok, server}`; missing and disabled
registrations return an actionable error. `/reload-mcp` retries all enabled
registrations rather than only previously connected clients. Removing a
registration while reconnect is in flight prevents late client registration.
These controls do not reread configuration files or enable disabled entries.

MCP registration removal/replacement now interrupts pending reconnect backoff.
Shutdown removes all registrations, cancelling their retry timers. Already-running
transport connections still settle under their transport lifecycle; superseded
clients cannot publish capabilities and are disconnected when they return.

### Read-only skill inspection

Native slash `skills inspect <name>` returns `{ok, skill}` with `name`, `source`,
`metadata`, `platform_supported`, `execution_readiness: "not_checked"`, literal
`instructions` (at most 100,000 characters), and `truncated`. It emits a slash
inspection notification but does not append a model turn or expand instructions.
Only skills admitted by normal discovery/trust can be inspected. Missing names
and invalid arguments return `{ok:false,error}`. `complete` supports
`/skills inspect <prefix>` with actual registry names, without subcommand suffixes.
The TUI forwards inspection to this native handler and retains the transcript.

Native `skills diagnostics` returns `{ok:true, diagnostics, total, truncated}`.
Each diagnostic has registry `kind`, `path`, `detail`, and optional `name`.
The daemon refreshes discovery before projecting at most 200 entries, each with
at most 1,000 detail characters. This is a discovery snapshot rather than an
execution-readiness check. `/skills` argument completion offers `list`, `inspect`
and `diagnostics`; extra diagnostics arguments fail with usage guidance.

### Plugin registration provenance

`plugins [list]` preserves `plugins` and `slash_commands` and adds `source`
(`host-registry` or `unconfigured`), `inventory`, and
`execution_readiness: "not_checked"`. Inventory entries carry name, version,
description, dependency strings, capability-name arrays, and `source` as either
`{kind:"module",path}` or `{kind:"host-registration"}`. Module paths describe
successful discovery registration; programmatic registrations have no invented
filesystem source. `plugins inspect <name>` returns one entry as `plugin` or an
error for an unknown registration. Completion resolves inspection names from the
current registry. Neither command invokes plugin callbacks or enables modules.

MCP tools in daemon-backed sessions now appear in the native tool inventory under
stable `mcp__` names derived from exact server/tool identity. Startup connection
success and `/mcp reconnect` or `/reload-mcp` rebuild the runner inventory.
Namespaced calls use ordinary tool schemas, events, permissions and errors; no
new tool-execution RPC bypass is introduced. Reconnect invalidates old-client
handlers instead of silently redirecting them. The CLI owns MCP shutdown cleanup.

### MCP settings host

`mcp.settings.get` reads the explicitly injected user MCP settings file. It returns
`ok`, `source`, `revision`, and `servers` containing `name`, `enabled`, `transport`,
`timeout_ms`, `configured_fields`, and current connection `state`. Launch arguments,
URLs, environment values and headers are write-only on this surface. No settings
read or save emits transcript notifications. A missing settings host or an invalid
file returns `ok: false`; project-only registrations are not editable here.

`mcp.settings.save` accepts `{name, revision, changes, create?: boolean}`. With
`create: true`, the name must be absent from both the file and live manager;
the supplied fields define a new server. A missing user file reads as an empty
configuration with a stable revision and is created only after successful
candidate discovery. Shutdown/removal invalidates pending creation. Omitted fields retain
existing values; null removes a field; supplied `env`/`headers` objects replace the
whole corresponding record. Server rename is rejected. Validation and candidate
connection precede the guarded file commit and live registry swap. Success returns
`{ok: true, revision, warnings, server}` and refreshes runtime tool inventory.
A refresh failure is a successful settings commit with a warning to restart.
Stale revisions, file locks, invalid configuration and failed connections return
`ok: false`, preserving the current live registration. Client disconnect cancels
a pending candidate before commit; an active connect must settle before cleanup.
The transport must retain its normal authenticated/trusted client boundary: this
RPC can change executable commands and must never be exposed as a model tool.

Monitor summaries may include `deliveryError`, a bounded diagnostic from the most
recent failed event-notification callback. It is distinct from execution/storage
`error`: delivery failure preserves the monitor state, committed events and already
queued reaction. A subsequent successful output notification clears it. This field
is currently in-memory health, not durable retry state.

### Language-server health

`lsp.status` uses the active session (or the normal session selector) and returns
`{ ok: true, servers: [...] }`. Each server contains `name`, `languageId`,
`extensions`, `state` (`disabled`, `idle`, `starting`, `stopping`, `ready`, `failed`, `closed`)
and optional fixed `detail` text. Inspection is read-only and never starts a
server. Missing sessions, unavailable host integration and workspace lookup
failures return `{ ok: false, error }`. Command/argument/environment values are
not included. `/lsp [status]` renders this health data through the existing slash
output path and does not replace the transcript. Manual edits to user `lsp.json` require restart; `/config lsp` applies edits live through the settings RPC.

`lsp.release` takes `{ name }` and the normal session selector. It releases only
the named server host in that session's workspace. Success returns
`{ ok: true, name, message }`; missing names/configurations/host support and
cleanup failures return `{ ok: false, error }`. `/lsp release <name>` exposes the
same lifecycle through the native slash handler. It may interrupt requests using
that host. It does not modify configuration, replay requests or immediately start
a replacement. The next request may initialize a new host after cleanup succeeds.

### Language-server settings RPC

`lsp.settings.get` requires an active session and returns `{ ok: true, revision,
servers, warnings }`. Each masked server row contains `name`, `enabled`,
`languageId`, `extensions`, effective `timeoutMs` and `configuredFields`. Saved
commands, arguments and environment values are not returned.

`lsp.settings.save` accepts `{ revision, name, action, changes? }`, where action is
`create`, `update` or `remove`. Create/update require a changes object; remove
rejects one. Omitted fields preserve saved values; null clears optional args, env
or timeoutMs. Names cannot be changed through changes. Saves validate the complete
candidate, stop changed hosts, persist with revision checks, publish live settings
and refresh tool inventory. Responses use the same masked shape. Post-commit
refresh/durability warnings accompany success. Validation, stale revisions,
unavailable hosts and cleanup failures return `{ ok: false, error }`; raw saved
configuration and server errors are not echoed. Disconnect cancels uncommitted
saves. Other clients' simultaneous edits may fail and require reload. These are
host settings operations, not model tools. `/config lsp` opens the native settings overlay, which preserves failed drafts and uses confirmation before removal.

### Goal evidence inspector

`goal.inspect` includes `token_usage` (null when no ledger is available) with
`inputTokens`, `outputTokens`, `measuredCalls`, `settledCalls`, `pendingCalls`
and `complete`. The goal may include `maxTotalTokens`. F10 keeps missing,
pending and incomplete usage distinct from a verified zero count.
`/goal --tokens <positive integer>` edits the cap without resetting spend;
native create/edit tools accept `max_total_tokens`. `get_goal` exposes the
ledger projection too. Admission is checked for each tracked provider call and
before further goal rounds; previously admitted calls may overshoot. Capped
resume refuses unknown prior usage or foreign pending calls. Legacy goals with
no ledger do not receive an invented complete baseline.

Goals may carry optional `maxDurationMs`, a wall-time limit from `createdAt`.
The deadline includes pauses and process downtime. `/goal --duration 30m`
sets or raises that total duration on the current goal (whole `s`, `m`, or `h`).
The native `create_goal` and human-authorized `update_goal` edit actions accept
`max_duration_ms`; `get_goal` returns the configured duration and `deadlineAt`.
F10 shows elapsed/remaining time and expiry. At expiry the daemon aborts live
goal work and blocks further rounds with `time-limit`; raising the duration
does not implicitly resume a blocked goal. Legacy goals have no time limit.

`goal.inspect` is read-only and scoped to the connection's active session. It
returns `{ok, session_id, goal}`; goal is null or the durable goal view including
revision, phase, objective, roundsStarted/maxGoalRounds, blocker and optional
criteria. Each criterion has id, description and optional evidence with
toolCallId, summary and recordedAt, or `{kind: "user-decision", decisionId,
summary, recordedAt}`. F10 refreshes this view while open and on R;
failed refreshes preserve the last valid view.

`goal.decision` records an explicit user acceptance note for one criterion. The
daemon generates its decision ID and timestamp, checks the displayed goal ID and
revision, and persists before acknowledging. The session ID must match the
connection's active session; it cannot retarget the request. A running turn or
pending session operation refuses the action. Active, paused and blocked goals
accept decisions without changing their phase; completed goals are immutable.
The UI distinguishes user decisions from model-assessed tool relevance. Failed
submission retains the note for review. This RPC is exposed to trusted daemon
clients, not through model goal tools or MCP tool registration. The transport
does not attest that a human physically supplied the input.

`/goal resume` is the human authorization that stages continuation. It records
one durable, per-session continuation receipt (`goal_wake`) for the current goal
ID and revision in state `queued`; it does not persist a prompt. The background
admission path waits for already queued human work, then reserves the next goal
round, regenerates its prompt from the current goal, and persists the claim and
round together before sending the prompt to the provider. Completing a human
turn also stages continuation for an active, armed goal. `goal.inspect` includes
the current continuation receipt when it belongs to the displayed goal.
The `continuation` value is null when no receipt belongs to that goal; otherwise
it is the persisted `goal_wake` record with `version`, `id`, `sessionId`,
`goalId`, `revision`, `state`, `queuedAt`, and applicable claim or settlement
fields.

Queued receipts are disarmed across restart and are not replayed automatically;
another `/goal resume` is required. A running receipt owned by a different
daemon is recovered as `interrupted` because its provider result is unknown, so
the round is never replayed. Settled, interrupted and cancelled receipts remain
durable for inspection, while a later explicit resume may stage a new receipt.

Model goal tools accept `criteria: [{id, description}]` on create/edit and
`update_goal` action `record_evidence` with criterion_id, tool_call_id and
evidence_summary. Evidence resolves against a unique completed tool call in the
same session. Failed, denied, pending and goal-management calls cannot qualify.
The host checks execution outcome; the relevance claim remains the model's
assessment and is labelled as such. No command-name verification whitelist is
used. Declared criteria must have evidence before completion. Changed criteria
or objective invalidate affected evidence; revisions and the change log retain
the edit history. Legacy goals without criteria keep their prior behavior.

Goal payloads may include `currentMilestone`, a human-readable checkpoint of up
to 1,000 characters. Native `create_goal` and human-authorized `update_goal`
action `edit` accept optional `current_milestone` (`string` or `null` on edit).
`update_goal` action `milestone` requires `current_milestone` (string or null),
goal_id and revision; other edit fields are rejected. It is available to a
direct human turn or the goal's own current continuation round, never to
subagents or unrelated background turns.
The host records milestone changes as goal revisions and durable change history;
they do not complete a goal, add criterion evidence, spend a round or alter a
budget. `/goal milestone <text>` sets it, bare `/goal milestone` displays it,
and `/goal milestone clear` removes it. Active, paused and blocked goals accept
these changes; completed goals retain their last milestone and reject setters.
An objective change clears the old milestone unless that edit supplies a
replacement value.

### Context inspector

`context.inspect` reads only the connection's active session; a caller-supplied
session_id cannot select another owner. Optional `section` is instructions,
memory, conversation or tools; `offset` is a nonnegative integer. Responses
include sections with availability, provenance, count and estimated_tokens;
entries with index, title, text, truncated and estimated_tokens; generation,
section, offset, next_offset, captured_at and a scope/estimate note. Pass the
returned generation when paging. A changed content fingerprint rejects a stale
page. Pages contain at most 20 entries and 8000 text characters per entry.
This inspects latest scaffold plus retained transcript, not an exact live wire
request. Native `/context` returns the summary without provider calls or
transcript changes; the TUI opens a restorable inspector. `/usage` is unchanged.

Memory entries may include `control: {scope, path, pinned, excluded}` and the
page includes `controls_revision`. These are optional source snapshots, separate
from the assembled reference; source-preview estimates are not added to section
totals. In Memory, J/K selects, I pins/unpins, and X excludes/includes a source.
`context.control` accepts `action` (pin, unpin, exclude, include), `scope`, `path`,
`revision`, and inspector `generation`. It applies only to the connection's active
session and validates both revisions and source identity. Pin contents come from
the daemon snapshot, never caller-supplied text. Mandatory assembly layers and
self-memory are not controllable through this RPC. Excluding a pinned source
removes the pin. Including restores eligibility for the next retrieval; another
turn may be required before a source can be pinned again.

Controls require an idle session with conversation history and are persisted in
`metadata.context_controls`. Success returns `applies_next_turn: true`. Pins are
bounded snapshots (16 pins, 8000 UTF-8 bytes each, 32000 total); exclusions are
scope/path identities (128 maximum). They survive source edits, restart and
compaction. After restart, controls remain visible even though assembled source
snapshots are unavailable until the next turn. Exclusions govern automatic memory
recall, not explicit memory-reading tools. Full branches inherit controls;
historical branches omit them along with other current derived metadata.

The tools section lists retained `role: tool` results in transcript order, then
the latest assembled tool schemas. Result titles identify their original message
position and tool name when present. Results are available even before a scaffold
has been assembled. They remain in the conversation section too; section token
estimates overlap and must not be summed. Inspection does not delete, exclude or
replay any tool result, and the standard excerpt and pagination bounds apply.

Conversation entries label retained user-turn numbers. Native slash
`/branch --through-turn N [title]` creates an independent branch through the chosen
completed retained turn. Invalid/incomplete boundaries fail before allocation.
The branch excludes later messages and mutable derived metadata, marks aggregate
usage incomplete, and records `branch_through_retained_turn` and
`branch_message_count` in lineage metadata. Current model/reasoning/permission
settings are retained; workspace files are not restored. The TUI resumes the new
branch while keeping the source open. Branching requires an idle source session.

`context.inspect` additionally accepts section `compaction`. Its section summary
has `available: true`, zero `estimated_tokens` (history is not model context),
and provenance `Persisted compaction metadata`. At most 100 successful stamps are
retained in session metadata, displayed newest first through the existing bounded
paging contract. Main and subagent compaction record history before their normal
transcript flush. Legacy `last_compaction` is retained and is used if history is
absent. Recorded archive paths are not verified or read by this RPC. History changes
invalidate the generation just like transcript changes.

Schedule create/update accepts optional `max_runs`: integer 1–10000 or null to
remove the lifetime admission limit. Omission preserves the current limit on edit.
Job payloads include `max_runs` and `runs_started`. The counter is host-owned and
cannot be reset through schedule settings. All scheduler admissions, including
manual executions and retries, share it; exhausted admissions fail before execution.

Schedule create/update accepts optional `expires_at` (ISO timestamp with explicit
timezone, or null to clear). Omission preserves the existing value on edit. New
expiry values must be future instants; persisted past values remain readable.
Payloads return the normalized UTC timestamp. At or after expiry, new admissions
are rejected without consuming `runs_started`; scheduled eligibility checks pause
the job and retain `metadata.schedule_expired_at`. Existing active work is governed
by its execution timeout/cancel control, not the new-admission expiry.

Schedule create/update accepts `target: "session" | "independent"`. A new session
target is bound server-side to the authenticated active session ID; caller-supplied
IDs are not accepted as binding input. Session mode requires max_runs and expires_at.
An omitted target preserves the binding on edit. Editing a bound job with session
mode retains its original ID; independent clears it. Payloads expose target_session_id.
Execution serializes through that conversation's operation queue and revalidates
identity/workspace on open/resume. Queued expiry and cancellation prevent a late
provider turn. Events are delivered to clients viewing the bound conversation.

`/loop` is a registered native slash command. Its bare TUI form opens a restorable
conversation follow-up panel; daemon slash dispatch supports list, pause, resume,
cancel and run. `schedule.list` and schedule control RPCs accept `scope: "session"`
to filter/authorize against the authenticated active session's target_session_id.
Other scope values are rejected. The unscoped workspace schedule interface remains
available. New /loop drafts use the existing schedule.create contract with target
session, paused=true, interval_seconds=600, max_runs=10 and a 24-hour expires_at.

### Live goal display

`status_update` may carry `goal` and `goal_phase`. Strings replace the displayed
objective and phase; explicit `null` clears them. Omitted fields leave the current
goal display untouched, allowing partial telemetry frames. The daemon sends goal
changes during a running turn and after `session.goal`, so the header, Tasks card,
and goal inspector do not need a session reload. Goal tools and human commands use
the same live compare-and-set history during a turn.

### Conversation follow-up status

For `schedule.list` with `scope: "session"`, the response includes
`owner_session_id`. Each job includes `latest_attempt` (null if unavailable), a
bounded projection of the run ID, state, start and end times from run history.
No archived output is loaded for this projection. Clients may supply
`owner_session_id` on scoped schedule requests; the daemon rejects a mismatch
with the currently attached conversation before listing or acting on jobs.

The session header and F10 inspector read this scoped status without model calls.
They distinguish active/cancelling work from future eligibility: paused, expired,
and attempt-exhausted jobs do not promise a next wake. L in F10 opens /loop;
existing pause/cancel/history controls retain their meanings. Late responses for
a previous conversation are ignored, and failures replace stale wake information.

### Follow-up stop conditions

Schedule create/update accepts nullable `stop_condition` (1–4000 characters when
set), restricted to session-bound follow-ups. Omission preserves it on edit;
null clears it. Records expose this field and retain completion reports in
metadata. The model-only `manage_schedule` action `complete` requires the active
host-owned follow-up context, matching schedule/session IDs and nonempty
`evidence` (up to 8000 characters). It is not a general external schedule RPC.
Reports are explicitly model-reported; up to 20 are retained on the job.
Completion pauses new wakes, survives restart, and requires explicit resume.
Cancellation and completion retain separate meanings.

Scoped `schedule.list` accepts `summary: true` for polling: prompts are excerpted
to 500 characters and metadata contains status flags only, excluding report
history and evidence. Inspectors request the full record on demand.

Schedule create/update additionally accepts `max_total_tokens: integer | null`;
job payloads return the same nullable field. It is a lifetime measured-token
admission threshold, including cache input, across attempts and descendants.
In-flight calls may overshoot. Unknown historical usage prevents admission when
configured. Clearing the threshold retains cumulative usage. Durable accounting
is stored in job metadata as `total_token_usage` (version 1, attempt, usage).

Every schedule job payload, including session `summary: true` responses, now has
`token_budget: { used: number | null, complete: boolean, maximum: number | null,
blocked: boolean }`. Null usage denotes invalid/unavailable accounting; incomplete
usage can contain a known lower bound. `blocked` means configured token admission
cannot currently proceed. Running/cancelling state takes display precedence because
existing in-flight work may still settle. Clients must not promise a future wake
for an idle token-blocked job, even before the scheduler persists its paused state.

`monitor.create` accepts optional positive-safe-integer `max_total_tokens` when
`react: true`. A configured reaction health payload includes
`tokenBudget: { maximum: number, blocked: boolean }` alongside cumulative `usage`.
Idle budget-blocked grants use state `exhausted`; active work keeps its existing
running/cancelling state. The threshold includes parent, child and auxiliary calls.
Already-admitted calls may overshoot. Grant replacement remains a new watch.

`monitor.update` edits reaction limits for the selected owner's watch. Required
fields: `monitor_id`, `revision`, `max_reactions` (1–10),
`reaction_timeout_seconds` (1–120), and `max_total_tokens` (positive safe integer
or null). The revision and current settings are returned as `reactionHealth.policy`.
The atomic update retains attempts, consumption and usage. It rejects stale edits,
unresolved executions, cancelled/expired grants and edits below attempts spent.
Successful host edits dispatch already-pending eligible evidence through normal
session admission. Source match/expiry changes still require a new watch.

### Model routing notes

`model.routing_note.get` accepts `provider_profile` (an existing profile name)
and optional `model` (blank means provider-wide). Returns
`{ok:true,routing_note:{provider_profile,model,note,revision}}`; absent notes have
revision 0. `model.routing_note.save` also requires `note` (at most 2000 characters)
and the last observed `revision`. It atomically saves and increments the revision;
stale revisions fail. Empty notes retain a revision tombstone. Notes are global
user preferences shared across sessions, editable through `/config` F6, and read
by the model inventory host. They never modify provider credentials or quotas.

### Authenticated webhook monitor extension

`monitor.sources` resolves the current authenticated session owner and returns
`{ok:true,webhooks:[{name}]}`. Names refer only to explicitly host-configured
sources; secrets and environment values are never returned. An unconfigured host
returns an empty list. `monitor.create` additionally accepts
`source_kind:"webhook"`, `webhook_name`, `trigger:"output"`, `match` and the existing
expiry/reaction settings. Other source selectors (`terminal_id`, `file_path`,
`websocket_url`) are rejected for webhook creation. Summary `source` is
`{kind:"webhook",name}` and `terminalId` is empty. Existing owner scoping,
inspection, policy editing, event retention and stopping semantics are preserved.

The daemon's explicit monitor HTTP host receives signed UTF-8 deliveries as
documented in the configuration guide. Delivery data is untrusted evidence, not
new authorization. Shutdown interrupts the watch and closes its subscription;
restart requires a new watch and does not claim recovery of missed deliveries.

### Project specialist editing and machine handoff

Additive v35 RPCs under `agentPreset.*` use the selected session's project:

- `agentPreset.projectList {}` → `{ok, agents:[{id,description,error?}]}`.
- `agentPreset.projectRead {id}` → `{ok,id,content,revision}`.
- `agentPreset.projectWrite {id?,content,revision}` → `{ok,id,content,revision}`.
  New definitions use `revision:null` and may derive the id from YAML `name`.
  Existing definitions require the read revision. Validation/conflict errors
  return `ok:false`; the client preserves its draft. Successful saves refresh
  runtime definitions.

`slash.exec` with `command:"machine list"` returns `{ok,machines,output}`.
Each machine is `{alias,target,workspacePath}`. `machine add <name> <ssh-target>
<absolute-path>` and `machine remove <name>` mutate the user registry;
`machine connect <name>` resolves a saved machine and returns `{ok,machine}`.
`machine hosts` returns `{ok,hosts:string[]}` discovered from concrete Host aliases
in the daemon user's `~/.ssh/config` and Include files. No config is modified.
`machine browse <ssh-target> [base64url-utf8-path]` returns
`{ok,path,directories:string[],truncated:boolean}`; an omitted path means remote home.
Directory names are immediate children, including hidden folders. Listings stop at
1,000 entries and 1 MiB, with a 15-second SSH deadline. Browsing uses BatchMode and
strict host-key verification. Browsing, TUI setup and its RPC tunnel explicitly
disable SSH agent and X11 forwarding even when an SSH alias enables them.
TUI setup and transport also require an already verified host key; establish the
host's identity with ordinary SSH before connecting a new destination.
Setup, tunnel and folder browsing use a temporary mode-0600 SSH configuration
that includes the normal user/system files by reference. It retains aliases and
identity-file references, then clears `SendEnv`; a fixed `SetEnv XERXES_SSH=1`
entry prevents configured environment assignments from being forwarded.
The private configuration also disables inherited control-connection reuse,
LocalCommand, and all configured port/socket forwards. The long-lived tunnel
creates its own private control master with no forwards, then adds only the
selected daemon socket through `ssh -O forward` with no user configuration.
This avoids reopening unrelated LocalForward, RemoteForward or DynamicForward
entries. Master readiness and socket creation have a bounded startup deadline;
cancellation and failures close the owned master. A dropped master follows the
same bounded same-destination recovery path and does not replace the renderer.
No credential or configuration contents are copied into that wrapper. It is
removed when the connection/browse ends. SSH failure diagnostics are classified
into fixed messages rather than exposing subprocess output. The managed
bootstrap's private `setup.log` records only known stage messages; installer,
dependency and build output is not retained there or tailed into the TUI.
Errors return `{ok:false,error}`; cancelled TUI
pickers discard late replies. F2 on the host/folder field opens these pickers;
Escape restores the unfinished form. These are read-only slash RPC extensions.
Arguments may be single- or double-quoted. Resolution does not prove connectivity
or change daemon session ownership. The terminal client prepares the remote daemon,
forwards its Unix socket through SSH, and starts a local renderer with a connect-only
GatewayClient. That client never starts or signals a local daemon on tunnel failure.
The previous local renderer is restored on exit. Errors return `{ok:false,error}`.

An established TUI handoff retains that renderer during transient tunnel loss.
It retries the same destination and private socket at most three times (250 ms,
1 s, 2 s), without repeating bootstrap. Authentication/host-key failures stop
automatic retries. A private mode-0600 status file contains fixed transport
messages, never SSH stderr. Terminal failure leaves the renderer available for
draft inspection/copying; exiting does not silently start another renderer.
The gateway waits up to 30 seconds for the replacement socket and reclaims its
advertised connection lease before resuming. The UI retains its draft and view,
buffers live events until restored session ownership is installed, then consumes
missed events once and restores only currently pending interactions. If another
drop discards an undelivered journal, it reloads persisted/inflight state instead.
Expired leases reopen saved state without restarting cancelled work. Exhausted
recovery is visibly disconnected; it must not report idle or submit new work.

`workspace.diff` is an additive read-only RPC with no required parameters. It
uses the active session's cwd (or the daemon project before initialization). Optional
`path` selects one literal workspace-relative file with its own output budget; absolute
paths and traversal segments are rejected. It never changes the workspace or Git index.
It returns `{kind:'clean'}`, `{kind:'error',message}`
or `{kind:'ok',diff:{lines,files,insertions,deletions,truncated,untracked,untrackedTruncated}}`.
Lines have `kind`, `text`, and optional `oldLine`/`newLine`. Untracked text is
represented as additions, so it supports the same file navigation as tracked changes.
Git output, duration and rendered rows are bounded; symlinks are not followed.
Optional `untracked_limit` (integer 1–10000, default 50) expands the new-file
list without increasing the diff-content budget. `untrackedTruncated` identifies
remaining entries; GUI "Load more new files" and TUI `M` request a larger list.
Clients may use the existing bounded `workspace.filePreview` for a known untracked
file when an older daemon ignores `path` and omits it from a truncated overview.
They must not present the complete content of a tracked file as an added-file diff.

`background.status` is a read-only session-scoped RPC returning
`{ok:true,shells:number,watchers:number}`. It counts currently running terminal
registry entries and monitors in the `watching` state, without reading output or
consuming events. The gateway maps `session_id` to `session_key`. The composer
now uses the additive `background.activity` RPC and `background_changed` event.
The event carries no data: it invalidates the snapshot on shell, monitor or
schedule lifecycle changes, without forwarding output bytes. Clients subscribe
before reading and coalesce events received during requests. Disconnects clear
stale counts; reconnect initialization refreshes the snapshot.

`background.activity` returns `{ok:true,session_id,rows,omitted}`. Rows carry
`id,kind,title,detail,state,startedAt,endedAt,action,scope`; schedules also carry
`revision,nextRunAt,lastState,lastEndedAt`. Times are epoch milliseconds or null,
except nextRunAt (ISO). Shells/watches belong to the session; schedules include
its follow-ups and independent schedules in the same workspace. Other sessions'
follow-ups are excluded. At most 200 rows are returned, active work first.
`/activity` opens the unified panel; existing terminal.control, monitor.stop and
schedule.pause/cancel RPCs perform explicit user actions. A stopped watcher does
not kill its source shell. Finished/failed badges expire after thirty seconds;
retained history remains inspectable in the panel and /runs.

`set_model` accepts optional `provider_profile`, validated with the model before
mutating the session. The picker sends one request instead of provider_select
followed by set_model. Session metadata persists provider_profile (name only,
never credentials). Runtime routing resolves that profile for each turn and
inherited child. Ambiguous legacy routes fail with a request to use /model.

`slash.exec` also exposes local extension management: `skills search <query>`,
`skills browse`, `skills install <directory-or-SKILL.md>`, `plugins install
<module.ts>`, and `plugins enable|disable <name-or-path>`. Host availability,
validation and import errors are returned explicitly. Plugin management is an
injected daemon host capability; the production CLI supplies it.

Remote handoff now bootstraps a dedicated user-owned installation through SSH.
It checks GitHub main, builds a missing revision with locked dependencies in a
staging directory, verifies the CLI, and promotes it only after success. A setup
lock prevents concurrent installation; errors preserve previous releases. Bun
1.3+ is installed when absent/outdated. SSH transports RPC only; the local TUI
owns the terminal. Exiting restores the previous local renderer. No provider
credentials, project files, or existing Xerxes installations are overwritten.

Bang-command adapter responses preserve native `{code,stdout,stderr}` even when
`ok:false` denotes a nonzero process exit. The TUI displays these streams before
submitting one `turn.submit` through `prompt.submit`, carrying command/output as
context and retaining output in `display_text`. Session switches and explicit
interrupts suppress late automatic follow-ups. Interpolation remains output-only.

`agentPreset.projectGenerate {description}` returns `{ok,id,content,revision:null}`
for an unsaved, validated project-agent Markdown draft. It uses the session's
provider/model, accepts 1–12,000 characters, and bounds the provider call to 90
seconds. Failure returns `{ok:false,error}`. It does not write files or modify the
conversation. Review/edit and use `agentPreset.projectWrite` to save explicitly.

Compaction lifecycle uses `status_update` with `kind: "compressing"` and a
user-facing `text` while work is in progress, followed by `kind: "compaction"`
when the attempt ends. These events also cover mid-turn automatic compaction and
context-overflow recovery. The end event clears the activity indicator; it does
not imply success. Error reporting remains separate. Ordinary telemetry does not
clear an active compaction indicator.

Provider requests emit `status_update` with `kind: "provider_wait"` before
waiting for output and `kind: "provider_ready"` when output begins or that
attempt ends. These activity events do not replace usage or session metadata.

Network recovery uses `status_update` with `kind: "network_retry"` and text
`Retrying connection…`. It remains visible across provider attempts until
`provider_ready` or turn completion. These recoverable failures do not append
error messages to the transcript. Retries have capped backoff but no attempt
limit; cancellation still ends the turn. TLS verification remains enabled.

The `complete` RPC accepts optional `path_prefix` for directory browsing.
Unlike mention completion, this is a literal path prefix (spaces are preserved),
`./` lists the workspace root, and unreadable directories return an RPC error.
Results retain the existing `value`, `label`, and `meta: "file" | "dir"` format
and a 50-entry bound. Optional `path_offset` (safe integer 0–100000, default 0)
pages the sorted results. Hidden entries appear when the basename prefix begins with `.`.

### Idle runtime replacement

`runtime.restart_if_idle` returns `{ok:false,busy:true}` while sessions, queued
session operations, subagents, scheduled runs, shells, monitors, or configured
channels keep the runtime in use. When idle it closes request admission, stops
scheduled dispatch, acknowledges `{ok:true}`, and gracefully shuts down. A local
desktop transport reconnects using its bundled runtime and resumes the same
session. Remote transports require an update on the remote host. Older daemons
that do not recognize this method are never restarted automatically. An explicit
desktop restart click can migrate a legacy daemon after checking runtime status,
all active sessions, terminals, and monitors; unavailable activity checks block
the restart. This legacy check is advisory rather than atomic across clients.
The desktop attempts replacement once per app instance; busy work defers
it, while failures remain visible and require retry rather than a restart loop.

### Shared local daemon ownership

TUI and desktop derive the same control address under `$XERXES_HOME/daemon` from the resolved Xerxes home, independent of project directory. Projects remain explicit on `initialize` and `session.open`; sessions retain their own cwd, permissions and instructions. Skill registries, MCP managers, agent presets and turn runners are scoped to the workspace within this one process. The v35 frames remain unchanged.

Closing a TUI detaches its connection; it does not terminate the daemon. Build updates request `runtime.restart_if_idle`, which checks work across every session atomically. Explicit socket overrides and remote endpoints remain supported.

Migration preserves old work: clients first attach to an already-running legacy project socket. The global daemon refuses to take that workspace while its old socket is alive, preventing two writers to a saved session. Clients migrate an idle legacy runtime only after its atomic restart endpoint approves shutdown. Busy runtimes and older servers without that endpoint remain attached. Once a legacy runtime exits, subsequent clients use the global socket.

### TUI parity workflows (2026-09-15)

The following are native terminal workflows, not additional wire protocols:

- `/runs` → **V** opens durable event history. **N/P** page using the daemon cursor;
  **R** retries the failed page. Inspecting events does not acknowledge the run.
- `/forge` discovers and inspects packages. **Enter** opens parameter inputs,
  **N** defines a package, **Ctrl+S** reviews and then confirms a definition,
  and **D**, then **Y**, removes the selected version. Invalid inputs and immutable
  version errors retain the draft. `/forge inspect <name> [version]` is also supported.
- `/preset manage` reads complete composition YAML; **E** edits a user composition,
  **Ctrl+S** performs a guarded save, **Ctrl+D** discards a draft, and **O** reports
  its document location. Shipped presets can be copied with `/preset copy`.
  `/custom-agents` remains the separate project-specialist editor.
- `/file <path>` previews numbered workspace text. Completion uses literal
  `complete.path_prefix`, including spaces. `/undo-edits <recorded path|--all>`
  confirms reversal of recorded text edits; it does not discard arbitrary Git changes.
- `/terminals` → **Enter**, **V** reads retained output with cursor paging.
  Retention gaps are explicit. Failed input writes preserve the draft; repeated
  submission while pending cannot duplicate writes.
- `/tool-output [list|last|number|id]` pages the complete tool result received in
  this client. Compact transcript summaries remain short. Historical daemon
  replay can contain only summaries; unavailable historical output is not fabricated.
- `/search [--session <id>] [--limit 1–500] <text>` displays results in a pager,
  with conversation identities, resume instructions and index-coverage warnings.
- `/daemon status` reports shared-runtime identity and readiness. `/restart`
  and `/daemon restart` use `runtime.restart_if_idle` after confirmation;
  `/daemon stop` explicitly confirms stopping every workspace through `shutdown`.
  Quitting one terminal only detaches that client.

On same-session transport recovery, the TUI keeps the inspector identity and
cursor page, composer draft and manual transcript position. Stale confirmation,
approval and question state is cleared; reconnect does not grant authority to an
old dialog. A deliberate session switch resets session-specific inspectors.

See `docs/daemon-tui-gaps.md` for the audit and acceptance evidence.

### Incremental desktop history

`initialize` and `session.open` optionally accept `history_limit` (integer 0–100).
Omitting it preserves the existing v35 transcript and replay behavior. With 1–100,
`session.history` in the returned session payload contains `{ actions, before,
has_more, total_actions }` instead of `transcript`, `tool_executions` and
`thinking_content`; initialize does not additionally emit the full historical
replay. Each action contains an `id`, ordered `messages`, `executions`, and
`thinking`. One complete tool call/result pair is one action.

The `session.history` RPC accepts `{ session_key?, before?, history_limit? }`
and returns `{ ok, session_id, history }`. Limits default to 100 (maximum 100);
`before` is the opaque cursor from the preceding response. Pages are chronological
and exclusive of the cursor boundary. Appending new messages does not shift
older pages. A rewritten boundary or cursor from another session is rejected;
clients must retain their current display and offer a reload/retry rather than
silently splice unrelated history.

`session.status` and `session.active_list` also accept `history_limit: 0` to omit
transcripts, executions and thinking. Counts, active-turn state, telemetry, goals,
todos, agent state and a bounded first-user-message `preview` remain available.
Metadata reads do not change a connection's session selection. Explicit export
requests may still retrieve the full transcript.

### Local provider relay authority (additive v35)

`initialize` advertises `provider_relay_control_supported: true` for the local
authority RPCs below. These are client control operations, not model tools or
slash commands. The remote binding and SSH review flows below use this authority.
A renderer must not read local profile files or provider keys to use this API.

- `provider.relay.inventory {}` returns `{ok:true, profiles}` with profile name,
  provider, model, supported status, output-limit mode, credential source and an explicitly
  unverified readiness label. It omits provider endpoints and credentials.
- `provider.relay.authorize` requires `{consent:true, destination, workspace,
  profile, model, expires_at, max_requests, max_output_tokens, max_concurrent}`.
  The caller must obtain explicit user consent for that complete scope first;
  selecting a machine is not consent. Workspace must be absolute, expiry is a
  future Unix timestamp in milliseconds (at most eight hours), request limit is
  1–10000, numeric output limit is 1–1000000 tokens per request, concurrency is 1–16.
  Numeric limits are checked against the final native provider payload, after
  thinking expansion or minimum-output floors. Exceeding the approved limit
  fails with `output_limit` before submitting that request; lower reasoning or
  explicitly authorize a larger limit before retrying.
  Codex subscription does not accept an output cap. Numeric grants for that
  transport fail with `output_limit_unsupported`. It requires the separate
  policy `{max_output_tokens:null, consent_provider_controlled_output:true}`
  after explicit consent to provider-controlled output. Omitting the numeric
  field or the extra consent flag is not consent. Other transports cannot use
  this null policy. Expiry, request, concurrency and revocation limits still apply.
  Returns `{ok:true, grant:{id, ...}, capabilities?}`. The optional capability
  snapshot describes reasoning controls for the exact authorized local model;
  it contains no credentials, endpoints, provider prose or profile defaults.
  Local catalog discovery is bounded to three seconds and falls back to the
  local bundled catalog/provider table with explicit provenance. Authorization
  is rechecked after discovery; expiration or revocation cannot return an active
  grant. The grant view shows destination,
  workspace, exact profile/model, limits, usage, status, local provider execution
  and memory-only persistence. `outputLimitMode` is `request-bound` or
  `provider-controlled`, with `maxOutputTokens:null` only for the latter.
  `id` is an opaque connection-owned handle, not a
  bearer token. Neither the engine token nor provider credentials leave the daemon.
- `provider.relay.next {id, frame}` accepts a native relay pull or cancellation
  frame and returns `{ok:true, reply}`. The bounded codec is defined in
  `security/providerRelayProtocol.ts`; provider diagnostics are replaced with
  fixed failure codes. Pending pulls do not serialize revocation or cancellation
  behind a provider response.
- `provider.relay.status {id}` returns `{ok:true, grant}` with current usage/state.
- `provider.relay.revoke {id}` returns `{ok:true}` and aborts active calls. Pending
  and future pulls report `grant_revoked`.

Revoked or expired grants release their authority resources after backend work
settles. The daemon retains at most 128 recent terminal status records, without
tokens or client resolvers; an evicted handle reports `grant_unavailable`.
Recent repeated revocation is idempotent. Noncooperative backend work continues
to count as active until it settles. Exhausting the request allowance does not
discard an already-started request's buffered final output.

An authority/ownership failure returns `{ok:false, code, error}` with fixed text.
A provider-stream failure appears inside `reply.error`. No arbitrary exception
text is forwarded. The exact approved route is checked again before each request;
changing it requires renewed consent, with no fallback to another provider.

Grants belong to the requesting connection's existing lease owner. A client
without a reconnect lease loses its grants immediately on disconnect; an opted-in
client retains the owner until the existing lease expires. Lease expiry and daemon
shutdown abort and remove grants. A new owner cannot recover them by knowing an
id. Grants are never stored in a session or configuration file. Idle restart
refuses while a grant is active, including gaps between provider requests; explicit
shutdown still revokes authority. The production binding and consent lifecycle
is described below; reconnect never silently grants fresh authority.

The local-only `ui/lib/localProviderBroker.ts` bridge can own one authorized grant
on the parent TUI's existing daemon connection. It listens only on a Unix socket
inside a mode-0700 temporary directory, with mode-0600 socket permissions. One
bounded newline-delimited relay envelope is accepted per connection; it exposes
no general daemon RPC, grant id, token, profile configuration or provider credential.
Requests are limited to 16 MiB plus envelope overhead, replies to 1 MiB plus
overhead, concurrent sockets to 16, and each exchange to a hard 60-second deadline.
Incomplete and pipelined frames are rejected; UTF-8 decoding occurs after assembly.
Abandoned pending pulls send cancellation to the owning daemon. Expiry, explicit
close and the owner's abort signal revoke the grant and remove the private endpoint;
the broker never retries a request or renews authority. Lost daemon ownership cannot
be restored by reconnecting and retaining the old grant id.

This broker is a local host primitive, not a general daemon transport. Its caller
must obtain consent for the exact scope before
creation, keep the same owning gateway connected, give the socket path only to the
local renderer for that reviewed destination/workspace, and close the broker when
the handoff ends. Socket permissions protect against other OS users; they do not
isolate mutually untrusted processes running under the same local account.

### Partial turns and activity status

Native turns retain text and reasoning already emitted by a terminally interrupted
provider attempt. Cancellation, terminal failure, retry-backoff cancellation and
iterator abandonment persist that partial assistant content without inventing
tool calls or placing error diagnostics in model history. A successful retry still
replaces its failed attempt rather than concatenating replayed output. Resume
marker cleanup preserves assistant whitespace when no provider marker was removed.
This does not yet persist interrupted/failed outcome annotations for the history
view; preserving content alone is not proof that a turn completed successfully.

Terminal titles report current activity explicitly: idle, working, waiting or
disconnected. A disconnected connection takes precedence over retained busy or
approval state. Idle is neutral, including after failure or cancellation; it does
not show a successful-completion checkmark.

### Complete notices

Status notices wrap in a bounded preview below the composer. While a notice is
visible, **Alt+N** (or clicking it) opens its complete message in the existing
keyboard pager. Escape returns to the same draft and conversation. This shortcut
does not override an active approval, question or other blocking overlay.
Background follow-up polling owns its inline error display; stale poll failures
from another conversation do not write into the current transcript.

Explicit `initialize {resume_session_id}` rejects missing persisted history
instead of creating a replacement conversation. A supplied `session_key` that
still names the same live session can be reattached after lease expiry without
requiring its first saved transcript. A mismatched key never changes the requested
identity; workspace checks remain in force. Failed resume leaves the connection's
current selection intact. Explicit new-session creation is unchanged.

The TUI footer's **model selected** label reports configuration only. It does not
assert credential validity, provider availability or successful connectivity.

### Remote setup review

Daemons advertise additive `remote_provider_bundle_supported` capability metadata.
On supporting hosts, one TUI setup approval covers up to 32 supported local
profiles and their configured models for the selected remote task/workspace.
Limits are per provider, with one shared expiration time; provider-controlled
output is accepted once for all affected profiles. Credentials and refresh stay
local. Unsupported profiles are listed with setup guidance and are not granted.
Older hosts retain the single-provider path rather than silently dropping choices.

`provider.remote.bind` accepts optional `alternatives`, an array of up to 31
additional `{source, profile, model, capabilities?}` selections. Its returned
binding includes corresponding alternatives with independent private binding IDs.
All belong to the connection owner and session workspace. Disconnect/release
closes the entire group; partially failed preparation revokes all grants already
created. Persisted selections confer no authority after restart. Existing private
request/reply frames and single-provider calls retain their format.

`provider_list` and `fetch_models` accept `for_model_selection: true` to project
approved local choices for a locally bound task. Ordinary provider management
still addresses remote configuration. `list_available_models` projects the same
approved choices and treats empty/whitespace `provider_profile` as omitted.
`/model` and explicit delegated provider selections may use any approved pair
without another consent screen. Unapproved models cannot switch a same-named
local profile to remote credentials. `/provider` remains the explicit remote
override. New providers/models, reconnection, expiry and revoked access require
fresh setup; destination trust is not persisted.

`/machine` → Enter and `/machine connect <name>` open the same setup review
before handoff. Tab/Left/Right choose an integration; arrows, PageUp/PageDown,
Home and End scroll its explanation. Enter prepares the remote task and opens
the provider-location review; Escape returns without connecting. The review identifies the target host/workspace,
execution and configuration locations, persistence and remote setup commands.
It labels remote readiness uninspected. The next review offers existing remote
setup or explicitly approved local provider reuse.

The review reads no provider credentials. At handoff, the resolved host and
workspace must still match the reviewed record; a changed destination requires
reopening the machine list. Pending resolution is single-flight, and cancellation
ignores its late response. Existing daemon machine RPCs are unchanged.

### Opt-in remote provider binding transport

Hosts that supply the same `RemoteProviderBindings` instance to the server and
native runner advertise `remote_provider_binding_supported:true`. The ordinary
CLI daemon now supplies this registry and can execute bound sessions without a
remote provider configuration. The TUI machine review prepares the remote task
and requests explicit scoped consent before creating local authority. Its parent
owns the relay; the child receives only a public remote session ID/key.

`provider.remote.bind` accepts `{consent:true,source,profile,model,capabilities?}` for the
connection's idle session. It returns `{ok:true,binding}` with a nonsecret opaque
binding ID, display source/profile/model, workspace and version. No credential
or local grant token is accepted. The session stores a local-provider requirement;
that marker alone confers no authority. A runner without its matching live
binding fails explicitly even if a remote provider or fallback model is configured.

The optional version-1 capability snapshot has `{version:1,model,reasoning:
{shape,efforts,canDisable,provenance}}`. `shape` is `effort`, `toggle` or
`inherent`; provenance is `provider_reported`, `bundled_catalog` or
`provider_fallback`. The exact model, bounded effort names and allowed fields
are validated at the local broker, remote bind and persisted-read boundaries.
Only this public metadata is retained in the requirement marker. It never
restores authority after a disconnect or grants access to provider configuration.

Bound sessions do not inherit the remote daemon's provider sampling, output,
thinking, retry-route or context-window defaults. The local authority snapshots
validated sampling fields from the selected local profile at authorization and
applies them on the local host; explicit request overrides retain precedence.
Profile sampling edits take effect on a new grant, while same-route credentials
are resolved per request. Explicit session/agent reasoning is preserved, including
off; unrelated remote settings reloads do not change local default reasoning.
Context capacity remains `0` (unknown), and unpinned reasoning is displayed as
`local default`; the reasoning snapshot does not claim either value. These are display values,
not claims that local thinking is off or that remote model capacity applies.

The relay adds a local-only output authority annotation that cannot be serialized
or supplied through JSON. Native adapters validate their final requested output
cap before sending it. This validates Xerxes's outgoing request, not an external
provider's implementation of its API contract. Codex subscription is available
only under the separately consented provider-controlled policy above; a numeric
grant never silently changes to that policy. The consent panel requires a
separate provider-controlled-output acknowledgement before authorization.

The daemon sends `{jsonrpc:"2.0",method:"provider.remote.request",params:
{binding,request_id,frame}}` privately to the owning physical transport.
The gateway intercepts it before UI events, transcript conversion and reconnect
buffers. An explicitly supplied local authority callback handles the frame;
without one, the gateway replies `grant_unavailable`. The reply RPC is
`provider.remote.reply {binding,request_id,reply}`; another connection and late
or duplicate replies are rejected. Both ends bound concurrent pulls and waiting.
The native relay codec still validates model scope and completion/delta shapes.

Physical disconnect closes remote bindings immediately, rejects pending pulls,
and retains the persisted local-provider requirement. Reclaiming a session lease
does not reauthorize a provider. A fresh explicit binding is required; pending
model context is never replayed through the UI journal. `provider.remote.release`
closes this connection's bindings while retaining their requirements, so release
cannot silently select remote credentials. User-facing setup/revocation and
explicit remote-provider override flows remain unfinished.

Automatic titles, manual/pre-turn compaction and project-agent draft generation
honor the session's provider binding or saved provider pin. A local-bound title
uses only its authorized model, without a cheaper model substitution. Missing
or released local authority never selects remote credentials for these calls;
optional title generation retains the provisional title. Private provider replies
and release requests bypass a waiting auxiliary RPC on Unix and WebSocket
transports, while ordinary session mutations retain their queue ordering.
Release sends cancellation for tracked provider streams before rejecting pending
waits. The CLI planner, mid-turn compaction and native subagents also resolve the
source session binding before remote profiles. Child route fingerprints survive
persistence; a replaced binding cannot revive an old child authority.

The relay reports provider context overflow using the fixed `context_overflow`
error code so the native loop can compact and retry without forwarding raw
provider diagnostics. The codec validates and strips internal boolean compaction
summary provenance before sending history to the provider.

### Client-only appearance

`/appearance` toggles **Chrome** and **Transparent**. `/appearance chrome` and
`/appearance transparent` select one explicitly. The choice is applied live and
stored in the TUI host's `$XERXES_HOME/tui-appearance.json` (default
`~/.xerxes/tui-appearance.json`), independently of daemon/session configuration.
It stays local when the TUI connects to an SSH daemon. No RPC or daemon catalog
change is required for this client-only command; it is available in local
completion and `/help`, including with older daemons.

Transparent uses the terminal's default background, including its configured
opacity. It does not change terminal window opacity. Foregrounds still follow
terminal light/dark detection; popup, selection and diff grounds remain filled
for readability. A failed save leaves the prior appearance active and reports
an actionable local notice. Switching appearance does not navigate or replace
session, transcript, draft, viewport or running-work state.

### Durable turn outcomes (additive v35 fields)

A native terminal turn carries optional `stop_reason` on `turn_end`, using the
same reason vocabulary as the streaming `status_update`. It distinguishes
`completed` / `objective_verified`, `aborted`, `provider_failed`, `turn_failed`,
`context_overflow`, `output_limit`, `tool_budget_exhausted`,
`objective_guard_exhausted`, and `unconfigured_tools`. Absence means unknown,
not successful completion. `unstarted` still suppresses transcript artifacts.

The last persisted message of a turn can carry `turn_outcome`:
`{ "version": 1, "reason": "aborted", "turn_id": "abcdef012345" }`.
It can be on a user message when no assistant output was received, or on a tool
result when no final narration followed. The optional ID is an opaque hex turn
identity. These are presentation fields, excluded from native model messages;
no diagnostic, credential, or assistant text is placed in this object.

Full history replay emits `notification` with `category: "history"`,
`type: "replay_outcome"`, and the validated object in `payload`. Paged history
represents each outcome once as its own action (an empty assistant record carrying
`turn_outcome`), after its message/tool actions. Clients render a status receipt,
not an empty assistant reply. Old records remain readable and have unknown
outcomes. Compaction archives retain outcome metadata; a summary does not invent
an outcome for the turns it replaces. A process crash before final persistence
can still leave an unknown outcome.

Session/init payloads may include `local_provider_label`, a nonsecret requirement
label rather than a connectivity claim. An empty string clears a previous label.
Explicit idle-task `provider_select` or `set_model` with `provider_profile` can
select remote credentials and remove the local requirement; implicit model
changes cannot. Active provider work prevents the override. Failed selection
retains the local requirement and does not restore revoked authority.

The machine consent defaults to keeping current setup. Enter opens a local
profile review; A authorizes only its displayed scope. C additionally accepts
provider-controlled output where supported. Exit, expiry or owner disconnect
revokes the in-memory grant. Reopening requires fresh review; no provider is
silently substituted. Other integrations continue on their documented host.

SSH task preparation uses `initialize` with `history_limit:0` while inspecting a
saved task, followed by metadata-only `session.status`. It does not need replay
rows to request consent. The child TUI still resumes its transcript normally.
Gateway resume adopts the authoritative returned `session.key` before subsequent
scoped RPCs, whether initialize attached a prepared live task or loaded history.
An explicit resume ID first resolves an existing live task, including an untouched
task without durable history. The workspace check still runs before attachment
or configuration changes. This does not save empty conversations or reconstruct
them after daemon loss; missing saved history remains an error.

Explicit reasoning changes on a task with a local-provider requirement stay on
that task and never update a remote profile's defaults. Native task reasoning is
saved before acknowledging success when the task has history. A runtime without
a task reasoning setter refuses the local change rather than reloading global
defaults. Failed effort saves retain the previous live and durable choice, including
context deltas. Writes are serialized per session so a pending setting or concurrent
flush cannot persist a rejected effort; storage failures return an actionable
bounded message and allow retry.

For locally bound tasks, `reasoning_levels` and effort validation use the negotiated
local snapshot exclusively. The picker names its provenance and explains that
fallback tables are not live-verified. Missing or damaged legacy metadata yields
`shape:"unknown"`, `source:"unavailable"` and no selectable levels, with an
instruction to reopen/review through an updated local TUI and daemon. It never
guesses from the remote profile. Long explanations and recovery instructions
are scrollable with PageUp/PageDown, including at 40 columns. Automatic remote
model discovery skips locally bound tasks and discards results after the task
or provider route changes; it cannot inject an unrelated remote-provider warning.

Model and interaction-mode selections use the same per-session write ordering.
For tasks with history, a successful `set_model` or `set_mode` response means
the choice is saved. A failed write leaves the previous model/provider pin,
mode/plan flag and context deltas intact; a later flush cannot persist a rejected
choice. The host mode callback runs only after the mode has been saved and
published. Storage errors explain that the previous choice is unchanged and
direct the user to check session storage and retry. Empty tasks retain the
existing no-phantom-history behavior.

Pasted slash-command arguments are not file-drop hints: model versions, search
terms and path arguments do not trigger the image-attachment suggestion. File
paths remain ordinary composer text; the hint neither attaches nor sends them.
