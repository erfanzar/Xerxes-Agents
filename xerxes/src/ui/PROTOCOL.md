<!--
Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
Licensed under the Apache License, Version 2.0.
-->

# Xerxes TUI ⇄ Daemon Wire Protocol

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

1. Resolve the project dir to the nearest Git root when available; otherwise
   use `realpath(cwd)` (falling back to the absolute path).
2. Compute the per-project socket path through the native `daemonPaths()`
   helpers (default layout):

   ```md
   $XERXES_HOME/daemon/projects/<sha256(project_dir)[:16]>.sock
   ```

   where `$XERXES_HOME` defaults to `~/.xerxes`. (`.pid` sits next to `.sock`.)
   `XERXES_DAEMON_SOCKET` can explicitly override the socket path; the gateway
   and daemon use the same deterministic project-path calculation.
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
  "params": { "type": "text_part", "payload": { "text": "hello" } } }
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
| `turn.submit`                        | `{ session_key?, text, mode?, plan_mode? }`    | `{ ok }`                                                       | Lower-level submit; `prompt` wraps this.                                          |
| `turn.steer` / `steer`               | `{ session_key?, content }`                    | `{ ok }`                                                       | Inject steer text into the active turn.                                           |
| `turn.cancel` / `cancel`             | `{ session_key? }`                             | `{ ok }`                                                       | Cancel this connection's turn.                                                    |
| `cancel_all`                         | `{}`                                           | `{ ok, cancelled }`                                            | Cancel every session.                                                             |
| `session.open`                       | `{ session_key?, agent_id?, project_dir? }`    | `{ ok, session }`                                              | Open/attach a session within the active or explicit project boundary.             |
| `session.list`                       | `{}`                                           | `{ ok, sessions }`                                             | For `/resume` picker.                                                             |
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
| `fetch_models`                       | `{ profile_name }`                             | `{ ok, models, source, warning? }`                              | Discover one exact stored profile without exposing its endpoint credential.       |
| `provider_list`                      | `{}`                                           | `{ ok, profiles }`                                             |                                                                                   |
| `provider_save`                      | `{ name, base_url, api_key, model, provider }` | `{ ok, profile }` + emits `InitDone`                           |                                                                                   |
| `provider_select`                    | `{ name }`                                     | `{ ok }` + emits `InitDone`                                    |                                                                                   |
| `provider_delete`                    | `{ name }`                                     | `{ ok }` + emits `InitDone`                                    |                                                                                   |
| `shutdown`                           | `{}`                                           | `{ ok }`                                                       | Stops the daemon.                                                                 |

**Migrated/removed** (return `{ ok: false, error: MIGRATED_ERROR }`): `task.submit`,
`task.cancel`, `task.list`, `task.status`, bare `submit` / `list` / `status`.

> Slash routing: the TS client handles presentation-only commands locally and
> sends daemon-owned commands to the Bun daemon. The v35 daemon provides
> completion, session controls, approval/question replies, profile CRUD/model
> discovery, browser CDP attachment, plugins, skills, runtime controls, and
> cron/session workflows. Unknown or unavailable operations return an explicit
> native error; neither layer fabricates success.

---

## 3. Events (daemon → client)

Every event arrives as `params: { type, payload }`. The `type` is the
discriminator. Over the **daemon socket** transport `type` is **snake_case**
(see the casing note in §1). The optional native bridge can emit PascalCase
aliases for compatible external clients, so the table below lists both. **The
TS client keys on the snake_case column** and maps the PascalCase column to the
same handler as a tolerated alias.

### Event name map

| PascalCase (bridge alias) | snake_case (daemon wire) | Payload fields                                                                          |
| ------------------ | --------------------- | --------------------------------------------------------------------------------------- |
| `InitDone`         | `init_done`           | `model, session_id, cwd, git_branch, context_limit, agent_name, skills[]`               |
| `TurnBegin`        | `turn_begin`          | `user_input: string \| Part[]`                                                          |
| `TurnEnd`          | `turn_end`            | —                                                                                       |
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
| `StatusUpdate`     | `status_update`       | `context_tokens, max_context, mcp_status, plan_mode, mode, reasoning_effort`            |
| `Notification`     | `notification`        | `id, category, type, severity, title, body, payload`                                    |
| `PlanDisplay`      | `plan_display`        | `content, file_path?`                                                                   |
| `SubagentEvent`    | `subagent_event`      | `parent_tool_call_id?, agent_id?, subagent_type?, event: WireEvent` (nested, recursive) |

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
