# System architecture

The native implementation is a Bun/TypeScript event pipeline. Every public surface consumes the
same serializable turn and protocol contracts rather than starting a second runtime.

```text
CLI / OpenTUI / channel / HTTP API / ACP
                |
                v
      bootstrap + session projection
                |
                v
      provider routing and async stream
                |
                v
  permission + policy + tool execution boundary
                |
                v
  transcript, memory, cost, and daemon events
```

## Main packages

| Area | Native location | Responsibility |
| --- | --- | --- |
| CLI | `src/typescript/src/cli.ts` | One-shot, TUI, daemon, ACP, diagnostics, export. |
| Streaming | `streaming/` | Normalized async events, thinking parsing, cancellation repair, tool turns. |
| Runtime | `runtime/` | Bootstrap, profiles, budgets, sessions, diagnostics, update/install planning. |
| Executors and tools | `executors/`, `tools/` | Schema lookup, policy/permission gates, native operations, host ports. |
| State | `session/`, `memory/`, `context/` | Durable transcripts, replay, retrieval, bounded prompt context. |
| Services | `daemon/`, `api-server/`, `acp/`, `mcp/` | v35 local protocol, OpenAI-compatible API, ACP, MCP transports. |
| Integrations | `channels/`, `extensions/`, `cortex/`, `skills/` | Channels, plugins, skills, topology orchestration, native skill modules. |
| Terminal client | `src/typescript/src/ui/` | React + OpenTUI renderer and v35 gateway client. |

## Turn lifecycle

1. The caller opens or selects a session and submits a user turn.
2. Runtime bootstrap composes the configured agent, provider profile, session context, memory, and
   available tool definitions.
3. The LLM adapter normalizes its response into native streaming deltas.
4. The loop emits text and thinking events, collects requested tools, and checks each request
   against policy and permission before execution.
5. Tool results are serialized back into the transcript. Cancellation produces explicit synthetic
   results for skipped calls so the next provider request remains valid.
6. The terminal event, usage, and durable session state are delivered to the caller.

## Trust boundaries

The core never silently treats an external operation as local. Provider clients, browser/computer
automation, hardware training, media APIs, remote channels, and external memory stores are explicit
ports. A host chooses and supplies a real adapter; tests supply a fake. Browser automation is a
native CDP attachment to an explicitly supplied, already-running Chromium-compatible endpoint;
Xerxes does not launch or own that browser process. Security controls remain at the call boundary:
path validation, URL checks, prompt scanning, policy, approvals, redaction, and the selected sandbox
route.

## Compatibility contracts

The Bun runtime preserves intentional public shapes, including daemon protocol v35,
OpenAI-compatible completion frames, persisted session records, YAML agent/skill formats, and tool
schemas. A missing handler must report that fact rather than launching another runtime or returning
a simulated success.
