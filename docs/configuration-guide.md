# Configuration guide

Xerxes keeps host-dependent configuration explicit. The native value models live in
[`xerxes/src/core/config.ts`](../xerxes/src/core/config.ts), while daemon settings
are read by [`xerxes/src/daemon/config.ts`](../xerxes/src/daemon/config.ts).

## Local runtime home

`XERXES_HOME` chooses the directory used for profiles, daemon state, sessions, and agent memory.
`~` and `~/…` are expanded before the native paths are resolved. If it is unset, Xerxes uses its
normal per-user default. `bun run xerxes doctor` reports the resolved location without exposing
credentials.

## Provider configuration

Configure a provider with a profile or a deliberately supplied environment variable. Common
variables include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and provider-specific
keys defined by the native provider registry. A profile can also hold a model identifier and an
OpenAI-compatible base URL.

Do not commit credentials. Tests and embedding hosts should pass a provider client or a synthetic
environment map rather than reading ambient secrets.

## Daemon settings

The daemon reads `$XERXES_HOME/daemon/config.json` when it exists. Its top-level native sections
are `runtime`, `control`, `workspace`, and `channels`; `maxConcurrentTurns` controls concurrent
turn processing. Settings can reference a named environment value with `env:NAME` or a `*_env`
key, which is resolved at the host boundary.

Useful process-level overrides include:

| Variable | Effect |
| --- | --- |
| `XERXES_DAEMON_HOST` | WebSocket control host. |
| `XERXES_DAEMON_PORT` | WebSocket control port. |
| `XERXES_DAEMON_SOCKET` | Local daemon socket path. |
| `XERXES_DAEMON_TOKEN` | Control-plane token. |
| `XERXES_MAX_TURNS` | Maximum concurrent daemon turns. |
| `XERXES_MODEL` | Runtime model override. |
| `XERXES_BASE_URL` | Runtime provider base URL. |
| `XERXES_PERMISSION_MODE` | Runtime permission mode (`accept-all` by default; also `auto`, `manual`, or `plan`). |

Run `bun run xerxes daemon --project-dir .` to use the native daemon with an explicit workspace.

Interactive sessions start in YOLO mode (`accept-all`). The TUI shows `YOLO ON`
beside the active model while it is enabled; `/yolo` switches between `accept-all`
and `auto`. Explicit tool-policy denials remain final in every permission mode.

## Embedded configuration

`XerxesConfig`, `ExecutorConfig`, `MemoryConfig`, `SecurityConfig`, and `LLMConfig` validate
typed input at the application boundary. They accept the documented snake_case persistence shape
and expose immutable native values. Prefer an explicit object passed by the embedding application
over global environment mutation.

## Security defaults

Tool execution remains subject to policy, permission, path safety, prompt scanning, and the
configured sandbox boundary. Enabling a model or channel does not enable high-power tools by
itself. Keep network, browser, media, accelerator, and remote-channel integrations behind an
explicit host port.
