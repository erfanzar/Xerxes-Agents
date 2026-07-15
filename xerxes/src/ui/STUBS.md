# Xerxes TUI Stubbed RPCs

The TypeScript TUI contains controls whose daemon-side behavior is not exposed
by the Bun runtime. Unsupported compatibility calls fail explicitly with a
typed error (or, for bang commands, a nonzero result) so the UI never invents
a successful no-op response.

OpenTUI and the Bun daemon are the only supported TUI/runtime path. This
document records the remaining unsupported control surfaces for that path.

## Explicitly Stubbed in the TUI

- `terminal.resize`: needs a daemon-visible terminal geometry update endpoint.
- `shell.exec`: bang-command execution is not exposed through daemon slash.
  The UI reports an explicit nonzero result instead of a false successful
  exit; use an agent turn with the native process tool surface instead.
- `image.attach`: needs a file/image attachment endpoint.
- `clipboard.paste`, `paste.collapse`, `input.detect_drop`: local TUI helpers only.
- `voice.toggle`, `voice.record`: needs voice capture/transcription endpoints.
- `plugins.manage`: needs plugin install/enable/disable RPCs.
- `skills.reload`, `skills.manage`: needs structured skill reload and management RPCs.
- `delegation.status`, `delegation.pause`: needs delegation campaign state endpoints.
- `subagent.interrupt`: needs targeted subagent cancellation by id.
- `spawn_tree.save`, `spawn_tree.list`, `spawn_tree.load`: needs persisted spawn-tree endpoints.
- `process.stop`: needs targeted long-running process control endpoint.
- `reload.mcp`, `reload.env`: needs explicit MCP/env reload endpoints.
- `rollback.list`, `rollback.diff`, `rollback.restore`: needs rollback/snapshot RPCs.
- `tools.configure`: needs runtime tool allow/deny configuration RPCs.
- `model.disconnect`, `model.save_key`: needs provider credential management RPCs.

## Supported by the Bun v35 Cutover

- `session.create`, `session.resume`, `session.activate` map to `initialize`.
- `prompt.submit` maps to `turn.submit`.
- `session.interrupt` maps to `cancel`.
- `session.active_list`, `session.list`, and `session.status` use their v35
  daemon methods directly.
- `runtime.status` exposes the Bun runtime build/protocol metadata.
- `session.steer` uses native `steer`, including safe-boundary injection during
  an active turn and next-turn persistence while idle.
- `session.compress`, `slash.exec`, and `command.dispatch` use native slash
  handling for the documented Bun command subset.
- `approval.respond` and `clarify.respond` resolve native pending requests.
- `complete.path` and `complete.slash` use native completion.
- `model.options` uses native `provider_list`; profile mutation uses
  `provider_save`, `provider_select`, `provider_delete`, and `fetch_models`.
- `config.set` model/mode changes use `runtime.reload` / `set_mode`.

Unsupported slash/plugin/skill commands return an explicit Bun-daemon result;
they are not delegated to Python.
