# Xerxes TUI Stubbed RPCs

The TypeScript TUI contains controls for features that the current Xerxes daemon
does not expose yet. The gateway returns safe empty/no-op responses for these
calls so the UI renders without crashing.

## Stubbed Until Python Endpoints Exist

- `terminal.resize`: needs a daemon-visible terminal geometry update endpoint.
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
- `browser.manage`: needs browser session management endpoints.
- `rollback.list`, `rollback.diff`, `rollback.restore`: needs rollback/snapshot RPCs.
- `tools.configure`: needs runtime tool allow/deny configuration RPCs.
- `model.disconnect`, `model.save_key`: needs provider credential management RPCs.

## Mapped To Existing Xerxes RPCs

- `session.create`, `session.resume`, `session.activate` map to `initialize`.
- `prompt.submit` maps to `turn.submit`.
- `session.steer` maps to `steer`.
- `session.interrupt` maps to `cancel`.
- `slash.exec`, `command.dispatch`, `shell.exec` map to `slash`.
- `approval.respond` maps to `permission_response`.
- `clarify.respond` maps to `question_response`.
- `complete.path`, `complete.slash` map to `complete`.
- `model.options` maps to `provider_list`.
