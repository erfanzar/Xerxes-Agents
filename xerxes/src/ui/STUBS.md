# Native TUI capability limits and legacy names

Updated 2026-09-15. A rejected legacy RPC name does not imply that the native
capability is missing. The TUI uses the v35 daemon and never delegates to Python.

| Legacy name | Supported workflow or actual limit |
| --- | --- |
| `plugins.manage` | `/plugins list`, `inspect`, `install`, `enable`, `disable`. Mutations require the injected plugin-management host. |
| `skills.manage`, `skills.reload` | `/skills` refreshes discovery; inspect, diagnostics, trust, search, browse and local install are available. |
| `reload.mcp` | `/reload-mcp`; `/config mcp` edits settings and `/mcp` inspects health/reconnects. |
| `process.stop` | `/terminals` controls advertised writable/interruptible/killable processes; `/runs` cancels supported runs. |
| `tools.configure` | `/preset manage` edits composition tools. `/permissions` controls the active permission mode. No generic legacy mutation RPC. |
| `model.disconnect`, `model.save_key` | `/providers` and `/model` manage profiles and model selection. |
| `rollback.list`, `rollback.diff`, `rollback.restore` | `/snapshots` and `/rollback` preview and restore supported snapshots. |
| `terminal.resize` | No daemon terminal-geometry endpoint. TUI resizing does not imply remote PTY resizing. |
| `voice.toggle`, `voice.record` | No native microphone capture/transcription implementation. `/voice` may report an embedding host's UI action; it is not proof of capture. |
| `reload.env` | No live environment reload; restart an idle daemon after changing its environment. |
| `delegation.status`, `delegation.pause`, `subagent.interrupt` | No targeted native delegation pause/interrupt endpoint. `/agents` inspects and retries supported failed agents; cancellation of the containing turn remains available. |
| `spawn_tree.save/list/load` | No persisted legacy spawn-tree RPCs. Native saved agent manifests and local replay do not recreate a complete historical child-tool timeline. |
| `forge.stop` | Forge templates execute synchronously. There is no asynchronous Forge run to cancel. |
| Clipboard/drop helpers | Local terminal features, not daemon capabilities. |

Session creation/activation/resume maps to `initialize`; prompt submission maps
to `turn.submit`; interruption maps to `cancel`; steering maps to `steer`.
Approvals and questions use the request-owned native response endpoints. Model
and reasoning controls use native provider/profile RPCs. Shell results retain
stdout, stderr and exit status; failures do not become successful no-ops.

See [the capability matrix](../../../docs/daemon-tui-gaps.md) for the current
TUI parity audit and its verification status. External-provider, remote-machine,
and platform acceptance must be distinguished from deterministic tests.
