# Daemon capabilities not fully exposed in the TUI

Source audit: 2026-09-15. This records confirmed gaps, not an exhaustive claim of
parity or live acceptance. RPC aliases and slash-command equivalents count as
support; missing literal method names alone do not prove a missing feature.

| Capability | Daemon | TUI gap |
| --- | --- | --- |
| Durable run evidence | `run.events` returns cursor-paged persisted events | Run inspection shows retained output and reaction health, but does not browse this event stream. |
| Declarative Forge packages | `forge.list`, `inspect`, `run`, `define`, `undefine` | `/creator-trace` shows session trace; no equivalent package management workflow. `forge.stop` itself reports unsupported synchronous cancellation. |
| Full agent compositions | `agentPreset.read`, `write`, `openDocument` | `/preset` supports listing, selecting, copying, defaulting and removing. It does not directly read/edit the full composition or open its document. The separate project specialist editor is supported. |

Do not confuse these with shared runtime limits: targeted `subagent.interrupt`
is unavailable on the native path, and saved agent snapshots do not replay a full
historical child tool timeline. Neither is evidence of a daemon feature hidden
only by the TUI.

`src/ui/STUBS.md` under `xerxes/` describes legacy compatibility method names.
Some descriptions are outdated: MCP reload is available through `/reload-mcp`,
and process inspection/cancellation uses `terminal.control` and run controls.
Likewise LSP status/release have slash equivalents. Those legacy names should
not be used as a current feature-gap checklist.

Evidence: `xerxes/src/daemon/server.ts`,
`xerxes/src/ui/app/slash/commands/presets.ts`,
`xerxes/src/ui/app/slash/commands/integrations.ts`, and the TUI run overlay and
gateway implementation. Windows and external vendor acceptance remain separate
verification gaps documented in `desktop-parity.md`.
