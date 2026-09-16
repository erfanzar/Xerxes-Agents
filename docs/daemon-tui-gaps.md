# Daemon → TUI capability matrix

Audit dated 2026-09-15. This matrix separates implemented TUI paths from
runtime limitations and unavailable external acceptance. Sources below are repository-relative. Native acceptance
means keyboard interaction with the built OpenTUI client connected to an isolated
real `DaemonServer`; seeded data is identified separately from external calls.

## Confirmed gaps addressed in this work

| Requirement | Source and reachable workflow | Evidence / remaining checks |
| --- | --- | --- |
| Durable event history | `daemon/server.ts` `run.events`; `ui/opentui/runEvents.tsx`: `/runs`, V, N/P pages, R retry, Esc return. Cursor advances only after success; reads never acknowledge. | `ui/__tests__/runEvents.test.tsx`: invalid frames, paging, errors, retry, stale replies, 140/80/40 columns. Actual built TUI pages 1/2 at 120×36, 60×24, 180×48 captured. Native socket loss preserved page 2; page 3 loaded after reconnection. |
| Forge list/inspect/run/define/remove | `extensions/declarativeForge.ts`, `daemon/server.ts` `forgeRpc`; `ui/opentui/forgeOverlay.tsx`: `/forge`, input forms, definition review, parameter declarations, confirmed removal, readable output. | `ui/__tests__/forgeOverlay.test.tsx` (5 passed); `test/daemonServer.test.ts` Forge RPC contract passed including persistence, immutable versions, required input and confirmation. Native browse/run/output captured at 120×36 and 40×18. Native definition review/save and confirmed removal verified against the persisted package file. |
| Full composition editing | `agents/presets.ts`, `daemon/server.ts` `agentPresetRpc`; `ui/opentui/presetEditor.tsx`: `/preset manage`, E edit, Ctrl+S save, O document folder; shipped presets remain read-only and can be copied. | `ui/__tests__/presetEditor.test.tsx`, preset command and daemon contract tests passed. Native editor saved YAML through RPC and resulting file was checked. Drafts are scoped by session/preset; optimistic-content conflict guard prevents overwriting a changed document. |
| Workspace file preview | `daemon/filePreview.ts`; `/file <path>` in `ui/app/slash/commands/files.ts`, numbered text in existing pager, bounded-preview notice. Daemon enforces workspace/symlink/UTF-8 restrictions. | Focused command tests passed for multiline/truncation/errors. Native keyboard capture saved in the evidence directory. |
| Reverse recorded edits | `daemon/server.ts` `undoChanges`; `/undo-edits <exact recorded path|--all>`, explicit confirmation, per-file partial results. This is not general Git discard. | Focused command tests passed: confirmation, stale-session refusal, partial errors. Native confirmation and result captured; the isolated file changed from “After edit” back to “Before edit”. |
| Incremental terminal output | `daemon/server.ts` `terminal.output`; `ui/opentui/terminalOutputPages.tsx`: `/terminals`, open terminal, V, N/P/R; retention losses shown. Existing tail view remains available for legacy records. | Focused pager tests passed at 140/80/40 columns, failure retains current page and retries cursor. Native output pages 1/2 captured; resizing from 40 to 180 columns survived after the root-clipping fix. |
| Terminal errors/input | `ui/opentui/terminalPanel.tsx`: list/output polling failures now visible; failed writes retain input and duplicate pending writes are blocked. | Existing terminal panel tests passed; write failure/duplicate submission regression passed; native output pages and repeated 40/60/120/180-column resize verified. |
| Command argument discovery | `ui/hooks/useCompletion.ts`, `daemon/server.ts` `complete`: daemon argument completion now reachable for config/plugins/skills/Forge/presets; full command inserted. | Focused completion tests passed; native filename completion inserts the complete /file command. |
| Scoped transcript search | `daemon/server.ts` `session.search` and slash `searchTranscripts`: `/search --session <id> --limit <count> <text>`. Existing global search preserved. | Daemon search tests passed, including scope/limit and invalid-limit rejection; native scoped results displayed in the scrollable search pager. |

All `daemon/`, `ui/`, `agents/`, and `extensions/` source paths above are under
`xerxes/src/`; test paths are under `xerxes/`.

## Existing workflows traced during audit

These rows establish reachability. The full UI suite exercises these existing workflows. A source path establishes
reachability; deterministic test coverage is distinct from live external acceptance.

| Daemon surface | TUI route / source | Audit status |
| --- | --- | --- |
| initialize, session.open/list/active_list/status/most_recent/delete/save/title/undo | `ui/gatewayClient.ts`, session picker, `/resume`, `/new`, `/save`, `/title`, `/undo`; `ui/app/useSessionLifecycle.ts` | Session lifecycle, resume, queue and gateway suites; native saved-session resume and reconnect verified. |
| turn.submit/background/cancel/steer, cancel_all, mode/plan/model/reasoning | Composer, steering/interrupt, `/background`, `/stop`, `/cancel-all`, model/mode controls; `gatewayClient.ts`, slash commands | Aliases traced; turn/controller and native daemon contract suites. |
| permission_response, question_response | Owned approval/clarification dialogs, `createGatewayEventHandler.ts`, `gatewayClient.ts` | Request IDs preserved; clarification, approval and daemon interaction contract suites. |
| session.goal, goal.inspect/decision | `/goal`, F10 goal panel, `goalOverlay.tsx` | Criteria, evidence, user decisions, token usage and continuation views exist. |
| session.usage, context_breakdown, context.inspect/control | Usage/status and `/context`, `contextOverlay.tsx` | Native structured context controls exist. |
| workspace.diff/worktree/list/checkApply/apply/integrations/recover | `/diff` F7; `/worktree`; `/workspaces` and recovery panel | Review/apply uses reviewed state and explicit confirmation. |
| terminal.list/inspect/control | `/terminals` F8, `terminalPanel.tsx`, `lib/terminals.ts` | Tail, filtering, searching, input and advertised control actions exist. Cursor gap addressed above. |
| background.activity/status | `/activity`, composer summary, `activityOverlay.tsx` | Counts and actionable rows refresh on invalidation events. |
| run.list/inspect/acknowledge/cancel | `/runs`, `runOverlay.tsx`, `lib/runs.ts` | Filters, workspace/session scope, paging, revision-owned actions, reaction health and token usage exist. Identity, revision, source, start/end times, terminal kind and exit code are now also readable in the inspector. |
| monitor.sources/list/inspect/create/update/stop | `/monitors`, `monitorOverlay.tsx` | Source forms and reaction controls exist; external sources require live resources. |
| schedule.* | `/schedules`, `/loop`, `scheduleOverlay.tsx` and schedule helpers | CRUD, preview, pause/resume/run/cancel, deliveries and history exist; aliases traced. |
| snapshot.* | `/snapshots`, `/rollback`, `snapshotOverlay.tsx` | Timeline, diff, guarded file restore and restore-attempt history exist. |
| capabilities.*, tool.inventory | `/features`, tool picker, `capabilitiesOverlay.tsx` | Inspection and inventories exist. |
| agent.settings.*, model.routing_note.* | `/config agents`, `agentSettingsOverlay.tsx` | Tier settings, profile selection and routing notes exist. |
| provider_*, fetch_models, reasoning_levels | `/providers`, `/model`, reasoning picker | Credential mutations stay daemon-side; live provider acceptance separate. |
| mcp.status/settings.*, slash mcp/reload-mcp | `/mcp`, `/config mcp`, `/reload-mcp` | Legacy `reload.mcp` rejection is not a missing native reload workflow. |
| lsp.status/release/settings.* | `/lsp status`, `/lsp release`, `/config lsp` | Slash equivalents traced. |
| channel.list/enable/disable | `/channels`, `ui/app/slash/commands/integrations.ts` | Host errors remain explicit; no external test destination available. |
| agentPreset.project* and remaining composition actions | `/custom-agents`, `/preset use/default/copy/remove`, `customAgentEditor.tsx`, `presets.ts` | Project editor remains distinct from full composition editor. |
| skills, plugins, hooks, browser and machine slash actions | `/skills`, `/plugins`, `/hooks`, `/browser`, `/machine`; corresponding overlays/slash commands | Local management exists; browser attaches only to supplied endpoint; remote acceptance requires host. |
| daemon.wipe_memory/history | `/remove-memory`, `/remove-history`, `maintenance.ts` | Explicit destructive confirmation exists. |
| runtime.status/update_status/restart_if_idle/shutdown | Gateway lifecycle, `/daemon`, `/update`, `/restart`; `ui/app/slash/commands/daemon.ts` | Added /daemon status with daemon identity/readiness/fields, /daemon stop with global confirmation, and guarded /restart. Native busy restart refused without stopping the stream. Accepted explicit shutdown exits this TUI rather than triggering crash recovery. |
| creator_trace | `/creator-trace` | Existing formatted session trace; independent from package management. |

## Event and restoration audit

`ui/gatewayClient.ts` maps native events into `ui/app/createGatewayEventHandler.ts`.
The handler covers message/reasoning deltas, tool generation/start/progress/result,
approvals/questions, subagent lifecycle/progress/results, notifications/errors,
background invalidation, usage/status and session titles. Session reattachment
restores todos, goals, transcript and saved agent manifests. Native reconnect acceptance found and fixed an additional lifecycle gap: same-session recovery now keeps inspector keys, cursor pages and non-bottom transcript position, while clearing stale interaction authority. `turnController.test.ts`, `createGatewayEventHandler.test.ts` and `gatewayClientLifecycle.test.ts` cover the boundaries.

## Genuine limits / unavailable acceptance

- `forge.stop`: synchronous text-template execution has no asynchronous job to stop.
- Targeted subagent interruption and complete persisted child tool replay are not
  native daemon capabilities; do not label them TUI-only omissions.
- Terminal output retention is bounded; lost output is reported, not reconstructed.
  Legacy archived runs may have only a tail, without an incremental cursor.
- Native terminal geometry RPC and microphone capture are unavailable.
- Windows acceptance cannot run on this Mac without a Windows host.
- External provider/MCP/channel/SSH acceptance has no designated test resources.
  Deterministic injected tests are not live external acceptance.

## Captures and checks

Current terminal captures live outside the repository at
`/Users/erfan/Documents/Projects/xerxes-desktop-verification/tui-parity/`.
The acceptance host is `xerxes/test/fixtures/daemon/tuiParity.ts`: isolated home,
socket, sessions, run database and scheduler lease. It uses seeded records and
an in-memory runtime; no paid provider or external message is sent.

Full root `bun run check && bun run test && bun run build`: passed in the September 16 review. Runtime: 3,899 passed, 3 skipped; TUI: 1,376 passed across 139 files. `git diff --check`: passed. The review caught a schedule-form test typing before a pending save finished; stop-condition and token-threshold tests now wait for the form to become editable before sending their next edit. The focused 19-test schedule suite and subsequent full gate passed.

The review also rebuilt and signature-verified the macOS development package. Native
desktop acceptance exercised six workspace switches in one window, retained drafts
and empty folders, and confirmed both workspace streams stayed active with blue
indicators. `test/fixtures/daemon/globalClients.ts` passed again with a local provider:
desktop and TUI shared one daemon, workspace instructions stayed isolated, an idle
legacy daemon migrated, and detaching the TUI left the shared daemon running. An
actual built TUI at 80×24 also opened Forge, submitted its parameter form, and
displayed the executed template output. These checks do not establish live external
provider or Windows acceptance.

## Additional audit findings and their resolution

- Full received tool output was previously reduced to 600 characters in the gateway. The compact summary remains bounded; the original result now survives in the tool record. Expand a tool or use `/tool-output list|last|number|id`. Native acceptance reached line 180; archived-tool mapping and adapter tests cover old/current-turn separation. Historical daemon replay that only supplies a summary cannot recover missing text.
- Run records carried meaningful identity and timing fields that the inspector discarded. Source/session IDs, revision, started/ended times, terminal kind and exit code are now available in its scrollable detail.
- File preview, recorded-edit reversal and terminal output now explicitly bind a supplied session ID to the native session key in the gateway. The gateway boundary regression prevents connection-default ownership from winning.
- Native terminal resizing exposed an OpenTUI buffer-drawing crash. The renderer root now clips intermediate resize frames. Repeated actual terminal resizing from 40 to 180 columns passed after reproducing the original crash.
- Search uses the native scoped search endpoint and a pager, including result limits, conversation IDs/resume instructions and index-coverage warnings.

## Returned fields and event review

The audit traced the dispatch branches in `daemon/server.ts`, including prefix-dispatched Forge/composition methods and schedule delivery actions. Lifecycle, permissions and revision fields remain attached to their requests instead of becoming decorative UI labels. Results expose actionable identities, errors, partial/truncated coverage, usage, continuation state, delivery/recovery state and paging cursors through the workflows above. Credentials remain daemon-owned.

Actual wire event producers were checked against `gatewayAdapter.ts` and `app/createGatewayEventHandler.ts`: streamed text/reasoning, tool call/progress/results, approval/question ownership, status/retry/compaction, subagent lifecycle/results, replay notifications, plan output, session title and background completion. Steer/approval/question response frames are acknowledgements of controls whose result is already represented by their owning workflow. Older protocol-table names such as image/audio/video URL parts and hook/step begin/end are not emitted by the current native daemon/streaming producer; they are not independently implemented runtime capabilities.

Existing regression families include `sessionLifecycle`, `gatewayResumeIntegration`, `sessionQueue`, `runOverlay`, `monitorOverlay`, `monitorCreate`, `monitorPolicy`, `scheduleOverlay`, `scheduleForm`, `snapshotOverlay`, `workspaceOverlay`, `contextOverlay`, `agentSettingsOverlay`, `capabilitiesOverlay`, `gatewayAdapter` and `gatewayClientLifecycle` under `src/ui/__tests__`. Daemon, production-runner, provider, MCP, channel, schedule, workspace and security contracts run in the full runtime suite.

Production host integration: `test/fixtures/daemon/globalClients.ts` passed with a local deterministic provider. The real TUI gateway and desktop client shared one daemon, completed simultaneous turns with isolated project instructions, migrated an idle legacy daemon, and left the shared daemon running when the TUI detached. This does not establish acceptance against an external provider.

The final explicit-shutdown acceptance also caught a recovery UX issue: a deliberate daemon stop was treated as a transport crash. Accepted lifecycle actions now close the initiating TUI after acknowledgement; rejected/busy actions leave it running. The full TUI suite, UI typecheck and TUI build were rerun after this final client-only correction.
