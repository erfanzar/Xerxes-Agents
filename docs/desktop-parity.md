# Desktop and TUI capability parity

Parity means equivalent user actions, state visibility, recovery, and keyboard access. Desktop layout follows native window conventions; it does not reproduce terminal glyphs. A source-level implementation is not live verification.

## Inventory

| Capability | Desktop surface | Current assessment |
| --- | --- | --- |
| Conversation, streaming, tool arguments/results | Conversation | Implemented; populated browser checks exist |
| Agent identity, status, summary, usage, touched files | Activity | Implemented; snapshot fields retained |
| Live per-agent reasoning, progress, paired tool calls | Activity agent details | Implemented through recursive `subagent_event`; deterministic event and browser tests passed. Historical snapshots do not replay a full tool timeline |
| Agent retry/follow-up and stop | Activity agent details | Implemented; accepted/rejected retry and unsupported stop verified with injected runtime responses. Native retry now verified through the production daemon and local HTTP provider: initial 400 failure, retained follow-up, completed retry and persisted completed snapshot. External vendor retry remains unverified |
| Parent/child agent relationships | Activity agent details | Parent identity shown; recursive child-event isolation tested |
| Sessions, global history/search, resume, rename/export | Sidebar and search | Current browser acceptance passed for rename/rejection retry, active-session/draft preservation, delayed 160-message resume, switching and reading-position retention. Native resume/reload and Markdown download verified against a copied populated session. |
| Independent workspaces | Native windows, Workspace | Local window workspace/session, normal bounds and maximized state restore across quit/relaunch. Native two-workspace populated test passed, including individual close and stable shared-daemon PID. Native fullscreen restoration also passed with two populated workspace windows; Native SSH identity/session restoration succeeded on a second launch; the first launch rendered blank and remains an intermittent defect. |
| Composer, models, reasoning, commands and skills | Composer and discovery | Current browser acceptance passed for long drafts, model/effort menu bounds and focus at 760/1200/1800px, 167-skill filtering and complete command insertion. |
| Approvals, questions, plan decisions | Inline conversation | Plan-review keyboard approval and revision feedback verified with single-response assertions; overlapping generic handler fixed. All three tool-approval responses and batched option/freeform rejection/retry verified in browser fixtures |
| Goal inspection and scoped decisions | Goal → Criteria & evidence | Implemented using the shared TUI inspection parser. Browser verified rejection/note retention, retry and human evidence. Native human acceptance verified on an isolated copy of a saved session. |
| Context inspection and controls | Activity → Inspect context & memory | Five sections, paging, generation validation and memory pin/exclude controls implemented. Browser interactions verified; native read-only inspection verified on a resumed session. Native exclusion/inclusion verified against seeded controls on a copied session; persisted revision 3 was independently reopened. |
| Files, changes, checkpoint/snapshot restore | Files, Changes, Snapshots | File-preview races/errors/keyboard scrolling verified in browser. Native snapshot capture, preview and file restore verified with backup/journal checks. Shared numbered diff checked at three widths. Native 650-line file and 1,300 changed-line diff verified at three widths. Real macOS Cancel and OK confirmation actions verified in a disposable repository; Cancel preserved the edited file, OK restored the snapshot, and the completed recovery journal and backup were independently checked. |
| Agent presets/custom agent authoring | Agents and Settings | Current browser acceptance passed for specialist editing, invalid-write draft retention, save/reopen and insertion into conversation. Delegation description is shown separately from full instructions. Native generation through a local provider, malformed-output recovery with retained description, review-before-save, persistence and reopening verified. |
| LSP configuration | Settings → Language servers | Create/edit/remove/release implemented. Masked launch values are preserved; focused contract tests and normal/narrow browser interactions passed. Native enabled clangd configuration, actual hover request and host release verified; separate installed-server navigation and versioned diagnostics test passed. |
| MCP, providers, channels, permissions | Settings | Browser channel/MCP rejection and recovery, provider save refusal/draft retention/retry verified. Approval responses separately verified. Native provider filesystem refusal retained its draft and displayed the error. Native MCP stdio discovery, reconnect, failed launch and recovery verified with an actual local subprocess. Remote provider/channel/MCP services remain unverified. |
| Scheduled jobs and runs | Scheduled jobs → Deliveries | Delivery inspection, explicit send, uncertain-result resolution and confirmation implemented. Browser fixture and TUI delivery tests passed; Native delivery integration verified through a local HTTP channel: Cancel transmitted nothing, 503 produced uncertain state, explicit retry sent the output, and persisted state recorded two attempts. |
| Monitors, creation, policy editing | Activity → Monitors | Terminal/file/WebSocket/configured-webhook creation, inspection/stop and reaction-limit editing implemented with TUI helpers. Browser verified file creation, policy rejection/retry, stop and webhook discovery; native metadata-only file watch start/stop verified with reactions off. Native loopback WebSocket creation, plaintext non-loopback rejection, event reception, reconnect/gap reporting and stop/socket closure verified; Native configured-webhook execution verified through the actual daemon listener with signed requests, unsigned rejection, duplicate suppression and rejection after stop. |
| Unified run history, acknowledgements | Scheduled jobs → All run history | Workspace/session scope, kind/state/unread filters, paging, results, acknowledgements and supported cancellation implemented with shared TUI RPC helpers. Browser interactions and native inventory verified; native file-watch cancellation and acknowledgement of a seeded completed run passed, including persisted read state. Native terminal process cancellation now verified; native schedule creation, Run now, output inspection and operator cancellation verified on the same connection. Independent execution preserves the caller session. |
| Workspace review/apply | Workspace → Review isolated work | Managed workspace inventory, numbered diff, guarded check/apply and recovery inspection/actions implemented through TUI helpers. Browser fixture and 16 TUI regression tests passed; native check, stale-apply rejection, successful apply and controlled interrupted-integration recovery passed in a disposable repository; Git index preserved. |
| SSH hosts and remote workspaces | Workspace | Browser bridge acceptance passed for alias discovery failure/manual entry, browsing, refused save with draft retention/retry and visible SSH authentication error. Native connection to the configured SSH host succeeded with strict host-key checks. Saved remote identity/session survived quit/relaunch; a first blank launch remains unresolved |
| Terminals and live output | Activity and Settings | Browser inspect, input rejection/draft retention/retry, explicit output refresh and interrupt passed at two widths. Native real background process output and cancellation through run history verified. Native PTY input/output and Ctrl+C process exit verified; controls update automatically on lifecycle events. Confirmed Ctrl+C exit is classified as interrupted in history and Activity. |
| Native rendering, reload, accessibility | Desktop window | Fresh separately packaged build inspected with a real 53-agent session; expanded Activity and conversation restoration exercised. Earlier blank rendering after replacement of a running bundle remains unresolved |

## Acceptance requirements

- Exercise the same daemon messages and RPC failures as the TUI, without changing v35 public contracts or silently inventing success.
- Preserve session/window isolation, cancellation, queued work, security checks, and drafts.
- Test loaded history, streaming, failures, eight agents, deep paths, and narrow/normal/wide windows.
- Inspect the rebuilt native application and record actual interactions. Browser fixtures alone are insufficient for full parity.
- Keep remaining gaps explicit. This checklist is not a declaration of completion.

## Latest verification

- Latest stable-source root gate: check, 3,880 runtime tests passed (three environment-dependent skips, zero failures), 1,338 TUI tests passed, and build passed. Build 97dfa85333ed2cc2 matches agent-retry-live-package.
- Desktop typecheck and focused suite passed: 217 tests, 996 assertions.
- Browser fixture exercised live event pairing, decoded results, rejected/accepted retry, unsupported stop, expanded Activity, and narrow layout.
- Native QA copy loaded the newly built renderer from `/private/tmp/Xerxes Parity QA.app`; resumed a real saved session at the bottom, opened agent failure details, expanded Activity, and restored the conversation. Active project daemons were not restarted.
- Screenshots and reproducible browser script are under the sibling `xerxes-desktop-verification/rework` evidence directory: `118-agent-timeline.png`, `119-native-agent-workspace.png`, `120-native-resumed-conversation.png`, and `scripts/agent-parity.ts`.

### Continuing parity pass: language servers

Settings now exposes LSP create/edit/remove/release and masked launch-field editing. 63 focused Bun tests, desktop typecheck, runtime/desktop builds, and deterministic browser interactions passed. The rebuilt native app loaded configuration from an isolated real daemon after resuming a copied populated session; Add server opened successfully. Native configuration writes and launching a language server were not exercised. Evidence is in the sibling `xerxes-desktop-verification/parity-final` directory. This does not supersede the remaining gaps above or constitute a full root gate.

### Packaged verification pass

Latest native acceptance used an isolated home and a copy of a real saved session: disabled LSP configuration create/edit/remove, goal criterion acceptance, metadata-only file monitor start/stop, run inventory and managed-workspace inventory. The renderer and daemon were rebuilt together and build identity verified (`7c2d509a41a95944`). Stored tool results opened in the rebuilt native app. Notification-only history replay now has typed rows with explicit preview-only output; its separate branch has deterministic regression coverage.

A new macOS arm64 DMG was built in the sibling `xerxes-desktop-verification/parity-final/package` directory, preserving running bundles. Image checksums passed and the mounted Finder layout was inspected. It is ad-hoc signed, not Developer ID signed/notarized. Native Windows verification is unavailable: the user has no Windows machine or VM.

### Current browser acceptance sweep

243 desktop tests passed across 28 files (1109 assertions). Re-ran 13 existing interaction scripts against the current compiled fixture and added discovery/authoring acceptance. All 14 flows passed after updating stale test selectors to distinguish toolbar controls from inspector tabs and scope todo text to its section. The responsive test now checks usable conversation width rather than assuming the old fixed 850px inspector cutoff; the runtime-popover flow closes explicit full-width Activity before checking its retained draft. These are fixture interactions, not live provider/integration claims.

Coverage includes session rename/retry, 160-message resume, model/effort anchoring and keyboard focus, eight agents and paired tool output, retry rejection/acceptance and unsupported stop, single compaction failure/recovery without resubmitting user work, queued-message hiding, fixed goal/todo navigation, structured output/raw details, long output expansion/wrapping, wide tables, Activity/statistics defaults and manual choices, runtime restart gating, offline recovery, high-contrast fallback, 167-skill discovery and specialist editing. Screenshots and scripts: sibling `xerxes-desktop-verification/parity-final/acceptance`.

Historical per-agent timing absence is shared with the TUI, whose inspector reports that older agents ran before timing recording; a complete historical timeline is not claimed by either client.


### Native workspace integration and current installer

The current packaged renderer was exercised through a real isolated daemon and disposable Git repository. Check/apply rejected destination drift without replacing the newer edit; a fresh check and confirmation applied the reviewed patch. A deliberately injected interrupted-integration record was inspected and recovered through the UI. Original files were restored and the Git index remained byte-for-byte unchanged. This verifies recovery of the controlled crash boundary, not an actual process crash in this native run. The existing killed-worker contract tests cover process termination. Screenshots were inspected.

The installer in sibling `xerxes-desktop-verification/parity-final/current-package` includes the latest Activity history grouping (build `32279c2bed37fc0c`). Packaging, deep strict signature verification and DMG checksums passed. It remains ad-hoc signed and not notarized. Windows native verification remains unavailable.


### Native run controls

Native packaged run-history controls stopped a real metadata-only file watch and reached cancelled state. A completed run seeded with the runtime RunHistory API was acknowledged through the real daemon; reopening its SQLite history verified unread=false. No terminal command or external message was sent for that seeded record. An immediate stale Unread label found during acceptance was fixed: confirmed detail now updates the corresponding list row and respects active filters. The current packaged renderer passed the immediate-row assertion.

Desktop typecheck, three focused run-history tests (17 assertions), native acceptance, packaging, strict signature verification and disk image checksums passed. Build 371d3ff2f70d4df5 is available in run-feedback-package. No full root gate was rerun for this small renderer-only change. Scripts/logs and native-run-acknowledged.png are retained here.


### Files and snapshot acceptance

File preview browser interactions passed at 720 and 1200px: delayed-response isolation, read-error recovery, 240 retained lines and keyboard scrolling. Native packaged snapshot capture, file selection, preview and restore passed against a disposable real Git workspace/daemon. The harness accepted window.confirm after asserting its file/backup wording; the native OS confirmation dialog itself was not exercised. The restored content, completed recovery journal and backup containing the replaced content were independently checked.

Snapshot and managed-workspace previews now share a numbered, highlighted, keyboard-focusable DiffPreview component. Native captures at 760/1200/1800px passed viewport bounds and were inspected. Six focused rendering/layout tests (27 assertions), desktop typecheck and 30 snapshot contracts (146 assertions) passed. The snapshot suite covers stale revisions, file-scoped preservation, durable recovery and killed workers. Current installer build 755e706f813abd4b in snapshot-package passed packaging, signature and disk-image verification.


### Large native files and plan keyboard audit

The packaged app browsed a deep real path with 650 source lines and rendered a 1,300 changed-line diff with line numbers. Native 760/1200/1800px windows retained bounded scrolling after layout settled; screenshots were inspected. All work used a disposable repository and isolated daemon.

Plan-review keyboard handlers overlapped with the generic question card. The generic handler now stays inactive during plan review. Browser interaction checks assert exactly one question_response for Enter with revision feedback and for keyboard approval, with the expected answer payload. 54 desktop shell tests (312 assertions) and desktop typecheck passed. The current downloadable installer still predates this keyboard-only fix; a new full check/test/build gate is in progress and packaging will follow.


### Approval and batched-question interactions

All three approval keyboard choices sent the correct wire response (approve, approve_for_session, reject); injected rejection kept the card available and retry succeeded. Batched option/freeform questions retained the custom answer after rejection and submitted the correct payload on retry. These are deterministic browser interactions, not live provider requests. Plan keyboard approval/revision was verified separately.

The settings/offline audit found stale per-project-daemon wording. Offline help now describes the shared daemon with task-local workspaces/sessions; Permissions correctly identifies current-session scope. No runtime behavior changed. Full runtime gate completed with 3,863 passed and zero failures; the same gate is now running TUI checks. Final packaging awaits completion.


### Latest completed full gate and package

Root check, 3,863 runtime tests (zero failures), 1,338 TUI tests across 132 files, and root build passed. The final copy-only changes also passed desktop typecheck and 54 shell tests. Installer build cf6da207b16d905f includes the plan-keyboard and daemon-wording corrections; signature verification and image checksums passed. The packaged renderer resumed a copied populated session and expanded saved tool results. Its isolated daemon reported protocol 35 and the matching build ID; no provider was configured in that test home, so this was saved-session native acceptance rather than a live-provider turn. The isolated daemon was shut down after testing.

Current installer: approval-package/Xerxes-Agents-0.4.5-macOS-arm64.dmg. Ad-hoc signed, not notarized; native Windows remains unavailable. Full feature acceptance is still tracked in docs/desktop-parity.md.


### Settings recovery and provider draft fix

Browser interactions at 760/1200px verified channel rejection/enable/disable, MCP reload rejection followed by status refresh, terminal input rejection with draft retention, successful input with explicit output refresh, and interrupt. All used controlled fixtures; no external gateway or terminal process was operated by those browser checks.

ProviderForm previously closed optimistically before provider_save completed. It now waits for a confirmed save, retains all draft fields on refusal, and shows the error next to Save & activate. Store returns a typed error message for refused/invalid/mid-turn saves. Browser refusal/retry and 120 focused store/shell tests (566 assertions) passed with desktop typecheck. The new regression compares initialization count with its setup baseline; its first assertion incorrectly assumed no setup initialization and was corrected.

Build 0f22bc738f1dc64c in provider-package includes the fix. Installer and strict signature checks passed. Its native renderer reopened a copied populated session; the isolated daemon reported the matching build ID. The native provider-save interaction itself remains unverified. The previously completed full root gate predates this local provider-form change.


### Native context, provider refusal and export

Native memory exclusion and inclusion saved through the real daemon. Reopening the copied session verified revision 3 with no excluded entries or pins: exclusion removes a pin, and inclusion does not repin it. Native provider save encountered a real EISDIR filesystem refusal in a disposable home; name/model drafts remained, and the inline error screenshot was inspected. The first capture was invalid because the settings form had disappeared; the clean rerun passed without focusing the test window. No provider request was sent.

Electron downloaded the copied populated session as Review-and-Understand-the-Project.md (362,736 bytes). The test selected an isolated output path through Electron will-download and checked the saved Markdown title/content. It did not exercise an OS save dialog. The daemon reported build 0f22bc738f1dc64c and protocol 35. Native test daemons were shut down.


### Native terminal cancellation correction

A real disposable Bun process was launched by the production executor through a loopback scripted provider. Native Settings displayed its output. Stop process terminated it, but the first acceptance run exposed failed instead of cancelled history. TerminalRegistry now records confirmed cancellation while retaining the exit code, waits for the signalling result when exit races with it, and leaves rejected stop errors observable. Activity uses that same cancellation outcome without changing terminal wire summaries.

40 terminal/background tests passed before the Activity follow-up, then 15 terminal tests passed including persistence, archived state, ownership isolation, exit-during-signal and rejected cancellation. Native retest passed output, cancelled run history and no failed Activity count. The isolated daemon confirmed build 8b8c6a6c2e29523a. Screenshots inspected; temporary daemon shut down. Strict package signature and DMG checksums passed. terminal-final-package contains the updated ad-hoc signed installer, not notarized.

The prior root gate completed with 3,864 runtime tests, 3 skips, 0 failures and 1,338 TUI tests plus check/build. A new root gate is running after this runtime correction; its results are not yet claimed.


The post-cancellation full gate exposed two stale daemon assertions (expected succeeded/completed after an explicit stop) and a Git-hook test startup issue. Daemon assertions now expect cancelled and both focused cases passed. On this Mac, launching a newly written executable could consume the 800ms timeout before the shell trap ran; the hook test now invokes /bin/sh explicitly and requires positive helper output before checking that it stops growing. Its focused rerun passed with all five assertions. The full gate is being rerun; no pass is claimed yet.

Native PTY input/interrupt acceptance remains open. The production default agent request offered exec_command and ToolSearchTool but not pty_open; the attempted scripted call was correctly rejected as not configured. Investigation should reconcile default-agent terminal tool names with the native registrations before claiming interactive acceptance. No tool restriction was bypassed.


### Native interactive terminals

Built-in execution profiles referenced older operator terminal names while the Bun daemon registered native PTY/background tools. The native names are now included in source YAML, bundled fallback and Bash alias expansion; older names remain available for other hosts. Reviewer allowlists remain restricted. 41 agent/prompt/compatibility tests passed. The prompt stays within its existing size ceiling.

A real /bin/cat PTY was opened through the production executor using a local scripted provider. Settings sent NATIVE_PTY_INPUT, showed its output, and Ctrl+C terminated it with exit 130. The first native test exposed stale live controls after asynchronous exit; the open terminal view now listens to daemon lifecycle events, and stale list responses cannot overwrite a newer state. 121 desktop store/shell tests passed, including delayed-refresh regression, plus desktop typecheck. The rebuilt native retest passed with daemon build c2a3e923cb93a335; screenshot inspected. Ctrl+C currently yields a failed history record despite the intentional interrupt and remains a classification follow-up.

pty-live-package contains this build. The fresh full gate is running after these changes; completion is not claimed.


### Confirmed interrupt and installed language server

Terminal history now records confirmed Ctrl+C with exit 130 as interrupted. A refused interrupt remains an error; a process that continues and later exits with another failure code remains failed. Completion waits for an in-flight control acknowledgement, including rejection. 41 terminal/background tests passed (164 assertions), and runtime typecheck passed. The native app verified PTY input, Ctrl+C exit and interrupted run history. The packaged daemon matched a62121edc97438f9.

The installed Apple clangd 21.0.0 passed the opt-in real-server test: hover, definition, references, symbols, versioned edit diagnostics and shutdown (12 assertions). Native Settings then created an enabled clangd configuration; the production executor returned a real hover result for square in a disposable C++ file, and Release host succeeded through the UI. Screenshots inspected. No download or external provider was used.

The preceding root gate passed with 3,868 runtime tests, 3 skips, no failures, 1,338 TUI tests and check/build. A fresh gate is running for the interrupt correction. interrupt-package is the latest installer; strict signature and DMG verification passed, ad-hoc signed and not notarized.


### Native schedule execution and cancellation

Scheduled jobs now exposes Run now. Native acceptance found that a pending schedule.run blocked run history and cancellation on the same RPC connection. Run reads and revision-checked cancellation can now bypass that queue after session initialization, while session mutations retain serialization. A new socket regression verifies live inspection, stale-revision rejection, cancellation, separate session ownership and event isolation. The daemon suite passed 158 tests (1,052 assertions before the additional ownership assertion); the final ownership regression and delivery test also passed. Root typechecks passed after the final changes.

Manual independent execution previously reused the caller session and leaked turn events into its transcript. The schedule RPC now uses a separate cron session and existing cron_event framing; legacy slash streaming is retained. Intentional cancellation has a neutral status and retains its reason/output. Other RPC failures still surface.

The rebuilt native app created a schedule, started a real production turn through a local scripted SSE provider, inspected and cancelled it, refreshed the retained output, and left the original chat unchanged. Screenshot inspected. The daemon matched build 4223926fe0a512b9; the isolated test daemon was stopped. schedule-isolated-package is the latest installer. Packaging and DMG verification passed; ad-hoc signing is not notarization. The full suite is still running, so this is not a final completion claim. Windows native testing is unavailable as confirmed by the user.


Native successful schedule completion also passed on build 4223926fe0a512b9, with retained production-turn output and the original conversation unchanged. The local scripted SSE provider supplied the response; no external provider account was used. Both success and cancellation screenshots were inspected.


The final stable-source root gate completed: bun run check, bun run test and bun run build passed; 3,870 runtime tests passed, 3 environment-dependent skips, 0 failures, and 1,338 TUI tests passed across 132 files. git diff --check passed. The root build emitted 4223926fe0a512b9, matching the natively tested installer. The earlier in-flight test run overlapped the isolation edit and failed its newly added ownership assertion; this fresh stable-source run supersedes it. This closes the schedule acceptance gap, not the external/native-platform and other inventory gaps above.


### Window restoration and desktop workspace binding

Desktop now saves an atomic per-window layout separately from the legacy last-workspace file. It retains workspace, selected session, normal bounds, maximized/fullscreen state and SSH identity. Offscreen geometry is clamped to connected work areas; malformed records are rejected. Quit preserves the pre-close arrangement; closing an individual window removes only that window. IPC session observations are scoped to the current connection, so a stale connection cannot replace the saved selection. Remote reopening uses SSH identity and reports connection errors rather than interpreting a remote path as local.

Native testing uncovered that desktop initialization omitted project_dir on the shared daemon. DaemonRpc now supplies its host-owned workspace for initialize/session.open, including resumes. The test now checks the actual rendered workspace rather than only the layout file. Populated relaunch also exposed duplicate messages: structured transcript hydration raced legacy history notifications across Electron IPC. The transport suppresses those redundant notifications only while initialize is pending; live warnings and explicit slash replay remain available.

24 focused transport/window tests passed (76 assertions), plus desktop typecheck. Native build 19aa5c9eee16f297 opened two workspaces, completed distinct turns via a local scripted provider, quit and relaunched, verified the exact saved sessions/geometry/maximization and one copy of each answer, then closed one window and verified the remaining record. The same daemon PID survived both app quits; the disposable daemon was stopped at the end. Both screenshots inspected. The initial offscreen position assertion correctly exposed clamping; the final exact-position fixture used on-screen bounds.

window-final-package is the latest installer; strict signature and DMG checksums passed. It is ad-hoc signed, not notarized. SSH/fullscreen restart and native Windows remain unverified. The final root gate is running; no final parity completion is claimed.


Final stable-source root gate passed: bun run check, bun run test and bun run build; 3,875 runtime tests passed, 3 environment-dependent skips, 0 failures; 1,338 TUI tests passed across 132 files. git diff --check passed. Build 19aa5c9eee16f297 matches the tested native package. The preceding typecheck caught a test-only private-helper call, corrected to the fixture public raw helper before this gate. No production daemon or installed app was replaced. Remaining external/platform acceptance is still listed in the inventory.


### Native WebSocket lifecycle acceptance

The native app created a WebSocket monitor against a disposable local Bun server, received a matching event, reported a real disconnect gap, reconnected and received another event, then stopped the watch. Server counters confirmed two connections and two closures, with no application messages sent to the server. Plaintext non-loopback URLs were rejected before connection. Notifications-only mode used no provider credentials.

This exposed delayed Activity state after stopping. Activity now subscribes to lifecycle changes and stopped sources no longer retain a connected label. The rebuilt native app moved the watch into past activity and removed its stop action. Build a4fb8b1bf34d5fcf was verified; screenshots native-websocket-events.png and native-websocket-stopped.png are in the sibling parity-final evidence directory. Ten focused tests passed (42 assertions). The full root gate subsequently passed: 3,875 runtime tests, three skips, 1,338 TUI tests, typechecks and build. Native Windows remains unavailable; external services are not covered by this loopback test.


### Activity layout and native fullscreen restoration

Build 742e1a89193c607b fixes statistics wrapping by choosing columns according to the inspector content width. Activity polling retains one fallback timer when lifecycle notifications overlap. After the preceding full gate, these small renderer changes passed desktop typechecking and nine focused tests (30 assertions), packaging, and repeated native WebSocket lifecycle acceptance at 900px and 1560px. Statistics labels and unavailable values remain readable in both captures. A narrow monitor notification transcript still exposes horizontal scrolling; this is a remaining presentation defect.

A two-workspace native test entered fullscreen, quit the application, and relaunched it. The actual native fullscreen flag, both workspace identities and distinct conversations were restored exactly once. Closing one window preserved the other; the global daemon PID remained unchanged across both quits. Native screenshots and saved layout evidence were inspected. Local scripted SSE responses were used, so this does not qualify an external provider.

Latest installer: sibling parity-final/activity-final-package. This is ad-hoc signed, not notarized. Windows, real SSH, external integration acceptance, native delivery/webhook execution and other inventory gaps remain open.


### Notification wrapping and real snapshot confirmation

The shared wrapping notification row now breaks long URLs and identifiers without horizontal overflow. Native acceptance at 900px and 1560px asserted scrollWidth <= clientWidth and checked the computed overflow-wrap value to prove the rebuilt CSS was displayed. The WebSocket receive/reconnect/stop path passed again; screenshots were inspected. Latest installer is notice-final-package (runtime build remains 742e1a89193c607b; this change is CSS-only).

The actual macOS snapshot confirmation sheet was operated through native accessibility, with no window.confirm override: Cancel preserved changed after snapshot in a.txt, then reopening and selecting OK restored original a. The completed journal and pre-restore backup were independently checked. This closes the native OS confirmation gap; the test used only a disposable local Git workspace.


### Native agent authoring acceptance

The packaged desktop used the production projectGenerate path against a local HTTP provider. A malformed first result produced a visible error while preserving the description. Retrying returned a valid draft, which was not written before review. Save specialist persisted the Markdown and Edit instructions reopened the identical content. Exactly two local provider calls were observed. The first background capture after reopen was blank; a foreground confirmation run rendered correctly and passed all interactions again. This does not qualify a remote provider or model output quality. No repository source changes were needed.


### Native MCP lifecycle acceptance

A real local Bun MCP subprocess completed initialization and tool discovery through the packaged native desktop. Reload reconnected it. A controlled launch failure exercised the normal retry backoff; this exposed that reload errors skipped refreshing MCP status. The desktop now fetches current status after failed reload, preserving the reload error (and both causes if refresh also fails). The native failure view showed disconnected with the subprocess error; a subsequent reload recovered connected status and tool discovery. All owned MCP subprocesses were independently confirmed exited after isolated-daemon shutdown.

68 focused desktop store tests passed (258 assertions), desktop typecheck and packaging passed. Latest installer is mcp-final-package, build 8c50f6a1a9d94925. The full root gate completed successfully: 3,876 runtime tests, three skips, 1,338 TUI tests, typechecks and build. This local integration test does not qualify remote vendor services.


### Signed webhook and delivery native acceptance

The actual configured webhook listener rejected unsigned input (401), accepted a signed event (202), suppressed its duplicate (200), and rejected delivery after native Stop watching (410). The monitor showed exactly one matching event. The daemon build matched 8c50f6a1a9d94925. The generic monitor notification currently labels webhook input as Terminal watch; source identity in the monitor itself remains correct.

For deliveries, a paused test schedule and saved outbox record were seeded in an isolated home. Native Cancel caused zero network requests. Confirm send reached a real local generic-webhook receiver that returned 503; Refresh deliveries showed uncertain state. Allow retry and Confirm retry restored pending state; a second confirmed send succeeded. An independently reopened outbox recorded sent with exactly two attempts, and the receiver verified the saved content and destination on both requests. No external recipient was contacted. Screenshots inspected; no source changes were needed for these paths.

The complete stable-source gate passed: 3,876 runtime tests, 3 skips, 1,338 TUI tests, typechecks and build. Remote vendor integration and Windows remain unverified. The configured SSH host n_server_spot_m is reachable and has Bun 1.3.12 and Xerxes installed; native SSH acceptance is still pending.

## Native SSH acceptance — 2026-09-15

The current mcp-final-package connected through the actual desktop Workspace flow to n_server_spot_m and a disposable /tmp workspace. The Workspace dialog reported SSH online, and session.status returned successfully. The existing EasyDeL project daemon PID 176349 remained unchanged. No provider turn was sent. Remote bootstrap uses its managed published release, so this does not verify the local unpublished runtime on Linux.

Quit persisted the SSH target, workspace, session and bounds. The first relaunch showed a blank native window; a second relaunch restored the correct workspace and online connection. This is partial acceptance, not resolution of the intermittent blank-rendering issue. Evidence: native-ssh-connected.png, native-ssh-restored.png and native-ssh-restore.log in the sibling parity-final directory. Native Windows remains unavailable as confirmed by the user.

## Monitor source label and SSH follow-up — 2026-09-15

Webhook notifications now identify themselves as Webhook watch rather than Terminal watch. Build e70458e386bd2e1b was packaged separately in monitor-label-package. Runtime typechecking, two focused monitor RPC tests (eight assertions), packaging and DMG checksums passed. Native configured-webhook acceptance confirmed the visible label, unsigned rejection, signed acceptance, duplicate suppression and rejection after stop. The initial native assertion omitted the existing [Webhook message] prefix; correcting that expectation passed without further source changes. Screenshot native-webhook-label-event.png was inspected. The prior full root gate remains the latest full-suite result; this one-label change received the focused checks above.

A further SSH relaunch rendered the restored workspace correctly with no renderer exception logged. The first blank launch has not been reproduced or explained, and remains open. No user daemon was stopped or installed application replaced.

## Native renderer recovery and draft identity — 2026-09-15

Added a native recovery dialog for renderer termination and main-frame load failure, excluding expected aborted navigations. Concurrent failures share one prompt; Reload reconnects the current session and does not restart its daemon. Closing/cancelling never initiates reload. Seven focused window tests passed (32 assertions), followed by desktop typechecking and packaging. Native crash injection displayed the real macOS dialog and its Reload action loaded the renderer again.

The fault test also identified unstable draft identity across session bindings. Composer drafts now use the durable session ID when available. On build 950bd62283032567, after a deliberate renderer crash and native Reload, DOM diagnostics confirmed the exact unsent draft was restored and the saved SSH session ID remained a120b2c67bf2. Four focused recovery/draft tests passed (14 assertions) and desktop typechecking passed. The separate recovery-draft-package installer completed checksum verification.

Native visual recovery is not fully verified: some post-reload native captures remained blank while DOM reads reported populated content and an intact draft; Chromium capture reported Current display surface not available for capture. Reacquiring the window restored visual access in one earlier diagnostic run, but the final run still showed this discrepancy. Do not interpret the DOM check as proof that the visible blank-window issue is fixed. The isolated renderer test was stopped; user daemons were not restarted.

## Visible crash recovery confirmed — 2026-09-15

Build 368d9a5699afe54e explicitly shows and focuses the native window and its web contents after the user chooses Reload. The repeatable renderer-crash test now restored visible content immediately, retained the exact unsent draft, and the Workspace dialog confirmed the same SSH workspace online. The saved session remained a120b2c67bf2. Both accessibility state and a native screenshot confirmed the result; native-recovery-confirmed.png is the evidence. The existing remote EasyDeL daemon PID 176349 remained untouched.

This resolves the tested blank state after crash recovery. The earlier blank startup observation has not been proven to have the same cause. Four focused recovery/draft tests, desktop typecheck, separate packaging and DMG checksums passed. recovery-focus-package is the latest installer. A fresh full root gate is running; no full-gate result is claimed yet for this build.

## Packaged SSH startup check — 2026-09-15

Launched the actual recovery-focus-package executable (not the stock Electron test harness), using an isolated XERXES_HOME and Chromium user-data directory. It visibly rendered, restored the saved remote workspace, and its native Workspace dialog reported SSH online. Native screenshot native-packaged-ssh-startup.png records this check. The instance was quit normally. The earlier single blank startup observation was not reproduced on this shipped launch path; its original cause remains unproven. Full root validation is still progressing, with no failure reported at this checkpoint.

## Stable-source recovery gate — 2026-09-15

The full root check, test and build completed successfully on build 368d9a5699afe54e: 3,878 runtime tests passed, three skipped, zero failed; 1,338 TUI tests passed across 132 files. git diff --check passed. The fingerprint matches the native recovery-focus-package used above. Logs recovery-root-check.log, recovery-root-test.log and recovery-root-build.log are saved in the sibling parity-final evidence directory.

## Native agent retry integration — 2026-09-15

A local scripted HTTP provider drove a real AgentTool child through failure and the native Retry agent action. This exposed two defects: an optimistic acknowledgement timestamp hid terminal snapshots, and idle parent session.status reads retained the old snapshot after a retry completed. The GUI now treats acknowledgements separately from live events and polls retry progress; the runtime refreshes manifests from the workspace host, filtered by the exact owning session ID. No v35 field changed.

The rebuilt native app on 97dfa85333ed2cc2 completed the same test. Exactly two child provider requests occurred, and the second preserved the entered follow-up. Both conversation and Activity showed Completed with NATIVE_RETRY_SUCCEEDED. The parent session JSON was independently read after shutdown and retained the completed snapshot. Screenshots native-agent-retry-failed.png and native-agent-retry-completed.png were inspected.

69 store tests passed (261 assertions); seven runtime-refresh/ownership tests passed (15 assertions); runtime and desktop typechecks and separate agent-retry-live-package packaging passed. The initial new test used the wrong cleanup method; it was corrected to runtime.shutdown before the final focused run. The full root gate is running for this cross-cutting change.

## Final retry gate — 2026-09-15

Root check, test and build passed on stable source 97dfa85333ed2cc2: 3,880 runtime tests passed, three skipped, zero failed; 1,338 TUI tests passed across 132 files. The build matches agent-retry-live-package. git diff --check passed. The latest root logs are retry-root-check.log, retry-root-test.log and retry-root-build.log in sibling parity-final. Native vendor services and Windows remain unverified; historical full child tool timelines remain limited by the shared persisted snapshot contract. The earlier isolated startup blank was not reproduced in the packaged executable; native crash recovery is separately verified.

## Push gate — 2026-09-15

The authorized push received another completed root check, test and build: 3,880
runtime tests passed, three skipped, zero failures, and 1,338 TUI tests passed.
Staging exposed whitespace in two previously untracked files; removing it changed
the source fingerprint to 4e7e7ff0aed5f71e without changing behavior. Native package
evidence above remains tied to 97dfa85333ed2cc2; no new native package is claimed.
The staged diff whitespace check passed. Confirmed daemon/TUI feature gaps are
recorded separately in `daemon-tui-gaps.md`.
