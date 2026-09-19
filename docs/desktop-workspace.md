# Desktop workspace

When both builds support reconnect leases, the desktop preserves its owned turn
for up to 30 seconds after a socket or SSH tunnel drops. Retry rebuilds the
transport, reclaims the same owner, and restores the session and unanswered
permission requests. Expired leases and daemon shutdown still cancel work;
Retry never silently resubmits a cancelled instruction. Older runtimes retain
their immediate disconnect cancellation until updated.

The desktop client renders locally and uses the same daemon and persisted sessions as the terminal client. Start a development build with `bun run --cwd xerxes build:desktop`, then `bun run --cwd xerxes desktop` (requires Electron and Bun).

## Working in the app

- **Sessions** opens existing conversations and creates new ones. The composer sends real turns; approvals, questions, cancellation, model selection, and session search use the daemon.
- **Agents** opens project specialists. Select one to read or edit its instructions, choose **New** for a manual definition, or press **G** to describe a specialist and generate a draft with the current model. Generation requires a configured provider. Review and save the draft before it becomes available for delegation.
- **Extensions** shows installed skills and tools, their instructions, and plugin inventory. Plugin installation and enable/disable actions use the runtime. MCP configuration opens the existing settings editor.
- **Scheduled work** previews cron timing before creation. Schedules can be edited, paused, resumed, inspected, and removed. History comes from persisted runs. Example: `0 8 * * 1-5` runs at 08:00 on weekdays in the selected timezone.
- **Activity** shows running shells, monitors, agent and goal details, and terminal output. The top Activity control also shows live background counts and compaction progress. These indicators reflect daemon events, not assistant prose.
- **Tool sections in chat** appear from the first tool call, while it is still running. They start collapsed; click the header or focus it and press Enter to expand. Running status and failure counts remain visible when collapsed. Your expansion choices survive new tool calls and turn completion. Pending permission requests remain visible outside the fold.
- **Changes** reviews the working tree, including selectable untracked files. **Snapshots** captures file states, previews changes, and restores a reviewed file using the preview revision. Restore requests use the runtime's backup behavior.
- **Add context** searches project paths and inserts file mentions into the draft. Choosing a directory browses its children.
- **Preferences** retains provider, model, permission, MCP, personal agent preset, theme, and native notification settings.

Dialogs support Escape, keyboard focus containment, visible errors, and narrow windows. Errors do not erase the conversation.

## Working in multiple windows

Use **File → Open Workspace in New Window…** (Cmd/Ctrl+Shift+O), or **Workspace → Open workspace in new window…**, to keep another project open alongside the current one. **File → New Window** (Cmd/Ctrl+Shift+N) opens an unassigned window where you can select a local or SSH workspace. Each window owns its connection, session view, draft, reconnect and dictation state. Local windows share the global daemon; switching or closing one window does not retarget another window or stop its work. Use the native Window menu to switch between named workspaces. The app saves window arrangements, workspace targets and selected sessions for relaunch.

Session ownership is checked before initialization can replace a binding. A session from another workspace cannot be relocated by opening it under the wrong folder. If an older desktop build saved an invalid workspace/session pairing, startup opens a fresh session in the selected folder and explains that the original conversation remains unchanged. RPC validation, authentication and configuration errors are displayed as workspace problems rather than an offline daemon, and are not automatically retried as network failures.

## Local and SSH workspaces

The workspace control selects local folders or saved SSH workspaces. **Add remote workspace** can discover host aliases from the local SSH configuration, browse remote folders, and save the selected project. Connecting installs or updates the managed remote runtime, then forwards its Unix socket. The desktop renderer stays local.

Selecting SSH keeps the local conversation alive in a retained view inside the same window. The sidebar exposes **This Mac** and other connected SSH hosts, including their saved chats and live working status. Selecting one returns to its existing view without reloading or cancelling work. Drafts and remembered sessions are scoped to the endpoint as well as the workspace, so identical paths on different machines cannot overwrite one another. Failed SSH setup leaves local navigation available; disconnected endpoints retain their last known chat list.

SSH uses existing authentication and known-host records. Unknown host keys are rejected; establish trust using SSH before connecting. No provider credentials are copied to the remote machine. Configure providers there as needed.

If the SSH tunnel drops, automatic recovery and **Retry now** recreate the tunnel before retrying RPC calls. The existing renderer and session binding stay in place, preserving the draft. Concurrent requests share one recovery attempt; closing the view cancels it. Authentication and host-key failures remain visible.

Reconnect first probes the existing remote daemon and reuses its address. A live
daemon does not require a GitHub check or installation during Retry. Only a missing
or refused daemon socket falls back to remote setup; a stalled or invalid RPC reply
is surfaced as an error. Explicit runtime updates still check the managed release.

**Reconnect** retries the selected workspace. Reconnecting the same remote host and folder resumes the current session and reloads its transcript. **Cancel connection** stops the current setup attempt. A failed replacement connection preserves the previous workspace. Disconnecting the local tunnel does not delete remote sessions.

## Dictation

Dictation appends transcribed text to the draft; it does not send a message automatically. It requires microphone permission and an explicitly configured OpenAI-compatible transcription endpoint in the desktop process environment:

```bash
export XERXES_DICTATION_BASE_URL=https://your-provider.example/v1
export XERXES_DICTATION_API_KEY=your-key
export XERXES_DICTATION_MODEL=whisper-1
bun run --cwd xerxes desktop
```

The model is optional; `whisper-1` is the default. Recording is limited to two minutes and 8 MiB. Stop recording to transcribe, or cancel to discard it. Audio goes to the configured endpoint only after recording stops. Chat-provider credentials are not silently reused.

## Verification scope

Automated desktop tests cover transport, session replay, RPC errors, specialist data, diff indexing, remote address validation, subprocess cancellation/timeouts, and transcription input validation. The local walkthrough uses an isolated real daemon. Live provider generation, remote-host setup, and microphone transcription also depend on external configuration and must be exercised in that configured environment.

## Install the macOS app

Build a drag-to-Applications disk image from a source checkout:

```bash
bun install --frozen-lockfile
bun run --cwd xerxes build:installer
```

The output is `xerxes/dist/Xerxes-Agents-<version>-macOS-<architecture>.dmg`. Build on Apple Silicon for `arm64`, or an Intel Mac for `x64`; the image is not universal. Open the image, drag **Xerxes Agents** to **Applications**, open the installed app, and eject the disk image.

The packaged app includes the Bun executable used for the build and its upstream license notice. End users do not need a separate Bun installation or shell PATH configuration. Explicit `XERXES_BUN` / `XERXES_TUI_BUN` overrides still take precedence. Maintainers should keep the bundled license notice aligned with the Bun version used for packaging.

On first launch, choose a folder. The setup checklist then shows runtime connection and model selection, links to the existing provider credential editor, and offers tool-permission review. **Start working** becomes available only after a workspace, connected runtime, and model are present. This checks configuration, not provider authentication: setup never sends a paid test prompt. **Later** dismisses the checklist; **Preferences → General → Open setup checklist** brings it back.

To update, quit the desktop app and replace the application in Applications. Settings and sessions are stored outside the application and remain intact. Removing the app does not delete `~/.xerxes` or an explicitly configured `XERXES_HOME`.

Replace the complete `.app` bundle, not only `Contents/Resources/app`: the renderer,
bundled runtime, and build identity must come from the same package. A partial copy
can leave a build mismatch that restarting the daemon cannot repair. After a full
replacement, use the runtime status control to update an idle daemon; allow active
work to finish before restarting it.

The desktop automatically requests an update when its connected runtime is out of
date. The daemon's atomic idle guard checks all workspaces; a busy runtime stays
connected and the desktop retries after work finishes. Unsupported legacy updates
and setup errors remain visible and require attention rather than a forced stop.
An older desktop never automatically replaces a newer protocol/version daemon.

Managed SSH workspaces prepare the current remote release and use the same idle
guard before replacing a running daemon. The existing SSH tunnel and saved session
are reused. Build compatibility is checked against the release installed on that
host, which may differ from a local development checkout. Closing the workspace
cancels an outstanding SSH setup operation. Custom external sockets without a
managed updater still require an explicit host-side update.

Distribution still requires the existing Developer ID signing and Apple notarization configuration in `packageDesktopMac.ts`. A locally ad-hoc-signed image is a development artifact, not a notarized public release.

Connection recovery keeps retrying transport failures (including certificate verification
failures) until the connection recovers or the turn is cancelled. Backoff grows from one
second to a maximum of 30 seconds. TLS verification remains enabled on every attempt;
incorrect certificate trust must still be repaired. Authentication, validation, and explicit
budget failures remain terminal. Retry events use `maxAttempts: 0` for unbounded network
recovery; HTTP provider errors retain their bounded retry policy.

First-run setup opens in a focused dialog. Later dismisses it, and Preferences can reopen
it. Opening provider or permission settings temporarily hides setup so the editor remains
accessible.

## Conversation-first desktop

New session and Cmd/Ctrl+N open a blank conversation directly. The optional
plan-first control remains below the composer; provider and reasoning controls
remain beside the message input. Changing settings does not create a turn.

Skills & tools and Agents are main-workspace destinations. The skill catalog
has a searchable master/detail view; selecting Use in conversation inserts a
request into the current draft without sending it. Saved agents expose their
delegation name and the same draft insertion flow. Generate still invokes the
runtime generator and opens an editable draft before saving.

Files, Changes and Activity open as nonmodal task context. They do not replace
the conversation or skill/agent page. Close task context returns the space to
the conversation. On smaller windows the pane overlays the edge of the window.
Tracked and untracked diffs use the same runtime-backed review component.

Artifacts lists files changed during the current session and offers transcript
export; it is not a global artifact index. Scheduled jobs and workspace setup
remain focused dialogs. Creator mode remains available through the plugin
workflow rather than occupying primary session navigation.

The macOS DMG uses a compact 680-point Finder window with a Retina background,
a real application bundle, and an Applications shortcut. Instructions appear
in the background; no separate “Start here” document is required. Building the
DMG requires Finder and permission for `osascript` to configure its window.
The builder uses a uniquely named temporary volume to avoid other mounted
installers, persists its layout, converts to a read-only image, and verifies its
checksum before replacing the previous artifact. Installer artwork lives in
`assets/installer/`.

## Shared runtime for desktop and terminal

Desktop and the TUI now connect to one local daemon per Xerxes home, regardless of workspace. Each workspace has independent skills, MCP connections, agent definitions and turn runners; sessions retain their own working directories, instructions, permissions and persisted history. Closing either client leaves the daemon available to the others.

Idle legacy project daemons migrate automatically through the atomic idle-restart endpoint. Busy project daemons stay attached while their work is running. The global runtime refuses to reopen a workspace still owned by a legacy daemon, so migration cannot overwrite running work. After an old runtime finishes and exits, its next connection joins the shared daemon. Explicit custom sockets and remote connections continue to select independent runtimes.

### Package without replacing a running bundle

When an app or daemon is running from `xerxes/dist/Xerxes Agents.app`, build into a separate directory:

```bash
XERXES_DESKTOP_PACKAGE_DIR=/absolute/path/to/package bun run --cwd xerxes build:installer
```

Compiled inputs still come from `xerxes/dist`; the branded application and DMG are written to the selected directory. Use a fresh directory for each verification build. This does not install the app, restart a daemon, or update the copy in `/Applications`.

## Loading long conversations

The desktop requests the latest 100 historical actions when opening a session and starts at the bottom. A message/reasoning entry or a complete tool call/result pair is one action. Scrolling near the top loads the preceding 100 actions; **Load 100 older actions** is also keyboard accessible. The visible reading position stays anchored while earlier content is inserted. Failed pages can be retried without replacing the conversation or draft, and results from a previously selected session are ignored.

Background session/fleet refreshes request metadata only. The daemon retains the complete working conversation for the model; paging limits what is sent to and rendered by the desktop. An older running runtime cannot honor server paging until safely updated; the desktop limits rendering and pages its legacy response locally in the meantime. Existing v35 clients retain the original response unless they request `history_limit`.

### Switching folders while work runs

**Add folder**, workspace headings and cross-project sessions switch retained workspace views
inside the same native window. Each view keeps its own daemon connection, draft, scroll position,
and live response while hidden. Returning to it does not reload it. **Open Workspace in New Window**
is the explicit alternative for another native window.

Explicitly opened local folders remain in `desktop.json` even without chats. Workspace views and
the active selection are restored together on relaunch. Older saved workspace windows migrate
into one window. New explicitly opened windows retain separate window groups.

The sidebar refreshes live session state every five seconds, including tasks in other workspaces.
Running tasks use blue; a failed refresh preserves the last known state instead of declaring them idle.
