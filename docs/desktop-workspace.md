# Desktop workspace

The desktop client renders locally and uses the same daemon and persisted sessions as the terminal client. Start a development build with `bun run --cwd xerxes build:desktop`, then `bun run --cwd xerxes desktop` (requires Electron and Bun).

## Working in the app

- **Sessions** opens existing conversations and creates new ones. The composer sends real turns; approvals, questions, cancellation, model selection, and session search use the daemon.
- **Agents** opens project specialists. Select one to read or edit its instructions, choose **New** for a manual definition, or press **G** to describe a specialist and generate a draft with the current model. Generation requires a configured provider. Review and save the draft before it becomes available for delegation.
- **Extensions** shows installed skills and tools, their instructions, and plugin inventory. Plugin installation and enable/disable actions use the runtime. MCP configuration opens the existing settings editor.
- **Scheduled work** previews cron timing before creation. Schedules can be edited, paused, resumed, inspected, and removed. History comes from persisted runs. Example: `0 8 * * 1-5` runs at 08:00 on weekdays in the selected timezone.
- **Activity** shows running shells, monitors, agent and goal details, and terminal output. The top Activity control also shows live background counts and compaction progress. These indicators reflect daemon events, not assistant prose.
- **Changes** reviews the working tree, including selectable untracked files. **Snapshots** captures file states, previews changes, and restores a reviewed file using the preview revision. Restore requests use the runtime's backup behavior.
- **Add context** searches project paths and inserts file mentions into the draft. Choosing a directory browses its children.
- **Preferences** retains provider, model, permission, MCP, personal agent preset, theme, and native notification settings.

Dialogs support Escape, keyboard focus containment, visible errors, and narrow windows. Errors do not erase the conversation.

## Working in multiple windows

Use **File → Open Workspace in New Window…** (Cmd/Ctrl+Shift+O), or **Workspace → Open workspace in new window…**, to keep another project open alongside the current one. **File → New Window** (Cmd/Ctrl+Shift+N) opens an unassigned window where you can select a local or SSH workspace. Each window owns its connection, session view, draft, reconnect and dictation state. Switching or closing one window does not retarget the others or stop project daemons. Use the native Window menu to switch between named workspaces. Closing the entire app currently restores only the last saved local workspace on relaunch, not the complete window arrangement.

## Local and SSH workspaces

The workspace control selects local folders or saved SSH workspaces. **Add remote workspace** can discover host aliases from the local SSH configuration, browse remote folders, and save the selected project. Connecting installs or updates the managed remote runtime, then forwards its Unix socket. The desktop renderer stays local.

SSH uses existing authentication and known-host records. Unknown host keys are rejected; establish trust using SSH before connecting. No provider credentials are copied to the remote machine. Configure providers there as needed.

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
