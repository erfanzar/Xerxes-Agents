# Session work survives client closure — 2026-09-20

The global daemon already ran separately from the TUI and desktop processes, but
foreground turn ownership still followed the client. Closing an unleased client
cancelled its turn immediately; closing a leased client cancelled it after the
30-second reconnect window. Process independence did not imply work independence.

The TUI and desktop now opt into session-owned turns when initializing. The
daemon retains running work after the client closes, including after lease
expiry. A client reopening the same session restores its transcript and running
state. Stop remains an explicit cancellation action. This is an additive v35
contract: clients omitting the flag retain their previous disconnect behavior.

## Capability and workflow matrix

| User expectation | Behavior and source | Confirmed defect or limit | Correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Close the UI while an agent works | `src/daemon/server.ts`, `submitTrackedTurn`, `disconnectOwner`; `src/ui/gatewayClient.ts`; `src/desktop/main/daemon.ts` | Client ownership cancelled foreground work despite a surviving daemon | Stable session owner; desktop and TUI initialize with `session_owned_turns`; initialized observers receive session events | Regression covers unleased closure and saved output. Actual source TUI submitted work, exited during execution, and reopened the same completed session | Native desktop window-close interaction was not exercised in this check; its production transport was exercised separately |
| Reconnect after more than 30 seconds without starting another agent | `server.ts`, lease ownership and session snapshots | Lease expiry previously cancelled the turn | Lease replay expires independently of session work | Regression forces lease expiry, resumes the same public session and checks one runner invocation. Actual SSH desktop transport reopened after 82.941 seconds and restored completed output | No new guarantee for daemon crashes, reboot or host sleep |
| Stop still stops, and approvals remain deliberate | `server.ts`, `canAnswerInteraction`, permission/question handlers | A disconnected client cannot answer its pending interactions | Same-session observers can recover pending permission/question prompts; foreign sessions cannot answer | Regressions cover explicit cancellation after reconnect, recovered approval, question response and cross-session rejection | Unattended work requiring a decision waits; it does not approve itself |
| Background work remains observable | `server.ts`, session owner event dispatch | Replacing the owner must not remove parent task identifiers | Preserve background session/task tags and parent delivery while parent remains attached | Regression checks parent event tags and completion after parent closes | Exhaustive combinations of concurrent background tasks and multiple observers were not tested live |
| Remote work continues without silently weakening credential handling | Existing remote-provider bindings and relay cleanup | A provider relayed through the UI needs that UI connection | Keep immediate relay authority revocation on disconnect; session lifetime does not extend credential grants | Existing remote-provider/reconnect suites passed | UI-owned local-provider forwarding remains connection-dependent; daemon-host providers are the independent path. No live provider service was invoked |

Paths in the table are relative to `xerxes/`. The wire contract is documented in
[`xerxes/src/ui/PROTOCOL.md`](../xerxes/src/ui/PROTOCOL.md).

## Actual-use evidence

The owned fixture `xerxes/test/fixtures/daemon/detachedTurn.ts` runs a real daemon
and durable session store with a deterministic gated runner. It does not invoke
external providers, use user credentials or execute workspace tools.

- Local actual OpenTUI at 100×28: submitted work, exited using `/exit` while it
  was running, released the runner after 111.343 seconds, then reopened session
  `2c4484ca31d1`. The terminal displayed `FINISHED_WITHOUT_CLIENT`. Daemon PID
  21091 remained unchanged and recorded `cancelled: false`. The fixture lacks a
  canonical outcome record, so its restored footer said `ended · outcome unknown`.
  Evidence: `/tmp/xerxes-detach-tui.raw` captures submission/exit (not the reopened
  screen); `/tmp/xerxes-detach-local-20260920/started.json` and `completed.json`
  record daemon identity and completion. The reopened screen was inspected in
  the actual PTY.
- SSH host `n_server_spot_m`: a separate daemon, isolated home and Unix socket
  under `/tmp/xerxes-detach-ssh-20260920`. Production desktop `DaemonRpc` connected
  through SSH, submitted work, disposed the client, and opened a new connection
  after lease expiry. Session `7d65b573e2be` restored its completed output with
  daemon PID 1943491 unchanged. Evidence:
  `/tmp/xerxes-detach-ssh-result.json`, `/tmp/xerxes-detach-ssh-finish.log`,
  `/tmp/xerxes-detach-desktop-ssh.ts`, and `/tmp/xerxes-detach-desktop-resume.ts`.
  This establishes desktop transport behavior, not native GUI interaction or
  live-provider acceptance.

User daemons and running tasks were not restarted. The isolated fixture daemons
were stopped after acceptance. Temporary evidence is outside the repository and
may be removed by normal temporary-directory cleanup.

## Completed checks and rollout boundary

`bun run check`, `bun run test`, `bun run build` and
`bun run --cwd xerxes build:desktop` passed in this worktree. Runtime tests:
4,126 passed, three environment-dependent skips, zero failures. TUI tests:
1,495 passed. Build ID: `4626e514d72db074`. Logs:
`/tmp/xerxes-detach-final-{check,test,build,desktop}.log`.
Focused daemon, desktop transport and gateway/reconnect regressions also passed.

Both the updated client and daemon must be loaded to enable this behavior.
Already-running old daemons and turns are not migrated by rebuilding; update
them when idle. The combined verified desktop build was installed on disk at
`/Applications/Xerxes Agents.app` on 2026-09-20; existing application and daemon
processes were not restarted. The broader workflow audit remains in
progress.
