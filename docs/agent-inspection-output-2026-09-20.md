# Agent inspection and readable output — 2026-09-20

Conversation agent rows were noninteractive text. The Activity roster could
expand limited details but desktop state discarded the reported base agent,
provider profile and reasoning effort. A runtime ID arriving without a status
change also failed to update its conversation member. Command output had a
separate narrow, unwrapped viewer; large notifications could fill the transcript.

## Capability and workflow matrix

| User expectation | Current source and confirmed defect | Implemented correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- |
| Click an agent and inspect its work | `desktop/renderer/App.tsx`, `AgentRoster.tsx`: conversation and header rows were plain divs; roster detail did not navigate | Clickable conversation/header rows and keyboard-operable roster summaries open the same inspector in Activity. Back returns to activity without changing the conversation/session | Native Electron clicks from conversation and roster; rapid agent switching; 720px and 1440px captures. Packaged renderer opened a retained agent through a real daemon | Native acceptance uses controlled agent records, not an external provider's live child |
| Know the base agent and assigned task | `agentEvents.ts`, `store.ts`, `types.ts`: base/provider/effort metadata was dropped; spawn arguments retained only a title | Preserve reported assignments and original prompt. Inspector shows base agent, model, provider profile, effort, creator and task; missing values remain “Not reported” | Event regression preserves assignments through partial updates and cancellation. Store regression checks snapshot metadata and runtime ID admission when status remains working | Old records that never retained the assignment cannot reconstruct it |
| Follow one identity as the runtime starts or renames it | `store.ts:syncAgentMembersFromFleet`: runtime ID was only copied when status changed | Update identity independently of status, keep request keys associated with runtime rows, and resolve a pending selection when identified | Store and row reconciliation regressions; native rapid selection checks reject stale detail responses | Unidentified historical requests retain their original task but have no runtime controls |
| Read retained work after reconnect | `daemon/server.ts`: parent snapshots intentionally exclude child prompt/output bodies | Add parent-scoped `subagent.inspect`; return only whitelisted metadata and retained prompt/output. Poll while inspecting active work, retain evidence on failures and disconnect, expose retry | Real socket regression covers other-session rejection, missing IDs, reconnect and failed/interrupted/completed states. Production desktop RPC passed against separate local and SSH daemons. Native failure/retry/disconnect checks passed | Retained prompt and output are bounded to 16,000 characters each. This is not the full historical child transcript; live recent tools/progress are also bounded |
| Read commands and logs comfortably | `CommandActivity.tsx`, `Execution.tsx`: unwrapped command/output regions and serialized results; `App.tsx`: long notices unbounded | Shared OutputViewer: wrap by default, copy, line count and expanded reading surface. Full command gets a separate disclosure; valid JSON string/output envelopes decode safely. Long notices start collapsed | Rendering regressions cover encoded strings, literal paths, malformed output and HTML safety. Native tests exercise long logs, wrap toggling, expanded view and reconnect. Packaged renderer reads output from a real Bun command | Command viewer reads the existing 24,000-character retained tail and labels truncation; expanding does not retrieve discarded output |
| Escape closes output without stopping an agent | `App.tsx:GlobalKeys` intercepted Escape before native dialog dismissal | Open native dialogs own keyboard input; closing restores focus to the expand control | Native Escape and focus assertions passed in preview and packaged renderer | Other unrelated custom modal shortcuts were not exhaustively re-audited |

Source paths above are relative to `xerxes/src/`. The additive v35 endpoint is
documented in [`PROTOCOL.md`](../xerxes/src/ui/PROTOCOL.md). Its response never
includes arbitrary manifest metadata or provider configuration. Inspection does
not spawn, resume, cancel or approve agent work.

## Native and daemon evidence

Durable captures and result files are in the sibling directory
`../xerxes-desktop-verification/agent-inspection-2026-09-20/` (relative to the
repository root): `agent-packaged.png`, `command-packaged.png`,
`agent-inspector-narrow.png`, `command-expanded-wide.png`,
`agent-inspection-result.json`, and local/SSH result JSON files.

- Production components in an isolated native Electron window passed 15 checks
  driven by `xerxes/test/fixtures/desktop/agentInspectionVerification.cjs`.
  Run with `bun xerxes/scripts/previewDesktop.ts --verify-agent-inspection`.
  The existing `--verify-work-monitor` path also passed: three active agents,
  58 collapsed historical agents, a 91px collapsed long-command row,
  output errors/retry/disconnect/reconnect, and narrow rendering.
- The built desktop renderer and production preload/main transport were loaded
  with a private Electron profile and a separate real daemon. Saved agent
  inspection, base-agent display, retained output, an actual Bun subprocess's
  80 output lines, expanded reading and Escape focus restoration passed.
  `/tmp/xerxes-agent-inspection-native-20260920/result.json` records this run.
  The host fixture is `xerxes/test/fixtures/daemon/agentInspection.ts`.
- Production `DaemonRpc` passed the same metadata/output access, foreign-session
  rejection and resume checks locally and through SSH to `n_server_spot_m`.
  SSH ran an isolated home, socket, daemon and Bun subprocess. The agent record
  was seeded deterministic data. No external provider calls or user-agent
  control actions were made.

Early harness attempts exposed a quoting error and a selection/read race in
the test, then confirmed the Escape/focus defects corrected above. Packaged
testing initially let automatic runtime migration replace the test host; the
final controlled host advertised the matching build ID and retained its session.
Only the final successful result files establish acceptance. User daemons and
running tasks were not restarted. Owned test hosts and SSH forwarding were
stopped after verification.

## Completed checks and limits

The full root `bun run check`, `bun run test`, `bun run build`, plus
`bun run --cwd xerxes build:desktop` passed. Runtime: **4,130 passed, three
environment skips, zero failures**. TUI: **1,495 passed**. Runtime build ID:
`84827f73dc5bcbb9`. Logs: `/tmp/xerxes-inspector-gate-{check,test,build,desktop}.log`.
The final focus change was also verified with desktop typechecking and the
packaged native interaction test; the strengthened identity assertion passed
its focused store regression. `git diff --check` passed.

The verified build was installed at `/Applications/Xerxes Agents.app` on
2026-09-20. Its signature passed verification; all 600 files and 14 symlinks
match the built package. The previous package is retained at
`/Applications/.Xerxes-Agents-before-agent-inspection-20260920.app`.
The running application and daemons were not restarted. Older daemons can still provide existing
live activity, but retained detail reads need the updated daemon. An unsupported
inspection RPC remains an explicit error. Updating a busy daemon should wait
until the user's work is idle. The broader repository/workflow audit remains
in progress; this record does not claim live-provider or complete GUI/TUI parity.
