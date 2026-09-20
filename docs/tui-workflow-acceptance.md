# TUI workflow acceptance — September 19, 2026

Status: **in progress**. Earlier matrices and completion claims are leads, not
acceptance evidence for this checkout. No external service or SSH destination has
yet been designated for this audit. Tests using injected ports are not live acceptance.
Paths below are relative to `xerxes/src` unless prefixed with `test/`.

Latest checkpoint: finished spawned agents now accept follow-up messages while
retaining their identity and saved conversation. See the final September 20
checkpoint. One SSH setup approval now covers all supported configured local
provider/model pairs for that task, and blank model-inventory provider fields no
longer fail. See the September 20 checkpoint for scope, regressions and actual
loopback SSH evidence. Local reasoning capability negotiation, narrow-picker details,
failed model/mode/effort saves, stale remote discovery notices and pasted-command
file hints are corrected and verified below. The isolated baseline probe at
`/tmp/xerxes-tui-audit-20260919/setting-failure-probe.{ts,json}` records the original
model/mode save defects; the setting persistence checkpoint records their correction.
The permission persistence defect from `permission-persistence-probe.{ts,json}`
is now corrected, including failure/cancellation boundaries, session-scoped
reporting and resumed footer initialization. See the repository-review checkpoint
below and [repository review](repository-review-2026-09-19.md).
Older dated rows retain historical uncertainty; use the latest matching checkpoint
when determining the current status, not the existence of a passing older gate.

## Workflow matrix

| User expectation | Current behavior / source | Confirmed defect or limitation | Correction | Automated / actual-use evidence this audit | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A failed, interrupted or merely idle task does not look successfully completed | `ui/app/useMainApp.ts:terminalTitle`; `ui/domain/paths.ts:terminalActivityMarker`; `ui/opentui/messageLine.tsx:170` | Terminal title formerly used `✓` whenever not busy. A tool-free assistant message formerly received a `done` footer even after interruption or provider failure. | Title now explicitly reports idle, working, waiting or disconnected; idle makes no success claim and disconnect takes precedence over stale busy/approval state. Outcome receipts now replace the inferred done footer; successful, interrupted, failed, limited and unknown endings have distinct labels. | Two title regressions; actual `partial-persistence-terminal.raw` shows working then idle after cancellation and grant refusal. `partial-persistence-resumed-terminal.raw` shows disconnected after owned-daemon shutdown. Earlier captures preserve the misleading success marker. | Outcome persistence and receipts verified in the September 19 durable-outcome checkpoint below; older records without metadata remain explicitly unknown. Waiting-title path has automated but not new actual-use evidence. |
| A remote renderer can use approved local provider authority without receiving credentials | `ui/lib/localProviderBroker.ts`; documented `provider.relay.*` RPCs | Parent and remote renderer lacked a private local bridge to the parent-owned daemon grant | Added a Unix socket broker: private directory/socket, exact connection-owned grant, bounded frames/connections/deadlines, cancellation propagation, expiry and idempotent revocation/cleanup, fixed diagnostics, no request replay | Ten regressions include two actual daemons and native remote turn routing, provider cancellation at three boundaries, expiry, late authorization, malformed/oversized traffic, stalled connections and split UTF-8. Actual 80×24 TUI exercised native Anthropic HTTP success, Escape cancellation and refusal after revocation; checkpoint below. | Now reachable from the machine review: the parent owns the broker and both relay connections; the child receives only the remote session ID/key. Real loopback SSH acceptance is recorded below. No external provider/SSH destination acceptance is implied. |
| Resume retains received partial work and explains interrupted or failed turns | `streaming/loop.ts:persistUnfinishedRound`; native runner synchronization; TUI replay | Terminal stream failure/cancellation skipped assistant persistence. A subsequent provider-grant failure also lost its outcome. | Preserve the terminal attempt's emitted text/reasoning on cancellation, terminal failure, backoff abort and iterator abandonment. Exclude incomplete tool calls and synthetic diagnostics; successful retries still replace failed attempts. Validated outcome metadata persists separately from provider messages and replays through both full and paged history. | Native runtime regressions verify cancellation and failure through disk restart and the next provider request; stream tests cover tool-call safety, timeout, backoff and iterator abandonment. Actual session `7755a628a319` now retains its interrupted text after daemon restart, seen in `partial-persistence-resumed-terminal.raw`; old before-capture is retained. | Interrupted/failed outcomes now survive restart; hard crashes before final persistence remain unverified and may leave an unknown outcome. Historical text already discarded by old code cannot be reconstructed. External providers remain unverified. |
| Resume preserves code whitespace exactly | `streaming/toolMarkers.ts:stripAssistantToolCallMarkers`; `session/resumeRepair.ts` | Marker cleanup trimmed every assistant message even when it removed no marker, changing partial code and falsely reporting a repair | Return unchanged text when extraction only trimmed whitespace; actual marker removal stays in place | Native failure/cancel restart tests retain trailing spaces, newlines and indentation byte for byte; sanitizer regression covers whitespace-only messages, invalid marker text, repair counts and repeated loads | Actual terminal capture verifies recovered text; exact whitespace is established by persistence/model-request assertions rather than visual appearance alone. |
| A locally routed task uses local provider defaults, while explicit task choices still work | `daemon/turnRunner.ts`, `subagentHost.ts`, `providerRelays.ts`, `runtime.ts`, `server.ts`; `security/localProviderRelay.ts` | The main runner inherited remote output/sampling/thinking/retry/context settings; children inherited remote default effort. Remote context and effort were displayed as local task settings. | Ignore remote provider defaults for local bindings; snapshot validated local profile sampling at authorization and apply it locally; preserve explicit session/agent effort including off. Reload cannot import remote effort into a bound session. Report local default effort and unknown local capacity instead of remote values. | Production CLI regression with and without conflicting remote configuration; main, child, planner, overflow compaction, cancellation, release, explicit off and saved metadata; local authority/profile tests and reload/resume regression. Actual source TUI completed delegation/planning and explicit off; seven local requests, zero remote-provider requests; captures and details in the local-defaults checkpoint below. | Local capability negotiation and local reasoning-picker validation still required. Actual providers and external SSH remain unverified. Provider adapter token-ceiling behavior is a separate confirmed defect below. |
| An approved output ceiling remains effective at the actual provider boundary | `security/localProviderRelay.ts`; `llms/anthropic.ts:anthropicRequestPayload`, `bedrock.ts:buildBedrockConverseInput`, `client.ts:responsesPayload` | Relay bounds the neutral request, but Anthropic/Bedrock can expand the wire output cap for thinking; Codex subscription removes it entirely. | Native adapters now check a local-only authority annotation against their final output field before sending. Oversized translations fail with fixed output_limit; Codex numeric grants are refused, and provider-controlled output requires a separate explicit consent flag plus null cap. Request/expiry/concurrency/revocation limits remain intact. | No-network adapter capture `/tmp/xerxes-tui-audit-20260919/relay-output-cap-audit.json`: approved 1024; Anthropic wire 14096; Bedrock wire 11024; Codex wire cap absent. Anthropic/Codex exercised through relay + native clients with an injected fetch that captures then aborts; Bedrock through its production payload builder. | 40 focused regressions and actual native-adapter/production-daemon TUI refusal-and-retry capture are recorded below. No external provider contract acceptance; plugin adapters are not supplied to the production local relay factory. The machine consent panel now distinguishes numeric limits from explicitly accepted provider-controlled output. External-provider enforcement remains unverified. |
| Connect only to an authenticated remote destination | `ui/lib/machineHandoff.ts`; `/machine`; `daemon/machineDiscovery.ts` | Setup/tunnel inherited host-key policy and SSH agent-forwarding configuration; browser also inherited agent forwarding | Explicit strict host-key checking, agent forwarding off and X11 forwarding off for TUI setup/tunnel and folder browsing | 11 handoff tests including real `ssh -G` precedence checks; 4 browser tests; pinned-host-key loopback SSH acceptance | External-host acceptance; inherited environment and other forwarding settings still need audit |
| Resume remote work without losing drafts or position | `ui/lib/machineHandoff.ts`, `ui/gatewayClient.ts` | Tunnel close killed the renderer and discarded its draft | Keep the renderer through bounded same-destination tunnel retries; fixed private status messages; terminal failure remains inspectable | 11 handoff tests; real loopback SSH tunnel killed during a 30-part turn: replacement tunnel, unchanged renderer PID, unsent draft retained, response completed | External host, managed bootstrap, exhausted retries in actual TUI, draft persistence after explicit exit |
| Keep live work through a brief disconnect | `ui/gatewayClient.ts`, `ui/app/useMainApp.ts`, `useSessionLifecycle.ts`; daemon `connection.lease` | TUI did not negotiate leases; loss reset turns and could cancel work; restoration could discard a journal on a second drop | Opt into advertised leases; reclaim before resume; deliver missed and buffered live events after UI adoption; restore current interactions; snapshot fallback after interrupted restoration; explicit disconnected state and clear stale recovery activity | Real-daemon tests cover ordered exactly-once delivery, question ownership, lease expiry without restart, and second-drop snapshot recovery; lifecycle tests retain turn/todos/viewport; actual loopback SSH streaming recovery | Repeated drops before initialize response, all approval types and local native-terminal acceptance need further stress testing |
| Reuse local setup deliberately | `security/localProviderRelay.ts`, `localProviderEndpoint.ts`, `providerRelayProtocol.ts`; `llms/localRelayClient.ts`; `daemon/providerRelays.ts`, `remoteProviderBindings.ts`, `server.ts`; `ui/gatewayClient.ts` | Managed remote runtime still uses remote configuration; no reachable per-integration reuse/consent path | Scoped memory-only local authority; opt-in remote session binding with private bounded request/reply transport; persisted local-provider requirement prevents native-runner fallback after disconnect/restart; private context bypasses reconnect journals | Earlier 24 authority/transport checks and fixture-wired TUI evidence; 3 real-daemon/native-runner gateway binding regressions plus 11 focused binding/transport tests cover scope, ownership, cancellation, timeout, revocation, reconnect and persistence | Production CLI now shares the binding registry across main, child, planner and compaction paths, verified without remote credentials. TUI consent/revocation, broker lifecycle and SSH handoff wiring remain required; local reuse is not yet available from the picker |
| Understand remote setup before connecting | `ui/opentui/machinePicker.tsx`, `machineSetupReview.tsx`, slash `core.ts` | Picker previously handed off without integration scope or configuration guidance | Both picker and named connect open a 12-page review with destination, execution/configuration/credential source, persistence, setup paths, and explicit uninspected status; reject changed destinations and duplicate handoffs | 12 machine-picker plus 8 overlay-state tests; actual source TUI traversed all pages at 80×24 and returned locally with Escape; renderer coverage at 40×18 | Review describes boundaries; it does not inspect remote health or enable local setup reuse. Destination preflight probes, scoped consent/binding and live services remain unfinished |
| Keep remote setup diagnostics safe | `ui/lib/machineHandoff.ts`, `ui/lib/remoteBootstrap.ts`, `daemon/machineDiscovery.ts`, `security/sshDiagnostics.ts` | Setup/browse failure exposed arbitrary subprocess output; malformed handshake parsing could echo input; bootstrap retained and tailed raw build output | Fixed classified SSH errors; safe handshake parsing; private bootstrap log contains only fixed stage messages and no installer/build output | Setup and browse sentinel regressions; actual shell bootstrap fixture verifies failed build exits unsuccessfully, reports its stage and leaves a mode-0600 log without synthetic secrets | Independent daemon/provider logging paths and remote third-party processes need separate audit |
| Keep local environment values local over SSH | `security/sshConnectionConfig.ts`; TUI handoff and daemon folder discovery | Command-line SendEnv clearing does not prevent later configured entries; configured SetEnv can send values independently of process environment | Private temporary configuration includes existing files by reference, blocks configured SetEnv with one nonsecret marker and clears SendEnv after includes | Real `ssh -G` test preserves alias/user/identity references without environment assignments; real loopback SSH with AcceptEnv enabled receives neither synthetic sentinel | External hosts and proxy chains remain unverified |
| Apply connection security independently of existing SSH sessions | `security/sshConnectionConfig.ts`; setup, tunnel, and folder discovery | Setup and browsing could inherit ControlMaster/ControlPath and PermitLocalCommand from user configuration | Private config selects ControlMaster no, ControlPath none and PermitLocalCommand no before user/system includes; tunnel explicitly owns a private master, never reuses a user master | Effective `ssh -G` regression; disposable loopback SSH with an existing master confirms a new TCP connection and no multiplexed session, configured LocalCommand marker absent (`ssh-control-acceptance.json`) | External hosts and ProxyCommand/ProxyJump require further acceptance; local SSH configuration remains trusted code |
| Open only the selected daemon socket over SSH | `security/sshConnectionConfig.ts`, `ui/lib/sshSocketTunnel.ts`, `machineHandoff.ts` | Inherited LocalForward, RemoteForward and DynamicForward could expose unrelated services | Clear all inherited forwards for setup, browsing and the owned master; add only the daemon socket via a config-free private control request, with bounded startup and fixed failure diagnostics | Effective OpenSSH config rejects all three forwarding types; 16 handoff/tunnel tests include failure, cancellation and retry; actual TUI loopback session with unwanted local/remote Unix forwards kept them absent, survived master termination, preserved draft and saved all 30 stream segments | External SSH versions, proxy chains, unavailable stream-local forwarding and exhausted retries require wider actual-use acceptance |
| Discover/switch concurrent sessions and workspaces | `ui/app/useSessionLifecycle.ts`, session picker | Under audit | — | Not yet exercised | Running work, identity and drafts across switches |
| Reject a missing explicit resume without creating another conversation | `daemon/runtime.ts:initializeSession`, `daemon/server.ts:initialize` | A missing transcript fell through to fresh-session creation | Explicit missing resume returns an actionable error; matching known live keys survive lease expiry; running-session refusal remains distinct | 41 focused runtime tests; monitor-restart regression; 10 gateway/footer checks; actual 80×24 source TUI retained saved transcript and original session after `/resume missing123` | Deletion during concurrent resume and filesystem failures still require wider stress coverage |
| Distinguish configuration from provider health | `ui/opentui/appChrome.tsx:WorkspaceFooter` | A green provider-ready label was inferred only from a model string | Neutral model-selected label; no provider-health claim without evidence | Footer regressions and actual terminal before/after capture | Dedicated provider connectivity/status diagnostics remain under audit |
| Resume and page large histories | `ui/gatewayClient.ts`; protocol `session.history` | TUI history paging path not yet found; protocol currently labels it desktop history | — | Source search only | Bounded loading, search, viewport restoration |
| Compose multiline drafts, queue, steer, cancel, retry | Composer, turn controller, session queue | Under audit | — | Not yet exercised | Failures, interrupted sends, persistence |
| Keep the selected provider after resume | `daemon/sessionProvider.ts` | Empty API-profile inventory skipped validation of an existing pin and allowed the runtime default client | Validate the pin before the empty-inventory fallback | Persist/restart regression; actual source TUI resume and submission, zero provider calls, visible error and idle recovery | Real external profile removal not exercised; error instruction is truncated at 80 columns |
| Keep titles and auxiliary calls on the session's provider | `daemon/server.ts:maybeGenerateTitle`, `sessionAuxiliaryClient`, `compactSessionByKeyUnlocked`, `agentPresetRpc` | Automatic titles used the global profile; compaction and agent generation bypassed local bindings, and missing compaction pins could fall through to runtime credentials | Resolve the session pin or live local binding before client construction; local titles stay on the grant's model; absent authority retains provisional titles/history instead of falling back | 10 real-daemon gateway regressions include local-bound titles, compaction, draft generation, missing pins and release during compaction; actual source TUI displayed Pinned Profile Title using pinned profile while global profile remained unchanged | CLI planner, mid-turn compaction and subagents are now routed and verified by the production CLI regression below. Full local-reuse UI remains unfinished; synthetic service ports do not establish external provider acceptance |
| Cancel local-provider work while an auxiliary RPC waits | `daemon/remoteProviderBindings.ts`, socket dispatch, `daemon/websocketGateway.ts` | Serial RPC queues blocked private replies; releasing a binding rejected the remote wait without cancelling the local stream | Private replies/releases bypass the occupied queue while retaining limits; release cancels tracked streams through the private path and prevents fallback | Actual socket/native-runner regression releases pending compaction, observes local provider abort and retained history; WebSocket queue regression confirms replies/releases proceed while ordinary mutations remain ordered | Actual TUI/SSH cancellation for the binding path remains unverified until production handoff/consent wiring exists |
| Read complete recovery instructions | `ui/opentui/appLayout.tsx:NoticeBanner` | Actual 80-column provider error hid `/model` in middle truncation | Bounded wrapped preview and Alt+N/click to open full notice in existing pager | Assembled renderer checks at 80×24 and 40×18; actual 80×24 source TUI shows `/model`, opens pager, and returns to draft | Extremely large notices, resize while pager is open and external-terminal modifier mappings remain to exercise |
| Resume without unrelated refresh errors | `ui/opentui/sessionFollowups.tsx`, `ui/app/useMainApp.ts` RPC wrapper | Stale follow-up poll during resume emitted conversation-changed errors into transcript | Poll opts out of global reporting; its current owner retains inline failure display and rejects stale results | Delayed rejection/navigation regression; actual source TUI resume no longer emitted the stale message | Other background pollers and repeated reconnect/navigation remain under audit |
| Preserve exact streamed content in saved state | `daemon/runtime.ts:updateFallbackSession` | Fallback runners trimmed every text/reasoning chunk, joining words and corrupting code whitespace | Preserve chunks verbatim, including whitespace-only chunks | Real-daemon reconnect snapshot test; Bun regression checks saved code and reasoning whitespace | Provider-owned session-state runners use a separate path; their persistence needs separate acceptance |
| Read tool output and reasoning | `ui/opentui/appLayout.tsx`, `/tool-output` | Older history may contain only retained summaries | — | Previous evidence not rerun | Full-result access and populated rendering |
| Inspect and control delegated work | Agent panel; `subagent.interrupt` | Protocol and prior gap matrix disagree about targeted interruption | — | Documentation conflict identified | Verify runtime implementation and reachable controls |
| See goals/todos above input, working status above them | `ui/opentui/appLayout.tsx`: `ComposerTaskSummary`, `LiveProgressPill` | Empty goal/todos hidden in source; actual layout under audit | — | Source traced | Empty, completed, blocked and long-content states |
| Manage shells, runs, schedules and monitors | `/activity`, `/terminals`, `/runs`, `/schedules`, `/monitors`, `/loop` | Under audit | — | Source TUI in real 80×24 PTY: `/runs`, V, N displayed seeded event 21 on page 2; Escape returned to Runs | Remaining actions, cancellation and recovery boundaries; external sources |
| Discover commands and arguments | Shared command registry, `ui/hooks/useCompletion.ts` | Under audit | — | Source TUI `/help` opened 267-line keyboard/command pager; Escape restored conversation in real PTY | Every reachable command and native fallback |
| Manage providers/models/effort and permissions | Provider/model pickers, settings, approval prompts | Under audit | — | Not yet exercised | Failure visibility, persistence, secret handling |
| Manage integrations | `/features`, `/skills`, `/plugins`, `/mcp`, `/config`, `/hooks`, `/browser`, `/channels` | Host-dependent integrations must not claim execution readiness from inventory alone | — | Protocol read | Actual configured-service paths |
| Navigate with keyboard, scroll/select, resize | Shared OpenTUI components | Under audit | — | Not yet exercised | Narrow and heavily populated terminal captures |
| Start, stay idle, stream and recover efficiently | Entry/gateway, stores, virtual history | Under audit | — | Source TUI 80×24 idle overlay: CPU time +0.34s over 10.009s (~3.4% of one core), RSS 254560→255392 KiB; concurrent full tests noted | Heap, startup/stream latency, long-session growth and post-fix comparison; short sample does not establish a leak |

## Local-to-remote integration inventory

This is the initial source-backed boundary map, not a readiness verdict. Each row
requires a separately verified setup/reuse path; availability must not be inferred
from a copied profile or a listed tool.

| Integration | Current execution/configuration location | Remote requirements / intended verification |
| --- | --- | --- |
| Provider API credentials | Executing host's `bridge/profiles.ts:ProfileStore` reads `$XERXES_HOME/profiles.json`; `auth/storage.ts` owns the separate credentials store | Retain remote credentials and exact profile overrides; design scoped local proxy/reuse consent without persisting local provider secrets remotely |
| OAuth/subscription providers | Authentication resolver on executing host | Verify refresh/expiry/revocation; do not copy login stores or promise generic API-key forwarding works |
| Model/effort and agent-tier choices | Session metadata plus daemon user's settings/catalog | Validate remote availability; preserve explicit assignments and reject unavailable routes without fallback |
| Public text search | `tools/duckduckgoEngine.ts:DuckDuckGoInstantAnswerProvider` uses executing host's `PublicWebClient` and public no-key endpoint | Does not require local credential forwarding; requires remote public-network access. Limited to text instant answers and related topics, not general full search. Live usefulness/availability unverified |
| Full search, image/news/maps/video search | `tools/duckduckgoEngine.ts:DuckDuckGoSearchProvider` is an injected host port; default public adapter rejects non-text requests | Requires a separately configured provider implementing those capabilities; no blanket search-key forwarding claim or current generic remote reuse path |
| MCP | `mcp/configured.ts` loads executing host's user `mcp.json` and trusted workspace `.mcp.json`; `mcp/client.ts` owns stdio/HTTP transports; settings snapshots may contain credentials | Audit stdio versus HTTP/OAuth independently; local executables/files do not become remote resources. Do not expose credential-bearing settings snapshots in preflight inventory |
| Skills and assets | Executing daemon's admitted user/project/bundled discovery | Assess explicit selected bundle reuse, trust, relative assets and executable host dependencies |
| Agent definitions/compositions | Executing host's user and workspace definitions | Review selected definitions and referenced assets; do not copy configuration roots implicitly |
| Plugins/hooks/LSP | Executing daemon and workspace-owned processes/configuration | Require remote dependencies/trust; report unsupported proxy paths honestly |
| Browser | Explicit endpoint on executing host | No browser launch or implied access to local browser; explicit endpoint required |
| Channels/webhooks | Executing host configuration and lifecycle | No message sends authorized for this audit; inventory and offline failure checks only |
| SSH authentication | Local SSH client, keys/agent remain local | Explicit host-key verification; disable agent forwarding; audit inherited environment and forwards before reuse claims |

## Evidence and scope

- Initial dirty files: desktop build script, desktop main/renderer files and
  desktop tests; untracked `.codex/` and `.github/hooks/`. Preserve these changes.
- Read `AGENTS.md`, protocol transport/RPC/event and extension contracts, the
  daemon/TUI matrix, wiring audit, roadmap and relevant implementation history.
- Root `bun run check && bun run test && bun run build` passed: 3999 runtime
  tests passed, 3 skipped, 0 failed; 1376 TUI tests passed across 139 files.
  Runtime build ID `f2f1a5f420629dad`. Logs:
  `/tmp/xerxes-tui-audit-20260919/{check,test,build}.log`.
  The folder-browser argument fix was also tested independently after its edit
  (`ssh-discovery-tests.log`); final typecheck rerun tracked in `check-final.log`.
- Real local PTY acceptance used the production source TUI against the isolated
  `test/fixtures/daemon/tuiParity.ts` daemon, with seeded data and no provider
  calls. `/help`, `/machine`, `/runs` and saved-event paging exercised so far.
  Raw ANSI capture: `/tmp/xerxes-tui-audit-20260919/terminal.raw`.
  Idle measurement: `/tmp/xerxes-tui-audit-20260919/idle-baseline.json`.
- Rebuilt `dist/ui/entry.js` also resumed the isolated saved session, streamed a
  seeded turn with tool output and eight subagent events, accepted Escape to
  interrupt, showed `interrupted`, and preserved `Keep this unsent draft` across
  F10 goal inspection and Escape. Capture:
  `/tmp/xerxes-tui-audit-20260919/built-terminal.raw`. These are actual keyboard
  interactions with a real daemon and fixture runner, not real model calls.
- Live daemon command inventory saved to
  `/tmp/xerxes-tui-audit-20260919/command-catalog.json`; listing does not prove
  that every command has been exercised. Final typecheck rerun and
  `git diff --check` passed.
- Native iTerm interaction was denied by computer-use policy. No alternative
  tool was used to control iTerm. PTY evidence is not native screenshot acceptance.
- Real loopback SSH acceptance completed using production handoff/tunnel code and a rebuilt TUI against an isolated real daemon with a seeded runner. Capture: `/tmp/xerxes-tui-audit-20260919/ssh-terminal.raw`; process evidence: `ssh-before-drop.json` and `ssh-processes.json`. Renderer PID 64964 survived tunnel 64960 being killed and replacement 65028; `Keep SSH draft` remained and all 30 response parts completed. Setup injection started the fixture daemon through real SSH; managed remote installation was not exercised. This is not external-host or real-provider acceptance.
- SSH used disposable host/client keys, explicitly pinned known_hosts, StrictModes and password authentication disabled. Owned daemon and sshd were stopped after acceptance; disposable private keys and authorized_keys directory were removed. No user SSH configuration or credentials were copied.
- Recovery-specific initial checks: 1383 UI tests passed in `reconnect-ui-tests.log`; build `8b2f73cf713fee33` in `reconnect-build.log`. The SSH capture used this build. Subsequent fixes (stale recovery status, second-drop snapshot fallback and whitespace preservation) have focused tests; the final gate below must cover them.
- No local credential has been sent to a remote host by this audit.
- Recovery checkpoint validation: root `bun run check` passed; runtime suite
  passed **4000 tests, 3 skipped, 0 failed** (`test-recovery-final.log`). The
  following UI run caught an overstrict socket-identity guard in a mocked
  lifecycle test. After narrowing it to actual identity changes, all **1384 UI
  tests across 140 files** passed (`ui-recovery-latest.log`), root check passed
  again (`check-recovery-latest.log`), and root build passed
  (`build-recovery-final.log`, build **334f17c354ae27a8**). Runtime source did
  not change after the passing runtime suite. These logs are under
  `/tmp/xerxes-tui-audit-20260919/`; the initial combined command exited on the
  UI failure, so successful component reruns are reported explicitly.
- Loopback SSH evidence summary:
  `/tmp/xerxes-tui-audit-20260919/ssh-acceptance-summary.json`.
  No post-change performance improvement is claimed: the prior short idle
  baseline remains the only performance measurement.
- Environment isolation actual-use evidence:
  `/tmp/xerxes-tui-audit-20260919/ssh-env-acceptance.json`. A disposable real
  loopback sshd accepted the tested variable names, but the remote command
  received neither the local SendEnv sentinel nor configured SetEnv sentinel.
  It received only the nonsecret marker. Pinned disposable keys authenticated
  the connection; StrictModes stayed enabled. The owned server was stopped and
  keys/authorization directory removed. This verifies environment forwarding,
  not external-host/provider acceptance or other configured SSH forwards.
- SSH diagnostic/environment checkpoint: the complete root
  `bun run check && bun run test && bun run build` command passed after these
  changes: **4002 runtime tests passed, 3 skipped, 0 failed; 1385 UI tests
  passed across 140 files**. Build **a1a131b68122eae9**. Logs:
  `/tmp/xerxes-tui-audit-20260919/{check,test,build}-ssh-security.log`.
  `git diff --check` also passed. This is a validation checkpoint, not overall
  TUI or external-service acceptance.

## Next implementation boundaries

### Local provider relay integration under construction

`LocalProviderRelay` is a local-only authority engine, with no listener and no
UI entry point yet. Its host must obtain consent before calling `authorize`.
The grant view names destination, workspace, profile, model, expiration and
limits; tokens and client instances remain separate. Only a locally owned peer
object can use its grant. A resolver must return the exact approved provider
route identity on each call, allowing credential refresh but rejecting route
changes. This primitive never executes model-requested tools locally.

The next wiring must validate untrusted completion frames before the typed
engine, carry them over the authenticated SSH connection, and install a
session-scoped daemon client without persisting bearer authority or changing
global profiles. The TUI must display local-provider/remote-tool execution,
obtain explicit scoped consent, and provide renewal and revocation. Session
restore without the original grant must be actionable and must not fall back.
Subagents and housekeeping calls need explicit routing coverage too.

Source integration points verified in this checkout: `cli.ts:daemonRuntime`
constructs workspace runners, `AgentTurnRunner.resolveSessionProvider` selects
the main request client, `daemon/agentProvider.ts` resolves subagent routes,
and daemon compaction/title paths independently create clients. A remote host
without any configured model can currently return no workspace runner, so a
relay binding must enable that session without inventing a global fallback
profile. Updating only the main request client would leave subagents and
housekeeping on different or unavailable routes and is not sufficient.

`security/providerRelayProtocol.ts`, `security/localProviderEndpoint.ts` and
`llms/localRelayClient.ts` now provide a validated native completion codec and
bounded pull transport. The local endpoint holds the grant token; request and
reply frames carry no grant authority. Reasoning/signatures, tool replay,
effort, grammar and cache segments are retained. Provider cache session IDs
are namespaced by grant to prevent collision with local tasks. Channel close
and request cancellation settle waiting callers even if a provider/transport
ignores cancellation; an occupied backend continues consuming its grant's
concurrency slot until it actually settles. Unknown errors are not forwarded.

Current limits: provider-specific `extraBody` overrides and network image URLs
are rejected; inline images remain eligible. The original one-delta pull
protocol amplified latency; bounded batching now collects up to 32 deltas
with a 4 ms collection window, a byte limit and at most one pending prefetch.
Partial output is retained before a terminal provider failure. Production SSH binding, consent UI, routing and
revocation integration remain pending. No live provider was called.
The full root check/test/build command passed after this addition: **4012
runtime tests passed, 3 skipped, 0 failed; 1385 UI tests passed**. Build
**96de2c32d9f298cf**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-provider-relay.log`.
This checks the implementation in the worktree; it is not relay transport or
TUI acceptance, because the engine has not yet been connected to those paths.

Transport checkpoint evidence: 16 focused Bun tests cover grant authority and
transport/codec behavior, including a real Unix-socket request path through
`AgentTurnRunner` and a coding tool executed only in the fixture workspace.
Actual keyboard use of the built TUI against an isolated real daemon with
that relay installed by the harness completed the tool and rendered the final
response before returning to idle. Capture:
`/tmp/xerxes-tui-audit-20260919/relay-terminal.raw`; verified file/result summary:
`/tmp/xerxes-tui-audit-20260919/relay-tui-acceptance.json`. The owned daemon was
stopped after this run. This proves fixture-wired native turn/render behavior,
not a reachable production SSH consent flow or a real configured service.

The batched version also completed an actual TUI coding turn with the isolated
fixture daemon. Capture: `/tmp/xerxes-tui-audit-20260919/relay-batch-terminal.raw`;
result summary: `relay-batch-tui-acceptance.json` in the same directory. That
owned daemon was stopped afterward. Focused grant/transport tests now total 18.

Before/after synthetic relay benchmark (20 deltas, five samples): an injected
20 ms delay per pull took **450–471 ms before** and **22–23 ms after batching**.
Warm local Unix-socket samples decreased from roughly **0.90–1.33 ms** to
**0.14–0.29 ms**; first samples were slower. Data:
`/tmp/xerxes-tui-audit-20260919/relay-benchmark-before.json` and
`relay-benchmark-after.json`. These use a zero-latency fixture provider and
uncontrolled machine load; the delay is simulated, not measured SSH RTT.
Real SSH/provider throughput remains unverified.

The complete root check/test/build gate passed for the batched transport:
**4020 runtime tests passed, 3 skipped, 0 failed; 1385 UI tests passed**.
Build **cd2380120ad2a969**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-relay-batch.log`.
`git diff --check` passed. Production local-setup reuse remains unimplemented
until the SSH/daemon binding and explicit user consent controls are wired.

### Remaining acceptance work

1. Extend actual-use coverage of the implemented renderer-preserving recovery to
   failed authentication, exhausted retries, repeated drops and pending approvals.
2. Stress-test the implemented lease recovery at each handshake boundary and
   verify local-daemon reconnect, long offline intervals and cancellation authority.
3. Build a per-integration preflight and explicit reuse selection. Prefer a
   destination/workspace/session-scoped local provider proxy over copying keys.
   Existing remote credentials and explicit overrides must remain available.
   Grants require visible destination, scope, expiration and revocation; no
   credential reuse is authorized merely by selecting a saved machine.
4. Complete each matrix row through isolated actual use, then run the final gate
   again after further changes. This checkpoint does not meet final acceptance.

### Local daemon authority and pinned-provider checkpoint

The local daemon now owns relay authorization, pulls, status and revocation.
The renderer-facing handle is connection-owned; neither provider credentials nor
the internal grant token cross that boundary. Revocation can interrupt a pending
pull, disconnect removes unleased authority, route changes fail before calling a
replacement client, and active grants prevent idle restart between requests.
The API is documented in `ui/PROTOCOL.md`. Production SSH binding and user
consent controls remain unfinished. Six real-daemon socket tests supplement the
18 grant/transport tests (24 focused tests total).

A separate routing defect was confirmed and fixed in `daemon/sessionProvider.ts`:
when every API profile disappeared, its early empty-inventory return skipped an
existing session provider pin. The pin is now validated first. A regression in
`test/sessionProvider.test.ts` saves and resumes a conversation, verifies that no
default client is called, preserves the pin and ends the failed turn at idle.

Actual source TUI use against an isolated real daemon at 80×24 resumed the saved
fixture conversation and submitted another message. The provider failure was
visible and the working state returned to idle; the fixture recorded **zero
provider calls**. Capture: `/tmp/xerxes-tui-audit-20260919/missing-provider-terminal.raw`.
Result: `missing-provider-result.json` in the same directory. The owned renderer
and daemon were stopped. This used an injected provider inventory, not an actual
external service or remote host.

That actual-use run also confirmed two **open presentation defects**:

- Resuming can briefly emit “The active conversation changed; refresh its
  follow-ups before acting” into the transcript from a stale refresh.
- At 80 columns the provider error is truncated in the middle, hiding its
  `/model` recovery command. Returning to idle does not make that path coherent.

These are remaining fixes, not accepted limitations or completion evidence.

Final gate for this checkpoint passed after the provider-pin fix: **4027 runtime
tests passed, 3 skipped, 0 failed; 1385 UI tests passed across 140 files**. Root
check and build passed, build ID **03b8f8579356b710**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-relay-control-final.log`.
`git diff --check` passed. This is an in-progress checkpoint; the confirmed
presentation defects and unfinished remote workflow remain open.

### Notice and background-refresh correction

The two presentation defects found in the preceding checkpoint are corrected.
The focused UI suites pass **53 tests**. Actual 80×24 source TUI verification
resumed a saved conversation against an isolated real daemon, triggered the
pinned-provider error, displayed its `/model` instruction, opened the complete
notice with Alt+N, and restored **Keep my draft** after Escape. The stale
conversation-changed follow-up error did not appear. Capture:
`/tmp/xerxes-tui-audit-20260919/notice-after-terminal.raw`.
The fixture again recorded zero provider calls; the owned daemon and renderer
were stopped. This is fixture-service evidence, not external-provider acceptance.

Further observations to audit: the header still says **provider ready** after
the missing-provider failure; rapid unbracketed input in this PTY appeared to
lose characters, while bracketed paste preserved the command. The latter needs
a controlled reproduction before attributing it to the renderer. An unknown
resume identifier also appears to create a new session; validate its intended
contract and distinguish explicit new-session behavior from failed resume.

Source follow-up confirmed the next two defects: `ui/opentui/appChrome.tsx`
labels a configured model as “provider ready” without testing provider authority;
`daemon/runtime.ts:openSession` rejects corrupt transcripts for explicit resume
but still creates a session when `loadResult.kind` is `missing`. Both need
correction and regression/actual-use verification. They are not completion blockers
requiring external resources; work can continue locally.

Full root gate after the notice/polling fixes: **4027 runtime tests passed,
3 skipped, 0 failed; 1388 UI tests passed across 140 files**. Check and build
passed; build ID **a6c6fca1a4e0d2d7**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-notice.log`.
`git diff --check` passed. Structured terminal evidence:
`/tmp/xerxes-tui-audit-20260919/notice-after-acceptance.json`.
The overall goal remains in progress, including the confirmed next defects,
production SSH provider binding/consent and the rest of the workflow matrix.

### Missing resume and provider-status correction

Explicit resume now rejects a missing saved conversation and directs the user
to `/resume` or `/new`, preserving the current session. A matching known live
key is still honored after lease expiry; a live turn under a different key
retains its distinct busy error. The footer reports **model selected**, without
a green provider-readiness claim.

Actual source TUI at 80×24: resumed the saved fixture conversation, submitted
`/resume missing123`, read the actionable failure with the transcript intact,
and submitted another message to the original pinned-provider session. Capture:
`/tmp/xerxes-tui-audit-20260919/resume-footer-terminal.raw`. Zero provider calls;
owned renderer and daemon stopped. This used an isolated real daemon and injected
inventory, not a live external provider.

The stricter behavior exposed two previously false-positive test setups: the
permission-resume fixture read the wrong store, and the monitor-restart fixture
never saved conversation history. Both now use real persisted data. An implicit
ID-shaped open concurrency fixture retains its intentional create-if-absent
semantics; only explicit resume rejects missing history. The initial full run
failed those fixtures plus a live-session error expectation. Focused reruns
passed after corrections: 41 runtime/persistence checks, the monitor restart
check, and 10 gateway/footer checks. A fresh full gate follows.

Fresh complete gate after all resume corrections passed: **4028 runtime tests,
3 skipped, 0 failed; 1389 UI tests across 140 files**. Check and build passed;
build ID **80b84b30d394f2d1**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-resume-final.log`.
`git diff --check` passed. Structured actual-use result:
`/tmp/xerxes-tui-audit-20260919/resume-footer-acceptance.json`.
No commit or push. Production SSH setup reuse and broader acceptance remain open.

### Remote setup review checkpoint

A reachable setup review now precedes both picker handoff and the explicit
`/machine connect <name>` shortcut. Its 12 pages distinguish provider keys,
subscription login, model/effort, search, MCP, skills, agents, plugins/hooks/LSP,
browser, channels, and SSH/session persistence. Remote configuration remains
the active path. The review explicitly says remote readiness is uninspected and
local reuse is not yet available; this is not acceptance of the setup-reuse goal.

Twenty focused UI checks passed (12 machine-picker, 8 overlay-state), including
40×18/80×24 navigation, failed retry, cancellation with a late response, changed
host rejection, and single-flight Enter. Actual source TUI against an isolated
real daemon opened `/machine connect audit`, traversed all 12 pages, and returned
through the machine list to the local session with Escape. No SSH connection or
external provider call was initiated for this review-only run. Capture:
`/tmp/xerxes-tui-audit-20260919/machine-preflight-terminal.raw`. Owned renderer
and daemon were stopped. Full gate logs follow under `*-machine-review.log`.

The first `*-machine-review.log` full gate passed checks but stopped at
**4027 runtime passed, 3 skipped, 1 failed**: the unrelated desktop hung-Git
lookup test could not find its fixture PID file. That test passed in isolation;
no desktop source or test was changed for this failure. UI/build did not run in
that attempt. A fresh gate is recorded separately as `*-machine-review-final.log`.

The SSH config audit also confirmed that setup/browsing could inherit a user
control connection and enabled LocalCommand. The private config now disables
both before including user/system settings. Six focused SSH/discovery tests pass.
A disposable real loopback server and an existing disposable control master were
used to verify a separate TCP connection, absent configured environment
sentinels, and no LocalCommand marker. Evidence:
`/tmp/xerxes-tui-audit-20260919/ssh-control-acceptance.{ts,json}`. Owned processes
stopped and disposable key/authorization directories removed in `finally`.
This does not establish external-host or ProxyJump acceptance, and inherited
port forwarding still needs correction or an explicit supported policy.

Fresh full gate passed: **4028 runtime tests passed, 3 skipped, 0 failed;
1394 UI tests passed across 140 files**. Check and build passed, runtime build
ID **09725bd477bef737**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-machine-review-final.log`.
`git diff --check` passed. No commit or push. The earlier failed run remains
recorded above; this passing rerun does not resolve its intermittent desktop
fixture failure or establish acceptance of unfinished remote reuse workflows.

### Scoped SSH forwarding checkpoint

Configured forwards are now cleared for setup, discovery and the owned tunnel
master. Only the daemon Unix socket is added through a private control request
that reads no user configuration. This deliberately uses an owned master;
the earlier prohibition concerned reusing an unrelated user's master. It does
not copy keys/configuration or create a general-purpose remote forwarding grant.

Focused verification: 16 UI handoff/tunnel tests passed, including failure,
cancellation before readiness, startup deadline, dropped-master replacement,
and forwarding refusal. Real `ssh -G` verification clears configured local,
remote and dynamic forwards while retaining host/user/identity references.
UI typecheck passed. Actual 80×24 TUI against real loopback OpenSSH and an isolated
real daemon: configured unrelated local and remote Unix forwards stayed absent
while connected; master PID 46387 was terminated during a 30-part turn and
replaced by 46412; renderer PID 46392 remained unchanged. The unsent draft
`Keep scoped SSH draft` survived and all 30 stream parts completed and persisted
in `/tmp/xerxes-tui-parity-dULCWq/sessions/9144ce0af471.json`.

Evidence: `/tmp/xerxes-tui-audit-20260919/socket-tui-terminal.raw`,
`socket-tui-live-check.json`, `socket-tui-acceptance.{ts,json}` in the same
folder. The managed installer handshake was substituted with the isolated
daemon address; provider activity was synthetic. No external service acceptance
is inferred. Owned renderer, fixture daemon and SSH server stopped, disposable
keys removed. Full gate results follow under `*-scoped-tunnel.log`.

Full root gate for scoped forwarding passed: **4028 runtime tests passed,
3 skipped, 0 failed; 1398 UI tests passed across 141 files**. Checks and build
passed; runtime build ID **4657b5a1922e6404**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-scoped-tunnel.log`.
`git diff --check` passed. The overall goal remains in progress: production
provider-relay session binding/consent/revocation and broader workflow acceptance
are still required. No commit or push.

### Private remote provider binding checkpoint

Added the opt-in remote side of session binding and private gateway transport.
A persisted requirement outlives the ephemeral binding: restart, revocation or
physical disconnect produces an explicit error rather than using the native
runner's configured remote provider/fallback. Private model requests are excluded
from transcript conversion and the reconnect journal. The server advertises this
capability only when the host supplies the matching registry; ordinary production
CLI routing is deliberately still unsupported until all auxiliary model callers
are routed. This is implementation progress, not live setup-reuse acceptance.

Three real-daemon/native-runner gateway regressions pass for local authority,
revocation, disconnect/reconnect, and persisted restart with zero remote fallback
calls. Eleven focused binding/transport tests pass for ownership, workspace/model
scope, cancellation, late replies, deadlines, native coding tools and journal
exclusion. Logs: `/tmp/xerxes-tui-audit-20260919/binding-{integration,unit}.log`.
UI typecheck initially exposed an unwanted provider-runtime dependency in the
gateway; it now transports opaque envelopes and leaves native request validation
to the local authority. The corrected UI typecheck passes.
No actual TUI/SSH use of this new binding path is claimed yet. Production host,
consent/revocation UI, explicit remote override, and external service verification
remain required. Full gate logs follow under `*-remote-binding.log`.

The first `check-remote-binding.log` gate stopped during runtime typecheck because
new binding tests used `.next()` directly on an `AsyncIterable`. Those tests now
obtain its async iterator explicitly. The earlier runtime execution passed, but
that does not override the failed check. The fresh gate uses
`{check,test,build}-remote-binding-final.log`.
The initial transport deliberately fails closed on physical disconnect. It does
not yet resume an in-flight provider pull under the same surviving local grant;
automatic same-grant recovery, explicit rebind/retry presentation, and retention
of running work through that transition remain acceptance work.

Fresh full gate passed: **4031 runtime tests passed, 3 skipped, 0 failed;
1401 UI tests passed across 142 files**. Checks and build passed; runtime build
ID **fb3cb3b6cbacc935**. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-remote-binding-final.log`.
`git diff --check` passed. No commit or push; overall goal remains in progress.

### Auxiliary provider routing checkpoint

Fixed automatic titles selecting the global profile instead of the session pin.
Manual/pre-turn compaction and project-agent generation now resolve the same
local binding or pinned profile. Missing/released authority cannot fall back to
runtime credentials. Local-bound titles use their authorized model only.
Private replies/releases can bypass a blocked auxiliary RPC on both transports;
release now also sends cancellation to local streams, bounded to 16 tracked IDs.

Focused evidence: 9 real-daemon gateway integration tests, 37 compaction/title
runtime tests, 10 WebSocket tests, and 3 binding tests passed. The first auxiliary
integration run exposed the queue deadlock (replies only handed the queue back
after reaching its head); dispatch now bypasses that occupied queue for the two
private control methods. A profile fixture initially omitted its required URL;
that fixture was corrected using an unreachable synthetic URL and injected clients.

Actual 80×24 source TUI against an isolated real daemon resumed session
`36efcf1ac936`, submitted a turn and displayed **Pinned Profile Title**.
The injected title factory recorded `{model:"gpt-4o-mini",profile:"pinned"}`;
the global active profile stayed `global`. No external provider call was made.
Capture: `/tmp/xerxes-tui-audit-20260919/auxiliary-title-terminal.raw`.
Fixture and result: `auxiliary-title-{fixture.ts,result.json}` in the same folder.
Owned renderer/daemon stopped. Binding-path TUI/SSH acceptance remains unfinished.
The first full gate began before the late release-cancellation correction;
only the subsequent `*-binding-auxiliary-final.log` gate can validate the final
source. Production CLI planner/mid-turn compaction/subagent routing, UI consent,
revocation and reconnect recovery are still required before enabling local reuse.

The first broad runtime run failed six cases: CLI LSP/MCP resume, command
completion reaction, explicit-runtime compaction, and two title lifecycle tests.
The initial auxiliary resolver inferred and mutated an unpinned session's profile;
it could also reject explicit daemon connection settings. The resolver now reads
an existing pin without mutating it, otherwise preserves the matching runtime
route. Title client construction uses that same route. Focused reruns passed
all 13 CLI/command/compaction cases and 3 title/generation cases. A new regression
asserts optional titles never create a pin. The gateway suite now has 10 passing
tests, and binding tests have 4 passing checks including bounded paused streams.
The first broad run's counts were 4026 passed, 3 skipped, 6 failed; its UI/build
stages did not run. These failures are not treated as completed verification.

Fresh full root gate passed after the routing and cancellation corrections:
**4033 runtime tests passed, 3 skipped, 0 failed; 1408 UI tests passed across
142 files**. Checks and build passed; runtime build ID **d788ef04f803fa0f**.
Logs: `/tmp/xerxes-tui-audit-20260919/{check,test,build}-binding-auxiliary-final.log`.
`git diff --check` passed. No commit or push. The goal remains active, including
production planner/mid-turn compaction/subagent routing, user-facing consent and
revocation, relay reconnect recovery and the rest of the workflow audit.

### 2026-09-19 — Optional transparent TUI canvas

| User expectation | Current behavior and source | Confirmed gap | Implemented correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Switch between the current Chrome design and a terminal-transparent background, preserving the choice | `ui/theme.ts`, `app/appearance.ts`, `app/uiStore.ts`, `app/slash/commands/session.ts`, `opentui/entry.tsx`, `opentui/appLayout.tsx` | No user-selectable transparent canvas; renderer and composer painted an opaque ground | `/appearance` toggles; explicit `chrome`/`transparent` arguments; local atomic preference write before applying; live renderer/theme update; help and completion; readable filled popup/selection/diff grounds | Four new tests cover both light/dark and all interaction modes, session/busy-state preservation, skin replacement, exact Chrome restoration, round-trip persistence, failed writes, malformed/missing file, offline completion. Actual source TUI at 80×24 against an isolated real Unix daemon: explicit transparent, process restart loading transparent, Chrome restoration, no-argument toggle, invalid argument retains preference. Captures: `/tmp/xerxes-tui-audit-20260919/appearance-terminal.raw`, `appearance-restart-terminal.raw`. Restart capture uses SGR49 for canvas and composer; Chrome uses SGR48 fill. Session b71026d5fc6f and prior transcript remained visible. | Physical emulator opacity is user-controlled and not changed. No physical translucent-window screenshot; terminal-byte behavior verified. No new external SSH/provider acceptance claimed. |

This is a client-only option, saved on the TUI host under
`$XERXES_HOME/tui-appearance.json` (default `~/.xerxes/tui-appearance.json`).
It does not send an appearance configuration change to the daemon. The first
terminal pass exposed a leftover opaque composer surface; the restart pass
verified its correction. The isolated daemon/renderer were stopped afterward.

Appearance checkpoint full root gate completed successfully: `bun run check &&
bun run test && bun run build`; **4,033 runtime passed, 3 skipped, 0 failed;
1,412 UI passed across 143 files**. Build ID `f914e2f4ca18d9b2`.
Logs: `/tmp/xerxes-tui-audit-20260919/{check,test,build}-appearance.log`.
Structured terminal evidence: `appearance-acceptance.json` in the same directory
records both ground transitions, persisted choice, unchanged session identity,
and zero provider calls. `git diff --check` passed. The broader audit remains
in progress; this checkpoint completes the requested appearance option only.

### 2026-09-19 — Delegated local-provider routing

| User expectation | Current behavior and source | Confirmed defect / limitation | Implemented correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Delegated work uses the parent's authorized provider rather than a remote default | `daemon/subagentHost.ts`, `daemon/remoteProviderBindings.ts` | Native child allocation/execution resolved profile names independently; no host-owned local-client route existed | Added `resolveSourceClient`, checked before allocation and execution, across live generation/recovered retry. Bound sources use the exact authorized model; agent profile overrides cannot replace the user binding. Child snapshots retain a SHA-256 route fingerprint, never the client. Replacement binding cannot revive old authority. Local child failure directs the user to check the parent's local connection/authorization. | Five focused tests exercise success, incompatible models/profiles before allocation, active disconnect, cancellation, and retry after binding replacement. Existing provider/workspace/native host suite also passes. Actual source TUI + isolated real Unix daemon + native AgentTool + private Gateway relay frames completed a coder child and parent synthesis; `/tmp/xerxes-tui-audit-20260919/subagent-local-coder-terminal.raw` displays the child result. | The port is implemented and injected in this live isolated harness. Production CLI wiring, relay-only remote runtime creation, planner/mid-turn compaction, user-facing consent/revocation and reconnect continuation still unfinished. No external service or new external SSH destination used. |

The first live harness had no durable saved transcript; resume correctly refused
it. The next harness requested `default`, which the parent's delegation policy
correctly rejected. Neither is counted as successful delegation. The final
harness used a saved conversation and the allowed `coder` profile. A synthetic
provider supplies deterministic content; this establishes real TUI/daemon/tool
routing, not external-provider acceptance. The full gate's first check caught
two newly written test typing errors (`type` instead of `promptProfile`, and a
missing agent descriptor `id`); these were corrected before the fresh full gate.

Final live evidence (`subagent-local-result.json`) records calls in the order
parent → child → parent, **zero remote fallback calls**, completed child
`subagent_d41e0b29f86b`, and session `94b5c32a0a32`. The persisted parent JSON
contains the child's stable ID, route fingerprint and result. All isolated
renderer/daemon processes were stopped. Full root gate passed: **4,038 runtime
passed, 3 skipped, 0 failed; 1,412 UI passed across 143 files**; checks and build
passed, build ID `f91ab7e577c1652e`. Logs in
`/tmp/xerxes-tui-audit-20260919/`: `check-subagent-corrected.log`,
`test-subagent-final.log`, `build-subagent-final.log`. `git diff --check` passed.
The full objective remains active; production local-to-SSH setup continuity is
not complete merely because the isolated host port works.

### 2026-09-19 — Production daemon local-provider execution

| User expectation | Current behavior and source | Confirmed gap | Implemented correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A remote daemon can run with an authorized local provider even without remote credentials | `cli.ts` daemon construction and workspace runner factory; `daemon/runtimeConnection.ts` | The production CLI never supplied remote bindings and refused to create a workspace runner without a remote connection | One registry shared by server, native turn runner, source-routed subagents, planner and mid-turn compaction. An unconfigured execution template has an empty model and a failing default client; bound sessions supply the exact model/client. No fake credentials or automatic provider fallback | `cliLocalProviderBinding.test.ts` starts the actual CLI daemon with an isolated HOME and no model/provider/key configuration. It verifies parent calls, child delegation, title, planning, overflow recovery, cancellation, release and persisted local requirement without grant token | User-facing local setup consent, broker lifecycle and SSH handoff/reconnect still unfinished. The fixture supplies consent and a broker; this is not the final user path |
| Structured planning is actually reachable when the native host registers it | Default YAML and fallback definitions, `agents/subagentManager.ts` | PlanTool was registered in the daemon but omitted from default agent tools; native calls failed as unconfigured | Added PlanTool to the default agent's tool vocabulary. Kept it in the child orchestration blocklist so it cannot bypass delegation restrictions | Production CLI regression reaches the local plan generator; actual TUI invokes AgentTool and PlanTool in one turn | External provider plan quality not evaluated |
| A context overflow can compact and resume across the relay without disclosing provider diagnostics | `security/localProviderRelay.ts`, `security/providerRelayProtocol.ts` | Relay changed overflow into generic failure; after compaction its codec rejected the internal summary marker | A fixed `context_overflow` code preserves recovery semantics without raw diagnostic text; known boolean summary provenance is validated and stripped from model requests | Transport regressions reject malformed/unknown metadata and omit a synthetic diagnostic secret. Production CLI test grows history, induces overflow, confirms local summary call and resumed parent output | Other provider failure classes still have the existing fixed generic error vocabulary; external provider-specific overflow cases remain untested |

Actual source TUI at 80×24 connected to the normal source CLI daemon, rather
than an injected InMemoryDaemonRuntime host. An isolated broker answered private
provider requests using a synthetic LLM. The terminal displayed the native coder
result, two completed tools and final planning/delegation response. Capture:
`/tmp/xerxes-tui-audit-20260919/production-relay-terminal.raw`; launcher and result:
`production-relay-tui.ts`, `production-relay-info.json`,
`production-relay-result.json` in the same directory. The remote runtime config
contains only `permission_mode: accept-all`; no provider credentials or model
were configured there. All fixture processes were stopped afterward.

Harness corrections: Unix socket readiness uses `existsSync`, not Bun's regular
file existence check. Gateway completion is `message.complete`, not the raw
protocol's `turn_end`. Those early timeout runs were not accepted as evidence.
The first successful parent/child test then exposed the real PlanTool filter
failure; overflow testing exposed both real relay recovery failures above.

Production checkpoint: 26 focused runtime checks passed (relay authority,
transport, native child binding and normal CLI execution). The first full root
gate stopped at **4,040 runtime passed, 3 skipped, 1 failed**: the unrelated
`desktopProjectLookup.test.ts` timeout fixture did not create `git.pid` before
its assertion. UI/build did not run in that failed gate. The same test passed
in isolation (`test-production-desktop-lookup-rerun.log`); a fresh full gate is
required and recorded below when completed. No desktop source or test was edited.

Actual production result (`production-relay-result.json`) records parent, title,
parent-tools, child, planner and parent calls, with only `permission_mode` in the
remote runtime configuration. Session `7a90485ea669` remained the same in the
TUI. This supersedes the earlier matrix's assertion that ordinary CLI hosts
advertise remote bindings as unsupported; it does not supersede the unfinished
consent, broker and SSH lifecycle limitations.

Remaining confirmed source gap: `daemon/turnRunner.ts` selects the bound client
but still computes `maxTokens` from `this.options.maxTokens`/the remote runtime's
output-cap function and carries runtime sampling fields into the request. A
credential-free daemon has no such override in the successful test above. A
daemon with an existing provider can therefore contribute inappropriate limits
or sampling defaults to a locally bound session. This needs a regression with
conflicting remote settings and explicit safe local capability/sampling routing;
the successful no-credentials case does not prove mixed-configuration acceptance.

Fresh full root gate passed: **4,041 runtime passed, 3 skipped, 0 failed;
1,412 UI passed across 143 files**, checks and build passed, build ID
`feb2add1c75f37e0`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-production-relay-rerun.log`.
`git diff --check` passed. The preceding failed gate remains documented above.
No commit or push. The broader objective remains active, including mixed remote
settings, consent/broker UI, SSH lifecycle and remaining workflow coverage.

### 2026-09-19 — Local provider defaults and truthful status

The preceding production-wiring checkpoint left mixed configuration: the local
client was selected, but main turns still took remote token/sampling/thinking
settings, retry routing and context metadata. Native child creation also inherited
remote default effort. This checkpoint isolates those defaults, snapshots a safe
sampling whitelist from the local profile at authorization, and preserves explicit
session/agent effort. Invalid numeric local sampling refuses authorization with a
fixed error; arbitrary profile fields cannot enter the completion request.
Credentials continue to resolve on the local machine per request, with route-change
rejection before constructing a replacement client.

The production CLI tests now run both without remote credentials and with a
conflicting remote configuration (65536 output tokens, temperature 1.9, top-k 999,
top-p 0.99 and high reasoning). Local main/child requests use 4096, 0.2, 20, 0.8
and low reasoning. Planner, title and compaction retain their explicit bounded
request settings. A local output default is capped by the grant at the relay
boundary; this is **not** yet proof of the final provider wire cap (see below).
Cancellation reaches the local provider, release fails closed without a remote
fallback, explicit off travels as `thinking.effort: none`, and persisted session
metadata retains the explicit off choice. Additional native child tests preserve
explicit agent effort; runtime reload/resume tests prevent remote effort adoption.

Actual TUI verification used a normal isolated source CLI daemon, a local
`DaemonProviderRelays` instance with a saved synthetic profile, and an injected
provider. No external provider, credential or SSH destination was used. The first
80×24 PTY run completed delegation and planning but exposed a stale footer saying
`reasoning: high`. After correcting init/status/reasoning display paths, the second
run showed `reasoning: local default`, `/budget` reported unknown capacity,
AgentTool coder and PlanTool completed, and `/thinking off` changed the footer and
next provider request. The session remained `ebe76504940a`. Its seven local
requests included parent, title, child and planner; remote provider calls: **0**.
Owned TUI, broker and daemon processes were stopped after verification.

Evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `local-defaults-terminal.raw`: before the footer correction.
- `local-defaults-terminal-after-footer.raw`: corrected footer, budget, delegation,
  planning and explicit off in the actual source TUI.
- `local-defaults-result-before-footer.json`, `local-defaults-result.json`: captured
  effective request settings and zero remote calls; no transcript or credentials.
- `test-local-defaults-focused.log`: 25 focused tests passed before adding the
  reload/explicit-off regressions. `test-local-defaults-persistence.log` and
  `test-local-defaults-status.log` verify the final persistence/status corrections.
- `{check,test,build}-local-defaults-final.log`: first full root gate passed, 4046
  runtime passed/3 skipped/0 failed; 1412 UI tests passed; build `952820d7923cf4c9`.
  This first gate preceded the footer correction; the final gate is recorded below.

A separate no-network adapter audit confirmed a remaining authority defect:
`relay-output-cap-audit.json` records a 1024-token grant becoming 14096 on the
Anthropic wire and 11024 in Bedrock's Converse input; Codex subscription drops the
wire output field altogether. The relay request check alone is insufficient.
This is the next security correction before exposing local-reuse consent. Other
remaining work includes local capability negotiation and reasoning-picker
validation, the consent/revoke UI and SSH broker lifecycle, same-authority
reconnect behavior, and the broader workflow audit. The overall goal remains
in progress.

Final gate after the footer correction: `bun run check && bun run test && bun run
build` passed; **4046 runtime passed, 3 skipped, 0 failed; 1412 UI passed across
143 files**. Runtime build ID `18493afe683dea16`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-local-defaults-status-final.log`.
`git diff --check` passed. No commit, push, stage, PR or external account change.

### 2026-09-19 — Enforce output authority after adapter translation

The preceding adapter capture established a real limit bypass. `CompletionRequest`
now carries an optional local-only symbol annotation from the relay. It cannot
arrive in JSON, appear in a provider JSON body, or enter a saved transcript.
Anthropic, OpenAI chat/Responses, Azure, Bedrock, Gemini, Vertex and Pi-messages
validate the final output field before sending. A thinking expansion or minimum
floor above the grant throws a fixed error which the relay maps to `output_limit`.
Ordinary unannotated requests retain their existing provider behavior. Explicit
limits are never automatically enlarged, and thinking is not silently reduced.

Codex subscription has no request-level output cap. Inventory now identifies its
output-limit mode and explains this limitation. Numeric grants are refused with
`output_limit_unsupported`. The alternative is an explicitly different policy:
`max_output_tokens:null` plus `consent_provider_controlled_output:true`, in addition
to normal complete-scope consent. Missing/false consent or null for another
transport is refused. Grant status discloses `outputLimitMode: provider-controlled`;
request, expiry, concurrency, cancellation, disconnect and revocation controls
remain effective. The Claude Code subprocess is also refused before allocating
relay authority, matching its unsupported inventory status. No TUI consent flow
or external service acceptance is claimed by this internal control implementation.

Automated evidence: `test-output-limits-focused-final.log` has **40 passed, 0
failed**. Coverage includes all native cap fields, before-network refusal,
serialized authority exclusion, safe error transport, explicit-off successful
retry, Codex's separate consent, revocation, and both policies' active cancellation,
disconnect, expiry and close. Full root gate results follow below.

Actual 80×24 source TUI used a normal production CLI daemon and a local authority
with the native Anthropic client calling an isolated loopback HTTP/SSE provider.
A 20000-token grant seeded a durable conversation (two 14096-token requests,
including title). A fresh explicitly authorized 1024-token grant then refused the
configured 10000 thinking budget **without reaching the HTTP provider**. The TUI
showed the actionable error, returned idle and retained history. `/thinking off`
followed by a new prompt succeeded through the same session, with wire max_tokens
1024 and thinking disabled. The new grant records two attempts, zero active calls;
only one HTTP request was sent under that grant. Session: `f0436dfaac41`.

Evidence in `/tmp/xerxes-tui-audit-20260919/`:

- `output-limit-terminal.raw`: actual error, reasoning change and successful retry.
- `output-limit-result.json`: grant policy and provider wire measurements.
- `output-limit-tui.ts`: isolated production daemon/native-client harness.
- `relay-output-cap-audit.json`: original before-fix adapter measurements, retained
  as historical evidence rather than overwritten.

All owned fixture processes were stopped. No external provider credentials or
SSH destination were used. This checks outgoing requests and the local lifecycle;
it does not certify that an external provider implements its declared cap.

The actual refusal exposed a separate status issue: `ui/app/useMainApp.ts` chooses
`✓` for every idle terminal title, including the failed turn shown in this capture.
The actionable error and idle state are correct, but the title implies success.
This remains a confirmed display defect to correct in the broader status audit.

Final verification also covers normalized provider aliases (`codex` and
`openai_codex`) so their consent policy cannot disagree with native routing.
Invalid saved routes remain inspectable with an actionable unsupported state.
Focused final result: **42 passed, 0 failed**, recorded in
`test-output-limits-alias-focused.log`.

Full root `bun run check && bun run test && bun run build` passed after the alias
correction: **4060 runtime passed, 3 skipped, 0 failed; 1412 UI passed across 143
files**. Build ID `f4d074b8d7dfddf3`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-output-limits-alias-final.log`.
`git diff --check` passed. All owned acceptance processes are stopped; no commit,
push, stage or external account changes. The full goal remains active.

### 2026-09-19 — Private local provider broker

Added `ui/lib/localProviderBroker.ts` as the parent/child local transport primitive
needed by remote setup. The parent authorizes an exact, already-consented scope
through its existing local daemon connection and retains the connection-owned
grant handle. The child-facing socket accepts only the documented relay envelope;
it exposes neither general RPC access nor credentials, provider configuration,
grant handles or bearer tokens. No credentials are read by this module.

The socket resides in a mode-0700 temporary directory and is mode 0600. Input and
output frames, concurrent connections and hard exchange deadlines are bounded.
Parsing preserves UTF-8 across chunk boundaries. Invalid, oversized, incomplete
and pipelined traffic cannot invoke unrelated RPCs or echo raw diagnostics.
Closing an abandoned pull cancels the local provider. Explicit close, abort and
expiry revoke once and remove the endpoint; reconnecting the owning daemon
connection cannot resurrect the old grant. There is no automatic request replay,
provider fallback or authority renewal. EOF without a response is an immediate
failure, including when the broker refuses excess connections.

Ten focused UI regressions pass in `local-broker-focused.log`, including two
actual daemon sockets and native remote turn execution through the bridge,
provider cancellation when the caller, owner or broker disconnects, expiration,
cancellation during authorization, malformed/oversized traffic, stalled-client
bounds and split UTF-8 replies. Test stores and scheduler leases are isolated.

Actual-use evidence uses source TUI at 80×24, a normal CLI remote daemon with no
provider credentials, a separate local daemon and the native Anthropic adapter
calling an isolated loopback HTTP service. The fixture owns the binding and
broker while the TUI attaches to its saved session; **this is not acceptance of
the unfinished consent/handoff UI**. It completed a submitted prompt, displayed a
waiting streamed response, handled Escape cancellation and rejected another
prompt after the broker was revoked. The HTTP service observed four calls:
seed, title, submitted success and the waiting call; the waiting call was
aborted, and no call followed revocation. Broker cleanup removed its socket.
No external provider, account or SSH destination was used.

Evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `local-broker-tui.ts`, `local-broker-info.json`, `local-broker-result.json`
- `local-broker-terminal.raw`, `local-broker-daemon.log`
- `local-broker-resumed-terminal.raw`, `local-broker-resume-daemon.log`
- `local-broker-persistence.json`

The restart acceptance exposed a separate confirmed defect: session
`c1aad181e361` persisted all four user prompts and two successful assistant
messages, but omitted the interrupted partial response and provider failure
outcome. The resumed TUI showed consecutive unanswered user prompts. It also
confirmed misleading `done` footers after failures/interruption in the original
TUI. Both defects are open in the matrix; the successful broker test does not
establish durable outcome fidelity. Saved remote history contained no synthetic
credential, broker path or private RPC envelope.

Full root `bun run check && bun run test && bun run build` and `git diff --check`
passed: **4060 runtime passed, 3 skipped, 0 failed; 1422 UI passed across 144
files**. Build ID `d91be448dba93601`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-local-broker.log`.
All owned fixture, TUI and resumed-daemon processes are stopped.

The broker is not yet wired into the machine consent screen or SSH handoff.
That wiring must bind its private path to the reviewed destination/workspace,
preserve explicit remote credentials/overrides, expose grant state and limits,
and handle reconnect/revocation without implicit renewal. External services,
external SSH and bridge performance under large contexts remain unverified.
No staging, commit, push or account changes. Goal remains active.

### 2026-09-19 — Preserve partial replies and report activity honestly

Confirmed root cause for lost partial replies: `streaming/loop.ts` exited the
provider-attempt loop before committing an assistant message whenever a provider
failed terminally or the caller cancelled. The native runner then synchronized
that incomplete state into the durable session, overwriting the live-only view.
The loop now retains the final attempt's already-emitted text and reasoning on
terminal failure, cancellation, cancellation during retry backoff and iterator
abandonment. It excludes incomplete tool calls and synthetic error diagnostics.
A successful retry continues to replace its failed attempt; no duplicated
partial text is appended to the successful assistant message.

The disk-resume regression uncovered a second defect: provider-marker cleanup
trimmed every assistant message even when it removed no marker. The sanitizer
now returns the original text when extraction only removed whitespace. This
preserves indentation, trailing spaces, newlines and whitespace-only partials,
and avoids falsely reporting them as repaired provider markers. Actual marker
cleanup remains tested and unchanged.

Terminal titles now report `idle`, `working`, `waiting` or `disconnected`.
Disconnect takes precedence over retained busy/approval state. Idle has a neutral
marker, replacing the success checkmark previously shown after cancellation and
provider errors. This does not change the unresolved transcript-footer defect.

Focused verification: **56 runtime tests passed** across five files, including
native daemon cancellation/failure, real disk reload and the next model request;
**two UI title regressions passed**. Logs:
`partial-persistence-focused.log`, `partial-title-focused.log` under the audit
directory. Native persistence tests assert exact partial code whitespace and
reasoning, no synthetic diagnostics in model history, and a successful next turn.
Stream tests cover terminal timeout, pending tool-call safety, backoff cancellation,
iterator abandonment and retry replacement. Marker tests also verify idempotent
repair and zero false repair counts.

Actual-use acceptance repeated the isolated native Anthropic HTTP/local-broker
fixture with source TUI at 80×24 and normal CLI remote daemon. Session
`7755a628a319` received text, was cancelled with Escape, then refused another
request after grant revocation. The native HTTP stream observed cancellation.
After stopping all fixture services, a fresh daemon and TUI resumed that same
session: the previously lost partial assistant reply was visible. Stopping the
owned resumed daemon changed the terminal title to `disconnected`; cancellation
and refusal had changed it to `idle`, without a success glyph.

Evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `partial-persistence-tui.ts`, `partial-persistence-info.json`
- `partial-persistence-terminal.raw`, `partial-persistence-result.json`
- `partial-persistence-resumed-terminal.raw`, `partial-persistence-saved-check.json`
- `partial-persistence-daemon.log`, `partial-persistence-resume-daemon.log`

Full root check, test and build passed: **4065 runtime passed, 3 skipped, 0 failed;
1424 UI passed across 145 files**. Logs: `{check,test,build}-partial-persistence.log`.
Build was repeated after a comment-only clarification; final runtime build ID
`ef2c23e0dcba85c2`. `git diff --check` passed. All owned fixture/TUI/daemon processes
are stopped.

Remaining confirmed defects are explicit: interrupted/failed outcome annotations
are not yet durable, and transcript footers can still say `done` after failure or
cancellation (including after the newly preserved partial text is resumed).
Outcome storage must remain separate from provider diagnostic text and be
available to history paging/replay. The machine consent/SSH broker wiring remains
unfinished. No external provider or SSH acceptance was added, and no commit,
staging, push or account change was performed. Goal remains active.

### 2026-09-19 — Durable turn outcomes and truthful transcript receipts

| User expectation | Current behavior / source | Confirmed defect or limitation | Implemented correction | Automated and actual-use verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A completed, failed or interrupted turn retains its actual outcome after reopening | `types/turnOutcome.ts`; `streaming/loop.ts`; `daemon/turnRunner.ts`; `daemon/runtime.ts` | Terminal reason was transient; resumed partial text looked like a normal completed reply, and failure before output had no durable explanation | Validated versioned outcome on the terminal raw message; native messages carry non-enumerable WeakMap metadata which the daemon explicitly serializes; setup failure preserves its submitted prompt exactly once | Six native runtime regressions cover partial cancellation/failure, no-output and setup failure, first-failure discovery, disk restart, next provider request and late cancellation. Actual session `20ff16a3c48b` retains completed, failed, interrupted and failed-before-output outcomes across daemon restart | Historical records without a reason remain unknown. Hard crash before final snapshot can lose the outcome; no crash-durability acceptance claimed. No external provider tested |
| Transcript receipts do not label failed or interrupted work done | `ui/gatewayAdapter.ts`; `domain/messages.ts`; `app/createGatewayEventHandler.ts`; `opentui/messageLine.tsx`; `app/useMainApp.ts` | A receipt inferred success from the presence of assistant text; no-output failure had no assistant row on which to show its state | Dedicated outcome receipt independent of assistant content; distinct completed/interrupted/failed/limited labels, neutral unknown for legacy records; no success bell for failed or stopped turns; outcome closes the rail before later notices | Five adapter/render tests at 28/80 columns; four explicit-outcome handler cases; prior interruption assertions strengthened. Actual source TUI shows completed, failed and interrupted receipts; restart displays interrupted partial text and a failed receipt directly below the no-output request; disconnect retains both | Physical emulator screenshot not taken; real PTY escape streams captured. Old client versions ignore additive metadata |
| Paging and context maintenance preserve the outcome once, at its proper boundary | `daemon/historyPage.ts`; `daemon/server.ts`; `context/toolResultPruner.ts`; `context/screenshotSuperseder.ts`; `daemon/compactionRunner.ts` | Tool-only history paging discarded the original result record; pruning cloned messages without native side metadata; compaction archives omitted that metadata | Validated replay notification and one outcome action per history boundary; retained metadata through tool/screenshot pruning and compaction archives; summaries do not invent outcomes for removed history | One-action paging regression covers user, assistant and tool endings; real Unix gateway resume and page response retain outcome; three metadata/context regressions check validation, pruning, screenshot replacement, retained compaction tail, archived history, and exclusion from summarizer/model serialization | Full historical rendering of compacted archives remains part of the broader history audit |
| Escape racing a completed response does not rewrite completion | `daemon/runtime.ts:submitTurnOwned` | A direct runtime reproduction reported native completed, then emitted/persisted aborted when cancellation arrived during teardown | A known terminal reason wins over later controller cancellation; `turn_end.cancelled` agrees with that reason | Before reproduction returned completed then aborted; the same native runtime script now returns completed with cancelled false and persists completed. Regression cancels synchronously on the native completed status; ordinary cancellation regressions still pass | No new physical-keyboard timing stress trial; deterministic native boundary and existing UI race regression are the evidence |

Actual source TUI ran at 80×24 against isolated real local daemons and a loopback
Anthropic HTTP fixture through the local provider broker. The first two waiting
requests hit the fixture's default Bun.serve 10-second idle timeout before Escape;
they correctly became provider failures. A third was promptly cancelled and
became interrupted. After broker revocation, the next request failed without an
additional HTTP call. Saved history contains no synthetic fixture credential.
A new daemon and renderer then reopened the same session; both interrupted text
and the failed no-output request kept their receipts. Stopping only that owned
daemon showed disconnected while retaining those receipts. All fixture and TUI
processes were stopped. This is not acceptance of the unfinished SSH consent UI
or of any external provider account.

Evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `outcome-terminal.raw`, `outcome-resumed-terminal.raw`: real terminal captures.
- `outcome-acceptance.json`, `outcome-saved-check.json`, `outcome-result.json`:
  receipt checks, stored outcomes, and HTTP/cancellation/revocation accounting.
- `outcome-tui.ts`, `outcome-info.json`, `outcome-daemon.log`,
  `outcome-resume-daemon.log`: isolated fixture and daemon evidence.
- `outcome-race.ts`, `outcome-race-after.json`: native late-cancellation reproduction.

Final full repository gate passed after the late-cancel correction:
`bun run check && bun run test && bun run build` — **4,073 runtime passed,
3 skipped, 0 failed; 1,433 TUI passed across 146 files**. Runtime build ID
`32105814a967c40e`. Logs: `check-outcome-complete.log`,
`test-outcome-complete.log`, `build-outcome-complete.log`. `git diff --check`
passed. Earlier `test-outcome.log` retains two exact-object assertion failures;
those tests now explicitly assert the outcome metadata instead of rejecting it.
The broader audit remains in progress; this checkpoint does not close the
remote setup consent, integration, or full workflow acceptance work.

### 2026-09-19 — Reachable local-provider consent over SSH

| User expectation | Current behavior / source | Confirmed defect or limitation | Implemented correction | Automated and actual-use evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Review local setup reuse before granting authority | `ui/opentui/machinePicker.tsx`, `remoteProviderReview.tsx`; `ui/lib/remoteTaskSetup.ts` | Broker infrastructure had no user-reachable handoff owner or consent flow | Prepare an idle remote task, inspect safe local inventory, default to current setup, review exact profile/default model/destination/workspace/task and limits, then require A to authorize. Provider-controlled output additionally requires C. Parent owns both connections and broker; only public remote ID/key reach child. | Keyboard tests at 80×24 and 40×18; real-daemon setup tests cover remote-only use, cancellation, disconnect and reopened task; actual production MachinePicker/child TUI at 100×30 over disposable loopback OpenSSH reached a synthetic native Anthropic HTTP service. | External hosts and real configured provider services unverified. Local selection currently uses profile defaults, not a local model catalog. Local capability/effort negotiation remains incomplete. |
| Open a newly prepared task without manufacturing saved history | `ui/lib/machineHandoff.ts`, `ui/gatewayClient.ts`, `ui/opentui/entry.tsx` | Empty task was passed as history resume; daemon correctly refused missing saved history | Pass reviewed public live-session ID/key to child; initialize attaches exact existing task. Preserve empty-history exclusion. | Native daemon regression submits via separate child GatewayClient and asserts one live task identity. Actual SSH child opens fresh task and completes a provider call. Handoff tests cover prepare-before-suspend, cancellation, preparation failure and tunnel loss. | Parent draft/transcript preservation across this complete flow is not established by this checkpoint: the parent harness mounts the production picker, not the full parent app. |
| Reopen saved remote work with fresh consent | `ui/gatewayClient.ts:sessionResume` | Client retained proposed key after daemon resumed history under a different authoritative key; later scoped status RPC failed | Adopt the returned session key after successful initialize | New real-daemon regression; existing resume/reconnect suites pass. Actual before capture reproduces setup failure. After capture reopens the same task `a2ff5928fe2f`, obtains fresh consent and produces another reply. | Resolved in the bounded-preflight checkpoint below: preparation requests metadata only; child resume still restores full history. |
| Disconnect revokes authority without losing work or selecting another provider | `remoteTaskSetup.ts`, `localProviderBroker.ts`, `remoteProviderBindings.ts` | Handoff needed explicit parent lifetime ownership | Close broker/revoke when owner transport drops or window exits; saved task retains its provider requirement and requires fresh review | Actual owned SSH master terminated; child reconnects with its draft intact. Submitting preserved draft yields `turn_failed`, no HTTP call; explicit reauthorization produces next reply. Two authorizations, two revocations, three HTTP calls total (reply, title, resumed reply). | Expiry, pending request cancellation and malformed input have automated coverage; this checkpoint does not add actual terminal acceptance for every expiry/cancellation boundary. |
| Explicitly choose remote credentials instead | `daemon/server.ts:selectProvider,setModel`; `remoteProviderBindings.ts:useRemote`; `ui/opentui/appLayout.tsx` | Persisted local requirement otherwise prevents deliberate remote overrides | Explicit named remote profile removes binding requirement when idle; running or paused provider work refuses; failed selection restores even a malformed requirement. Source label clears on remote selection. | Real-daemon tests exercise provider and model selection, saved-state restart, actual native remote runner route, busy refusal and failed persistence; binding regression covers pending and paused streams plus stale client rejection. | Resolved in the bounded-preflight checkpoint below: the current local requirement appears in the top summary and the keep-setup explanation no longer implies remote credentials. |

Terminal evidence under `/tmp/xerxes-tui-audit-20260919/`:
`remote-review-before-terminal.raw` retains the failed reopen;
`remote-review-live-terminal.raw` records cancellation, authorization, successful
reply, tunnel loss, preserved draft, failed submission, return, fresh consent and
successful same-task continuation. `remote-review-acceptance.json` checks the
saved message/outcome sequence, authorization/revocation counts and synthetic
credential absence. `remote-review-live-result.json` records actual HTTP caps
1024/512/1024 and method names, without context or credentials.

The loopback SSH host uses disposable Ed25519 keys and a pinned host key. Both
daemons and the HTTP service are real isolated processes. The production SSH
socket tunnel, preparation, broker and child source TUI are exercised. The
managed installer handshake is replaced by a fixture handshake for an isolated
source daemon; this checkpoint does not claim installer or external-host
acceptance. The source daemon is launched through real SSH. No real provider
credential was authorized. `remote-review-cleanup.json` verifies all four owned
fixture CLI daemon PIDs stopped; fixture SSH and renderer processes also exited.

Focused evidence: `test-remote-handoff-review.log`,
`test-remote-override-runtime.log`, `test-remote-override-ui.log`, and
`test-remote-reopen.log`. Broad audit and the concrete follow-ups above remain
open; this checkpoint is progress, not completion.

Final gate for this checkpoint completed after the authoritative resume-key fix:
`bun run check && bun run test && bun run build` passed. **4,074 runtime tests
passed, 3 skipped, 0 failed; 1,453 TUI tests passed across 148 files.** Runtime
build ID: `bab2f9945889a398`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-remote-review-final.log`.
`git diff --check` passed. The final remote JSON and JSONL transcripts were also
scanned for the synthetic credential sentinel and contained none.

### 2026-09-19 — Bounded preparation and accurate provider source

| User expectation | Current behavior / source | Confirmed defect | Correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Review remote setup without downloading the entire conversation first | `ui/lib/remoteTaskSetup.ts`, `ui/gatewayClient.ts:sessionResume` | Preflight replayed all history before its metadata-only status call; 2,000 messages generated 16,944,184 transferred bytes | Preflight resume requests the existing `history_limit:0`; ordinary child resume keeps its normal transcript behavior | Real-daemon proxy regression verifies no history sentinel or replay frame crosses the preflight socket, then verifies child resume restores all 101 messages. Five-sample before/after measurement below. Existing resume, disconnect, cancellation and fresh-consent regressions pass. | Full child resume still replays history; this change bounds setup review only. Older servers that ignore the optional history limit are not measured. |
| Know whether a saved task requires local access | `ui/opentui/remoteProviderReview.tsx`; `ui/lib/remoteTaskSetup.ts` | Bound task was labeled “remote configuration”; keeping its setup falsely implied remote credentials. Valid 512-character profile names produced labels longer than the input limit and could lose their source label | Top summary uses the local requirement, otherwise an explicit remote-provider label. Keep-setup copy explains reauthorization or an explicit remote override. Accept the complete bounded display label | 80×24 and 40×18 render regressions; actual SSH 40×18 review of reopened task `587b016b2503` displays the local requirement and explicit remote-override instructions. Real-daemon test covers the maximum profile-name length. | No new external-provider or non-loopback-host acceptance. |

Performance evidence: `preflight-before.json`, `preflight-after.json`,
`preflight-comparison.json`, and reproducible Bun harness
`remote-preflight-bench.ts`, all under `/tmp/xerxes-tui-audit-20260919/`.
The seeded conversation has 2,000 messages containing 8,216,890 content bytes.

| Measurement | Before | After |
| --- | ---: | ---: |
| Median daemon-to-preflight transfer | 16,944,184 bytes | 50,098 bytes |
| Historical replay rows | 2,000 | 0 |
| Median preparation latency, five samples | 1,083.83 ms | 80.98 ms |

This is an actual isolated Unix-socket daemon/client measurement, not SSH
latency. Raw samples also record RSS deltas, but daemon, client and proxy
instrumentation share a process; those deltas do not establish retained-heap
behavior or a production memory bound. No provider calls occur in the benchmark.

Actual narrow-terminal evidence: `remote-review-narrow-terminal.raw`,
`remote-review-narrow-result.json`, `remote-review-narrow-acceptance.json` and
`remote-review-narrow.ts`. Production picker/preparation/broker/child TUI ran at
40×18 through real disposable loopback SSH, with the same isolated source-daemon
handshake substitution documented above. Keyboard paging exposed the long
workspace path and controls; Tab/arrows changed grant scope to 15 minutes,
50 requests, 4,096 output tokens and one concurrent request before review and
explicit authorization. A native synthetic HTTP reply completed; exit revoked
access; reopening showed the saved local requirement. Cancellation of that new
review granted no further access. Two HTTP calls (reply and title), one grant,
one revocation. Credential sentinel absent from terminal, daemon log, JSON and
JSONL transcript; owned renderer, SSH and daemon PIDs verified stopped.

Newly confirmed follow-ups, **not fixed by this checkpoint**:

| Requirement | Source / reproducible evidence | Current defect | Next correction / uncertainty |
| --- | --- | --- | --- |
| Repeated SSH use must not exhaust authority storage after revocation | `daemon/providerRelays.ts:revoke`, `security/localProviderRelay.ts:authorize`; `local-route-followups.ts` and `.json` | 128 sequential authorize/revoke cycles on one owner leave every authority entry retained; the 129th fails with an unsupported-request diagnostic even though none are active | Release terminal authority resources while retaining bounded, truthful status/revocation behavior; test repeated handoff, stale handles, active streams and isolation |
| Effort changes on a local-bound task must not change remote provider defaults | `daemon/server.ts:setReasoning,reasoningLevels,sessionProfileName`; actual isolated RPC probe in `local-route-followups.json` | A local-bound task with a remembered remote profile changed that profile’s saved effort from low to high through `set_reasoning`; effort availability is also derived from remote configuration | Scope effort to the actual local route and task; complete safe local capability metadata negotiation. Probe made no provider calls or external changes. |

The broad workflow audit remains active. These confirmed defects preclude a
completion claim.

Full root gate completed for this checkpoint: `bun run check && bun run test &&
bun run build` passed; **4,074 runtime tests passed, 3 skipped, 0 failed; 1,456
TUI tests passed across 148 files**. Build ID `7f9d921d98dcc3e9`. Logs:
`check-preflight-bounded.log`, `test-preflight-bounded-full.log`, and
`build-preflight-bounded.log` in the evidence directory. `git diff --check` passed.

### 2026-09-19 — Relay retirement, durable effort and untouched-task resume

| User expectation | Current behavior / source | Confirmed defect | Implemented correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Repeated SSH handoffs do not consume permanent grant slots | `daemon/providerRelays.ts`; `security/localProviderRelay.ts` | Sequential authorize/revoke reached the 128-entry limit with no active grants | Release expired/revoked authority after backend settlement; retain at most 128 credential-free terminal status records; isolate owners and retain noncooperative work until settlement | Original probe before: 128/129 accepted; after: 129/129. Real-daemon socket regression: 300 cycles with a separate live grant unaffected. Lifecycle tests cover 128 expirations, noncooperative cancellation, stale handles, disconnect and buffered final output on exhausted grants. `test-relay-lifecycle-all.log`: 55 passed across five files | Truly live grants remain limited to 128. Old terminal status handles eventually return unavailable. This establishes bounded record counts, not a production retained-heap measurement |
| Local task effort does not rewrite a remote provider default | `daemon/server.ts:configureReasoning,setReasoning` | Both selection routes saved local task effort into a remembered remote profile; unsupported slash runtime could globally reload | Guard remote profile writes for every own local-requirement marker; refuse global fallback for local tasks | Real socket RPC/slash regressions, saved task restart, unsupported runtime and failed persistence/retry. Original probe now leaves remote low while task becomes high. Actual SSH `/thinking high` leaves remote profile low | **Still open:** effort availability is derived from remote profile/catalog, not safely negotiated local capabilities. This change isolates writes only |
| A successful effort selection survives a daemon restart | `daemon/runtime.ts:setSessionReasoning` | Setter acknowledged an in-memory change without saving until a later turn or flush | Save through the normal transcript store before success; empty task exclusion is unchanged; filesystem failures propagate | Runtime resume test now restarts without flushing. RPC/slash tests verify saved metadata and a separate runtime reload. Actual SSH transcript inspected immediately after `/thinking high`, before renderer or daemon exit, contains high | A failed save does not acknowledge success, but the in-memory setting currently remains changed; full transactional rollback of failed settings writes remains to audit |
| Reopening a remote task that has not sent a message preserves its identity | `daemon/server.ts:initialize`; `ui/lib/remoteTaskSetup.ts` | New preflight connection looked for saved history using the public ID, missing the live task stored under its original key | Resolve explicit resume ID to an existing live task before durable lookup; retain workspace validation before attachment/overrides | Before: actual SSH setup error after an untouched task exit; two new remote/local setup regressions fail. After: both pass, same ID/key/object and no phantom file. Separate real-socket test preserves active work, rejects another workspace without retargeting its connection, and cancels through the attached client. Actual SSH reopens untouched task `ad1d02c1d5a4` | Empty tasks do not survive daemon loss by design. Saved-session and connection-recovery suites pass; external-host acceptance still unavailable |
| Transparent appearance works in SSH and stays local | `ui/app/appearance.ts`, local child renderer | Additional actual-use verification of the earlier appearance implementation | No further appearance code change needed | Actual SSH `/appearance transparent`, child restart, same transcript and high effort; terminal uses SGR49. Preference exists in local home and not remote home. `remote-effort-appearance.json` | Physical terminal-window opacity remains controlled by the emulator; not visually inspected |

Evidence in `/tmp/xerxes-tui-audit-20260919/`:

- `local-route-followups-before.json` / `local-route-followups.json` and the Bun
  reproducer retain the 129-cycle and profile-isolation comparison.
- `test-untouched-resume-before.log` preserves both failing regressions;
  `test-live-resume-after.log`, `test-live-workspace-resume.log`, and
  `test-local-effort-isolation.log` record passing checks.
- `remote-effort-before-terminal.raw` retains the untouched-resume failure.
  `remote-effort-live-terminal.raw`, `remote-effort-live-acceptance.json`,
  `remote-effort-live-durable-before-exit.json`, and `remote-effort-live.ts`
  capture the corrected production picker/preparation/broker/child TUI flow at
  80×24 over real disposable loopback OpenSSH. Only the managed installer
  handshake is replaced with the isolated source-daemon handshake.
- Two synthetic native Anthropic HTTP requests (reply/title, caps 1024/512), one
  explicit grant and one revocation. No provider call was made for the effort
  change or unapproved reopened route. Synthetic credential sentinels are absent
  from capture, daemon log and durable JSON/JSONL transcripts. Owned renderer and
  SSH PIDs are stopped; process inspection found no remaining fixture daemon.

This checkpoint is progress. Local capability negotiation, failed-setting
transaction semantics and the wider supported-workflow audit remain unfinished.

Full root gate completed for this checkpoint: `bun run check && bun run test &&
bun run build` passed. **4,078 runtime tests passed, 3 skipped, 0 failed; 1,463
TUI tests passed across 148 files.** Runtime build ID `7f39a66832d9ea5f`.
Logs: `/tmp/xerxes-tui-audit-20260919/{check,test,build}-relay-retirement-full.log`.
`git diff --check` passed. No commit or push. The goal remains active.

### 2026-09-19 — Local reasoning controls and storage-failure recovery

| User expectation | Current behavior and source | Confirmed defect or limitation | Implemented correction | Automated and actual-use evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| An SSH task using a local provider offers that provider's reasoning controls | `daemon/server.ts:sessionReasoningLevels`, `daemon/localReasoningCapabilities.ts`, `protocol/localProviderCapabilities.ts`, `ui/lib/localProviderBroker.ts`, `ui/lib/remoteTaskSetup.ts`, `daemon/remoteProviderBindings.ts` | Picker and validation could derive controls from an unrelated remote profile | Authorization returns bounded, versioned, secret-free local metadata with exact model and provenance; broker/bind/persisted reads validate it. Local-bound tasks never infer controls from remote settings. Legacy/malformed snapshots show an actionable unavailable state. Three-second discovery deadline and post-discovery authority check preserve expiration/revocation | Runtime/broker/gateway regressions cover reported, bundled and fallback controls, mismatched model, extra credential fields, noncooperative discovery timeout, expiry during discovery, legacy/corrupt metadata, inherent reasoning and profile isolation. Actual 40×18 and 80×24 SSH source TUI uses local Anthropic fallback controls against a conflicting remote Deepseek profile; source is explicitly labeled not live-verified | Snapshots do not establish external provider acceptance or context capacity; context remains unknown. Capability changes require fresh review. Local profile defaults remain private |
| Failed effort changes preserve the last working choice and permit retry | `daemon/runtime.ts:setSessionReasoning`, `serializeSessionWrite`, `writeSession` | A failed disk save left the rejected effort visible in memory and available for a later flush | Stage reasoning and context delta privately; serialize session writes; publish only after success. Return an actionable bounded storage error without raw filesystem diagnostics | Two regressions failed before correction, then passed: failed write preserves live/durable state; failed pending choice cannot escape through a second choice or concurrent flush. Actual SSH saved `low`, replaced only its isolated session JSON path with a directory, observed failed `high` while RPC still returned `low`, restored storage and retried; JSON saved `high` before exit | This transaction correction covers reasoning. Model/mode/permission persistence and their concurrent mutations remain under audit; empty tasks still do not create phantom durable histories |
| Read capability source and recovery instructions in narrow terminals | `ui/gatewayClient.ts:reasoningLevels`, `ui/opentui/reasoningPicker.tsx` | Gateway dropped note/shape; long picker details were clipped | Preserve source note/shape, wrap in a bounded shared scroll box, support PageUp/PageDown even with no choices, reserve room for selected row and keyboard hints | 40×18 regressions reach the end of long recovery text, close with Escape, and select row 10 of 12 with details visible. Actual 40×18 SSH picker displays local fallback provenance and accepts a selection | Character-frame and PTY evidence; no physical terminal screenshot |
| Local-bound tasks do not report unrelated remote-provider failures | `daemon/server.ts:refreshActiveModelCapabilities` | Actual SSH child displayed a failed remote model-discovery warning after local authorization | Skip automatic remote discovery for local-bound tasks; ignore pending results after task/provider changes | Real-socket held HTTP discovery regression switches to local binding before releasing a failed response; no stale warning and resume makes no remote request. Actual before capture has the unrelated warning; corrected 80×24 child capture does not | Explicit provider-management queries still intentionally inspect the selected provider |

Evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `test-effort-transaction-before.log` records the original failed regressions.
  `test-local-capabilities-runtime-v3.log`: 21 runtime tests passed across four files.
  `test-local-discovery-v3.log`: 27 UI tests passed across three files.
- `local-capability-live-terminal.raw`: actual 40×18 picker and failed-write/retry
  trial, session `bdec529d495b`. The initial storage error exposed raw EISDIR;
  the later recovery trial verifies its replacement.
- `local-discovery-noise-before-terminal.raw`: actual unrelated discovery warning.
  `local-capability-recovery-terminal.raw`: corrected 80×24 run, session
  `5c4ebc6d4cfb`; matching `-failed-save.json` and `-after-retry.json` verify low
  after failure and durable high after retry.
- `local-capabilities-acceptance.json` and `verify-local-capabilities.ts` record
  six passing acceptance assertions, unchanged remote profile in all three runs,
  zero synthetic credential sentinels in captures/logs/durable transcripts, and
  stopped owned renderer/SSH PIDs. Process inspection found no remaining fixture
  daemon. The real user daemon was not interrupted.

These runs use the production picker, preparation, broker and child TUI with real
disposable loopback OpenSSH. Only the managed installer handshake is substituted
with the isolated source-daemon fixture. Each task trial made two synthetic native
Anthropic HTTP calls (reply/title, 1024/512 output caps), one grant and one
revocation; the warning-only trial made no provider calls. This is not external
service or external-host acceptance. The broader goal remains active.

Full root gate for this checkpoint completed: `bun run check && bun run test &&
bun run build`, followed by `git diff --check`. **4,085 runtime tests passed,
3 skipped, 0 failed; 1,472 TUI tests passed across 149 files.** Runtime build ID
`ed32cb79f171ab22`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-local-capabilities-full.log`.
No source edits occurred during that gate. No commit or push.

Next confirmed storage work, reproduced independently while the gate ran:

| User expectation | Source / current behavior | Confirmed defect | Correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Failed model selection leaves the working model and provider pin intact | `daemon/runtime.ts:setSessionModel` mutates before save | Injected storage failure returns an error but live model becomes `rejected-model`; a later flush writes it | Pending | `setting-failure-probe.ts` and `.json`, isolated actual transcript store/runtime | TUI failure/retry, provider metadata, concurrent flush and restart boundaries need regression and actual-use coverage |
| Failed interaction-mode selection leaves the working mode intact | `daemon/runtime.ts:setSessionMode` mutates before save | Failed `code` → `plan` save leaves plan mode active and later durable | Pending | Same isolated probe, with a separate saved session | Mode callback, running-turn interaction, cancellation and restart boundaries remain to verify |

### 2026-09-19 — Failed model/mode writes, restart continuity and pasted commands

| User expectation | Current behavior and source | Confirmed defect | Implemented correction | Automated and actual-use evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Failed model selection preserves the working route and can be retried | `daemon/runtime.ts:setSessionModel` | Previous checkpoint's isolated storage probe showed rejected model active and later persisted | Stage model/provider pin and delta privately inside the session write queue; publish only after successful save, with bounded actionable errors | `sessionSettingPersistence.test.ts`: rollback, provider pin, deltas, retry, restart, pending selection plus reasoning/flush, and cancellation. Actual 80×24 source TUI against isolated real daemon: blocked JSON destination makes `/model gpt-4.1 --provider pinned` fail; footer and RPC retain gpt-4o; restoring storage and retry saves gpt-4.1 with pinned profile | No real provider call is needed for this settings path; external model availability remains unverified. Provider-binding replacement lifecycle is a separate boundary |
| Failed mode changes preserve mode/plan flags and host behavior | `daemon/runtime.ts:setSessionMode` | Failed plan-mode save previously changed live and later durable state | Stage mode/plan/delta before writing; notify host only after durable publication; preserve retry and queued writes | Same regression suite verifies no callback on failed/pending writes and exactly one on success, plus cancellation and restart. Actual `/mode plan` storage failure leaves code mode; successful retry persists plan mode | Hard process death during filesystem replacement remains governed by the existing transcript store; no new power-loss trial |
| Recover the same task and draft after daemon restart | Gateway recovery with corrected model/mode persistence | Additional boundary verification for these corrections | No further recovery code change | Actual daemon stopped and a fresh daemon loaded the saved session at the same socket. Existing source renderer displayed disconnected then idle; F10/Escape redraw confirmed `Keep this draft across restart`, prior transcript, same session 1326432c449e, gpt-4.1 and plan mode | Local Unix-socket acceptance this checkpoint. Earlier loopback SSH recovery evidence is separate; external host unverified |
| Pasting commands does not produce unrelated attachment advice | `ui/app/useComposerState.ts:looksLikeDroppedPath` | Actual pasted `/model gpt-4.1 --provider pinned` matched a dot anywhere in the string and suggested `/image` | Only inspect the leading path token for the bare absolute-path heuristic; retain explicit URI, quoted, relative, home and Windows paths | Two regressions cover model/search/file/MCP/custom commands and real path forms; paste/clipboard/early-input suites: 55 passed. Before capture shows spurious hint; fresh source TUI repeats identical paste with no hint and correct model selection | Heuristic advice only, never automatic attachment; ambiguous unquoted root paths with spaces can be made explicit by quoting |

Evidence under `/tmp/xerxes-tui-audit-20260919/`: `setting-terminal.raw`,
`setting-paste-after-terminal.raw`, `setting-failed-status.json`,
`setting-retry-persisted.json`, `setting-restarted-info.json`, and
`setting-acceptance.json`. The fixture creates a real isolated daemon with seeded
history; the source TUI uses its ordinary GatewayClient and commands. No external
provider calls occurred. Owned daemon and both renderers were stopped. The user's
daemon was not restarted. `test-setting-cancellation.log` records six persistence
regressions, and `test-setting-persistence-after.log` records the initial broader
90-test pass. The full gate follows; this is progress, not goal completion.

Full root gate completed for the model/mode and paste corrections: `bun run check
&& bun run test && bun run build`, then `git diff --check`. **4,091 runtime tests
passed, 3 skipped, 0 failed; 1,474 TUI tests passed across 150 files.** Runtime
build ID `a237126baf230f31`. Logs:
`/tmp/xerxes-tui-audit-20260919/{check,test,build}-setting-full.log`. No source
edits occurred during the gate. No commit or push.

The settings audit found an additional open boundary during this gate:

| User expectation | Current behavior/source | Confirmed defect | Correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A stricter acknowledged permission choice survives daemon loss | `daemon/runtime.ts:setSessionPermissionMode` updates in memory without saving | After selecting manual, durable metadata has no permission choice and a fresh runtime resumes with effective auto permissions | Pending | `permission-persistence-probe.ts` / `.json`: live manual, durable null, resumed explicit null, effective resumed auto | Needs atomic persistence, failed-write/concurrent/cancellation regressions, session-scoped permission reporting and actual TUI restart verification |

### 2026-09-19 — Repository review: permission persistence and resumed policy

| User expectation | Current behavior/source | Confirmed defect | Implemented correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| An acknowledged policy survives daemon loss | `daemon/runtime.ts:setSessionPermissionMode` | In-memory manual choice resumed as daemon default before a later flush | Stage and serialize persistence before publishing mode, pin and context delta; fixed actionable save error | Four `permissionPersistence.test.ts` regressions cover restart without flush, failed save/retry, concurrent writes and turn cancellation. Actual 80×24 source TUI with isolated real daemon saves manual, rejects accept-all with blocked storage, retries plan and resumes plan in a fresh process | No power-loss filesystem trial; external SSH host not exercised for this particular correction |
| Permission reports and resumed footer describe the selected task | `daemon/server.ts` permissions command and session initialization | Bare command used daemon default; initial status event overwrote the correct task policy after resume | Use session policy before daemon default in both paths | Real socket regression fails before and passes after, checking two tasks and all resumed policy events. Actual restart trial caught the wrong footer; after correction the same renderer retains its draft/history/identity and displays `plan only, no writes` | Existing rendering coverage applies at other sizes; new actual-use trial is 80×24 |

Evidence: `/tmp/xerxes-tui-audit-20260919/permission-terminal.raw`,
`permission-durable-before-failure.json`, `permission-durable-retry.json`,
`permission-restarted-info.json`, and `permission-acceptance.json` (eight passing
assertions). The renderer and owned daemons were stopped; the user daemon was
not restarted. No external provider call occurred. The capture contains both
before/after footer behavior and no synthetic credential values. Focused logs:
`review-permission-cancellation.log`, `review-permission-resume-before.log` and
`review-permission-resume-after.log`. Final repository gates are recorded in the
linked repository review. This checkpoint does not complete the broader audit.

### 2026-09-20 — One approval for the local provider setup

| User expectation | Current behavior/source | Confirmed defect or limitation | Implemented correction | Automated and actual-use verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Approve the SSH task's local providers together, then switch without repeated consent | `ui/opentui/remoteProviderReview.tsx`, `ui/lib/remoteTaskSetup.ts`, `daemon/remoteProviderBindings.ts` | Previous binding authorized one profile/model; changing provider required preparing the task again | One destination review covers all supported configured profiles; each retains a separate local grant, common expiry, per-provider limits and private route. Advertised bundle capability preserves older daemon support. Partial failure/cancellation closes every prepared grant | Narrow/normal rendered approval tests; two real daemons test one review, two grants, switching, replies and revocation; binding tests cover ownership, bounds, disconnect and restart. Actual 80×24 TUI through isolated loopback OpenSSH approved once, completed turns using both providers and revoked both on exit | Up to 32 profiles, each profile's configured model only. Memory-only authority ends at expiry/disconnect/revocation; reconnect requires a new setup review. Unsupported integrations remain explicitly unavailable. No permanent host trust or arbitrary-model authority |
| Model picker and agents can see and use the approved local setup | `daemon/server.ts:modelInventoryToolRequest/setModel`, `daemon/subagentHost.ts`, `ui/gatewayClient.ts` | Remote inventory suggested remote credentials; explicit local child profiles were rejected; picker falsely described approved choices as fallback results | Session-scoped picker/inventory list approved local routes and preserve selected profile for native child work, including profiles sharing a model. Same-named remote profiles cannot silently replace an unapproved local model. Limits stay editable on the local workstation | Real-daemon integration covers discovery and a completed second-provider turn; native subagent test completes on the second profile with matching route fingerprint. Rendering tests at 76/140 columns suppress the remote limit editor. Actual SSH picker lists both providers, switches and completes without another approval; confirmation displays the correct discovery source | Catalog includes configured approved models only, not the provider's full live catalog. Additional models need local configuration and another setup review. Existing external services and the user's actual SSH host were not exercised |
| Provider selection survives failures and restart without changing credential source | `daemon/runtime.ts:setSessionModel`, `daemon/server.ts` remote override paths | New multiple-profile routing requires a durable selected local profile and correct rollback | Save local profile/model together before publishing; failed explicit remote overrides restore the previous local profile label and requirement; absent live authority remains an error | Storage failure/retry/restart regression; model/provider override failure regressions; group disconnect and restart tests cannot revive authority or invoke a remote fallback | A failed explicit override can require reopening local access, as before; authority is never restored from persisted metadata |
| Empty optional provider fields work in `list_available_models` | `runtime/modelInventory.ts` | Screenshot's `provider_profile:""` was treated as an unknown profile, blocking discovery | Trim provider input; empty/whitespace means provider inventory. Usage lookup still requires an explicit nonempty profile | Exact screenshot-shaped regression plus whitespace and usage validation; real local-bound daemon inventory returns both approved providers | Live quota/catalog lookup is unavailable through these scoped local grants and is not invented |

Actual-use evidence under `/tmp/xerxes-tui-audit-20260919/`:

- `provider-bundle-live-terminal-after.raw` and `provider-bundle-live-result.json`:
  the complete first-provider turn, picker switch, second-provider turn and exit.
- `provider-bundle-final-terminal.raw` and `provider-bundle-final-result.json`:
  confirmation of honest discovery labeling, second-provider success and cleanup.
- `provider-bundle-acceptance.json`: 19 passing checks across both trials, including
  unchanged remote profile defaults, absence of local credential sentinels in
  terminal/log/history, grant revocation, stopped owned processes and removed
  fixture SSH private keys.

These used production machine picker, setup, broker, child TUI and actual isolated
local/SSH daemons. The managed installer handshake was replaced with an isolated
source-daemon handshake; provider responses came from a local synthetic Anthropic
HTTP service. This establishes actual TUI/SSH integration behavior, not live
external-provider or external-host acceptance. The final limit-editor guard was
verified by rendering regressions after the confirmation trial. No active user
daemon or task was interrupted. No performance improvement is claimed.

Focused final regressions: 54 tests across model picker, remote task setup and
remote provider gateway integration. Final root `bun run check && bun run test &&
bun run build` and `git diff --check` passed on the final source: 4,099 runtime
tests and 1,484 TUI tests, zero failures. Three runtime tests remain skipped
(two Windows-only tests and installed-clangd acceptance). Logs:
`bundle-final-{check,test,build}.log` in the evidence directory. No source edits
occurred during that gate. The broader capability audit remains in progress.

### 2026-09-20 — Follow-up input to finished spawned agents

| User expectation | Current behavior/source | Confirmed defect | Correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A message reaches the existing agent even if it just finished | `daemon/subagentHost.ts:sendInput`, `tools/claudeTools/agentOps.ts:SendMessageTool` | A running-only admission check rejected completed children and required changing to AgentTool; completion could race a status check | Automatically continue terminal open children with stable id, saved history and original configuration. Serialize concurrent id/name admission through asynchronous recovery. Track continued work for parent result delivery | Four focused regressions cover completed/failed/interrupted children, concurrent input, persisted conversation after recovery, empty input, foreign ownership, closed handles and policy invalidation. Existing native-host/tool suites passed. Actual 80×24 source TUI over an isolated real daemon ran AgentTool to completion, then SendMessageTool succeeded and rendered the child's revised result | Live external providers and an external SSH host were not exercised for this correction. Explicitly closed or policy-invalidated agents do not silently restart. Existing provider/budget/workspace checks still apply |

Evidence under `/tmp/xerxes-tui-audit-20260919/`: `followup-terminal-final.raw`,
`followup-result.json`, `followup-acceptance.json`, `followup-focused.log`, and
`followup-last-tests.log`. The first fixture attempt used an incorrect injected
provider delta field and did not exercise a tool; only the corrected final capture
is acceptance evidence. Providers were deterministic injected clients. The TUI,
daemon, native child runner, tool dispatch and durable conversations were real.
The owned fixture processes were stopped without restarting user tasks.

Final gate on this source passed: `bun run check && bun run test && bun run build`
and `git diff --check`; 4,103 runtime tests and 1,484 TUI tests passed, zero failed.
Three environment-dependent runtime tests remain skipped (Windows-only behavior
and installed clangd). Logs: `followup-full-{check,test,build}.log`. The eight
actual-use assertions in `followup-acceptance.json` passed. No source edits were
made during that final gate. The broader audit remains in progress.

### 2026-09-20 — Automatic SSH setup recovery

The designated `n_server_spot_m` host exposed a stale directory lock blocking
managed remote updates. The shared TUI/desktop bootstrap now uses kernel-owned
locks, waits for concurrent setup, rechecks completed builds, and handles legacy
publication races without deleting unknown staging directories. The reported
desktop retry successfully loaded remote session discovery after repair.

The [SSH recovery matrix and terminal evidence](ssh-setup-recovery-2026-09-20.md)
records eight shell regressions on macOS and Linux, real isolated daemon update
and saved-history acceptance on both machines, the full Bun gate, installation,
and exact remaining limits. This is live SSH setup acceptance for that named
host; it does not establish live provider or integration acceptance. The broader
audit remains in progress.

### 2026-09-20 — Work groups, active agents, compact commands and workspace review

The [workflow repair matrix](desktop-workflow-repairs-2026-09-20.md) records the
reported desktop/TUI corrections, source references, native captures, real local
and SSH daemon checks, Git 2.34.1 worktree verification, and remaining limitations.
Operational groups now exist from their first event and start closed. Active
agents precede collapsed history; agent receipts reconcile identities with chat.
Commands use compact expandable rows. Recursive file branches, untracked-file
paging and individual previews are reachable in their respective interfaces.
The broader audit remains in progress.

### 2026-09-20 — Clickable agent inspection and output reading

The [agent inspection/output matrix](agent-inspection-output-2026-09-20.md)
records clickable desktop agent rows, reported base-agent assignments, scoped
retained detail access, a shared output reader, and Escape/focus protection.
Native interaction, packaged renderer plus isolated daemon, and local/SSH RPC
checks passed. Historical evidence remains bounded and external providers were
not exercised. The broader audit remains in progress.

### 2026-09-20 — Session work survives UI closure

The [session lifetime matrix](daemon-session-lifetime-2026-09-20.md) records the
client-disconnect cancellation defect, the additive session-owned turn contract,
same-session interaction recovery, and explicit-stop regressions. An actual TUI
and the desktop transport over SSH completed isolated work after client closure
and lease expiry, then restored the same sessions. UI-owned provider forwarding
still needs its connection; this change does not extend credential authority or
promise recovery across daemon termination. Both updated client and daemon are
required. The broader audit remains in progress.
