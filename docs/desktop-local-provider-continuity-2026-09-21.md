# Desktop local-provider continuity

The desktop SSH path previously attached directly to remote configuration. The
TUI's local provider review and relay existed, but the GUI never created local
authority or handled its private request frames. A local key edit therefore had
no effect on a GUI task using the remote host's credentials.

| User expectation | Current behavior/source | Confirmed defect or limit | Correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Reuse local providers in an SSH GUI task | `desktop/main.ts`, `main/providerForwarding.ts`, `renderer/RemoteProviders.tsx` | GUI had no local authority lifecycle | Workspace and Models & Providers settings offer one review for selected local profiles and configured models. Main process owns private brokers; credentials stay local | Real daemon/desktop transport integration tests exercise sharing and native turns | Packaged main/preload/renderer exercised through real SSH; evidence below |
| Rotate a saved local key without another provider approval | `daemon/providerRelays.ts`, `bridge/profiles.ts` | Existing per-request credential refresh was unreachable through GUI SSH | GUI uses the existing resolver on every new request; same provider/endpoint remains approved | Two real daemon instances return results before and after a saved-key change; the injected provider sees the changed key; no remote fallback | An already-started provider request keeps its original authentication. Environment-variable changes in another process do not update a running daemon's environment |
| Keep task identity and work safe during setup | `daemon/server.ts`, `main/providerForwarding.ts` | New desktop setup could race navigation or an admitted turn | Session-key precondition checked at the daemon boundary; new capability required; admitted turns/session operations and observed running agents block setup | Tests cover session changes, stale review, running work, failed batch preparation and cancellation | Capability is additive; older daemons must be updated when idle |
| Distinguish remote credentials from local access | `renderer/RemoteProviders.tsx`, `renderer/Overlays.tsx` | GUI gave no source/expiry explanation | Source, execution host, workspace, duration and limits are shown. Remote profile controls remain explicit. Revocation preserves local requirement | Native interaction acceptance below; authority tests prove fail-closed revocation | Local access depends on this desktop view and both daemon connections; remote-owned setup can continue independently |
| Keep credentials and provider context out of public UI events | `desktop/main/daemon.ts`, existing local broker | Desktop treated private provider frames as unknown diagnostics | Main-process interception, original-socket replies, bounded concurrency/deadlines, fixed diagnostic text | Transport regression checks private-frame isolation, malformed frames, concurrency refusal and sanitized provider failures | Provider responses themselves are task output; the relay does not make arbitrary provider text secret-free |
| Recover from disconnect without fallback | `main/providerForwarding.ts`, existing remote bindings | No desktop grant lifecycle | Either transport loss closes brokers. Reconnect retains session requirements but needs fresh review. Closed views revoke authority | Actual socket-loss regression aborts the local stream, resumes the same session and reports unavailable local access | Daemon restart cannot preserve in-memory authorization |
| Select approved models after sharing | `renderer/store.ts`, `daemon/server.ts:fetch_models` | GUI picker would otherwise still list remote models | Model-selection discovery returns the approved local models; ordinary provider management remains remote | Real-daemon catalog assertion | Unapproved models require a new review; no automatic provider aliasing |

Limits shown in the review: one to eight hours, at most 32 profiles, 10,000 requests
and 16 concurrent requests per profile, and 32,768 output tokens per request where
supported. Subscription providers require one acknowledgement for provider-controlled
output. No new destination trust is persisted. Existing TUI review uses the same
extracted public inventory parser.

Native acceptance used the packaged desktop main/preload/renderer, a production
local CLI daemon, and a real SSH tunnel to an isolated daemon on n_server_spot_m.
The remote bootstrap address was injected; the managed installer was not exercised.
A local synthetic HTTP provider accepted the first saved key, then the replacement
key without another review. Revocation prevented another provider call; the same
session was preserved. No real provider credentials or paid inference were used. The behavior run used
package build `d8f4d48933cf12d3`; after applying the existing shared button style,
wide/narrow review captures were repeated on package `112b17d67f622ba6`.
The final rebuild additionally preserves the pre-existing inventory-error wording.
All task-owned native test processes were stopped afterward; user daemons and
active work were not restarted.

Evidence is saved outside the repository at
`/Users/erfan/Documents/Projects/xerxes-desktop-verification/desktop-provider-continuity-2026-09-21/`:

- `gui-result.json`: native-path assertions and preserved session ID.
- `requests.json`: sanitized first/rotated credential labels only.
- `review-wide.png`, `review-narrow.png`: inspected approval UI at both sizes.
- `rotation-accepted.png`, `revoked.png`: resulting source/status screens.

Physical socket-loss/reconnect, expiry, cancelled preparation, stale-session review,
changed endpoint, old-runtime refusal, model discovery, persisted requirements and
private-frame isolation are covered by real-daemon and transport regressions.
Actual Codex subscription inference/auth rotation, other operating systems and the
managed remote updater remain unverified in this change. Codex subscription profiles
use the local ChatGPT/Codex login resolver, not a generic OpenAI API-key field.
Changing an environment variable in another process cannot update an existing
daemon environment.

The first UI gate found the extracted parser had changed its safe error wording;
the existing contract was restored. Two existing tests also observed intermediate
states: the broker test now waits for daemon turn admission to reopen, and the
transcript test waits for Markdown content before checking exact spacing.
Parallel UI runs also exposed unrelated intermediate-frame assertions in the
custom-agent, monitor, diff and context tests. The final complete UI rerun uses one
worker; default parallel execution remains load-sensitive. Completed checks:

- `bun run check`: passed, including runtime, TUI and desktop type checks.
- Full runtime suite: 4,143 passed, 3 environment-specific skips.
- Full TUI suite: 1,495 passed across 151 files with `--maxWorkers=1`.
- Final focused desktop relay/transport regressions: 30 passed.
- `bun run build`, `bun run --cwd xerxes build:desktop`, and
  `git diff --check`: passed. Final package build ID: `df60f287e996973b`.

The default parallel root test invocation was not clean: the passing complete
UI validation is the serial run recorded above. The app is rebuilt but has not
been installed, committed, or pushed by this follow-up. The broader workflow audit remains in progress.
