# Xerxes 24/7 superagent

Status: implementation in progress. Baseline: main `30f5bea9` (0.8.33).
All work is local on branch `xerxes-superagent`; no release or deployment is implied.

## Existing architecture (Step 1)

- `xerxes/src/hub/supervisor/schemas.ts` owns the sixteen logical tool schemas and provider aliases. `tools.ts` builds the isolated supervisor registry.
- `actions.ts` validates target/thread scope, hashes arguments, creates durable intents, then either announces a confirmation card or dispatches through an owner Autopilot grant. Confirm and dispatch revalidate authority and target revision.
- `threads.ts` computes observation scope from participants. `security/principal.ts`, `hub/commandRouter.ts` and `security/hubIdentityVerifier.ts` enforce delegated agent ceilings and signed target-bound requests.
- `audit.ts` records intent/confirm/dispatch/outcome through the identity audit log. Pre-mutation audit fails closed; post-outcome logging cannot undo execution.
- `autopilot.ts` handles attention events and periodic check-ins. `watch.ts` observes events, bounded step digests and anomalies/reviews. `index.ts` composes these with the runner and event subscription.
- `task.tail` currently resolves session keys using `session.active_list {}`. Omitting `history_limit: 0` requests all active histories before a second `session.history` request. Resolution does not propagate cancellation. The tail rejects pagination despite daemon support for `before`. Static evidence establishes these defects; no live latency measurement has been made.
- Existing delegated permission approvals and broad automatic owner actions require tightening under this contract.

## Safety contract (Step 2)

| Tier | Abilities | Authority |
|---|---|---|
| A SEE | Workspace file/directory/search/git reads; paginated transcripts and test/build results; health; GitHub reads; Xerxes UI screenshots at fixed widths | Existing thread/workspace read ACL, bounded bytes/time. No new authority from read content. |
| B REMEMBER/SCHEDULE | Persistent notebook (plans/promises/open items), one-shot wakes, event subscriptions, quiet unchanged check-ins | Scoped to owner/thread/workspace; stored provenance remains data, not an instruction grant. |
| C SCOPED DO | Commands and file edits | Owner-authorized action mandate; owned worktree plus enforced sandbox; branch/file leases, timeout, bounded output, cancellation. Never ambient Hub shell access. |
| D GATED | Push/publish/outside delete/spend/credentials/others' systems/permission modes/goal completion | Current Hub owner must explicitly confirm exact canonical action; no Autopilot shortcut. |

Each ability has its own module and tests. Tool schemas, policy/admission and host execution are separate layers. Additive RPCs preserve protocol v35. Production composition injects genuine workspace/OS/GitHub/screenshot ports; missing hosts return actionable unsupported results, never fake observations.

### Untrusted input

Task transcripts, file contents, web responses, notebook text and event reasons are untrusted observations. They cannot create a C/D mandate. Admission uses an out-of-band, persisted owner authorization bound to workspace, action kind and canonical request hash. A model-supplied boolean, source label, notebook sentence or event payload is never proof of authority. D confirmations bind the exact request, expire, are single-use, and require current owner identity at dispatch. C has no network/credential privileges and refuses when enforcement or task ownership cannot be established.

### Operational controls

Every admitted/denied action is audited with actor, tier, target, hash and result; raw secrets are excluded. Audit failure prevents execution. Global pause stops admission and wake delivery. Kill also aborts active owned executions. Durable UTC-day counters enforce configured wake/action/token/cost budgets; unknown prices remain unknown and cannot satisfy a monetary cap. Budget exhaustion never silently widens authority. Side effects are not replayed automatically after an uncertain outcome.

### Scheduling and silence

Use durable SQLite notebook, wake and subscription records. Events are delivered by the existing Hub journal subscription; timers are for deadlines, not polling task transcripts. Wakes recheck current owner/grants/pause/budget and carry data-only reasons. A crash after a claimed wake is observable rather than silently replaying mutations. Unchanged model-free check-ins append no thread message, invoke no model and send no push. Notebook entries are injected as bounded untrusted context on later turns.

## Verification and delivery

Use Bun-native tests from `xerxes/` so isolated-home preload applies. Write failing behavior tests first; run each slice's focused tests before its local commit. After integration, run the root check/test/build gate and `git diff --check`, independently review safety paths, and report failures/skips explicitly. No live provider, external GitHub mutation, cloud spend, service restart, push, publish or release is authorized by this implementation task.

## Implementation checkpoints

1. Map and tiered design posted. Separate local worktree created from default branch.
2. Pending: bounded/cancellable transcript reads and Phase 1 abilities.
3. Pending: scoped execution and locks.
4. Pending: mandatory gates and operational controls.
5. Pending: final integration, verification and report.
