# Xerxes Hub protocol (v1)

The Hub is a separate always-on process (`xerxes hub`) that aggregates every
workspace (the local daemon and remote runtimes over SSH) and serves phones and
the desktop supervisor tab. Its protocol is versioned on its own and is **not**
a passthrough of daemon protocol v35. Source of truth:
`xerxes/src/hub/protocol.ts`, `xerxes/src/hub/cursor.ts`,
`xerxes/src/security/principal.ts`, `xerxes/src/hub/supervisor/schemas.ts`,
`xerxes/src/client/bridge.ts`.

## Frames

Every frame is one JSON object with `v: 1`.

| Direction | Shape |
|---|---|
| client → Hub | `{v, kind:"request", id, method, params}` |
| Hub → client | `{v, kind:"response", id, ok:true, result}` or `{v, kind:"response", id, ok:false, error:{code, message}}` |
| Hub → client | `{v, kind:"event", type, payload}` |

- `id` is a non-empty string of at most 200 characters.
- `method` must be one of `HUB_METHODS`; anything else (including daemon method
  names) is refused with `invalid_request`.
- Frames over 1 MiB are refused in both directions (`frame_too_large`). Batches
  and pages target 256 KiB.
- Error codes: `unauthorized`, `forbidden`, `not_found`, `invalid_request`,
  `frame_too_large`, `unavailable`, `conflict`, `submission_conflict`,
  `interaction_already_resolved`, `gap`, `rate_limited`, `unsupported`,
  `unconfigured`, `internal`.

## Actors

The actor of every request is derived from the authenticated connection, never
from request parameters. Events carry `actor` (`HubActor`): principal kind
(`user` or `agent`), id, display name, device, and for agent actions
`requested_by` / `confirmed_by`, which are attribution only.

## Events and cursors

A `HubEvent` has `event_id`, `sequence`, opaque `cursor`, `workspace_id`,
optional `session_id`, `type`, `payload`, `occurred_at`, optional `actor`.

- Delivery is **at-least-once**. Clients dedupe by `event_id` and sequence
  (`EventCursorTracker`) and acknowledge only after applying an event.
- A cursor is bound to one Hub stream and one subscription scope (the sorted
  set of workspace and session ids). A cursor from a replaced stream yields a
  `hub.stream_gap` with reason `stream_replaced`; a cursor from another scope is
  refused.
- A sequence jump is a gap: the client resnapshots instead of guessing.
- `hub.stream_gap.reason`: `retention`, `stream_replaced`,
  `journal_unavailable`, `scope_changed`.

## Commands

`hub.command {workspace_id, session_id?, command_id, action, args, expected_revision?}`
runs one allowlisted action (`HUB_COMMAND_ACTIONS`). `command_id` is the
idempotency key: repeating it returns the recorded outcome. The result `state`
is `accepted`, `rejected`, `pending_confirmation` (supervisor intents), or
`unknown` when the outcome could not be established (for example SSH dropped
mid-call). `unknown` is resolved with `hub.command.status`, never by replaying.

## Scopes and roles

| Role | Scopes |
|---|---|
| viewer | `session.read`, `usage.read`, `presence.read` |
| steer | viewer + `session.steer` |
| operator | steer + `session.submit`, `session.cancel`, `session.configure`, `goal.write`, `permission.respond`, `question.respond` |
| owner | operator + `grants.manage`, `approvals.manage`, `devices.manage`, `supervisor.model` |

`supervisor.model` lets a principal run one model completion on a daemon's own
provider profiles (`hub.model.complete`). An agent principal may hold it; the Hub
signs the supervisor's calls with exactly this one scope.

Effective scopes = matching allow grants − matching deny grants, ∩ the device
ceiling. A hub owner starts from every scope. Session-scoped grants apply only
to that session and never reveal the workspace inventory. Agent principals are
capped at `AGENT_SCOPE_CEILING` and can never hold a human-only scope
(`permission.respond`, `question.respond`, `grants.manage`,
`approvals.manage`, `devices.manage`), whatever a grant says.

## Supervisor tools

The XERXES supervisor sees twelve tools, with underscore provider names
(`tasks_list`) and dotted canonical names (`tasks.list`): `workspace.list`,
`tasks.list`, `task.digest`, `task.tail`, `task.search` (reads);
`task.start`, `task.steer`, `task.stop`, `agent.message`, `goal.set`,
`model.set` (need a human to confirm); `notify.user` (thread audience only).
Approval, review, permission-mode, shell, Workflow and generic RPC tools are
refused by `assertSupervisorToolCeiling`.

## Machines and project discovery

A registry entry may be a **machine**: `connection.discover: true` (the registry
stays version 1; the field is optional). Its `projectDir` is stored as `/` and is
never used for routing. The pool connects one lane to the entry's endpoint and, on
every poll, reads `session.active_list {history_limit:0}` and a bounded
`session.list {scope:"global"}`. Each distinct absolute `cwd` (subagent sessions
and relative paths are skipped; at most 100 per machine) becomes a **derived
workspace**:

- id `m-<first 8 chars of the entry id>-<sha256(cwd) hex, first 12>`, so ids are
  stable across restarts and grants keep working;
- label `<machine label> · <basename(cwd)>`, `projectDir` = the cwd, same endpoint;
- listed in `hub.workspaces.list`, `hub.snapshot`, routable by `hub.command`, and
  visible to the supervisor. The machine entry itself is never listed to phones.

Per-project entries keep working. `xerxes hub machines add-local <label> --socket
<path>` registers a machine idempotently by socket path and removes older
per-project entries on the same socket.

`hub.machines.list {}` -> `{machines:[{id, label, health, read_only,
models_supported, workspace_ids}]}`. The owner sees every machine; others see
machines holding a workspace they can read broadly (a session-only grant never
reveals a machine). `models_supported` is the daemon's `hub_models_v1_supported`.

## XERXES model

The supervisor exists whenever identity does. Its model is one row in `hub.sqlite`
(`hub_supervisor_model`: machine, profile, model, effort or null, updated_by,
updated_at). Every model call is a signed `hub.model.complete` to the chosen
machine's daemon (principal `xerxes-supervisor`, kind `agent`, scope
`supervisor.model`, `requested_by` set on settings calls). The machine must have
enrolled the Hub (`xerxes hub link`). No provider key, token or base URL ever
reaches the Hub, a phone or a renderer. `XERXES_HUB_SUPERVISOR_*` is only a
fallback while nothing is stored.

| Method | Params | Result |
|---|---|---|
| `supervisor.config.get` | none | `{status:"configured"\|"unconfigured"\|"machine_offline"\|"invalid", config?:{machine_id, machine_label, profile, model, effort}, can_edit, message?}` |
| `supervisor.config.options` (owner) | `{machine_id, profile?, model?}` | `{machine_id, profiles:[{name, label, provider, default_model}], profile?, models:[{id, label?}], model?, efforts:[{effort, description?}], default_effort}` |
| `supervisor.config.set` (owner) | `{machine_id, profile, model, effort}` | the `supervisor.config.get` shape |

`config.set` validates the model and effort against the machine's options,
audits (a failed audit refuses the change), persists, hot-swaps the model port and
broadcasts event `supervisor.config_changed` (the `config.get` shape, with
`can_edit` per recipient) to every connected person. Owner means principal kind
`user` with `hubOwner`. With no stored choice (and no environment fallback)
`supervisor.thread.open` works and reports `config_status:"unconfigured"`, and
`supervisor.message.submit` is refused with code `unconfigured`.

## Identity and pairing

Source: `xerxes/src/hub/identity/` (`createHubIdentity` in `index.ts`). State
lives in `<hub state dir>/identity.sqlite` (0600, WAL) plus two 0600 files:
`identity-pepper` (keys every token and code hash) and `vapid-key.pem`. Only
keyed hashes of tokens, pairing codes and invite codes are stored.

**Owner bootstrap** is local only (`xerxes hub owner init --name <n>` or
`admin.bootstrapOwner`). It refuses when an owner exists. There is no remote
claim route.

**Pairing.** `admin.createPairing({userId, label, role?, confirm?})` returns a
URL `<publicOrigin>/pair#<base64url JSON {hub_id, pairing_id, code}>` (`hub_id`
is the base64url of the 32-byte signer fingerprint), a 6-digit comparison code,
a 2-minute expiry, an SVG QR and a terminal QR. The code travels only in the
fragment and a POST body.

- Consuming is single use and transactional; competing consumers produce exactly
  one winner. Five wrong codes lock the pairing. Attempts are rate limited per
  address and per pairing id.
- By default consumption creates a **pending** device. It cannot authenticate or
  refresh until the owner runs `admin.confirmPairing(id, comparisonCode)` (CLI:
  `pair confirm <id> <code>`) within 5 minutes. Five wrong comparison codes
  revoke it. Only a pairing created with `confirm: false` explicitly (CLI
  `--no-confirm`) activates the device immediately. The comparison code proves
  the phone holds this ticket; it does not stop someone who photographed the QR
  first, so check the device label before confirming.

**Cookies.** `__Host-xerxes_refresh` (device refresh token) and
`__Host-xerxes_session` (5-minute browser session), both `Path=/; Secure;
HttpOnly; SameSite=Strict`, no `Domain`. A CSRF token (returned in JSON as
`csrf_token`) is required in header `x-xerxes-csrf` on every mutating route that
uses a session.

**Lifetimes.** Session 5 minutes, device 7 days of inactivity, 30 days absolute.
Limits can only be tightened (`limits` option).

**Refresh rotation.** `POST /api/hub/v1/session/refresh` rotates atomically.
Presenting a retired refresh token revokes the whole device (family, sessions,
push subscriptions). Exception, the grace window (10 s by default): the immediately
previous token seen again within the window returns the *same* successor, so two
tabs or a retried request do not log the user out. The successor is held in memory
only; after a Hub restart a late retry is treated as reuse (the safe direction).

**Revocation.** `revokeDevice`, `revokeUser` and `revokeAll` mark rows and call
`onRevoked` listeners synchronously after commit (one device-level event plus one
per live browser session). Revocations written by the local CLI (another process)
are announced by polling SQLite `data_version` (default every 2 s, or call
`admin.syncRevocations()`).

**HTTP routes** (`/api/hub/v1/`, JSON in and out; errors are
`{error:{code,message}}`):

| Route | Auth | Request | Response |
|---|---|---|---|
| `POST pair/consume` | none, Origin allowlist if present | `{pairing_id, code, device_label?}` | `{status: active\|pending_confirmation, user, device_id, csrf_token, session_expires_at, comparison_code, hub_fingerprint}` + both cookies |
| `POST invites/accept` | none | `{invite_id, code, display_name?, device_label?}` | same as consume, creates a new user |
| `POST session/refresh` | refresh cookie, Origin required | none | same shape, rotated cookies; 202 `{status:"pending_confirmation"}`; 401 clears cookies |
| `POST session/logout` | refresh or session cookie, Origin required | none | `{ok:true}`, device revoked, cookies cleared |
| `GET session` | session | none | `{user, device_id, csrf_token, session_expires_at, passkeys}` |
| `GET devices[?all=1]` | session | none | `{devices:[...]}` (`all` is owner only) |
| `POST devices/revoke` | session + CSRF | `{device_id}` | `{ok:true}`; another user's device needs owner + step-up |
| `POST pairings`, `pairings/confirm` | owner + CSRF + step-up | `{user_id?, label?, role?}` / `{pairing_id, comparison_code}` | `{pairing_id, url, comparison_code, expires_at, qr_svg}` / `{ok, device_id}` |
| `POST invites` | owner + CSRF + step-up | `{workspace_id, session_id?, role, expires_in_ms?, label?}` | `{invite_id, url, comparison_code, expires_at, resources, role, qr_svg}` |
| `POST grants` | owner + CSRF + step-up | `{principal_id, grants[], mode?: replace\|add}` | `{grants}` |
| `GET passkeys` | session | none | `{passkeys:[...]}` |
| `POST passkeys/register/options`, `register/verify` | session + CSRF | none / `{clientDataJSON, attestationObject, label?}` | WebAuthn options / `{id, alg, label}` |
| `POST passkeys/assert/options`, `assert/verify` | session + CSRF | none / `{id, clientDataJSON, authenticatorData, signature}` | options / `{ok, step_up_at}` |
| `POST passkeys/remove` | session + CSRF + step-up | `{id}` | `{ok:true}` |
| `GET push/vapid-key` | session | none | `{public_key}` |
| `POST push/subscribe`, `push/unsubscribe` | session + CSRF | `{endpoint, keys:{p256dh, auth}}` / `{endpoint?}` | `{subscription_id}` / `{removed}` |

**WebAuthn step-up.** Challenges are single use, bound to the browser session and
expire in 5 minutes. The RP ID is the host of the public origin, the origin must
match exactly, and user presence and user verification are required. ES256 and
EdDSA keys are supported, attestation `none` only, and a non-increasing non-zero
`signCount` is refused. `requireRecentStepUp(sessionId, maxAgeMs)` guards
membership changes over HTTP. Without an enrolled passkey those HTTP actions are
refused (`step_up_required`) and must be done from the local admin CLI/API. The
first passkey can be enrolled by a live session without step-up (residual risk:
a stolen live session could enrol it; enrolment is audited).

**Invites.** The owner creates an expiring (max 7 days), single-use invite naming
one workspace or one session and a maximum role (never `owner`). Accepting creates
a new user and device with grants limited to that resource, so a session-only
invite never reveals the workspace inventory. Confirmation works like pairing.

**Grants.** Written through `parseGrant`. Agent principals are refused any
human-only scope at write time as well as at evaluation time.

**Audit.** `hub_audit_events` is an append-only hash chain (SQLite triggers refuse
UPDATE and DELETE; `verifyAuditChain` detects edits, reordering and deleted middle
rows; tail truncation needs an external anchor). `audit()` rejects on write
failure and rejects records containing token-, code- or key-shaped strings.

**Push.** `admin.sendPush(userId, {title, body, url, dedupeKey?})` encrypts with
RFC 8291 `aes128gcm` and signs with an RFC 8292 VAPID JWT through an injected
`fetch` port. Only 201/202 count as delivery; 404/410 delete the subscription;
failures are typed (`no_subscriptions`, `deduped`, `user_inactive`,
`vapid_subject_missing`, `all_failed`). Endpoints must be https on an allowed push
service host. Lock-screen text is a title, a short body and a same-origin path;
never transcript or tool arguments. Subscriptions are purged when a device or user
is revoked.

**CLI** (`runHubIdentityCli`): `owner init --name`, `pair [--label] [--user]
[--role] [--no-confirm] [--origin] [--json]`, `pair confirm <id> <code> [--json]`, `devices
list|revoke`, `users list|revoke`, `invite create --workspace <id> [--session
<id>] --role <role> [--expires 24h]`, `audit verify`.

**Machine-readable CLI.** These print exactly one JSON object on stdout (progress goes to stderr):

| Command | JSON |
|---|---|
| `hub info --json` | `{hub_id, fingerprint, public_key_pem, public_origin: string\|null, version}` |
| `hub pair [--label X] [--no-confirm] --json` | `{pairing_id, url, comparison_code, expires_at, confirm_required, qr_svg: string\|null}` |
| `hub pair confirm <id> <code> --json` | `{device_id}` |
| `hub machines list --json` | `{machines:[{id, label, endpoint_kind, discover, projects:[{workspace_id, project_dir, label, health}]}]}` (health from the running Hub, else `offline`) |
| `hub machines add-local <label> --socket <path> --json` | `{machine_id, created, removed_workspace_ids}` |
| `hub machines remove <id> --json` | `{removed}` |
| `hub link <target> --json` | `{machine_id, remote_socket, service_path?, enrolled: true}` |

`hub link <target>` registers one discover machine on the Hub (through the remote CLI,
idempotent by socket path, replacing older per-project entries on that socket), reads the
Hub key with `hub info --json` over ssh and pins it on the local daemon with
`hub.enroll {all_projects:true}`. `hub remote <target> <hub-args...>` runs
`xerxes hub <hub-args...>` on the Hub host over ssh (`BatchMode=yes`, `ControlMaster=no`,
`ControlPath=none`, every argument shell-quoted) and relays stdout, stderr and the exit
code. The remote Xerxes is `$HOME/.xerxes/bin/xerxes` when executable (written by
`hub setup`), else `xerxes`; `--remote-xerxes` overrides.
