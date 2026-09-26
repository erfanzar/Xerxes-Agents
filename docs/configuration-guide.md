# Configuration guide

## Accepting goal criteria

Open F10 to inspect the current goal and its acceptance criteria. Use Tab to
select a criterion, A to enter an acceptance note, and Enter to accept it. Esc
cancels the editor. Ctrl+R keeps the note while refreshing the criteria for
review; press A again after reviewing before submitting. This records
a **user decision**, distinct from the model's interpretation of a successful
tool result. The model can read the decision with `get_goal`, but its goal tools
cannot create one. Accepting a criterion does not complete or resume the goal.

The decision is tied to the displayed session, goal and revision. If the goal
changes while you are writing, refresh and review the changed criterion before
submitting again. Decisions survive session reload and may be recorded for
active, paused or blocked goals once their current turn has stopped. Completed
goals are immutable. Editing the objective or a criterion invalidates affected
evidence while retaining its history.

## Goal continuation

After a human resumes a paused or disarmed goal with `/goal resume`, the daemon records one
per-session continuation receipt and stages a queued background wake for the
current goal revision. The receipt is durable, so a queued continuation remains
visible through reload. At admission, a changed goal revision supersedes the old
queued brief. Finishing a human turn also stages continuation for an active,
armed goal.
Human requests already waiting on the session are admitted first.

When the wake is admitted, the daemon reserves the next goal round and claims
the receipt in the same persisted update before it sends the regenerated goal
prompt to the provider. The receipt records whether that round later settled or
was interrupted; it does not store a prompt.

A restart leaves queued work disarmed. Reopening a session does not replay it;
use `/goal resume` to authorize continuation again. A running receipt owned by a
different daemon is marked interrupted during recovery because its provider
outcome is unknown, and the round is not replayed automatically. Inspect the
goal and its continuation state with F10 or `goal.inspect` before deciding how
to proceed.

## Goal milestones

Use `/goal milestone <text>` to record the current human-readable checkpoint
for the active goal. `/goal milestone` displays the current value and
`/goal milestone clear` removes it. The value is limited to 1,000 characters;
setting or clearing it creates a host-generated goal revision and durable change
history, but does not spend a round, change a budget, add evidence or complete
the goal.

Milestones can be set while a goal is active, paused or blocked. A completed
goal's milestone cannot be changed; it remains visible in `/goal` and F10.
Changing the objective clears the prior milestone unless the same edit supplies
a replacement milestone. The model may provide `current_milestone` when a human
turn creates or edits a goal. The goal's own current continuation round may use
`update_goal` action `milestone` with `current_milestone` (or null to clear it).
That action updates progress context without permission to rewrite the objective,
raise budgets or supply completion evidence. Unrelated background work and
subagents cannot update the milestone.
Numeric budget fields supplied alongside a milestone are ignored and reported in
`ignored_fields`; they never change the goal's limits. Use an authorized `edit`
action to change limits.

## Goal token limits

Create a goal with `/goal <objective>`, then set `/goal --tokens 100000`.
F10 shows recorded input/output totals, the cap, pending calls and whether the
accounting is complete. Input totals include reported cache-read and cache-write
tokens. This is a provider-call admission limit: already-admitted calls may
finish above the cap; later calls and rounds are refused. It is not the remaining
quota of a provider subscription.

Usage is stored in `runs/goal-tokens.sqlite` beneath the Xerxes home, independently
of transcript compaction. Changing a cap preserves usage. A goal with missing
usage or a pending call from a previous daemon cannot resume capped work by
assuming that spend was zero. Goals predating this ledger have an unknown
baseline. Uncapped goals retain that uncertainty and can continue; setting a cap
does not retroactively create complete accounting.

For a goal created by a model tool, counting starts with subsequent provider
calls; the call that produced `create_goal` was admitted before that goal existed.
Delegated agents retain the original goal scope through queuing and in-process
retry. A child whose goal was replaced cannot charge the replacement goal.
Recovered subagents retain their original parent session and goal budget binding.
After a daemon restart, retry requires that same goal to be active and armed,
and charges new calls to its durable ledger using the current daemon owner.
Pause or replacement prevents retry. For capped goals, an exhausted cap or
unresolved usage from another process also prevents a new provider call.
Resume the original goal before retrying; if the
goal has been replaced, dispatch new work instead.

Older child records without budget ownership cannot be retried in sessions with
goal history. Dispatch new work from the parent so its budget is captured. Work
originally created without a goal remains explicitly unbound. Other enclosing
limits that cannot yet be restored (such as scheduled-run call scopes) also
prevent recovered retry rather than disappearing on restart.

Native goal tools expose `max_total_tokens` on create and human-authorized edit.

## Goal time limits

Create a goal with `/goal <objective>`, then use `/goal --duration 30m` to cap
its wall time. Whole seconds (`s`), minutes (`m`) and hours (`h`) are accepted.
The duration starts at the goal's original creation time and includes pauses
and restarts. F10 shows elapsed/remaining time and expiry. At expiry, current
active goal work is cancelled and the goal becomes blocked with `time-limit`.
Paused goals remain paused, but cannot resume after their deadline until the
limit is raised.
Raise the total duration and use `/goal resume` to continue; changing the limit
alone does not resume it. Unconfigured goals have no wall-time cap.

The model can set `max_duration_ms` in `create_goal`, or in a human-authorized
`update_goal` edit. Automatic goal rounds cannot increase their own limit.
This is a time limit, not a subscription quota or token budget.

Xerxes keeps host-dependent configuration explicit. The native value models live in
[`xerxes/src/core/config.ts`](../xerxes/src/core/config.ts), while daemon settings
are read by [`xerxes/src/daemon/config.ts`](../xerxes/src/daemon/config.ts).

## Local runtime home

`XERXES_HOME` chooses the directory used for profiles, daemon state, sessions, and agent memory.
`~` and `~/…` are expanded before the native paths are resolved. If it is unset, Xerxes uses its
normal per-user default. `bun run xerxes doctor` reports the resolved location without exposing
credentials.

## Provider configuration

Configure a provider with a profile or a deliberately supplied environment variable. Common
variables include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and provider-specific
keys defined by the native provider registry. A profile can also hold a model identifier and an
OpenAI-compatible base URL.

Do not commit credentials. Tests and embedding hosts should pass a provider client or a synthetic
environment map rather than reading ambient secrets.

When any profile uses the `claude-code` provider, each runtime start (local, or on an SSH host)
runs `claude update` in the background, or installs Claude Code with Anthropic's official
installer when the CLI is missing. Startup never waits for it, and the outcome is written to the
runtime log. Set `XERXES_CLAUDE_CODE_AUTOUPDATE=0` to turn this off. The Codex provider needs no
CLI: Xerxes calls the ChatGPT backend directly and only reads the Codex CLI's saved sign-in.

## Daemon settings

The daemon reads `$XERXES_HOME/daemon/config.json` when it exists. Its top-level native sections
are `runtime`, `control`, `workspace`, and `channels`; `maxConcurrentTurns` controls concurrent
turn processing. Settings can reference a named environment value with `env:NAME` or a `*_env`
key, which is resolved at the host boundary.

Useful process-level overrides include:

| Variable | Effect |
| --- | --- |
| `XERXES_DAEMON_HOST` | WebSocket control host. |
| `XERXES_DAEMON_PORT` | WebSocket control port. |
| `XERXES_DAEMON_SOCKET` | Local daemon socket path. |
| `XERXES_DAEMON_TOKEN` | Control-plane token. |
| `XERXES_MAX_TURNS` | Maximum concurrent daemon turns. |
| `XERXES_MODEL` | Runtime model override. |
| `XERXES_BASE_URL` | Runtime provider base URL. |
| `XERXES_PERMISSION_MODE` | Runtime permission mode (`accept-all` by default; also `auto`, `manual`, or `plan`). |

Run `bun run xerxes daemon --project-dir .` to use the native daemon with an explicit workspace.

Interactive sessions start in YOLO mode (`accept-all`). The TUI shows `YOLO ON`
beside the active model while it is enabled; `/yolo` switches between `accept-all`
and `auto`. Explicit tool-policy denials remain final in every permission mode.

`/ultra` toggles session ultra mode: every turn runs the maximum thinking
directive (32k token budget, high effort) until `/ultra off`. One-shot prompts
can escalate a single turn instead with the thinking keywords `think`,
`think hard` (alias `megathink`), `think harder`, or `ultrathink` — strongest
keyword wins. Session defaults come from `thinking`, `thinking_budget`, and
`reasoning_effort` in runtime settings or the active provider profile's
`sampling` block. Ultra mode is session-scoped and not persisted.

## Embedded configuration

`XerxesConfig`, `ExecutorConfig`, `MemoryConfig`, `SecurityConfig`, and `LLMConfig` validate
typed input at the application boundary. They accept the documented snake_case persistence shape
and expose immutable native values. Prefer an explicit object passed by the embedding application
over global environment mutation.

## Security defaults

Tool execution remains subject to policy, permission, path safety, prompt scanning, and the
configured sandbox boundary. Enabling a model or channel does not enable high-power tools by
itself. Keep network, browser, media, accelerator, and remote-channel integrations behind an
explicit host port.

## Agent intelligence tiers

`AgentTool`, `TaskCreateTool`, and each entry in `SpawnAgents.agents` accept an optional
`intelligence` value: `light`, `balanced`, or `smart`. The parent model chooses a tier according
to the delegated task; you control the actual model behind each tier. This selects a model,
not a reasoning-effort setting or a measured intelligence score.

Add the mapping to `$XERXES_HOME/daemon/config.json` (replace the illustrative model names
with IDs supported by your active provider/profile):

```json
{
  "runtime": {
    "agent_intelligence": {
      "default": "balanced",
      "light": "your-fast-model",
      "balanced": "your-normal-model",
      "smart": "your-strong-model"
    }
  }
}
```

Restart the daemon after editing its configuration. The same runtime setting is loaded by
one-shot and ACP CLI entry points. Embedded tool hosts can supply `intelligence` to
`registerClaudeAgentTools` using the same mapping object. Registered tool descriptions tell
the parent which tiers are configured.

- An explicit `model` overrides the configured default. Supplying both a model and a tier is an error.
- An explicit tier overrides the default and agent-definition model.
- With no tier or explicit model, the configured default applies. `"default": "inherit"`
  (or no configuration) preserves the existing agent-definition → parent → host model selection.
- Missing tier mappings and malformed settings fail explicitly. All swarm tier requests are
  validated before any child is launched. Provider errors remain visible; there is no silent
  downgrade to another tier.
- Tier selection uses the active provider transport and credentials; it does not switch profiles.
  The resolved model is carried by the existing agent snapshot and retained for retries.

For example, the parent can call:

```json
{
  "agents": [
    { "title": "Locate tests", "prompt": "Find relevant tests and report paths.", "intelligence": "light" },
    { "title": "Review races", "prompt": "Review cancellation races and propose a verified fix.", "intelligence": "smart" }
  ],
  "wait": true
}
```

### Cron execution ownership and limits

The daemon runs due cron jobs under its exclusive scheduler lease. `/cron run`
uses that same owner and refuses to overlap a job that is running or still
cancelling. A daemon that does not own the lease reports an actionable error for
manual execution instead of bypassing the owner.

There are at most four unsettled cron jobs globally and four per project by
default. Automatic jobs waiting for capacity retain their due time. Embedded
hosts can set `DaemonServerOptions.cronMaxConcurrentJobs`,
`cronMaxConcurrentJobsPerProject`, and `cronJobTimeout` (milliseconds; zero
explicitly disables timeout). These are host options, not new environment
variables or slash arguments.

Timeout requests cancellation of the turn, delegated children and unattended
goal continuation. A runner that ignores cancellation keeps its concurrency
slot and job ownership until it settles. Shutdown keeps the scheduler lease
until that cleanup finishes; if the process exits first, normal stale-process
lease recovery applies. A persisted job does not execute while its host is off.

### Durable run history

The production daemon stores run history in `$XERXES_HOME/runs/history.sqlite`.
The first integrated source is scheduled/manual cron turns. `/runs` lists the
current session's runs; `/runs unread` selects unacknowledged outcomes.
`/runs inspect <id>` shows retained output and its revision, and
`/runs ack <id> <revision>` acknowledges exactly that version without stopping
work. An old acknowledgement cannot hide a newer result.

History survives reconnect and daemon restart. A record left running by a dead
process becomes interrupted; its process is not recreated. Inspected output is
limited to the last 64,000 characters. This initial history is per session;
a cross-session runs dashboard and terminal/agent adapters remain separate work.

Terminal history is also recorded by the production daemon. Running output is
checkpointed at most once per 250 milliseconds and completion saves the final
tail. F8 includes archived terminal rows after reconnect or restart; those rows
are read-only and cannot send signals to a historical PID. The stored terminal
kind and exit code remain available. Successful foreground commands are saved
without creating unread inbox items; background completions and failures do
create them. Attached local and WebSocket clients receive an owner-scoped
completion notification, while disconnected clients can inspect unread results
later. These notifications do not yet wake an idle model turn.

In the TUI, bare `/runs` opens a dedicated inspector. List refresh errors,
inspection errors, and action errors remain separate: a successful background
list refresh cannot hide an inspection or cancellation failure. Retry an
inspection with R; a successful inspection clears its error.
Up/Down selects a run,
U switches between all and unread results, A acknowledges the inspected
revision, R refreshes, and Page Up/Down scrolls output. Escape closes the view
without clearing the conversation or draft. A completed turn preserves the
user-opened inspector. On wide terminals, the run list and output sit side by
side; narrow terminals stack them. Explicit `/runs list`, `/runs unread`,
`/runs inspect` and `/runs ack` retain their native text-command behavior.

The runs inspector now defaults to the current workspace, so results from
scheduled sessions and delegated agents are visible together. W switches to
current-session scope. The daemon resolves workspace scope from the active
session; supplying another workspace path does not grant access to its runs.
Agent executions are recorded separately for every retry, with their final
output or cancellation/failure. Retrying an agent does not overwrite an earlier
run or acknowledge its unread result.

Agent retries belong to the requesting chat. An exact task ID from another chat
is rejected; a shared agent name resolves only within the current chat, and
ambiguous names require an ID. New native agents capture the owning session's
workspace. Retries retain that project for tools, transcript loading and
worktree allocation, including after restart. Older records without a workspace
use their owning session's project; an unavailable owner or malformed saved
workspace produces an error instead of selecting the daemon's default project.

New native agents also record a nonsecret provider-route fingerprint. Recovered
retry and reset retain the saved model, explicit reasoning effort and provider
profile. If the inherited connection or the selected profile's provider/endpoint
has changed, execution is rejected: restore the original configuration or
dispatch new work. Profile selection is checked again after asynchronous
validation and immediately before constructing the child client. Inline runtime
connection overrides remain authoritative; an active profile does not replace
them during recovery. API keys are not persisted in the fingerprint, so current
credentials can rotate without changing the configured route. This identifies
configured routing, not the identity of the currently authenticated account.
Legacy recovered tasks without routing identity require new work rather than
guessing the original connection. Archived tasks retain the same selectors.

### Command completion follow-ups

In a direct user turn, `exec_command` can set `notify_on_completion: true`
when the user asks to hear back after background work finishes. The daemon
creates a completion watch automatically for explicit background commands and
commands adopted after their foreground timeout. A foreground command that
finishes within its timeout simply returns its result. Other hosts reject this
option before starting a command when no completion adapter is installed.

The result includes `completion_watch.id` and `expires_at`. The watch lasts
at most 24 hours and admits one follow-up lasting at most 60 seconds, queued
behind current session work. It uses the existing monitor inspector and Runs
history; stop it through `/monitors stop <id>`. These are time/count limits,
not a token budget. Background-origin turns cannot enable this option.
Exhausted, settled reaction grants release their admission slot while retaining
history. Unresolved executions still count toward the 16-grant session limit,
even after expiry or cancellation, until cleanup is confirmed.

If watch admission fails after the command starts, the result retains its
`procId` and includes `completion_watch_error`. Inspect that process rather
than launching it again. Daemon shutdown still ends process-local watching;
this option does not restore old shell pipes after a restart.

### Event-driven terminal watches

`monitor_terminal` watches future output from an already-running terminal owned
by the calling session. Supply `terminal_id`, a case-insensitive literal `match`,
and optionally `duration_seconds` (1–86400, default 3600) and `max_events`
(1–1000, default 50). There is no regular-expression or shell-evaluation step.
The monitor frames complete lines, suppresses recently repeated matching lines,
retains at most 100 events in its short display and expires automatically. Durable evidence keeps up to the configured 1000-event ceiling. Per-session and host
limits are 16 and 128 active watches respectively.

Use `/monitors` to inspect watches and `/monitors stop <id>` to stop one. The
model can also use `list_monitors` and `stop_monitor`. Stopping a watch does not
kill its command. Source exit flushes an unterminated final line and ends the
watch. Session eviction and daemon shutdown detach watches. Newly created watches
retain their configuration and evidence for the inspector after restart. Up to
100 stored watches appear as archived, interrupted, or detached; they are not
counted as attached watchers. A detached watch may belong to another live daemon
and must be controlled through that daemon. Older runs without stored watch
configuration remain accessible through Runs. Matching lines are
announced to attached users and recorded in `/runs`; unchanged output consumes
no model turns. Existing terminal output is not replayed as a new match when a
watch attaches. Authenticated webhook adapters remain pending.

For file changes, open `/monitors`, press N, choose the File source and enter
an existing regular file path within the session workspace. The model can use
`monitor_file` with `file_path`, the same duration and event limits, and optional
reaction settings. This watches metadata, not file contents: events report
changes, deletion and recreation, including atomic file replacement. Rapid
changes may coalesce. Paths outside the workspace, traversal and symlink
escapes are rejected. Moving or replacing the watched parent directory fails
the watch visibly instead of silently observing a stale directory.
An in-workspace symlink is resolved once when attaching. The inspector shows
the resolved target, and changing the original symlink does not retarget the watch.

File watches retain their events in Runs. After daemon shutdown or restart,
the inspector reports an interrupted watch and an observation gap. Create a
new watch to establish a new baseline; changes during downtime are not replayed.

To watch a server-pushed WebSocket feed, choose the WebSocket source in the same
creation form, enter its URL and a literal match, or call `monitor_websocket`
with `websocket_url` and `match`. Remote feeds require `wss://`; unencrypted
`ws://` is limited to loopback. URLs cannot contain credentials, query parameters
or fragments. This adapter does not send authentication headers or application
subscription messages; endpoints requiring them are unsupported.

Only text messages up to 64 KiB are accepted. The transport checks advertised frame lengths and cumulative fragmented-message lengths before buffering payloads; handshake headers are capped at 16 KiB. TLS certificate verification remains enabled even if the process environment disables Node TLS verification. Matching is case-insensitive;
duplicate matching content within the most recent 256 matching identities does
not trigger another action. Evidence over 8192 characters is visibly truncated.
Disconnects record gap events and permit three lifetime reconnect attempts at
250 ms, 1 s and 4 s, each with a five-second connection timeout. Gap events count
toward the watch event limit and can trigger an explicitly enabled reaction.
Missed messages are not replayed. Stop, expiry and shutdown close the socket and
cancel reconnect timers. As with file watches, daemon restart requires creating
a new watch. The inspector shows connection health and retained gap evidence.

When the user requests automatic investigation, a direct user turn can set
`react: true` on `monitor_terminal`. Notification-only remains the default.
Use `max_reactions` (1–10, default 3) and `reaction_timeout_seconds`
(1–120, default 60). The overall watch expiry also bounds reaction deadlines.
Background turns cannot grant themselves new reactive watches. Reactions queue
behind session work, coalesce evidence, use monitor origin, and record their
answers and outcomes in Runs. They do not receive direct-user goal authority.
Stopping a watch revokes its queued reactions and aborts its active reaction;
stopping a session revokes all its reaction policies. Cancellation retains
execution ownership until cleanup settles.

These are reaction-count and wall-time limits, not a total-token or spending
budget. Token accounting and automatic recovery of pending reactions after a
daemon restart remain unfinished. Unresolved executor claims stay fenced rather
than being silently retried.

The TUI's bare `/monitors` command opens a dedicated session monitor inspector.
Select a watch with the arrow keys to inspect reaction state, expiry, errors and
recent matching output. Press S to stop a live watch or cancel pending reactions on a stored entry,
as shown by the action label. Fully settled and detached entries have no local
cancellation action. Press R to refresh, Page Up/Down
to scroll evidence, or Escape to return to the conversation. Stopping revokes
its automatic reactions while leaving the source terminal running. Explicit
`/monitors list` and `/monitors stop <id>` remain available as text commands.

Press N in the inspector to create a watch from one of this session's live
terminals. Set a literal match and duration; notification-only is the default.
Automatic reactions can be enabled with an attempt limit and per-reaction
timeout. Tab/Shift-Tab moves between fields, left/right changes choices, Enter
creates the watch, and Escape returns. A rejected request keeps the entered
settings so they can be corrected. These limits do not cap total token spend.

### Schedule inspector

Open `/schedules` to inspect cron jobs associated with the current workspace.
The panel shows the prompt, schedule and timezone, next and last run, execution state,
and last recorded error. Arrow keys select a job. P pauses/resumes future runs,
G runs now, X requests cancellation of the active run, and R refreshes. Escape
returns to the conversation; Page Up/Down scrolls the details. Cancellation
remains available while a manual run is pending and does not claim cleanup has
finished. The daemon must be online to execute jobs.

Press N to create or E to edit. Tab/Shift-Tab selects the prompt, timing kind,
cron expression or one-shot timestamp, and paused/enabled state. Left/right
changes choices; Enter saves; Escape returns. New jobs default to paused.
One-shot times require an explicit timezone. Rejected saves preserve the form;
a stale revision requires returning and refreshing before editing again.
`/cron add` and `/schedules` with arguments retain text management. Trigger-store
migration is not yet available. Legacy jobs without a project association remain
accessible through `/cron list`.

Press H on a selected schedule to open its run history. This uses the Runs
inspector: select an attempt to read output and errors or acknowledge its
result. Escape returns to the schedule panel. The view initially includes only
that schedule's runs in the current workspace; unrelated activity cannot push
its history out of the query's result limit. At most 100 matches are listed.

Press D on a schedule to inspect retained output deliveries. Arrow keys select
an attempt; Page Up/Down scrolls the payload. Pending entries offer S to send
saved output without rerunning the model. For uncertain outcomes, first check
the destination: A records that it arrived, while T allows a retry. Both
reconciliation choices require Enter to confirm; Escape cancels. Allowing a
retry does not send it—S is a separate action after the state becomes pending.
Active sending entries cannot be reset. R refreshes and Escape returns to the
schedule panel. Failed actions keep their error visible.

Schedule forms expose an execution timeout (1–3600 seconds) and a one-shot
retry count (0–10). Zero disables automatic retries of failed one-shot prompt
runs. The form fills unset limits with 300 seconds and three retries; saving
makes those explicit. Existing jobs without limits continue using daemon
defaults until edited. Recurring failures wait for the next scheduled occurrence.
Timeout requests cancellation; the execution slot stays held until cleanup
settles. These controls are not token or spending caps.

Timing also offers Interval: set 1–86400 seconds between scheduled occurrences.
The daemon poll cadence controls actual dispatch precision; this is not a
real-time timer. After downtime, an overdue interval runs once and schedules
its next occurrence from the observed tick, without replaying every missed
occurrence. Resume starts a fresh interval. Interval jobs use the same limits,
run history and delivery controls as cron jobs.

### Scheduling from the CLI

`xerxes schedule` uses the same project daemon and job store as `/schedules`.
Start Xerxes in that project, or run `xerxes daemon --project-dir /path/to/repo`
in another terminal, then use:

```bash
xerxes schedule create --project-dir /path/to/repo --schedule 'interval:600' --objective 'Check the build status'
xerxes schedule create --project-dir /path/to/repo --schedule 'cron:0 9 * * 1-5' --timezone Europe/Istanbul --objective 'Review open work' --paused true
xerxes schedule list --project-dir /path/to/repo
xerxes schedule inspect --project-dir /path/to/repo --id <returned-job-id>
xerxes schedule disable --project-dir /path/to/repo --id <returned-job-id>
xerxes schedule enable --project-dir /path/to/repo --id <returned-job-id>
xerxes schedule fire --project-dir /path/to/repo --id <returned-job-id>
xerxes schedule cancel --project-dir /path/to/repo --id <returned-job-id>
xerxes schedule remove --project-dir /path/to/repo --id <returned-job-id>
```

New jobs are enabled unless `--paused true` is supplied, and receive a generated
ID. The project defaults to the current directory; `--socket` selects an explicit
daemon control socket. Execution requires the owning daemon to remain running.
Intervals are integer seconds from 1 to 86400. Cron uses five fields and defaults
to UTC. Legacy slash-separated cron is accepted only when its cadence can be
converted without changing day-of-month/day-of-week semantics.

`disable` prevents future automatic runs; `cancel` requests cancellation of the
current local run. `fire` waits for a manual execution, subject to the daemon's
lease and concurrency limits. A lost connection or timeout can leave the outcome
unknown: inspect the job and `/runs` before trying again. The CLI never retries a
mutation automatically. Removal requires an idle job without an unreconciled
execution receipt and leaves archived output/delivery history intact.

Legacy `--directory`, `--owner`, and `--delivery-id` options return migration
guidance rather than writing to the old trigger store. Event/webhook schedules
are not executable through this command and are rejected. Existing legacy
records are retained and are not automatically activated.

### Importing legacy time triggers

Use `/schedules legacy` to preview triggers in the legacy scheduler store. The
preview reports unsupported conversions and existing migration destinations
without disabling triggers. `/schedules migrate <trigger-id>` imports a selected
interval or compatible cron trigger into the current workspace as a **paused**
schedule. Inspect its timing and prompt in `/schedules`, then explicitly resume
it when ready. The legacy source is disabled before transfer and retained as a
recovery record. Repeating the command retries the same destination without
overwriting edits to the imported job.

Dependency workflows, event/webhook triggers, and cron expressions with both
restricted day-of-month and day-of-week need explicit conversion and are not
imported. A failed transfer keeps its source disabled; retry the migration to
its recorded destination. There is currently no automatic rollback command.

### Agent schedule tools

The normal daemon runtime registers `list_schedules` and `manage_schedule` in
its searchable tool catalog. They use the same store, workspace scoping, input
validation and execution limits as `/schedules`. `manage_schedule` supports
inspect, create, update, pause, resume, cancel and run. Create/update accepts a
prompt, explicit paused state, and exactly one cron expression (UTC by default), future ISO
one-shot time or interval in seconds. Updates require the latest revision from
list/inspect and an idle job.

Create, update, resume and run require a direct user turn and use the persistent
scheduling approval boundary, including in accept-all mode unless the host has
explicitly configured its existing approval bypass. Background monitor and
scheduled turns cannot recursively create or activate schedules. Pause stops
future occurrences; cancel requests cancellation of active work. Run executes
once in the cron session, archives its result and uses the existing delivery
route. Interrupting that tool requests cancellation while the scheduler retains
ownership until execution settles. Schedules require their owning daemon to be
running; the tools do not install a background system service.

Scheduled execution uses the job's stored project directory, even if the owning
daemon started in another workspace. A missing/non-directory project fails
before provider execution. If a job explicitly targets an existing session in
another project, execution fails rather than relocating that conversation;
choose a separate session or correct the schedule's project association. Legacy
jobs without a stored project retain their original host-directory behavior.

### Hooks across multiple workspaces

The daemon selects shell hooks per session workspace for turn, session lifecycle
and compaction events. User hooks run in that workspace; trusted project hooks
come from that project's configuration rather than the daemon startup project.
Workspace hook loading still requires `XERXES_ALLOW_WORKSPACE_CONFIG=1`, `true`,
`yes`, or `on` (case-insensitive words). Partial word matches do not enable it.
Hook runners are cached for up to 32 workspace paths. Restart the daemon to
reliably reload changed hook configuration across all cached workspaces.

Native shell hook deadlines include output collection. On POSIX systems hooks
run in a separate process group: the deadline sends SIGTERM, then SIGKILL after
a 200 ms cleanup grace. Pipe readers are cancelled so inherited output handles
cannot hold the turn open. An expired permission hook fails closed even if its
shell exited zero or printed an approval before a descendant kept its pipes
open. Windows terminates the direct shell process and bounds pipe collection;
POSIX process-group cleanup does not apply there.

### Recovering monitor reactions after restart

When a session is resumed or attached, the daemon reconciles its eligible
reaction policies with saved monitor event cursors. This includes events saved
just before a crash interrupted queueing. Already consumed cursors are not run
again. Expired, cancelled, exhausted and unresolved-executor policies do not gain
new authority from recovery. Evidence must belong to the resumed session's
workspace. Recovery waits for the normal session turn admission.

This recovery occurs when the owning session is loaded; it does not automatically
open every saved conversation at daemon startup. Terminal processes and watches
lost with the old daemon are not recreated by replaying saved events.

The Runs panel and schedule run history support **N** for older results and
**P** for the previous page, with up to 100 runs per page. The page number is
shown in the header. Scope and unread-filter changes return to page one;
refreshing retains the current page boundary. New arrivals appear on page one.

In Runs, **K** cycles all kinds, agents, terminals, schedules and monitors.
**S** cycles all statuses, running, failed, interrupted, cancelled and succeeded.
The active filters are displayed below the header and apply to the full stored
history. Changing either filter returns to page one. Schedule history keeps its
kind fixed to schedules while allowing status filtering.

Runs shows an **X** action in the selected result when its live owner supports
cancellation. The label distinguishes stopping a process or watch from cancelling
a schedule execution or a reaction and its watch. A successful request does not
mean cleanup has finished; the live run state continues to update. Stale rows
must be refreshed. Cancelling a schedule run does not pause future occurrences.

One-shot schedules require a valid calendar timestamp with an explicit timezone,
for example `2028-02-29T09:00:00+03:00`. Both slash commands and the schedule form
reject impossible dates, `24:00`, missing timezones and past times. Invalid dates
are never normalized into another day. Valid timezone offsets are converted to
the same UTC instant for persistence.

### Recurring schedule timezones

Recurring jobs accept an IANA timezone, defaulting to UTC. Set **Recurring
timezone** in the schedule form, pass `timezone` to `manage_schedule` or
`schedule.create`/`schedule.update`, or use:

```text
/cron add --schedule "0 9 * * *" --timezone "America/New_York" --prompt "Review changes"
```

The stored zone controls each recurrence, including after restart and resume.
An update that omits it preserves the existing zone. Invalid zone names fail
before saving. During daylight-saving transitions, nonexistent local times are
skipped and repeated local times can run twice. Next-run timestamps remain UTC.
Interval jobs use elapsed seconds; one-shot jobs use the explicit timestamp's
instant. Their timing is unaffected by the recurring timezone field. Supported
zones and transition rules come from the Bun runtime's ICU timezone data.

### Reviewing interrupted scheduled executions

Scheduled occurrences persist an execution receipt before invoking the model.
If the daemon restarts with an unfinished receipt, or saving completion fails,
the job pauses for review instead of automatically repeating possible side
effects. The Schedules panel shows the recovery warning. Inspect Runs and any
archived delivery output before choosing **Resume**. Resume acknowledges the
uncertainty and retains the previous receipt as metadata: recurring jobs move
to their next future occurrence, while an overdue one-shot can execute again.
Resume is rejected while a run is still running or cancelling.

This is conservative recovery, not an exactly-once guarantee for external
systems. Explicit manual runs and provider/tool retries have separate behavior;
the execution receipt fences automatic scheduled occurrences.

Monitor reaction usage in Runs includes measured parent usage and cumulative
counters for child agents started during that reaction. Repeated child events
are deduplicated; unrelated agents already running are excluded. Failed and
cancelled reactions retain their measured counts. Child accounting remains
marked incomplete because child events do not yet certify complete provider
usage. These counters describe observed work, not a hard token spending limit.

Cancelling one watch's reaction does not cancel other watches in the same
session. Once its provider/tool cleanup settles, the dispatcher admits the next
eligible queued reaction. A reaction deadline or provider failure follows the
same queue-draining rule. Session-wide cancellation and daemon shutdown still
stop the whole dispatcher. Failed reactions remain visible in reaction health;
their already-consumed evidence is not retried automatically.

### Inspect loaded shell hooks

Use `/hooks` or `/hooks list` to inspect shell hooks for the active session's
workspace. It shows loaded source paths, configuration errors, workspace trust,
event, matcher, timeout and whether a hook can deny an operation. Inspection
uses the runtime's cached configuration, not a fresh file preview; restart the
daemon after editing hook configuration. Untrusted workspace hooks remain
unloaded. Inspection does not run commands. Hosts without the native shell-hook
factory report that inspection is unavailable. `/hooks` shows the latest 20
execution outcomes from a bounded history of 100 per cached workspace runner:
event, hook number within the event, completed/denied/failed, timestamp and
duration. Unmatched hooks are omitted. Input and output are not stored in this
history. Failed entries distinguish timeouts, nonzero exits (including the
exit code), and other execution failures without capturing error output.
History is in memory and resets on restart or workspace-cache eviction.
A hook test mode is not yet exposed.

### Schedule timing preview

The `/schedules` create/edit form previews the next eligible run as a UTC ISO
timestamp using the daemon's timing validator, including the selected cron
timezone. Invalid timing is shown in the form; previews never save or start a
job. Paused jobs must still be enabled. The preview is computed when timing
changes, and saving recomputes it against the current clock.

The form also exposes **Missed runs** and **Lateness allowance seconds**. The
default runs once after downtime. Choose **Skip overdue occurrences** to advance
recurring jobs without executing work older than the allowance (default 300
seconds). Missed one-shots pause for review. The inspector shows when a skip has
occurred; it does not count as an execution. Exact allowance-boundary occurrences
remain eligible. Overlap is always forbidden, including timed-out work still
cancelling. Manual Run explicitly overrides timing, while retaining those limits.

### Schedule model-call limits

The schedule form's **Model calls per run** field accepts 1–10000; leave it blank
for no call limit. The limit is shared by the main agent, native child agents,
and auxiliary completions, including compaction. Exhausting it stops further
model calls and marks required work failed. Automatic titles are skipped when
no calls remain. This limits logical calls, not tokens, money, or transport-level
retries. Each retry of the scheduled job receives a fresh per-attempt allowance.
Nested budget scopes inside a run must satisfy every enclosing limit and report
usage to each scope. A separately admitted schedule attempt or monitor reaction
starts an independent scope; it does not inherit an expired notifying run.

### Schedule delivery destinations

The `/schedules` form includes **Delivery channel** and **Recipient or room ID**.
Focus the channel field to load configured adapters, then use Up/Down to choose
one. Disabled adapters are labelled; their status is not a delivery guarantee.
Enter the recipient explicitly. Choose `none` for archive-only output; selecting
it clears the recipient. Saving never sends a test message. Output is archived
on execution, and failed forwarding remains recoverable in the delivery panel
without rerunning the scheduled prompt. Narrow terminals show the current field
and its neighbors to keep the editor visible.

### Agent mode settings in the TUI

Open `/config` (or `/config agents`) to edit Light, Balanced and Smart.
F2 cycles the default between inherit and modes
with configured models. F4 disables the selected mode in the draft, resetting
the default to inherit if necessary. Enter saves those changes; Escape discards
unsaved changes. Reasoning discovery runs only while editing the effort field.

Tab and Shift-Tab move through each mode's provider-profile name, model ID and reasoning
effort. Up/Down selects a saved provider profile or an offered reasoning effort
in the corresponding field; typing a value is also supported. The model field
discovers models from the selected saved profile (or the active profile when
blank). Use Up/Down to select a model and F5 to refresh. Discovery warnings and
failures are shown without replacing your draft; custom model IDs remain valid
input. Reasoning choices
come from the same daemon resolver as the main reasoning picker: Codex uses its
authenticated catalog, with bundled catalog/provider fallback when unavailable.
The selected profile is used for both discovery and validation, matching save
validation. Enter saves; Escape closes. Provider profiles are listed on the screen;
create credentials through `/provider` first. Blank provider uses the parent
profile; blank effort leaves the effort unspecified. Clear a tier's fields to
disable it. Only configured tiers are advertised to the model.

Saved settings take precedence over `runtime.agent_intelligence` and persist in
`$XERXES_HOME/daemon/agent-settings.sqlite`. The daemon refreshes agent tools on
save; newly spawned agents use the settings, and existing children keep their
captured configuration. Concurrent stale edits are rejected. `/config runtime`
retains the text runtime configuration view. Model IDs are editable text;
provider authentication stays in the existing profile store.

The one-shot and ACP CLI entry points also load the saved agent-mode settings.
They share the daemon's provider-profile resolver: selecting a child profile
uses that profile's credentials, endpoint and model limits without changing the
parent's active profile. Missing or unsupported profiles produce an explicit
error. Embedded hosts must still supply their own provider-resolution port.

### Chat switching and task progress

Returning to a chat restores its latest checklist alongside its transcript and
running activity. The checklist belongs to the session and remains across turns
until the agent replaces or clears it. Replayed tool rows show readable command
or file summaries; full argument payloads belong in inspection views.

### Scheduled-run token accounting

The schedule inspector shows the last execution attempt's provider-reported input and output tokens across parent, native child, and auxiliary calls. This is collected even without a model-call limit. Missing usage, failed streams, cancelled requests, or calls still pending when the attempt ends make the report **partial**. A partial report contains only observed usage and must not be interpreted as zero cost or an exact total. Optional work finishing after the snapshot may remain unaccounted for in that snapshot. This reporting does not enforce a token or monetary cap. Retries have separate attempt snapshots; the schedule shows the latest one.

Scheduled attempts also preserve their usage snapshot in **Runs → inspect**, alongside that attempt's output and outcome. A later execution does not replace earlier usage records. Runs created before usage accounting was added show usage as unavailable. Partial snapshots remain partial after restart; they do not claim to include work that was still pending at finalization.

Running schedules checkpoint call admission before contacting the provider and observed usage after each call settles. If the daemon exits unexpectedly, run recovery retains that evidence and marks the run interrupted. Any admitted call without a settled usage record remains pending/unknown. Usage within an unfinished stream may not yet be reported, so the checkpoint is not a billing reconciliation. Checkpoint storage errors stop further model-call admission.

### Monitor reaction usage

Monitor-triggered turns now share the same parent/child/auxiliary call accounting as schedules, with live checkpoints and a durable per-attempt snapshot in Runs. The watch's reaction summary accumulates those attempt reports. Missing child usage, failed provider attempts, or auxiliary calls pending at finalization keep the report partial. Injected runners that only expose session/event counters retain those observations as incomplete; they do not establish coverage of every provider call. Usage reporting does not impose a total token cap.

Monitor usage is also checkpointed into the reaction mailbox before the run-history projection. A daemon crash preserves already-observed aggregate usage even if final settlement never occurred. Interrupted claims remain incomplete and disable automatic continuation for that owner rather than replaying uncertain effects. Missing final usage cannot erase checkpoints, and repeated checkpoints do not add the same tokens twice. The two stores are not a single transaction: a crash between writes may leave the detailed run snapshot behind the authoritative mailbox aggregate.

### Preview hook selection

`/hooks preview <event> [tool-name]` previews loaded hooks in execution order without running their commands or changing trust. For example, `/hooks preview PreToolUse ReadFile` shows which permission hooks match `ReadFile`, which are skipped by their tool-name matcher, and each timeout. Native lifecycle names such as `on_turn_end` also work. Event aliases use the same resolution as hook configuration. Omit the tool name to test against an empty tool name.

This is a selection dry run, not a sandboxed execution test: it cannot predict a hook's output, mutation, success, or permission verdict. The preview uses the current cached configuration, reports its sources and errors, and never enables untrusted workspace hooks. Restart to load configuration edits.

`/hooks failures` shows failures and permission denials from retained hook history, newest first. Use `/hooks failures PreToolUse` (or a native lifecycle event) to filter it. Execution failures retain their timeout/exit/execution classification, exit code where known, and duration; permission denials are listed separately because a policy decision is not an execution failure. Successful and unmatched hooks are excluded. Configuration errors remain visible.

This history is the latest 100 executions per workspace, not 100 failures; successful executions can age out older failures. It is held in memory and resets on daemon restart or workspace cache eviction. Empty results mean no matching failures in retained history, not proof that the hook always succeeded. Inspecting failures does not execute hooks or change trust.

### Explicit isolated agent requests

`AgentTool` accepts `isolation: "worktree"`; each `SpawnAgents` entry accepts the
same option. Daemon/TUI, one-shot and ACP CLI hosts supply the native Git adapter through
`worktreeForWorkspace`. Embedded hosts can supply that factory or a fixed
`worktree` port. Unsupported runners reject the request. Non-Git projects and
repositories without a committed HEAD produce an actionable setup failure.

The optional `worktree_ref` selects an existing branch, tag, commit ID or Git
revision such as `HEAD~1`; it requires `isolation: "worktree"`. Omission starts at
committed HEAD. The revision is resolved before allocating each attempt, including
retry; use a full commit ID when the baseline must remain fixed even if a branch
or tag moves. Missing or invalid revisions fail without creating a checkout. Parent
uncommitted edits are not copied for revision-based starts. File-tool resolution and the child's saved
working directory use that checkout, while its transcript remains owned by the
parent project. Retry allocates a new checkout and preserves dirty prior results.
Only unchanged checkouts can be removed automatically. A fixed injected adapter
and its owning cwd cannot change by reconfiguration. The workspace factory
selects the captured execution generation's cwd, so switching projects during
setup or retry cannot move the child to another repository. Existing tasks keep
their original execution configuration; new spawns use the latest settings. Review/apply, setup policy and retained-checkout recovery are
still pending.

To start from the parent's current files, set `worktree_source: "working-tree"`
with `isolation: "worktree"`. This is mutually exclusive with `worktree_ref`.
Capture includes tracked edits/deletions and untracked non-ignored files. Ignored
untracked files are excluded. It uses a unique temporary Git index and does not
modify the parent's index or files. Copied edits are unstaged in the child;
new files remain untracked, so the parent's staged/unstaged selection is not
imported. The captured tree is recorded in the ownership manifest without
creating a commit.

Capture rejects submodules/nested Git repositories and detects source/HEAD
changes observed during validation. It is not an atomic filesystem snapshot;
files written after capture belong to later work. Retry captures the parent's
current files again. Inherited dirty files prevent automatic cleanup just as
agent edits do. Setup failures preserve any allocated checkout for inspection.

### Reviewing retained agent workspaces

`/workspaces` lists retained checkout records in the active session's repository,
including damaged/missing checkouts as unavailable records. It reads persisted
manifests, so it works after a daemon restart. Each page has up to 100 records;
use the printed `/workspaces after <cursor>` command for the next page.

`/workspaces inspect <id>` shows the task ID, checkout path, branch, committed
baseline, captured tree when applicable, current HEAD, status and a diff from the
starting state. Non-ignored untracked files are included in the diff. Inspection
uses a temporary index; it does not change the checkout's index, files, commits
or the parent workspace. Ignored files appear in status but their new contents
are excluded. Diff output is bounded to 1 MiB; oversized output returns an error
instead of claiming a complete review. Inventory is limited to 4096 storage
entries; a specific workspace ID can still be inspected above that limit.

Records do not prove whether an agent is running. These commands do not apply,
merge, hand off or delete results. Those actions are not yet exposed. Git routing
variables inherited from the launcher cannot redirect these operations to a
different repository.

In the TUI, bare `/workspaces` opens the interactive review panel. Use Up/Down or
click a record to select it, Tab to focus the diff, Page Up/Down to scroll, and
Left/Right to read long lines. N/P pages the inventory, R refreshes it and Escape
closes the panel while preserving chat history. Narrow terminals stack the list
above the diff. A display truncation warning means the visible diff is incomplete.
Explicit `/workspaces list` and `/workspaces inspect <id>` retain textual output.

Press C in the workspace review panel (or click **Check apply**) to test whether
that exact reviewed patch applies to the parent checkout. The check includes
binary content and untracked non-ignored files. If the agent's reviewed content
or HEAD changed, refresh the review first. Conflicts are reported and both
workspaces remain unchanged. A successful check describes the destination at
check time; it does not reserve its state or approve a later operation. Apply
requires the separate confirmation below; handoff is not yet exposed.

After a successful check, **A** offers apply for the reviewed patch. The panel
shows the destination and requires **Y** to confirm; **Esc** cancels. Changes to
the agent result or affected destination files invalidate the operation and
require a fresh check. Apply preserves the parent index and agent checkout and
saves backups before writing. Failed partial applies restore only unchanged
integration output; concurrent edits are preserved with an error naming the
backup directory. Do not retry an unknown/disconnected result automatically.
Press **I** to inspect saved integrations. Select an interrupted operation and
press **B**, then **Y**, to restore its original files; **Esc** cancels. Recovery
verifies backups and refuses a live or unverifiable lock owner. Newer file edits
remain untouched and are reported as conflicts. Completed applies cannot be
undone here; B on a completed apply only releases a leftover lock belonging to a
verified dead owner. **R** refreshes, **N/P** pages, and **Esc** returns to workspace review.
Recovery guards release on process death; legacy guard files from earlier builds
still require owner inspection rather than automatic deletion.

An operation that stopped during preparation appears as `preparing`. **B** then
offers **Abandon preparation**, which releases its verified dead owner's lock
without changing destination files. A saved `applied`, `rolled-back` or
`abandoned` operation offers only **Release leftover lock**, preserving any newer
edits. Unreadable artifacts remain visible as errors and are never silently deleted.

Selecting an integration checks its saved backups and previews affected files:
`restore` matches the prepared result, `unchanged` already matches the original,
`conflict` contains different/newer content or a changed HEAD, and `preserve`
belongs to a completed operation. Use **PgUp/PgDn** for file pages and **Left/Right**
for long paths and explanations. This preview describes check-time state;
recovery revalidates files before restoring them.

### Setup for isolated agent workspaces

Opt in per repository in `$XERXES_HOME/workspace-setup.json` (normally
`~/.xerxes/workspace-setup.json`). Keys are canonical absolute Git top-level paths:

```json
{
  "workspaces": {
    "/absolute/path/to/project": {
      "command": ["bun", "install", "--frozen-lockfile"],
      "timeout_ms": 120000
    }
  }
}
```

This is user-owned executable configuration. Xerxes does not automatically run
setup instructions from repository files. The command receives literal arguments
and runs in each new isolated checkout after starting files are materialized,
before the agent starts. Timeout is 1–120000 ms; output is bounded. Cancellation
uses the existing foreground process-tree termination path and never backgrounds
setup. Retries that allocate a new checkout read the current setup policy again.

Failure prevents agent startup and retains the checkout plus its setup record.
`/workspaces` inspection displays completion/error and bounded stdout/stderr.
A started record without completion is shown as incomplete, not proof that a
process is still running. This is currently configured through the user file;
a dedicated setup-settings editor is not yet exposed.

A setup failure keeps the agent task retryable. Correct the user setup policy,
then use the existing agent retry action: Xerxes allocates a fresh checkout and
runs setup again under the same task identity. Earlier failed checkouts and
setup output remain available in `/workspaces`. Repeated setup failure updates
the task's error and retained-checkout path; no agent runner starts until setup
succeeds. This retry behavior applies to the live daemon; restart recovery is
not a guarantee that an interrupted setup process can be resumed.

#### Command completion watches

Choose **Trigger: Command completion** in the new-monitor TUI form, or call
`monitor_terminal` with `terminal_id` and `trigger: "completion"`. No `match`
is needed. The watch stores one exit result with a bounded final output tail;
it also handles a command that finished just before the watch was created.
Add `react: true` when the user requested a follow-up investigation or response.
The existing reaction mailbox queues that bounded turn behind the owner's work,
persists the evidence, and prevents repeated execution of the same claim.
Stopping the watch before completion prevents delivery. A daemon restart cannot
restore an unfinished process's pipes or an in-memory watch; already-recorded
completion evidence remains eligible for the existing mailbox recovery rules.

#### Incremental terminal output

The model can use `read_terminal_output` with `terminal_id` and then pass the
returned `cursor` on subsequent reads. Independent readers do not consume each
other's logs or the `check_command` buffer. `hasMore` indicates another retained
page, while `droppedChars` explicitly reports output lost to retention. This is
a read operation, not a wait; use a completion watch for an automatic follow-up.

Archived terminals use `run:<run_id>` from Runs and accept the same cursor after
reconnecting. Saved output retains the last 64000 UTF-16 code units. Active
output is checkpointed every 250 ms and on completion; a crash can lose output
since the last checkpoint. Old records without cursor metadata remain readable
through ordinary terminal inspection, but cannot invent a historical offset.

#### Finding terminal work in F8

Press `f` in the terminal list to cycle All, Running and Finished. Open a terminal
and press `/` to search the retained output using case-insensitive literal text.
Matching rows show their original line numbers. Enter applies the search; Escape
clears it before leaving the detail view. Search is limited to the retained tail,
so it cannot find output already discarded by retention. Search typing does not
send input or stop commands. PTY input and stop controls retain their existing keys.

#### Upcoming work in Runs

Workspace `/runs` shows a compact **Scheduled next** section above execution
history, ordered by the scheduler's recorded next due time. Paused schedules and
other workspaces are excluded. Wide terminals show up to three jobs; narrow
terminals show one, with the total count retained. Press `T` or click the section
to manage schedules. Session-only Runs does not show workspace-wide future jobs.
The displayed time is a due time, not a guarantee that execution will start then;
daemon availability, existing executions and misfire policy still apply.

#### Session attention in Runs

Runs shows pending tool approvals and user questions for the current session.
This section remains session-scoped even in workspace history, so another chat's
prompts are not exposed. Press `E` or click the section to return to chat and use
the existing prompt. The summary neither answers questions nor grants approval.
It refreshes with Runs and disappears when the underlying wait resolves or is
cancelled. Pending waits are live daemon state, not restart-restored approvals.

Incremental reads on a durable host commit the observed output offset before
returning a cursor. A process crash therefore cannot roll back a successfully
returned cursor solely because the periodic checkpoint had not run. A storage
failure rejects the read. Unread/uncheckpointed output may still be lost, and
retention can still overtake a cursor; this does not guarantee recovery from
storage-device loss or a power failure.

Full snapshot rollback removes only files captured in its pre-restore backup
and absent from the target snapshot. It preserves ignored files that were never
backed up and honors the restored ignore rules; ignored build output may remain
and may need an explicit rebuild or cleanup. Backup snapshots can restore removed
source files. A file path that has become a directory is refused during cleanup
rather than recursively deleting its new contents.

Preview a snapshot restore with `/rollback diff <snapshot-id>` before running
`/rollback <snapshot-id>`. The patch direction is current files to the snapshot:
added lines will be restored, removed lines will be removed. New files and deleted
captured files are included. Ignored files outside snapshot capture are excluded.
Preview uses a private temporary index and does not restore files, change the
workspace's Git index, or create another snapshot. It creates temporary Git tree
objects in the shadow repository. Git previews exceeding 2 MiB fail explicitly;
the transcript displays at most 100,000 characters and labels truncation.
The preview prints a guarded command, `/rollback apply <id> <revision>`. Use it
to reject captured-file changes since preview before writing workspace files.
The revision binds the target snapshot and current captured Git tree, including
new/deleted files and executable modes. A refused restore may save a backup but
does not change workspace files. Preview again after a rejection.
Legacy `/rollback <id>` still restores directly without a preview check. Ignored
uncaptured data is outside revision validation; arbitrary external writers are
not locked out during capture or restore. Full transactional failure recovery
remains incomplete.

Snapshot restores preflight destination paths before writing files. A directory
or special file at a target-file path, or a non-directory/symlink ancestor,
produces an error instead of letting Git overwrite earlier files and then fail.
This also applies to selected-file restore. Resolve the obstruction explicitly
and preview again; file-to-directory and directory-to-file transitions involving
such obstructions are not automatically dismantled. This check does not prevent
filesystem changes racing the subsequent restore or recover all I/O failures.

The CLI captures workspace snapshots before model turns and `!` shell commands.
Capture finishes before that work begins; a failed capture displays a warning
without blocking the requested work. Ignored files and excluded secret files
are not captured. Older sessions may have no automatic snapshots from before
this behavior was enabled; a new snapshot cannot recover an earlier file state.

`/snapshots` opens the snapshot timeline in the TUI. Entries show capture time and
linked turn/session information where recorded. Use Up/Down to select, Tab to
focus the scrollable diff, Page Up/Down to scroll, and Left/Right for long lines.
`A` offers a file-restore confirmation; `Y` executes the guarded preview revision.
Escape cancels confirmation before closing the timeline. Restore errors leave the
panel open and require `R` to refresh before trying again. Restoring files leaves
the conversation unchanged. `/snapshots list` prints the existing text list.
An empty timeline does not imply automatic capture is enabled: `/snapshot` takes
an explicit capture; automatic pre-turn capture remains opt-in.

In the snapshot timeline, `F` cycles through changed files and back to all files.
The heading and confirmation distinguish restoring a captured file from removing
a new file absent from the target snapshot. A selected-file restore checks that
file's preview revision, saves a backup, and leaves unrelated edits untouched;
unrelated edits made after preview do not invalidate the selected file. Removed
files can be restored from the resulting pre-restore backup. Directory and
submodule selection is rejected. This remains subject to the restore concurrency
and I/O recovery limitations described above.

Restores now write a private recovery journal before changing workspace files.
The snapshot timeline reports unfinished attempts; `B` selects the first affected
backup for preview without restoring it. A prepared record can mean a restore is
still running or that it was interrupted. Failed records preserve the error and
backup ID. Review current changes before confirming recovery. A successful restore
of that backup (the whole workspace, or the same affected file) marks the original
attempt recovered. Ordinary pruning retains unresolved target/backup snapshots;
an explicit snapshot reset still deletes snapshot history and recovery records.

This journal makes partial outcomes discoverable. Caught restore failures now
attempt bounded automatic reversal: files already matching the backup are left
alone; files still matching the intended target are restored to their backup
state. Different content or obstructed paths are preserved and reported for
review. The original operation still returns an error even when reversal succeeds.
A killed process cannot execute reversal; its prepared record remains available
for explicit recovery. This does not guarantee durability after a power loss.
Unresolved history is bounded: 128 outstanding attempts stop further restores
until recovery, and the journal retains the most recent completed history.

A confirmed dead snapshot-lock owner can now be recovered immediately; an empty
or malformed freshly created lock still gets the existing initialization grace
period. Recovery does not infer that a process is dead from elapsed time alone.
Repeated failed retries to the same recovery target/scope retain the original
unfinished record and the latest retry. Older retry backups remain ordinary
snapshots subject to retention; the original recovery backup stays pinned.

Automatic reversal checks each file as it proceeds and stops admitting further
files after ten seconds (an in-flight Git command retains its own timeout).
Unprocessed or conflicting files leave the attempt failed and its backup pinned.
Fully verified reversal records phase `reverted`. File identity uses captured
Git contents/mode; ignored uncaptured data and arbitrary writes racing a check
remain outside the guarantee. Reversal is not an exclusive filesystem transaction.

`/tools` now reads the native runtime's registered tool inventory when no embedding
host supplies a separate catalog. It labels schemas as loaded, deferred (available
through tool search), filtered by the current agent/mode, or unexposed (registered
but absent from a non-deferred runner's schemas). Registration does not prove that
a call has permission, authentication, a connected browser/server, or a healthy
external service. These checks still occur at the owning tool boundary. A daemon
without an inventory reports that source as unavailable rather than inventing a
list from its tool count. `/tools list` uses the same inventory.

### MCP connection health and retry

Use `/mcp` or `/mcp status` to inspect configured native MCP servers, including
failed initial connections and disabled entries. Each row reports connection
state, discovered tool count and the last redacted lifecycle error when present.
`/mcp reconnect <name>` retries one enabled server; `/reload-mcp` retries all
enabled configurations, including those that failed at startup. Neither command
enables disabled servers or reloads changed configuration files from disk.
Restart the daemon after editing `mcp.json`. A disabled user entry still reserves
its name against a trusted project entry. Connection health does not establish
tool authorization or provider readiness.

Configuration is validated before constructing a transport. Invalid entries are
skipped with field-specific warnings; valid siblings still load. Use JSON booleans
for `enabled` and `allowPrivateNetwork`, string arrays for `args`, and string-valued
objects for `env` and `headers`. Unknown settings are rejected so spelling mistakes
cannot silently change behavior. `timeoutMs` must be positive and at most
2,147,483,647 milliseconds. Disabled entries must also have valid settings.

The default transport is `stdio`, which requires `command`. URL servers must
explicitly set `transport` to `sse` or `streamable_http` and supply an HTTP(S)
`url` instead of `command`. Put authentication in `headers`, not URL credentials.
Malformed JSON diagnostics identify the file without quoting its contents.

Removing or replacing an MCP registration cancels its pending retry delay and
prevents future attempts. Daemon shutdown applies this to all registrations.
An already-started connection still has to settle under its transport's timeout;
if it succeeds after removal, the manager disconnects that late client. This is
not a guarantee of immediate transport termination.

### Inspect discovered skills without running them

`/skills inspect <name>` shows a discovered skill's source file, platform support,
declared tools/dependencies/subcommands, and literal instructions. Completion
suggests admitted registry names. Inspection does not activate a skill, call a
provider, execute setup commands, expand `$ARGUMENTS`, or evaluate shell snippets.
Instructions are capped at 100,000 characters; the response reports truncation
and points to the source file. Declared requirements are not proof that the
current session can execute them. Undiscovered or rejected skills return an error;
inspection does not bypass workspace trust. Use `/skill <name>` to activate a
skill deliberately. Installation and remote skill search remain separate features.

`/skills diagnostics` refreshes discovery and reports rejected or shadowed sources,
parse failures, discovery limits and fallback names. Each entry includes its
reason and source path; a shadowed entry identifies the winning source. Results
are capped at 200 entries with 1,000 characters of detail each. Diagnostics are
from the current discovery pass, so fixing/removing a source clears its old note.
An empty report means no discovery notes, not verified tool/dependency readiness.
This command does not trust, install, activate or modify a skill.

### Plugin registration ownership

A plugin module's `register()` callback may unregister only plugins created by
that callback. Trying to remove a previously loaded plugin fails registration;
partial registrations from the failing module are rolled back. Hosts can still
remove plugins outside registration, and modules may clean up their own temporary
registrations. This protects registry API lifecycle behavior; imported plugin
modules execute trusted code and are not isolated in a security sandbox.

### Plugin registration inventory

`/plugins` identifies whether the daemon received an embedding host's plugin
registry. The default CLI currently has no native module-loading lifecycle wired
into this registry and reports `unconfigured`; this is different from an injected
registry with no registrations. Slash-command plugins are listed separately.
`/plugins inspect <name>` shows the source module path (or programmatic host
registration), version, description, and registered tool/hook/channel/provider
names and declared dependencies. Inspection calls no capability and makes no
provider request. Registration is not proof of execution readiness or dependency
health. Module loading, persisted enable/disable and runtime reload management
remain unsupported by these inspection commands.

### Calling configured MCP tools

The normal CLI daemon now publishes connected MCP tool schemas to its turn runner,
using the same configured manager as `/mcp`. Successful startup connections and
explicit reconnects rebuild the runner's tool inventory. Each tool has a stable
provider-safe name beginning `mcp__`, with readable server/tool segments and a
hash of their exact names. Use `/tools` to see these names when configuring tool
policy. Identical tool names on different servers remain distinct.

Tools use native argument validation and normal permissions. Server safety
annotations do not grant read-only or auto-approval privileges. The shipped
coding and creator profiles include connected MCP tools; custom profiles and
restricted-mode allowlists retain their limits. Server error results fail the
tool call. An old runner cannot silently route an in-flight call to a reconnected
replacement with potentially different schemas: it reports a stale-connection
error, while subsequent turns receive the rebuilt inventory. The daemon host
closes its MCP manager on shutdown.

Daemon-backed sessions, one-shot commands, resumed one-shot commands and ACP
share the same user/project MCP configuration policy. One-shot and ACP startup
wait for initial discovery before publishing tool schemas and close their MCP
connections during shutdown or startup failure. Project configurations require
the exact opt-in value `1`, `true`, `yes` or `on` (case-insensitive) in
`XERXES_ALLOW_WORKSPACE_CONFIG`. User entries take precedence, including disabled
ones. Direct MCP resource/prompt tools remain separate work; connection status
alone does not establish those surfaces.

Cancelling an MCP tool that is queued behind another operation rejects its wait
promptly and prevents dispatch. Its bounded queue slot remains reserved until
preceding operations drain, so repeated cancellations cannot create an unbounded
queue. Once a call starts, cancellation is handled by the transport; the manager
does not report completion before that active operation settles.

The native MCP settings host is wired into the normal CLI daemon for
user `mcp.json` entries. Its RPC reads expose configuration field presence and
health without returning command arguments, endpoints or credential values. Saves
use a file revision and preserve omitted fields; connection failures leave the
working server intact. A missing file opens an empty editor and is created on the
first successful save. Project-only entries cannot be edited here. A leftover
`.settings-lock` requires checking that its writer is no longer active before
recovery; it is not automatically stolen.

### Edit MCP settings in the terminal

Open `/config mcp` to edit servers in the user MCP file. F2 starts a new server:
enter a unique name, then its command or HTTP transport and URL. Existing names,
including project registrations, cannot be overwritten through creation. Use Up/Down to
select a server, Tab/Shift+Tab to select a field, and Enter to toggle or edit it.
Ctrl+S validates, connects and saves the draft. F5 reloads and discards a draft;
Escape cancels the current field edit or closes the editor. Loading and saving
keep the editor open until the operation settles.

Saved launch and authentication values are hidden. Leaving an input blank keeps
its existing value. Enter arguments as a JSON string array, and environment or
headers as JSON objects of string values; those objects replace the entire field.
Delete marks the selected field for removal. When changing between stdio and HTTP,
remove the old command or URL and set the new transport's required field before
saving. Failed saves retain the draft. A stale revision requires reloading before
retrying. `/config` continues to open agent provider/model/reasoning settings.

Monitor notification delivery errors are shown separately in the monitor inspector.
A failed UI/notification callback does not change a completed watch to failed or
cancel a reaction that was already queued. Matching events remain in Runs history;
a later successful output notification clears the in-memory delivery warning.
This does not automatically retry delivery or reattach a terminal after daemon
restart. Interrupted-run recovery preserves records, not process handles.

## Native language servers

Create `lsp.json` in your Xerxes home directory (`XERXES_HOME`, normally
`~/.xerxes`) to enable native language-server tools. For example, after installing
`typescript-language-server` separately and making it available on PATH:

```json
{
  "servers": [
    {
      "name": "typescript",
      "command": "typescript-language-server",
      "args": ["--stdio"],
      "languageId": "typescript",
      "extensions": [".ts"],
      "enabled": true,
      "timeoutMs": 30000
    }
  ]
}
```

Commands execute directly with the configured arguments, in the active session's
workspace. Optional `env` supplies string environment overrides. These are trusted
user-configured executables, not sandboxed processes. Xerxes does not install
servers or automatically load executable settings from a project. Invalid user
configuration stops startup with a configuration error. Restart the runtime after
editing this file; `/reload-mcp` does not reload language-server configuration.

Each enabled suffix belongs to one server; the longest matching suffix wins
(for example, `.d.ts` over `.ts`). Matching is case-sensitive. Servers start lazily
and are shared only within the same canonical workspace and configuration entry.
At most 32 workspace/server hosts are retained until runtime shutdown.

Configured servers expose `LSPTool` to the built-in default and creator agents in
daemon, one-shot, resumed-session and ACP execution. Custom agent tool lists must
explicitly include `LSPTool`. Its actions are `definition`, `references`, `hover`,
`symbols`, and `diagnostics`; `file_path` is required and positions are zero-based
UTF-16 line/character offsets. The server must support document open/change
synchronization and UTF-16 positions. Missing binaries, unsupported capabilities,
and unconfigured file types produce errors; ordinary text search remains usable.

Diagnostics are fresh only when a pushed result matches the current document
version. `fresh: false` means unconfirmed, not zero problems. Automatic edit feedback is described below. Use `/config lsp` to edit server settings.

Native `LSPTool` diagnostics wait up to one second for a matching-version server
publication. A timeout remains unconfirmed. Cancellation stops waiting, and an
external source edit during the request rejects the result so it cannot be used
as evidence for the newer file contents.

Daemon, resumed-session, ACP and one-shot turns collect automatic LSP feedback around native
file-writing tools when edit diagnostics are enabled. Each turn owns its baseline;
read-only turns do not start a checker. New findings are compared against fresh
pre-edit diagnostics. If the pre-edit result was unavailable, findings are labelled
current rather than newly introduced. Server errors/timeouts leave LSP unconfirmed
and do not suppress checker findings. Reports are bounded to eight edited files
and appended to the daemon or ACP conversation for the next turn. ACP emits the
report before its turn-end event. One-shot execution includes the report in text
output, the JSON response, or streamed text events before the result record.
Set `XERXES_EDIT_DIAGNOSTICS=0` to disable automatic feedback; explicit `LSPTool`
remains available. Use `/config lsp` for settings and `/lsp status` for workspace connection health.

Use `/lsp` or `/lsp status` to inspect language servers for the active session's
workspace. It distinguishes idle configuration, startup, ready connections,
disabled servers and failures. Inspection never launches a server or replaces the
conversation. Use `/lsp release <name>` to stop and release a failed or idle workspace host.
The next tool request starts a fresh host lazily; this does not re-run the failed
request or change configuration. Release may interrupt requests sharing that
workspace/server. Other workspaces remain unaffected. Cleanup failures block
replacement until a later release succeeds. Configuration edits still require
runtime restart; this command is not a configuration editor.

### LSP settings editor

Open `/config lsp` to create, edit or remove user language servers. Use Up/Down
to select a server, Tab/Shift+Tab to select a field, and Enter to edit or toggle it.
Arguments and file suffixes use JSON arrays; environment uses a JSON object.
Saved command/argument/environment values remain hidden. Blank input preserves
a saved field; Delete clears optional arguments, environment or timeout.

F2 creates a server; supply its language ID, suffixes and command. Ctrl+S saves
and applies changes to the running daemon, refreshes tool inventory, and stops
affected hosts. Their next request starts them lazily. F4 asks for Y/N confirmation
before removing the selected server. F5 reloads and discards the current draft.
Escape cancels a field edit or closes the overlay. Failed saves retain the draft
and explain that reload discards it. Short terminals show the selected field so
keyboard editing remains visible. The overlay does not replace the conversation.

### Opt-in installed-server acceptance

To check an already installed clangd without downloading anything, run from the
repository root:

```bash
XERXES_TEST_CLANGD=/absolute/path/to/clangd bun test xerxes/test/lspRealServer.test.ts
```

The test creates and removes a temporary C++ workspace, checks semantic navigation,
version-matched clean/error/corrected diagnostics, automatic edit feedback and host
shutdown. It is skipped when `XERXES_TEST_CLANGD` is unset. This exercises the native
LSP host against clangd; it does not establish compatibility with every language
server or verify the native terminal UI.

### Context inspection

Open `/context` to inspect instructions, retrieved memory layers, retained
conversation and tool schemas. Tab or Left/Right changes sections; N/P pages
through entries; arrows and Page Up/Down scroll; R refreshes; Escape returns to
the conversation. This is read-only and makes no provider request. `/usage`
continues to show usage and subscription information.

Instruction, memory and schema data comes from the latest assembled scaffold;
conversation data comes from the retained session transcript. The live provider
request can differ while a turn runs. Counts are approximate token contributions,
not billing figures. Missing scaffold data is labelled unavailable. Pages hold
20 entries with excerpts limited to 8000 characters; content changes invalidate
older page generations and require a refresh.

Named memory layers are visible, but individual retrieval scores/source-file
attribution, pinning, exclusion and compaction-history controls remain unfinished.

MCP configuration reads at startup and in `/config mcp` are limited to 1 MiB
of valid UTF-8 in a regular file. Special files such as named pipes are rejected
without waiting for a writer. Startup reports a read warning and skips an unreadable
file; it still supports linked configuration files. Settings editing additionally
requires a non-symlink file with one hard link. Invalid encoding is rejected rather
than silently replacing bytes in server commands or credentials.

The `/context` compaction section shows the latest 100 successful compactions,
newest first, including manual, automatic and delegated-session records when
inspecting that session. Records persist with session metadata. Older sessions
may expose only their existing last-compaction stamp; earlier events are not
reconstructed. Each entry includes time, reason, message reduction, token estimates
and the recorded pre-compaction archive path if present. A recorded path is not
proof that the file still exists. These records are inspection metadata and do
not consume model context. This view does not restore archives or retry compaction.

Schedules support an optional **Lifetime attempts** limit in `/schedules` creation
and editing. `manage_schedule` and schedule create/update RPCs accept `max_runs`
(1–10000, or null for unlimited). `runs_started` reports persisted admissions;
manual runs, retries, failures and cancelled admissions all count. Existing jobs
start at zero when no counter was recorded; previous executions are not reconstructed.
Changing the limit, pausing or resuming does not reset the counter. Exhausted jobs
cannot run again until their limit is raised or removed; the next scheduled check
pauses them. The counter is saved before execution, and a failed save prevents
execution. This limit is distinct from per-attempt model-call limits or token usage.

The schedule editor also supports **Expiry ISO time**, an optional timestamp with
an explicit timezone. Create/update and `manage_schedule` accept `expires_at`;
null clears it and omission preserves it on edit. No new automatic or manual
attempt is admitted at or after expiry, and an expired job pauses at the next
scheduler check. Resume alone does not override expiry. Already-admitted work
keeps its execution timeout and cancellation controls. Expired records remain
inspectable, and restart does not extend their deadline. Expiry prevents missed
occurrences from running after a machine returns late; it does not implement a
model-evaluated stop condition or same-session follow-up routing.

In `/schedules`, **Run in → This conversation** binds a follow-up to the current
conversation's persistent ID. Set both a lifetime attempt limit and expiry before
saving. Model schedule tools accept `target: "session"` for the same behavior;
new jobs otherwise retain independent scheduling. Editing an existing bound job
preserves its original conversation, even from another tab; choosing independent
clears the binding. Follow-ups wait behind current conversation work, coalesce
missed occurrences through the scheduler, and stream into clients viewing that
conversation. Cancel controls also apply while waiting. The execution timeout
includes that wait, and expiry is checked again when the queue reaches the job.

After restart, an existing saved conversation is reloaded with its context and
identity preserved. Deleted, unsaved or identity-mismatched conversations fail
explicitly; no replacement chat is created. A blank chat with no completed exchange
may not yet have a persisted transcript. This target mode does not yet evaluate
stop conditions or provide a dedicated `/loop` command.

A model tool cannot immediately run a schedule targeting the turn that is awaiting
that tool. This would wait on itself. The tool receives guidance to create/resume
a future wake-up or use `/schedules` after the turn finishes; no attempt is consumed.
External controls retain queued execution and cancellation behavior.

### Conversation follow-ups with /loop

Open `/loop` to see only the current conversation's follow-ups. Press N to create
one: the draft starts paused, targets this conversation, checks every ten minutes,
allows ten lifetime attempts and expires after 24 hours. Edit these values and the
prompt before saving; enable it when ready. P pauses/resumes, X cancels active or
queued execution, G runs once, and H opens run history. Escape leaves the form or
closes the panel without replacing chat history. `/loop list`, `/loop pause <id>`,
`/loop resume <id>`, `/loop cancel <id>` and `/loop run <id>` provide command access.
All these scoped controls refuse jobs bound to another conversation. Stop-condition
evaluation is not yet implemented: use manual pause/cancel or the fixed bounds.

### Follow-up status in the conversation

When a conversation has follow-ups, its header shows the next eligible wake,
attempts used and remaining, and the latest recorded outcome when available.
Paused, expired and attempt-exhausted jobs are labelled explicitly. A running job
may still be waiting for the conversation to become idle; cancelling is shown
until its owner finishes cleanup.

Press **F10** for goal, todo and follow-up details, then **L** to open `/loop`.
There, **P** pauses or resumes future wakes, **X** requests cancellation of the
active attempt, and **H** opens run history. The status view makes no model calls
and refreshes while connected. If the daemon is unavailable it reports that
instead of continuing to promise an old wake time.

### Stop a follow-up when its condition is met

In `/loop`, create or edit a session follow-up and set **Stop condition**, for
example “The deployment health check passes and all replicas are ready.” Keep the
attempt cap and expiry configured. The model receives the condition on every
attempt and must check current evidence. A failed or inconclusive check does not
count as completion.

The running follow-up can call `manage_schedule` with `action: "complete"`, its
`schedule_id`, and `evidence`. The daemon limits this action to that active
attempt and conversation. It persists the report and pauses future wakes; the
current turn can finish its explanation. Duplicate calls in the same attempt do
not duplicate the report. Other sessions and late calls cannot complete it.

The UI labels this **model reported**, not independently certified. `/loop`
shows the condition and recent reports. Completion stays stopped across restart
and after a later provider failure. Manual Run is refused until you explicitly
resume; resuming keeps previous reports and attempts used. Clearing or editing
the condition alone does not rearm a completed follow-up. One-shot completed
follow-ups are retained for review.

### Lifetime token admission for schedules

The `/loop` and schedule editor exposes **Lifetime token threshold**. The native
`manage_schedule` tool and schedule create/update RPC accept `max_total_tokens`
(a positive safe integer, or `null` to remove the limit). Usage is retained when
changing or removing the threshold. It includes measured input, cached input and
output across attempts, retries, child agents and auxiliary model calls.

This is a measured-token admission threshold, not a hard billing cap. Calls already
in flight can overshoot it. Reaching the threshold blocks additional calls and
subsequent attempts. Missing historical usage or an incomplete receipt blocks new
work under the threshold rather than treating unknown consumption as zero. An old
schedule without cumulative accounting needs a new schedule or explicit removal
of this limit. Keep per-attempt model-call limits, attempt limits and expiry where
those bounds are also needed.

Automatic terminal monitors also accept `max_total_tokens` in `monitor.create`
and the native `monitor_terminal` tool, and expose a Lifetime token threshold
field in the monitor form. It requires automatic reactions. The immutable grant
keeps cumulative consumption across reactions and restart; an exhausted or unknown
budget cannot admit another reaction. Other eligible watches in that session can
still run. The inspector shows usage and token admission state. Child and auxiliary
calls share the current reaction's remaining measured budget. As with schedules,
already-running calls may overshoot; this is not a hard billing cap.

Press **E** in the monitor inspector to edit an existing reaction policy's attempt,
timeout and token limits. Saving retains attempts, usage and consumed evidence;
raising limits can release pending reactions. The editor uses a revision guard,
and active, cancelled or expired reactions cannot be edited. A stale edit leaves
the draft visible: go back and refresh before retrying. Source matching and watch
expiry are unchanged by this limits editor; create a new watch to change those.

### Model inventory for agents

The native daemon exposes `list_available_models` as a read-only tool (discoverable
through ToolSearchTool). With no provider filter it lists configured profiles and
provider counts. Set `provider_profile` to discover that profile's models, runtime
offered reasoning levels, context window and output limit when known. `query`
filters IDs; `limit` is 1–50. Use the returned revision with `offset` for subsequent
pages. A changed catalog requires restarting pagination. Discovery can refresh
provider capability caches but never switches the active profile or session model.

AgentTool, TaskCreateTool and each SpawnAgents entry accept explicit `model`,
`provider_profile` and `reasoning_effort`. Provider/reasoning selectors require an
explicit model and cannot be mixed with an intelligence tier. Provider access is
checked when execution starts; a configured profile or returned catalog entry is
not proof of account entitlement. Quota is currently unknown: context capacity
must not be interpreted as subscription tokens remaining. Routing notes and live
provider quota adapters are not yet implemented.

### Model routing preferences

Open `/config`, select an explicit provider profile and optionally a model ID,
then press **F6** to edit routing preferences. **Tab** switches between the
provider-wide note and the selected model note; **F2** saves the selected note.
**Esc** returns to the agent settings draft. Notes save independently of agent
mode settings. Leaving the editor discards unsaved note changes.

Notes are user preferences, not permissions, quotas, or enforced routing rules.
The model reads them through `list_available_models`: provider summaries include
provider notes, and model details include both applicable notes. Each note is
limited to 2000 characters. Saving a blank note clears its guidance. Concurrent
stale saves are rejected; leave and reopen the editor to load the latest version
before reapplying your draft. No credentials are included in model discovery.
Subscription allowance remains unknown unless an authoritative measurement is
available; model context capacity is not a subscription allowance.

Native daemon agents with an explicit provider profile revalidate their model and
reasoning selection before allocation and before execution. The model must be
configured on that profile or present in its discovered/cached catalog; reasoning
must be offered by the runtime's provider/model capability lookup. Profile removal
or changes during validation produce an error. This checks routing compatibility,
not subscription entitlement; the provider can still reject a request. Custom
endpoints without discovery can use their configured model. Other embedded hosts
can supply the `validateProviderSelection` host port.

Model inventory includes `reasoning_source`: `provider_reported`,
`bundled_catalog`, `provider_fallback`, or `unknown` (`unavailable` for unsupported
spawn routes). `reasoning_shape` distinguishes graded effort, on/off toggles, and
inherent model behavior. `default_reasoning_effort` is null when unknown. These
fields describe offered controls, not verified account access. An `off` choice is
Xerxes omitting the reasoning field, not a promise that the model does no reasoning.

Request `list_available_models` with `provider_profile` and `include_usage: true`
to fetch subscription usage for that profile. It currently supports dedicated-key
Kimi/Kimi Code and Z.ai/Zhipu profiles on recognized official HTTPS endpoints.
Results identify `scope: profile_credentials`, observation time, used percentage,
and reported reset times. `remaining_tokens` stays null: usage windows are not
converted into invented token balances. Missing keys, unsupported/custom endpoints,
and unavailable reports return unknown. Shared OAuth logins are not automatically attributed to arbitrary profiles;
the built-in Codex exception is described below. Ordinary inventory requests do not fetch
subscription usage. Requests have a 10-second network deadline and propagate
caller cancellation; provider error bodies are not sent to the model.

Model-facing usage parsing is stricter than the legacy `/usage` display: ambiguous
percentage units or out-of-range values return unknown. Scope labels retain the
provider's identifiers when the time window is not established, and untyped
remaining quantities are omitted rather than interpreted as tokens.

The built-in `codex` profile also supports optional usage lookup through the same
stored Codex OAuth session used for inference. It reports `scope:
shared_codex_login`: usage belongs to that authenticated account/workspace and can
include activity outside Xerxes. Account IDs and tokens are not returned. Custom
Codex profiles, changed endpoints, and explicit-key variants are not automatically
associated with this shared login. Cancelling a lookup stops waiting for credentials
without cancelling a shared refresh needed by other requests.

Embedded hosts can expose inventory through the shared core-tool composition:

```ts
const app = new Xerxes({
  llm,
  model: 'your-model',
  coreTools: {
    modelInventory: async (sessionId, params, signal) => {
      // Use the host's authorized catalog; preserve scope and cancellation.
      return inventoryForSession(sessionId, params, signal)
    },
  },
})
```

`ModelInventoryHost` is exported from the package tool surface. The callback is
optional; without one, the tool is absent. Embedded discovery does not implicitly
open the desktop daemon or read another host's provider credentials. QueryEngine
passes its stable session ID through tool execution, including named sessions.
The native daemon now registers its existing inventory callback through the same
core-tool path. One-shot and ACP composition is described below.

Native one-shot CLI and ACP now supply a profile-store inventory host automatically.
Built-in agent catalogs expose `list_available_models`; custom agents must allow
it in their tool configuration. Standalone discovery uses the native generic
model endpoint or Codex catalog, preserves source metadata, and reports catalog
failures explicitly. It reads existing routing notes and can request optional
profile-bound usage. It does not change the active model/profile.

### Images submitted during a running turn

Pasted images belong to the message submitted with them. When busy input is set
to steer, a message containing images queues as a complete text-and-image message
for the next turn, because live steering currently supports text only. The queue
shows an image count and the UI explains the delay. Press Enter on an empty
composer to interrupt and send the queued message. A later paste stays with the
new draft; it is not consumed by an older queued message.

### Authenticated webhook monitors

Configure a named source in `~/.xerxes/daemon/config.json` (or the active
`XERXES_HOME`), then restart its owning daemon. No listener is created by default.
Secrets come from environment variables available to that daemon, with at least
32 bytes per secret; they are never returned to the model or written into monitor
history. The endpoint names are shared by sessions on that host.

```json
{
  "runtime": {
    "monitor_webhooks": {
      "host": "127.0.0.1",
      "port": 11998,
      "sources": [{ "name": "build", "secret_env": "BUILD_WEBHOOK_SECRET" }]
    }
  }
}
```

Open `/monitors`, create a watch, choose **Configured webhook**, select its name,
and enter a literal match. The model can call `list_monitor_sources` and then
`monitor_webhook`. Notification-only is the default; automatic investigation
requires a direct user request and shares existing reaction budgets. Multiple
sessions can subscribe to the same source; their watches and controls remain
session-owned. Port conflicts fail startup visibly rather than silently disabling
the receiver. When running multiple project daemons, configure a distinct port
for each receiving daemon.

Send `POST /monitors/build` with UTF-8 text (JSON is also treated as text) and:

- `x-xerxes-timestamp`: Unix seconds.
- `x-xerxes-delivery-id`: unique sender ID, 1–128 letters, digits, underscores or hyphens.
- `x-xerxes-signature`: `sha256=` followed by hex HMAC-SHA256, using the configured
  secret over `timestamp + "." + delivery_id + "." + raw_body_bytes`.

Timestamps must be within five minutes of the receiver clock. Bodies are limited
to 64 KiB, body reads to five seconds, and concurrent body reads to 32. Compressed
bodies are unsupported. The receiver records an admitted delivery ID before
notifying watchers. Duplicate deliveries return 200 without redelivery; new ones
return 202. A full replay cache returns 429 instead of evicting a still-valid ID.
The cache is process-local and bounded to 4096 IDs per source. After restart,
old watches are interrupted; creating a new watch does not recover missed events.
This is not an exactly-once external-action guarantee.

A source without a live watcher returns 410. Authentication failures, invalid
bodies and oversized requests are rejected. Keep the default loopback bind or
explicitly configure a TLS reverse proxy for a remote sender; Xerxes does not
create a public tunnel. Native GitHub/Stripe/etc. signature formats are not
accepted by this generic protocol—use a sender that implements the contract.

### Finding activity panels in the TUI

Type `/` to open the command menu. Its first **Activity** group contains
`/agents`, `/goal`, `/loop`, `/monitors`, `/runs`, and `/schedules`. Type a prefix
such as `/mon` or `/sched` to filter the list. Tab fills the highlighted command;
Enter executes a complete command. `/help` and `/commands` also list these entries.
The TUI keeps its activity navigation locally, so an older or temporarily
unavailable daemon catalog cannot hide those panels. Their data still requires
an available daemon with the corresponding RPC support. Restart the TUI after a
rebuild to load the updated menu.

### Generate and use project specialists, skills and commands

Run `/init` in Code mode to have the current model inspect the repository and
create a relevant project setup. You can add a focus, such as `/init focus on
kernel correctness and benchmarking`. This submits a normal model turn: it uses
your selected provider and the current tool permissions. The TUI and daemon use
the same initialization flow.

The model updates `XERXES.md` with verified repository guidance and uses
`create_project_setup` to add missing definitions:

| Location | Purpose | Use |
| --- | --- | --- |
| `.xerxes/agents/<name>.md` | Specialist instructions and tool selection | The built-in Code and Objective agents select specialists by their descriptions; you can also ask for one by name. |
| `.xerxes/skills/<name>/SKILL.md` | Reusable domain workflows | Listed by `/skills`; loaded through SkillTool or invoked as `/<name>`. |
| `.xerxes/commands/<name>.md` | User-invoked prompt workflows | Type `/<name> arguments`; `$ARGUMENTS` expands to the supplied arguments. |

Setup preserves existing agent, skill and command files. It validates the whole
input batch before writing, refuses unknown tool names and reserved names, and
creates files exclusively. Omitted agent tools default to ReadFile, ListDir,
GlobTool and GrepTool. Generated agents inherit the runtime model unless an
explicit model or configured intelligence tier is chosen when spawning them.
Setup does not start specialists, schedule jobs, or run the generated workflows.
After the setup turn, the daemon refreshes discovery automatically. Inspect the
model's created/skipped report and `/skills diagnostics` for rejected files;
finishing a turn is not a guarantee that every requested artifact was created.

Launching Xerxes in a project automatically loads its existing `.xerxes` setup.
Startup does not generate, patch, or rewrite setup files. README and other Markdown
notes under `.xerxes`, including `ops/` and `projects/`, plus `repo-map.yaml` or
`repo-map.yml`, become bounded project context. Agent, skill, and command bodies
load on demand rather than filling the main prompt. Legacy `.agents` context
continues to load as well.

After editing definitions in a running session, `/reload` refreshes discovery. Project agents override
user agents of the same name; user agents override shipped definitions. The
built-in Code/Objective child catalog includes discovered user/project agents,
including overrides of shipped specialists such as `reviewer`. A Markdown agent
used as the main agent can also discover the catalog whenever its tools allow
delegation. Native YAML compositions keep their explicit child allowlists and
pinned definitions. Delegated workers cannot spawn additional workers.
Discovery scans agent subfolders recursively and walks from the current directory
to the repository root; the nearest project definition wins. Markdown documents
without frontmatter are ignored. No `.claude` directories are read.

The model sees agent names and descriptions in its delegation tool and prompt.
It chooses a type by description, or follows the exact type requested by the
user. A delegated Markdown agent receives its own system prompt plus environment
and project context, in a separate conversation. Its full instructions are not
injected into the parent's context.

Markdown frontmatter supports these settings:

| Field | Behavior |
| --- | --- |
| `name`, `description` | Identity and when to delegate. Description is required for discovery. Legacy Xerxes files without `name` use their filename. |
| `tools` | YAML list or comma-separated names. Omitted means inherit available child tools; `[]` grants no tools. Native names and Claude's `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, and `Skill` names resolve to native tools. |
| `disallowedTools` | Removes tools from the inherited or explicit set. |
| `model` | A configured model ID, or `inherit`. An explicit spawn model wins over the file; otherwise the file wins over the parent model. |
| `effort` | `low`, `medium`, `high`, `xhigh`, or `max`, subject to provider support. Inherits session effort when omitted. |
| `maxTurns` | Positive cap on model rounds, including output-length continuations. A stopped agent retains partial output and can be resumed. |
| `skills` | Skill names whose full instructions are preloaded. Missing skills fail before a provider call. Other skills remain available through SkillTool if permitted. |
| `background` | `true` keeps the agent in the background even if a single-agent call or swarm requested a foreground wait. |
| `permissionMode` | `default` inherits; `plan` restricts to planning; `acceptEdits`/`auto` use Xerxes auto permissions; `manual`/`dontAsk` deny anything requiring a child approval prompt; `bypassPermissions` requests accept-all. All remain bounded by parent permissions. |
| `isolation`, `max_depth` | Native worktree isolation and depth controls. |

AgentTool accepts `description` as an alternative to `title`, and `resume` with
an existing agent ID to continue its conversation. Resuming cannot change model,
provider, or isolation. Ownership checks prevent accessing another chat's agents.

These are the supported native behaviors, not blanket compatibility with every
Claude Code extension. Per-agent `hooks`, `memory`, `mcpServers`, `color`,
`initialPrompt`, and Claude model aliases are not implemented here. Unsupported
frontmatter fields fail visibly instead of silently losing their behavior.

```markdown
---
name: kernel-expert
description: Review this repository's kernel layouts and numerical correctness.
tools: [ReadFile, ListDir, GlobTool, GrepTool]
model: inherit
---
Inspect the kernel implementation and its tests. Report correctness issues
with concrete file references and verification steps.
```

Existing `.xerxes` skills and commands appear automatically in `/skills` and slash
completion, including after local edits. Ordinary prompt workflows need no trust
step. Shell preprocessing using `` !`command` `` requires explicit content-hash
trust: inspect the workflow and run `/skills trust <name>` to enable it. New
workflows authored through `create_project_setup` already receive that trust;
editing them removes shell-preprocessing permission until trusted again.
Startup never records trust or executes workflow commands. Size limits and
injection checks still apply; `/skills diagnostics` shows rejected files.
Other workspace skill roots retain their existing trust requirements.
Skill names and command names share a namespace; existing discovery roots keep
their precedence. Project commands cannot replace built-in slash commands.

### Discovering capabilities

The welcome screen includes a small `Explore capabilities · /features` hint.
Type `/features` to open the capabilities hub. Its Skills tab searches admitted
skills by name, description and tags and previews instructions without activating
them. Tools are grouped by purpose with exposure details. Counts reflect retained
session history, not lifetime usage; compaction may reduce skill activation counts.
Registration does not imply that a tool is connected or permitted to execute.
Use Tab to switch categories, arrows to select, PgUp/PgDn to scroll details,
Ctrl+S to sort by name or usage, and Ctrl+R to refresh. Esc restores the conversation.

The Controls tab opens model/provider profiles, reasoning, MCP connections,
schedules, custom agents and remote workspaces. The current model and reasoning
labels below the composer are clickable; `/model` and `/reasoning` remain their
keyboard entry points. Non-TUI `/features` callers still receive the text guide.

In `/schedules`, press B to browse examples such as a morning briefing, weekly
review, test report or documentation check. Enter copies a template into an
editable, initially paused draft. Review the prompt, timing, timezone and delivery
before saving. Common schedules show readable timing alongside cron and the
daemon's next-run preview; complex expressions retain their exact cron text.

### Remote workspaces and extension controls

`/machine` opens the saved-workspace picker. Add a workspace with:

```text
/machine add compute my-ssh-alias "/home/me/My Project"
/machine connect compute
/machine remove compute
```

Names use letters, digits, underscores or hyphens. The target is an SSH config
alias or `user@hostname`; configure identity files, ports and jump hosts in
`~/.ssh/config`. The remote project path must be absolute. Saved workspaces live
in `~/.xerxes/machines.json` (or `$XERXES_HOME/machines.json`).

Connecting prepares the remote daemon and forwards its private Unix socket over
SSH. A local TUI renders the remote workspace, so typing, scrolling and menus
stay local. The remote machine executes tools and holds its provider settings,
files and sessions. Exit this view to restore the previous local picker/chat.
Local work stays with its local daemon; files and conversations are not migrated.
SSH uses your configured identities and jump hosts with BatchMode authentication;
authenticate and verify the host with `ssh <alias>` first if necessary. Installation
and updates remain managed under `~/.xerxes/remote-runtime` on the remote host.

`/custom-agents` (also `/agents edit`) opens a project specialist editor. Press
**N** to create a Markdown definition, **Enter** to edit an existing definition,
**Ctrl+S** to validate and save, and **Esc** to discard the draft. Definitions
live in `.xerxes/agents/<name>.md`. The editor supports the same Claude-style
frontmatter as automatic discovery. Invalid files and concurrent edits are
reported without overwriting the saved definition or clearing the draft.
Saved definitions are refreshed for subsequent delegation.

`/skills search <query>` searches the discovered local and bundled catalog;
`/skills browse` lists it. `/skills install /absolute/path/to/skill` installs a
local bundle, or pass its `SKILL.md` path. Assets and references are copied, while
existing names, unsupported filesystem entries, oversized bundles and failed
skill scans are rejected. Use `/skills inspect <name>` to inspect the result and
`/skill <name>` to invoke it. Shell preprocessing still requires `/skills trust`.

`/plugins install /absolute/path/plugin.ts` registers and enables a local native
tool module exporting `register(registry)`. Its path and enabled state persist;
source files stay at their original location. Use `/plugins inspect <name>`,
`/plugins disable <name>` and `/plugins enable <name>` to manage it. Enabled tools
are exposed with the `plugin_` prefix on subsequent turns. Disabling prevents
new invocations, including from an existing tool registry; it does not undo
already-running plugin work. Installation imports executable code from the
selected module. Install only modules you intend to run. Provider, channel and
hook plugins require an embedding host and are rejected by this installer.
These commands install local bundles/modules; they do not download marketplace
packages.

## Automatic context recovery

Automatic compaction checks the estimated prompt size before a new turn and
between model/tool rounds. It uses the routed model's context window, reserves
output capacity, and honors `auto_compact_threshold` (default `0.8`; `0` disables
automatic checks). Large transcripts are summarized in bounded chronological
segments, then combined, instead of sending the entire oversized history to a
single request. The original transcript is archived before replacement.

Compaction collects streamed responses and uses low reasoning effort where the
model's reasoning controls support it. The prompt requests only the final summary,
so generated scratchpad text does not consume the summary's output budget.
Each provider call has a 180-second deadline. A timeout retries only its failed
segment with smaller inputs, down to a 4096-token request budget, retaining
earlier successful summaries. A truncated completion is rejected; failure leaves
the original conversation in place rather than saving a partial summary.

If automatic compaction cannot make room, the next model request is paused.
After three consecutive failures, use `/compact` to retry explicitly; the
failure reason is retained in session metadata. Failed compaction does not
discard the conversation. Manual recovery may take several provider requests;
the TUI allows up to 30 minutes for `/compact`, while status and follow-up lists
remain responsive.

### Generate a custom agent in the TUI

Open `/custom-agents` (or `/agents edit`), select a specialist to read its full delegation
summary, and press **Enter** to edit its Markdown. The browser shows names in a
separate list and stacks the detail panel on narrow terminals.

Press **G** or click **G Generate**, describe the specialist, then press **Ctrl+G**.
For example: “A JAX reviewer who checks array shapes and sharding and proposes
focused regression tests.” Xerxes uses the current session's provider and model
to generate a draft. Review or edit it, then press **Ctrl+S** to save it under
`.xerxes/agents`. Existing files are never overwritten by a new draft. **N** still
creates a blank agent; **Esc** discards an unsaved draft. Provider failures retain
your description so you can retry.

### Reconnecting remote workspaces

In `/machine`, press **R** (or click **R Retry connection** after an error) to
retry the selected workspace. Transient SSH drops retry automatically up to three
times, after 2, 5 and 10 seconds. Progress shows setup, tunnel opening and retry
status; **Esc** cancels. Authentication and host-key errors require fixing the
connection and are not retried automatically.

When the remote TUI recorded its active session, retries from this picker resume
that session. Reconnecting does not resend your last prompt or restart its tools.
A remote host reboot can still interrupt remote processes; session recovery does
not guarantee that those processes survived.

Explicit subagent models take precedence over `intelligence` tier hints. When
passing `model`, `provider_profile`, and `reasoning_effort` from model discovery,
a redundant tier does not override those choices. Provider/reasoning selectors
still require an explicit model. For worktree delegation, `working-tree` plus
`worktree_ref: HEAD` means capture the current working tree; a conflicting named
branch remains invalid.
Empty-string or null optional `worktree_ref` values from model tool calls are
treated as omitted. Nonempty refs still undergo Git-ref validation.
