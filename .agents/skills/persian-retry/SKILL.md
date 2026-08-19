---
name: persian-retry
description: Enable a session-persistent auto-retry mode that keeps retrying and continuing a task when a connection or provider error interrupts the agent, until the task actually completes. Configurable via arguments.
version: 2.0.0
tags: [retry, resilience, connection, workflow, session, xerxes]
---

# When to use

Use this skill when the user invokes `/skill:persian-retry` (or asks for
"persian retry"). Once invoked, the mode it describes stays active for the
**rest of the session** until the user explicitly turns it off — treat it as a
standing session policy, not a one-turn instruction.

# Activation and configuration

When invoked, parse optional `key=value` arguments from the user's message to
configure the mode for this session:

| Key | Default | Meaning |
|-----|---------|---------|
| `max_attempts` | `0` (unlimited) | Max retries per failed step before giving up. `0` = retry forever. |
| `backoff` | `2,5,10,30` | Comma-separated seconds to wait between attempts; last value repeats. |
| `notify` | `summary` | `summary` = one short line per retry; `silent` = only report final outcome. |

Examples of how the user may invoke it:

- `/skill:persian-retry` — enable with defaults (retry forever, backoff 2/5/10/30s).
- `/skill:persian-retry max_attempts=10` — give up after 10 attempts on one step.
- `/skill:persian-retry backoff=1,3,9 notify=silent`

**Session persistence rules:**

1. After activation, restate the active configuration in one line
   (e.g. `persian-retry: ON — unlimited attempts, backoff 2/5/10/30s`) and
   apply the behavior below to every subsequent turn of this session.
2. The mode remains ON until the user says `persian-retry off`,
   `stop retrying`, or similar. Re-invoking the skill with new `key=value`
   arguments updates the configuration in place and keeps it ON.
3. If context was compacted and you see this skill was activated earlier in
   the session, keep honoring it — do not silently drop the mode.

# Behavior while active

1. Treat the current task as unfinished until its goal is verifiably met
   (files written, tests passing, output produced). A connection drop, TLS or
   certificate error, timeout, 5xx, or truncated stream is never a reason to
   declare success or to stop silently.
2. On a recoverable provider/network failure, wait per the configured backoff
   schedule, then retry the failed step. Keep retrying until the task
   completes, `max_attempts` is reached, or the user cancels.
3. If the turn was cut off mid-work, resume exactly like a user-sent
   "continue": reconstruct state from the transcript and tool results, then
   proceed from the last confirmed step instead of restarting.
4. Never fabricate the missing work. Re-run any tool call whose result was
   lost before relying on it, and re-verify partial side effects (files,
   commits, requests) before continuing.
5. Report per the `notify` setting: `summary` = one short line per retry
   (attempt number + reason); `silent` = only report the final outcome.
6. Stop and report only when the task is done, `max_attempts` is exhausted,
   the user cancels, or the failure is clearly non-recoverable (invalid
   credentials, permission denial, validation error). In those cases state
   what failed, what was already completed, and the exact next step to resume
   manually.
