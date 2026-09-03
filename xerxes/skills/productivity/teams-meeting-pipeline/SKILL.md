---
name: teams-meeting-pipeline
description: Teams meeting summaries, job replay, Graph subscriptions.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [teams, microsoft-graph, meetings, productivity, operations]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/productivity/teams-meeting-pipeline/SKILL.md
---

# Teams Meeting Pipeline

Use this skill whenever the user asks about Microsoft Teams meeting summaries,
transcripts, recordings, action items, Graph subscriptions, or any operational
question about the Teams meeting pipeline. Works in any language — the
triggers below are examples, not an exhaustive list.

Everything operator-facing is a pipeline CLI subcommand run via the
shell/terminal tool. There are no dedicated model tools for this pipeline —
the CLI is the surface. The examples below use a `teams-pipeline` command
shape; if your Xerxes deployment exposes this pipeline behind a different
binary or MCP server, substitute that entry point and keep the same
subcommand semantics.

## When to Use

The user is asking to:
- summarize a Teams meeting / extract action items / pull meeting notes
- check pipeline status, inspect a stored meeting job, or see recent meetings
- replay / re-run a stored job that failed or needs a fresh summary
- validate Microsoft Graph setup after changing env or config
- troubleshoot "meeting summary never arrived" or "no new meetings are ingesting"
- manage Graph webhook subscriptions (create, renew, delete, inspect)
- set up automated subscription renewal (see the critical pitfall below)

Multilingual trigger examples (not exhaustive):
- English: "summarize the Teams meeting", "pipeline status", "replay job X"
- Turkish: "Teams meeting özetle", "action item çıkar", "toplantı notu", "pipeline durumu", "replay job"

## Prerequisites

Before using the pipeline, verify these are set in the environment:

```bash
MSGRAPH_TENANT_ID=...
MSGRAPH_CLIENT_ID=...
MSGRAPH_CLIENT_SECRET=...
```

If any are missing, direct the user to Microsoft's Azure app registration
documentation — they need an Azure AD app registration with admin-consented
Graph application permissions before the pipeline will work. Secrets belong in
the user's environment, never in source files, config committed to the repo,
or command output that gets logged.

## Command reference

### Status and inspection (start here)

```bash
teams-pipeline validate              # config snapshot — run first after any change
teams-pipeline token-health          # Graph token status
teams-pipeline token-health --force-refresh   # force a fresh token acquisition
teams-pipeline list                  # recent meeting jobs
teams-pipeline list --status failed  # only failed jobs
teams-pipeline show <job-id>         # full detail of one job
teams-pipeline subscriptions         # current Graph webhook subscriptions
```

### Re-running / debugging

```bash
teams-pipeline run <job-id>          # replay a stored job (re-summarize, re-deliver)
teams-pipeline fetch --meeting-id <id>   # dry-run: resolve meeting + transcript without persisting
teams-pipeline fetch --join-web-url "<url>"   # dry-run by join URL
teams-pipeline fetch --join-web-url "<url>" --organizer-user-id <id>   # organizer-scoped lookup (required for /meet/ short URLs)
```

### Subscription management

```bash
teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://<your-public-host>/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

teams-pipeline renew-subscription <sub-id> --expiration <iso-8601>
teams-pipeline delete-subscription <sub-id>
teams-pipeline maintain-subscriptions            # renew near-expiry ones
teams-pipeline maintain-subscriptions --dry-run  # show what would be renewed
```

## Decision tree for common asks

- "Why didn't I get a summary for today's meeting?" → start with `list --status failed`, then `show <job-id>` on the relevant row. If the job doesn't exist at all, check `subscriptions` — the webhook may have expired (see pitfall below).
- "Is setup working?" → `validate`, then `token-health`, then `subscriptions`. If all three pass, request a test meeting and check `list` for a fresh row.
- "Re-run summary for meeting X" → `list` to find the job ID, `run <job-id>` to replay. If it fails again, `show <job-id>` to inspect the error and `fetch --meeting-id` to dry-run the artifact resolution.
- "Add meeting X to the pipeline" → usually you don't — the pipeline is subscription-driven, not per-meeting. To summarize a specific past meeting, use `fetch` to pull the transcript, then `run` after a job is created.

## Critical pitfall: Graph subscriptions expire in 72 hours

Microsoft Graph caps webhook subscriptions at 72 hours and **will not
auto-renew them**. If `maintain-subscriptions` is not scheduled, meeting
notifications silently stop arriving 3 days after any manual subscription
creation.

When the user reports "the pipeline worked yesterday but nothing is arriving today":
1. Run `teams-pipeline subscriptions` — if it's empty or all entries show `expirationDateTime` in the past, that's the cause.
2. Recreate with `subscribe` as shown above.
3. **Set up automated renewal immediately** as a scheduled trigger (e.g. an `every 12h` scheduled-trigger job that runs `teams-pipeline maintain-subscriptions`), a systemd timer, or a plain crontab. 12-hour interval is safe (6x headroom against the 72h limit).

## Other pitfalls

- **Transcript not available yet.** Teams takes some time after a meeting ends to generate the transcript artifact. `fetch --meeting-id` on a just-ended meeting may return empty. Wait 2–5 minutes and retry, or let the Graph webhook drive ingestion naturally.
- **Delivery mode mismatch.** If summaries are produced (`list` shows success) but nothing lands in Teams, check the Teams delivery-mode config and its matching target (`incoming_webhook_url` OR `chat_id` OR `team_id`+`channel_id`) in the deployment's config or `TEAMS_*` env vars.
- **Graph app permissions.** A token acquires cleanly (`token-health` passes) but Graph API calls return 401/403 when permissions were added but admin consent wasn't re-granted. Have the user revisit the app registration in the Azure portal and click "Grant admin consent" again.

## Related Xerxes skills

- **meeting-action-items** — for turning an already-delivered meeting transcript or notes file into tracked action items.
- **weekly-review-planning** — for folding resolved action items into a weekly planning pass.

## Verification

- [ ] `validate` passes after any config or env change.
- [ ] `token-health` returns a healthy token before diagnosing anything else.
- [ ] `subscriptions` shows an unexpired entry for the transcript resource.
- [ ] Automated renewal is scheduled before leaving a manual `subscribe` in place.
- [ ] A replayed job (`run <job-id>`) produced a fresh delivery visible in `list`.

---

Adapted from the `teams-meeting-pipeline` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent + Teknium.
