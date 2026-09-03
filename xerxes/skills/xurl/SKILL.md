---
name: xurl
description: X/Twitter via the official xurl CLI — raw post search, posting, DMs, media, and any v2 endpoint as JSON.
version: 1.1.3
author: xdevplatform + openclaw + Hermes Agent (adapted for Xerxes)
platforms: [linux, macos]
tags: [twitter, x, social-media, xurl, official-api]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/social-media/xurl/SKILL.md
---

# xurl — X (Twitter) API via the Official CLI

`xurl` is the X developer platform's official CLI for the X API. It supports
shortcut commands for common actions AND raw curl-style access to any v2
endpoint. All commands return JSON to stdout.

Use this skill for: posting, replying, quoting, deleting posts; searching for
raw posts (actual post JSON with IDs you can engage with) and reading
timelines/mentions; likes, reposts, bookmarks; follow/block/mute; direct
messages; media uploads (images and video); raw access to any X API v2
endpoint; multi-app / multi-account workflows.

## Secret Safety (MANDATORY)

- Never read, print, parse, summarize, upload, or send `~/.xurl` into model
  context.
- Never ask the user to paste credentials or tokens into chat.
- The user fills `~/.xurl` with secrets manually on their own machine.
- Never recommend or run auth commands with inline secrets in an agent
  session.
- Never use `--verbose` / `-v` in agent sessions — it can expose auth headers.
- To verify credentials exist, use only `xurl auth status`.

Forbidden flags in agent commands (they accept inline secrets):
`--bearer-token`, `--consumer-key`, `--consumer-secret`, `--access-token`,
`--token-secret`, `--client-id`, `--client-secret`.

App registration, rotation, and `xurl auth oauth2` are user-run, outside the
agent session. OAuth 2.0 tokens persist to `~/.xurl` (YAML) and auto-refresh.

## Installation

```bash
# Shell script (installs to ~/.local/bin, no sudo; Linux + macOS)
curl -fsSL https://raw.githubusercontent.com/xdevplatform/xurl/main/install.sh | bash
# Homebrew (macOS)
brew install --cask xdevplatform/tap/xurl
```

Verify with `xurl --help` and `xurl auth status`. If auth status shows no apps
or tokens, direct the user to the setup below.

## One-Time User Setup (user runs these outside the agent)

1. Create or open an app at https://developer.x.com/en/portal/dashboard
2. Set the redirect URI to `http://localhost:8080/callback`
3. Register the app locally:
   `xurl auth apps add my-app --client-id ... --client-secret ...`
4. Authenticate (opens a browser for OAuth 2.0 PKCE):
   `xurl auth oauth2 --app my-app` — or pass the handle explicitly
   (`xurl auth oauth2 --app my-app YOUR_USERNAME`) if X returns
   `UsernameNotFound` or 403 on the post-OAuth `/2/users/me` lookup.
5. `xurl auth default my-app` then verify with `xurl auth status` and
   `xurl whoami`.

Common pitfall: omitting `--app my-app` from `auth oauth2` saves the token to
the empty built-in `default` profile; later commands fail with auth errors.
Re-run `xurl auth oauth2 --app my-app` and `xurl auth default my-app`.

## Quick Reference

| Action | Command |
| --- | --- |
| Post / Reply / Quote | `xurl post "text"` / `xurl reply ID "text"` / `xurl quote ID "text"` |
| Delete a post | `xurl delete POST_ID` |
| Read a post | `xurl read POST_ID` |
| Search posts | `xurl search "QUERY" -n 10` |
| Who am I / user lookup | `xurl whoami` / `xurl user @handle` |
| Timeline / Mentions | `xurl timeline -n 20` / `xurl mentions -n 10` |
| Like / Repost / Bookmark | `xurl like ID` / `xurl repost ID` / `xurl bookmark ID` (each has an `un` variant) |
| Follow / Block / Mute | `xurl follow @h` / `xurl block @h` / `xurl mute @h` (each has an `un` variant) |
| Send DM / List DMs | `xurl dm @handle "msg"` / `xurl dms -n 10` |
| Upload media | `xurl media upload path/to/file.mp4` |
| Media status (videos) | `xurl media status MEDIA_ID` or `xurl media status --wait MEDIA_ID` |
| Auth status / default app | `xurl auth status` / `xurl auth default APP_NAME [USER]` |
| Raw v2 endpoint | `xurl /2/users/me` or `xurl -X POST /2/tweets -d '{"text":"..."}'` |

Notes:
- `POST_ID` accepts full URLs (e.g. `https://x.com/user/status/123`) — xurl
  extracts the ID. Usernames work with or without `@`.
- For X Articles, use raw API mode with the `article` tweet field and read
  `data.article.plain_text` from the JSON response; do not put `read` before a
  `/2/tweets/...` endpoint.
- Every response is JSON (`{"data": {...}}`; errors as `{"errors": [...]}`),
  with a non-zero exit code on failure — parse output directly.

## Agent Workflow

1. Verify prerequisites: `xurl --help` and `xurl auth status`.
2. Before `xurl search`, check intent: reach for it when the task needs actual
   post objects, authenticated account context, or leads into an X write
   action — not when the user just wants a topic summary.
3. Check the default app (marked `▸`) has credentials. If it shows
   `oauth2: (none)` but another app has a valid user, tell the user to run
   `xurl auth default <that-app>` — the most common setup mistake.
4. If auth is missing entirely, stop and direct the user to the setup section;
   never register apps or pass secrets yourself.
5. Start with a cheap read (`xurl whoami`, `xurl user @handle`,
   `xurl search ... -n 3`) to confirm reachability.
6. Confirm the target post/user and the user's intent before any write action.
7. Only `xurl` command output proves a state-changing action happened. Never
   report a write as done based on search results, summaries, or prior
   context.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Auth errors after a successful OAuth flow | Token landed on the `default` app; re-run `xurl auth oauth2 --app my-app` then `xurl auth default my-app` |
| `unauthorized_client` during OAuth | Change app type to "Web app, automated app or bot" in the X dashboard |
| 401 on every request | `xurl auth status` — confirm `▸` points to an app with oauth2 tokens |
| `client-forbidden` / `client-not-enrolled` | Dashboard → Apps → Manage → "Pay-per-use" → Production |
| `CreditsDepleted` | X API balance is $0; the user must add credits in the Developer Console |
| `media processing failed` on images | Add `--category tweet_image --media-type image/png` |

Rate limits: writes are tighter than reads; a 429 means wait and retry. A 403
on a specific action usually means a missing scope — the user re-runs
`xurl auth oauth2`. Never paste `~/.xurl` contents into the conversation.

---

Adapted from the `xurl` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright xdevplatform + openclaw + Hermes Agent.
