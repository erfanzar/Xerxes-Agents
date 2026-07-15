---
name: google-workspace
description: Use Gmail, Calendar, Drive, Contacts, Sheets, and Docs through the native Bun Google Workspace client.
version: 0.3.0
tags: [google, gmail, calendar, drive, sheets, docs, contacts, oauth]
source: bundled
subcommands: [setup, gmail, calendar, drive, contacts, sheets, docs]
resources: [references/gmail-search-syntax.md]
---

# Google Workspace

All Google Workspace commands run through native Bun/TypeScript code. The runtime does not invoke Python, install a package, discover a token file, inspect environment variables, or forward requests to `gws`.

## Explicit adapter

Operational commands require an adapter selected by the caller:

```bash
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup status
```

The selected local module must export either `default` or `googleWorkspaceCliAdapter`. It must provide either a ready `GoogleWorkspaceClient`, or all of these caller-owned ports:

- `fetchImplementation` — the host's transport implementation.
- `oauthConfig` — OAuth client ID, redirect URI, scopes, and optional endpoint overrides.
- `storage` — an encrypted/keychain/in-memory implementation of `GoogleWorkspaceAuthorizationStorage`.
- `browser` — optional; required only for `setup begin --open`.

An adapter is executable code. Select only a module you control. Credentials and authorization data stay in the caller-provided storage boundary; the CLI never prints access tokens, refresh tokens, OAuth state, or the PKCE verifier.

## OAuth setup

Create and configure a Google OAuth desktop or web client in Google Cloud, enable only the APIs you need, then implement the adapter above. The native setup commands do not accept a client-secret file path or create a default credential file.

```bash
xerxes skill google-workspace setup guidance
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup status
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup begin --open
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup complete "http://127.0.0.1:4567/callback?code=...&state=..."
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup revoke
```

`setup begin` returns only an authorization URL. `setup complete` returns only authorization status.

## Commands

```bash
# Gmail
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail search "is:unread" --max 10
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail get MESSAGE_ID
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail send --to user@example.com --subject "Hello" --body "Message text"
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail reply MESSAGE_ID --body "Thanks"
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail labels
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts gmail modify MESSAGE_ID --add-labels STARRED --remove-labels UNREAD

# Calendar
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts calendar delete EVENT_ID

# Drive, Contacts, Sheets, and Docs
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts drive search "quarterly report" --max 10
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts contacts list --max 20
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts sheets get SHEET_ID "Sheet1!A1:D10"
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice",95]]'
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
xerxes skill google-workspace --adapter ./google-workspace.adapter.ts docs get DOC_ID
```

All command results are JSON. For Gmail search syntax, see [the reference](references/gmail-search-syntax.md).

## Safety rules

Never send email or create/delete a calendar event without the user's explicit confirmation. Show the final message or event details before executing a mutation. Use the smallest practical OAuth scope set and revoke authorization when it is no longer needed.
