---
name: airtable
description: Airtable REST API via curl. Records CRUD, filters, upserts.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [airtable, database, api, productivity]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/productivity/airtable/SKILL.md
---

# Airtable — Bases, Tables & Records

Work with Airtable's REST API directly via `curl` using the shell/terminal tool.
No MCP server, no OAuth flow, no Python SDK — just `curl` and a personal access
token.

## When to Use

- Creating, updating, deleting, or upserting records in an Airtable base
- Filtering or searching records by formula, view, or sort
- Inspecting base and table schema before mutating anything
- Batch-syncing external data into Airtable idempotently

## Prerequisites

1. Create a **Personal Access Token (PAT)** at https://airtable.com/create/tokens (tokens start with `pat...`).
2. Grant minimum scopes:
   - `data.records:read` — read rows
   - `data.records:write` — create / update / delete rows
   - `schema.bases:read` — list bases and tables
3. **Important:** in the same token UI, add each base you want to access to the token's **Access** list. PATs are scoped per-base — a valid token on the wrong base returns `403`.
4. Store the token in the environment (e.g. `AIRTABLE_API_KEY` exported in the shell session before the turn, or supplied by the user's env file). Never put it in source files, fixtures, or command output that gets logged.

> Note: legacy `key...` API keys were deprecated Feb 2024. Only PATs and OAuth tokens work now.

## API Basics

- **Endpoint:** `https://api.airtable.com/v0`
- **Auth header:** `Authorization: Bearer $AIRTABLE_API_KEY`
- **All requests** use JSON (`Content-Type: application/json` for any POST/PATCH/PUT body).
- **Object IDs:** bases `app...`, tables `tbl...`, records `rec...`, fields `fld...`. IDs never change; names can. Prefer IDs in automations.
- **Rate limit:** 5 requests/sec/base. `429` → back off and honor `Retry-After`.

Base curl pattern (keep `-s` set so tool output stays clean):

```bash
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?maxRecords=5" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

## Field Types (request body shapes)

| Field type | Write shape |
|---|---|
| Single line text | `"Name": "hello"` |
| Long text | `"Notes": "multi\nline"` |
| Number | `"Score": 42` |
| Checkbox | `"Done": true` |
| Single select | `"Status": "Todo"` (name must already exist unless `typecast: true`) |
| Multi-select | `"Tags": ["urgent", "bug"]` |
| Date | `"Due": "2026-04-01"` |
| DateTime (UTC) | `"At": "2026-04-01T14:30:00.000Z"` |
| URL / Email / Phone | `"Link": "https://…"` |
| Attachment | `"Files": [{"url": "https://…"}]` (Airtable fetches + rehosts) |
| Linked record | `"Owner": ["recXXXXXXXXXXXXXX"]` (array of record IDs) |
| User | `"AssignedTo": {"id": "usrXXXXXXXXXXXXXX"}` |

Pass `"typecast": true` at the top level of a create/update body to let Airtable auto-coerce values (e.g. create a new select option on the fly, convert `"42"` → `42`).

## Common Queries

### List bases the token can see
```bash
curl -s "https://api.airtable.com/v0/meta/bases" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

### List tables + schema for a base
```bash
curl -s "https://api.airtable.com/v0/meta/bases/$BASE_ID/tables" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```
Use this BEFORE mutating — confirms exact field names and IDs, surfaces `options.choices` for select fields, and shows primary-field names.

### Filter records (filterByFormula)
Airtable formulas must be URL-encoded. Let Python stdlib do it — never hand-encode:
```bash
FORMULA="{Status}='Todo'"
ENC=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$FORMULA")
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?filterByFormula=$ENC&maxRecords=20" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" | python3 -m json.tool
```

Useful formula patterns:
- Exact match: `{Email}='user@example.com'`
- Contains: `FIND('bug', LOWER({Title}))`
- Multiple conditions: `AND({Status}='Todo', {Priority}='High')`
- Not empty: `NOT({Assignee}='')`
- Date comparison: `IS_AFTER({Due}, TODAY())`

### Sort + select fields / named view
Query-string square brackets MUST be URL-encoded (`%5B` / `%5D`):
```bash
# sort by Priority asc, return only Name and Status
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?sort%5B0%5D%5Bfield%5D=Priority&sort%5B0%5D%5Bdirection%5D=asc&fields%5B%5D=Name&fields%5B%5D=Status" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY"

# a named view applies its saved filter + sort server-side
curl -s "https://api.airtable.com/v0/$BASE_ID/$TABLE?view=Grid%20view&maxRecords=50" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY"
```

## Common Mutations

### Create a record (batch capped at 10 records/request)
```bash
curl -s -X POST "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"fields":{"Name":"Task A","Status":"Todo"}},{"fields":{"Name":"Task B","Status":"In progress"}}],"typecast":true}' | python3 -m json.tool
```
For larger inserts, loop in batches of 10 with a short sleep to respect 5 req/sec/base.

### Update a record (PATCH — merges, preserves unchanged fields)
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fields":{"Status":"Done"}}' | python3 -m json.tool
```

### Upsert by a merge field (no ID needed)
```bash
curl -s -X PATCH "https://api.airtable.com/v0/$BASE_ID/$TABLE" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"performUpsert":{"fieldsToMergeOn":["Email"]},"records":[{"fields":{"Email":"user@example.com","Status":"Active"}}]}' | python3 -m json.tool
```
`performUpsert` creates records whose merge-field values are new, patches records whose merge-field values already exist. Great for idempotent syncs.

### Delete records (1 per call, or up to 10 via query params)
```bash
# single
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE/$RECORD_ID" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY"

# up to ten
curl -s -X DELETE "https://api.airtable.com/v0/$BASE_ID/$TABLE?records%5B%5D=rec1&records%5B%5D=rec2" \
  -H "Authorization: Bearer $AIRTABLE_API_KEY"
```

## Pagination

List endpoints return at most **100 records per page**. If the response includes `"offset": "..."`, pass it back on the next call; loop until the field is absent. A compact Python loop handles this cleanly:

```bash
python3 - <<'PY'
import json, os, urllib.request
base, table, key = os.environ["BASE_ID"], os.environ["TABLE"], os.environ["AIRTABLE_API_KEY"]
offset = ""
while True:
    url = f"https://api.airtable.com/v0/{base}/{table}?pageSize=100" + (f"&offset={offset}" if offset else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req))
    for r in d["records"]:
        print(r["id"], r["fields"].get("Name", ""))
    offset = d.get("offset", "")
    if not offset:
        break
PY
```

## Typical Workflow

1. **Confirm auth.** `curl -s -o /dev/null -w "%{http_code}\n" https://api.airtable.com/v0/meta/bases -H "Authorization: Bearer $AIRTABLE_API_KEY"` — expect `200`.
2. **Find the base.** List bases, or ask the user for the `app...` ID directly if the token lacks `schema.bases:read`.
3. **Inspect the schema.** `GET /v0/meta/bases/$BASE_ID/tables` — note the exact field names and primary-field name before mutating anything.
4. **Read before you write.** For "update X where Y", `filterByFormula` first to resolve the `rec...` ID, then `PATCH`. Never guess record IDs.
5. **Batch writes.** Combine related creates into one 10-record POST to stay under the 5 req/sec budget.
6. **Destructive ops.** Deletions cannot be undone via API. If the user says "delete all Xs", echo back the filter + record count and confirm before firing.

## Pitfalls

- **`filterByFormula` MUST be URL-encoded.** Field names with spaces or non-ASCII also need encoding (`{My Field}` → `%7BMy%20Field%7D`). Use Python stdlib, never hand-escaping.
- **Empty fields are omitted from responses.** A missing `"Assignee"` key means this record's value is empty, not that the field doesn't exist — check the schema first.
- **PATCH vs PUT.** `PATCH` merges supplied fields; `PUT` replaces the record entirely and clears any field you didn't include. Default to `PATCH`.
- **Single-select options must exist** unless `"typecast": true` is passed; otherwise you get `INVALID_MULTIPLE_CHOICE_OPTIONS`.
- **Per-base token scoping.** A `403` on one base while another works means the token's Access list doesn't include that base — not a scope or auth issue.
- **Rate limits are per base, not per token.** Monitor the `Retry-After` header on `429`.

## Important Notes for Xerxes

- **Always use the shell/terminal tool with `curl`.** Do NOT use the web fetch tool (it cannot send custom auth headers) for authenticated API calls.
- **`AIRTABLE_API_KEY` must be present in the shell environment** before the first `curl` call — confirm with a status-code probe rather than printing the token.
- **Pretty-print with `python3 -m json.tool`** (stdlib, always present) rather than `jq` (optional). Only reach for `jq` when you need filtering/projection.
- **Pagination is per-page, not global.** The 100-record cap is a hard limit; loop with `offset` until the field is absent.
- **Read the `errors` array** on non-2xx responses — Airtable returns structured error codes like `AUTHENTICATION_REQUIRED`, `INVALID_PERMISSIONS`, `MODEL_ID_NOT_FOUND` that tell you exactly what's wrong.

## Verification

- [ ] Status probe against `/v0/meta/bases` returns `200` before any data call.
- [ ] Schema was fetched and field names confirmed before the first mutation.
- [ ] Record IDs were resolved via `filterByFormula`, never guessed.
- [ ] Every write was read back and matched the intended field values.
- [ ] Deletions were confirmed with the user after showing filter + count.

---

Adapted from the `airtable` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright community.
