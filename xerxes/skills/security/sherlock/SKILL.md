---
name: sherlock
description: Find accounts for a username across 400+ social platforms with the Sherlock CLI.
version: 1.0.0
author: unmodeled-tyler (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [osint, security, username, reconnaissance]
source: https://github.com/NousResearch/hermes-agent/tree/main/optional-skills/security/sherlock
---

# Sherlock OSINT Username Search

Find social media accounts by username across 400+ social networks
using the [Sherlock Project](https://github.com/sherlock-project/sherlock)
CLI.

## When to Use

- "Find accounts associated with this username"
- "Check username availability across platforms"
- OSINT/reconnaissance research on a username

## Requirements

```bash
pipx install sherlock-project   # or: pip install sherlock-project
sherlock --version              # verify first, always
```

Alternative: `docker run -it --rm sherlock/sherlock <username>`.

## Procedure

1. **Verify sherlock is installed** (`sherlock --version`). If missing,
   offer one install method and proceed — do not cycle through
   alternatives.
2. **Extract the username** directly from the request when clearly
   stated ("find accounts for nasa" → `nasa`). Preserve case, numbers,
   underscores. Ask only when genuinely ambiguous ("search for alice or
   bob", "do an OSINT search" with no name).
3. **Run the default search** — do not add flags unless asked:

```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

   Optional flags only on explicit request: `--nsfw`, `--tor`
   (requires a running Tor daemon), `--site` to limit scope.
4. **Run via the shell tool** with a generous timeout — the scan
   typically takes 30–120 seconds.
5. **Parse and present results**:

```
[+] Instagram: https://instagram.com/username
[+] GitHub: https://github.com/username
```

   Summarize ("Found X accounts for username 'Y'"), group by platform
   type if helpful, present links, and mention the results file
   (`<username>.txt` by default).

## Pitfalls

- **No results** is often correct — the username may not exist on the
  checked platforms. Try spelling variants or the `?` wildcard before
  concluding.
- **Slow sites** raise timeouts: `--timeout 120`.
- **False positives**: some sites always report "found" due to their
  response structure — cross-check unexpected hits manually.
- **Rate limits**: for bulk searches, add delays or use `--local` with
  cached site data.

## Ethical Use

For legitimate OSINT and research only. Search usernames you own or
have permission to investigate; respect platform terms of service; do
not use for harassment or stalking; consider privacy implications
before sharing results.

## Verification

- Output lists found sites with URLs (only `[+]` lines with
  `--print-found`)
- The results file exists when file output is used
- A spot-checked link actually resolves to an existing profile

---

Adapted from the `sherlock` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright unmodeled-tyler and Hermes Agent contributors.
