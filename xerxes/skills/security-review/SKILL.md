---
name: security-review
description: Security-focused review of pending changes — injection, auth, secrets, unsafe deserialization, dependency risk. Use when the user asks for a security scan or before shipping code touching auth, network, files, or shell execution.
version: 1.0.0
author: Xerxes Agent
license: MIT
metadata:
  xerxes:
    tags: [Security, Review, OWASP, Git]
    related_skills: [code-review, bug-bounty-hunter]
---

# Security Review

Adversarial review of the pending changes. Assume the reviewed code will handle
hostile input. Findings are exploitable-or-not: each must name the attack
scenario, the vulnerable sink (file:line), and the fix. Do not report style.

## Scope

Default: `git diff HEAD` (or the branch/range/PR the user names). Empty scope →
say so and stop. If the change touches dependencies, also diff the lockfile.

## Threat checklist

1. **Injection**: shell/SQL/HTML/template/command construction from variables —
   especially strings crossing into `spawn`/`exec`, `$`-substitution, YAML/SQL
   builders. Unquoted expansions are findings even when "inputs look safe".
2. **Input boundaries**: every place external data enters (HTTP bodies, webhook
   payloads, RPC frames, files, env vars, provider responses) must be validated
   before use. Missing caps on size/depth/rate are findings.
3. **Auth & secrets**: new endpoints/channels without auth or with auth disabled
   by default; tokens/keys in code, logs, error messages, or test fixtures;
   credential files written without 0600.
4. **Unsafe handling**: deserialization of untrusted data, `eval`-family, dynamic
   import/require of non-constant paths, path joins without confinement
   (`../` escapes, symlink races, TOCTOU between check and use).
5. **Crypto & randomness**: `Math.random` for security decisions, home-rolled
   hashing/encryption, disabled TLS verification.
6. **Supply chain**: new dependencies or install scripts; pinned vs floating
   versions; network fetches without integrity checks.
7. **Denial of service**: unbounded loops/allocations over attacker-controlled
   sizes, missing timeouts on network calls, regex catastrophic backtracking.

## Output

```
## Security findings
- [critical] src/webhooks.ts:88 — signature compared with ==, timing-attackable.
  Use a constant-time comparison.
- [high] …
## Not vulnerable (checked)
- one line per attack class that was examined and cleared
```

Verdict: **ship** / **fix-first** / **do-not-ship**, with the blocking findings.
