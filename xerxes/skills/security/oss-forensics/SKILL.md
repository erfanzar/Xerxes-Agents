---
name: oss-forensics
description: GitHub supply-chain forensics — recover deleted commits, detect force-push tampering, extract IOCs, report responsibly.
version: 1.0.0
author: Teknium (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [security, forensics, supply-chain, github, incident-response]
source: https://github.com/NousResearch/hermes-agent/tree/main/optional-skills/security/oss-forensics
---

# Open-Source Supply-Chain Forensics

Investigate whether a GitHub repository was compromised: recover deleted
or rewritten history, detect force-push tampering, extract indicators of
compromise (IOCs), and produce a report fit for coordinated disclosure.

## When to Use

- "Investigate [owner/repo] for a supply-chain compromise"
- "Recover deleted commits from this repository"
- "Was this repo's release artifact tampered with?"
- "Check for suspicious force pushes or backdoor commits"
- "Extract IOCs from this malicious package"

Do not use for: investigating proprietary/internal repos without
authorization, correlating GitHub activity to real identities, or
harassing maintainers.

## Procedure

### 1. Establish the baseline

Record the repo's current state: default branch HEAD, latest release
tags, published package versions (npm/PyPI), and the timestamps the
user is concerned about. Note what "normal" looked like before the
suspected event.

### 2. Recover deleted/rewritten history

Use the GitHub API — most of the time a force-push does not delete
underlying objects:

- `GET /repos/{owner}/{repo}/events` — public event stream retains
  `PushEvent` entries with before/after SHAs, including for commits
  later removed from a branch
- `GET /repos/{owner}/{repo}/git/commits/{sha}` — fetch a dangling
  commit directly by SHA even if unreferenced
- Compare package registry artifacts with the repo tag: download the
  published tarball and diff against a build from the tagged commit

### 3. Detect tampering patterns

- Force-pushes: scan events for non-fast-forward transitions
- Backdoor signatures: unexpected postinstall scripts, credential
  endpoints, obfuscated payloads in diffs, maintainership changes
  shortly before the suspicious commit
- Timeline correlation: maintainer account compromise often shows as
  activity from new locations/devices plus repo setting changes

### 4. Extract IOCs

Collect from recovered artifacts: attacker-controlled URLs/domains,
IPs, wallet addresses, npm/PyPI package names, commit SHAs, maintainer
accounts involved, and unique strings useful for YARA-style matching.
Record each IOC with the artifact and commit it came from.

### 5. Validate before you claim anything

Anti-hallucination gate: every claim in the report needs evidence you
re-ran (API response, diff, artifact hash). If evidence is ambiguous,
say so — do not upgrade suspicion to assertion.

### 6. Report with coordinated disclosure

If a genuine compromise is confirmed:

1. Notify the repository maintainers privately first
2. Allow reasonable remediation time (typically 90 days)
3. Coordinate with package registries if published packages are
   affected
4. File a CVE if appropriate

Never publish investigation results without validated evidence.

## Pitfalls

- The events API only retains ~90 days of public events — archive early
- Cached/CDN copies (Google cache, archive.org) can supplement deleted
  content but are not primary evidence; cite them as such
- Registry tarballs can differ from repo builds for benign reasons
  (build tooling); diff with care before calling it tampering
- Attribution to real-world identities is out of scope — report account
  names and behaviors, not doxxed identities
- Principles: minimal intrusion; collect only the evidence necessary to
  validate or refute the hypothesis

## Verification

- [ ] Every recovered commit SHA was fetched and its content inspected,
      not inferred
- [ ] Every IOC traces to a specific artifact and commit
- [ ] The report separates confirmed facts, suspicions, and unknowns
- [ ] Disclosure followed the coordinated process (private first)

---

Adapted from the `oss-forensics` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Teknium and Hermes Agent contributors.
