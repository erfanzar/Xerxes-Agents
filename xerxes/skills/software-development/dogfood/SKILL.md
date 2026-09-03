---
name: dogfood
description: Exploratory QA of Xerxes itself to find bugs with evidence.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [qa, testing, dogfood, cli, evidence]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/dogfood/SKILL.md
---

# Dogfood: Systematic QA Testing of Xerxes

## Overview

This skill guides systematic exploratory QA of this repository's own
product: exercise the Xerxes CLI, daemon, and runtime surfaces the way a
real user would, capture evidence of issues, and produce a structured bug
report. Dogfooding means using the tool on itself, not just running its
test suite.

## Prerequisites

- A buildable Xerxes checkout (Bun 1.3+) and the shell/terminal tool
- A scope from the caller: which surfaces to exercise, or "full pass"

## Inputs

1. **Scope** — which areas to focus on (CLI, daemon, ACP, export, skills,
   sessions) or "full pass" for comprehensive coverage
2. **Output directory** (optional) — where logs and the report go
   (default: `./dogfood-output`)

## Native Commands to Exercise

```bash
bun install --frozen-lockfile     # only if node_modules is missing
bun run build                     # production bundles the CLI needs
bun run xerxes --help             # help surface and command list
bun run xerxes doctor             # local diagnostic report
bun run check                     # strict TypeScript checks
bun run test                      # full Bun test suite
bun run xerxes "answer with the single word: ok"   # only with a configured provider key
bun run xerxes export --list      # saved-session export listing
```

Never place credentials in fixtures, logs, or the report. Real model runs
require an explicitly configured provider key or profile; if none exists,
record that surface as "blocked: no credentials" instead of faking it.

## Workflow

Follow this five-phase workflow:

### Phase 1: Plan

1. Create the output structure:
   ```
   {output_dir}/
   ├── logs/          # captured command output as evidence
   └── report.md      # final report (Phase 5)
   ```
2. Identify scope from the caller's input.
3. Build a rough exercise plan: help text accuracy, `doctor` output on this
   machine, one-shot CLI paths, error paths with invalid flags, daemon
   startup and shutdown, export listing, skill discovery.

### Phase 2: Explore

For each surface in the plan, run it through the shell/terminal tool and
record exact commands plus output:

1. Run the documented command exactly as the README and `--help` state it.
2. Then probe the edges: unknown flags, missing arguments, empty stdin,
   interrupted runs (Ctrl+C behavior), repeated invocations.
3. Check that every failure is observable: a command with no handler must
   fail explicitly rather than silently pretending to succeed.
4. Compare observed behavior against AGENTS.md, XERXES.md, and `--help`
   text; doc drift is a real finding.

### Phase 3: Collect Evidence

For every issue found:

1. Save the full command output (stdout and stderr) to a log file in
   `{output_dir}/logs/`.
2. Record: the exact command, working directory, expected behavior, actual
   behavior, and the log path.
3. Classify the issue:
   - Severity: Critical / High / Medium / Low
   - Category: Functional / Documentation / UX / Error-handling / Packaging

### Phase 4: Categorize

1. Review all collected issues and de-duplicate (same root cause appearing
   in multiple commands is one issue).
2. Assign final severity and category to each.
3. Sort by severity and count by severity and category for the summary.

### Phase 5: Report

Write `{output_dir}/report.md` containing:

1. **Executive summary** — total issue count, breakdown by severity, scope
   covered, and which native commands were exercised.
2. **Per-issue sections** — number, title, severity and category, the exact
   command, steps to reproduce, expected vs actual behavior, log reference.
3. **Summary table** of all issues.
4. **Testing notes** — what was exercised, what was not, and any blockers
   (missing credentials, no TTY, platform limits).

## Tips

- **Exercise error paths deliberately.** Silent success on an invalid
  command is a high-severity finding.
- **Test help against reality.** Every command `--help` advertises should
  actually work as described.
- **Check cross-surface consistency.** The daemon, CLI, and export paths
  should agree on session and command behavior.
- **Do not stop at the happy path**: try empty inputs, very long prompts,
  rapid repeated invocations, and running from unexpected working
  directories.
- Keep every claim in the report backed by a saved log file.

## Verification

- [ ] Every documented native command was run at least once
- [ ] Every reported issue has a saved log and a reproduction command
- [ ] The report contains no credentials or secrets
- [ ] Findings were de-duplicated and severity-sorted

---

Adapted from the `dogfood` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Teknium (teknium1), Hermes Agent.
