# Xerxes-Agents Repository Harness (`.agents/`)

Shared developer harness discovered natively by both Antigravity and Xerxes (`projectWorkspace.ts`, `skills.ts`): domain rules, on-demand procedural skills, invocable commands, read-only specialist auditors, and deep lookup tables.

## Layer Model & Budget Rules

- **Root `AGENTS.md`** — Universal commandments (first 1,500 chars for `leanPrompt.ts`), sub-projects matrix, commands, turn invariants, and harness map (`< 14,500 B`).
- **Rules (`.agents/rules/*.md`)** — Always-on domain invariants (plain Markdown, no frontmatter, `< 5,800 B` each).
- **Skills (`.agents/skills/<name>/SKILL.md`)** — Progressive-disclosure procedural checklists loaded on demand (excluded from eager workspace scan).
- **Commands (`.agents/commands/<name>.md`)** — Guided routines accepting `$ARGUMENTS`.
- **Specialist Agents (`.agents/agents/<name>.md`)** — Read-only auditors for protocol, ACP isolation, surface parity, and release readiness.
- **References (`.agents/references/projects-map.md`)** — Dense lookup tables kept `< 5,800 B` so Xerxes's `projectWorkspace.ts` loads every non-skill `.md` cleanly (`< 6,000 B`/file, `< 96 KiB` aggregate).

## Directory Roles: `.agents/` vs `.xerxes/` vs `xerxes/skills/`

| Directory | Tracked | Loader | Purpose |
| --- | --- | --- | --- |
| **`.agents/`** | Yes | Antigravity + Xerxes (`projectWorkspace.ts`, `skills.ts`) | Shared workspace harness: non-skill `.md` files stay `< 6,000 B` each and `< 96 KiB` total; `.agents/skills/` loads on demand (hash-trusted in Xerxes). |
| **`.xerxes/`** | No (gitignored) | Local Xerxes runtime | Local sessions, config, trusted project plugins, local commands/skills. Never commit. |
| **`xerxes/skills/`** | Yes | Bundled runtime (`copyBundledSkills.ts` -> `dist/skills/`) | 119 shipped user-facing skills; enforced by `bundledSkillsLint.test.ts` and `bundledSkillAssets.test.ts`. |

## Complete `.agents/` Index

### Rules (`.agents/rules/`)
- `.agents/rules/core-runtime-daemon.md` — TS/Bun/Windows, v35 daemon RPC & queues, prompt-cache split, streaming loop, tools/providers, fail-closed security.
- `.agents/rules/client-surfaces-and-hub.md` — 5 surface tsconfigs, `types: []` browser purity, Desktop/VS Code bridge, Hub v1, OpenTUI, CSS/WCAG AA, zero-DOM SSR tests.
- `.agents/rules/acp-swarms-and-agents.md` — Antigravity/ACP isolation, Immortals swarms/worktrees/RPC boundary, subagents & `default/agent.yaml`, Goal Mode, Superagent A/B/C/D.
- `.agents/rules/testing-packaging-and-ci.md` — Test runner routing, `bun run verify`, Bun-only & SHA-pinned CI, 9+4 version sync, `packaging/runtime`, skills/docs/examples/playground.

### Skills (`.agents/skills/`)
- `.agents/skills/xerxes-verify/SKILL.md` — Tiered verification ladder from narrow checks/tests to CI gates.
- `.agents/skills/xerxes-daemon-protocol/SKILL.md` — End-to-end v35 daemon RPC, wire event, and slash command updates.
- `.agents/skills/xerxes-tools-and-providers/SKILL.md` — Native Bun tools (`default/agent.yaml` + 7 sets) and LLM providers/OAuth.
- `.agents/skills/xerxes-surfaces-dev/SKILL.md` — OpenTUI, Desktop, VS Code, Mobile PWA, Shared Client, and Hub/Supervisor/Slack.
- `.agents/skills/xerxes-acp-and-swarms/SKILL.md` — External ACP bridges, Immortals swarms, subagents/worktrees, Goal Mode, Superagent.
- `.agents/skills/xerxes-release-and-packaging/SKILL.md` — 9+4 version bumps, `packaging/runtime` sync, `.tgz` pack/check/smoke, Docker, DMG, VSIX.
- `.agents/skills/xerxes-playground-and-docs/SKILL.md` — Eval suites (`playground:*`, `sweCli.ts`, `real-use`), bundled skills, docs, offline examples.

### Commands (`.agents/commands/`)
- `verify.md`, `add-rpc-or-slash.md`, `add-tool.md`, `audit-acp-isolation.md`, `audit-surface-parity.md`, `debug-turn.md`, `release-prep.md`, `run-evals.md`.

### Specialist Agents (`.agents/agents/`)
- `protocol-auditor.md`, `acp-isolation-auditor.md`, `surface-reviewer.md`, `release-auditor.md` (strictly read-only).

### References (`.agents/references/`)
- `.agents/references/projects-map.md` — 6-tsconfig matrix, subsystem map, build graph, 9+4 version sites, budgets, and test runners.

## Harness Hygiene

Keep `AGENTS.md` `< 14,500 B`, every non-skill `.agents/**/*.md` `< 6,000 B` (aggregate `< 96 KiB`), zero `.py`/symlinks, single-line YAML frontmatter on skills/commands/agents, no unshipped `scripts/...` refs, and clean under `promptScanner.ts`.
