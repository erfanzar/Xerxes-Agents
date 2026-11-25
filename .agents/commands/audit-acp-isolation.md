---
name: audit-acp-isolation
description: Audit external ACP changes against Antigravity thin-client guarantees, read-only agyBrainBridge, env and credential boundaries, Claude/Codex isolation, and the ACP test matrix.
tags: [acp, antigravity, security, isolation, audit]
---

# Audit External ACP & Antigravity Isolation

Run a read-only audit of external ACP changes for: `$ARGUMENTS` (default: `git diff --name-only` across `xerxes/src/acp/`, `xerxes/src/daemon/{acpBridgeHost,agyBrainBridge,server}.ts`, `xerxes/src/auth/{agyRuntimeEnv,codexAuth}.ts`, `xerxes/src/desktop/main/{agyCredentials,providerForwarding}.ts`, `xerxes/src/llms/claudeCode.ts`, `xerxes/src/ui/PROTOCOL.md`).

## 1. Five Mandatory Isolation Checks

### Check A: Antigravity Thin-Client & Verbatim Human-Only Forwarding
- `daemon/server.ts` enforces `ANTIGRAVITY_COMPOSED_TURN_REFUSAL` when `isAntigravityBinding(acpBinding) && options.userAuthored !== true` (blocks goal rounds, `/loop`/schedules, restart resume, board wakes, `/init`).
- Zero Xerxes prompt injection: no workspace context, project instructions, expanded commands, `[steer from ...]` (`verbatim`), `[Note: ...]` omitted-image notices (`!isAntigravity`), or `@mention` inline fallbacks; `/btw` sends `plain: true`.
- Refuses forged actions & remote driving: `/goal` create/edit/resume (`ANTIGRAVITY_GOAL_REFUSAL`), `swarm.{post,kill,model}` (except coordinator model via `acp.set_model`), `/swarm kill`, `terminal.action`/`schedule.cancel` on `agy-task:*` (`AGY_BOARD_ACTION_REFUSAL`), `artifacts.update` on `agy:*`, and Hub remote `turn.{submit,steer,cancel}`/`permission_response`/`question_response`/`queueDurableSteer` (`antigravity-local-only`).

### Check B: AGY Protocol Parity, `acp.login` & Skill Isolation
- Omit `fs.writeTextFile` capability for Antigravity (`!isAntigravity`) to keep native diff edits.
- Never auto-approve `session/request_permission` when `toolCallId` starts with `interaction_` (even under host Turbo).
- Send PDFs as `resource_link`; validate `fs/*` via `resolveWorkspaceScopedPathReal` (no dangling symlinks, 8 MiB / UTF-8-no-NUL cap); advertise `clientCapabilities` only when handlers exist.
- `discoverRuntimeSkillEntries` loads ONLY the active runtime's dirs (AGY: `.agents/skills`, `~/.gemini/antigravity/{,builtin/}skills`; Claude: `.claude/{skills,commands}`), NEVER `xerxes/skills/` or `~/.xerxes/skills/`. `/` completion uses only `completeAcpSlashAndSkills`; colliding slash commands forward verbatim; bypass Xerxes auto-compaction and `maybeGenerateTitle`.
- `acp.login` (`QUEUE_BYPASS_METHODS`, `start|status|submit|cancel`, 5m TTL) drives native `agy acp --login`, strips `DISPLAY`/`WAYLAND_DISPLAY`/`BROWSER` on Linux/SSH, never reads/copies/plants OAuth tokens, and stays synced in `PROTOCOL.md`.

### Check C: Strictly Read-Only `agyBrainBridge.ts`
- Zero create/write/rename/delete under `~/.gemini/antigravity(-cli)/brain/`.
- Enforce tail caps (`1 MiB` inspect, `64 KiB` activity, `2 MiB` tasks/steps, `8 MiB` board), direct-child symlink containment (`listAllowedPreviewRoots`), magic-byte/SVG artifact sniffing, `Swarm Worker`/swarm ID exclusion from subagent lists, and `skipped -> completed` in `readLatestUpdateStepsSync`.

### Check D: Subprocess Env & SSH Credential Boundaries
- `buildAgySubprocessEnv` (`auth/agyRuntimeEnv.ts`) strips `AGY_OUTER_HOST_ENV_KEYS`, `AGY_STRIPPED_XERXES_ENV_KEYS`, `XERXES_*` (except `XERXES_TEST_*`), and outer `GEMINI_HOME`; reads only `AGY_ALLOWED_ENV_KEYS` from `0600` `~/.xerxes/provider-accounts/agy/env.json`.
- `desktop/main/agyCredentials.ts` NEVER reads, exports, or plants Google OAuth keyring/file tokens (`hasKeyringEntry` checks existence only — no `-w`, `GetSecret`, or `secret-tool lookup`); only `adc` and `gemini_api_key` transfer via stdin-only `bun -e` after 5m human review; renderer receives secret-free `AgyLocalStatus`.
- `providerForwarding.ts` adopts host-ready transferred named OAuth accounts (`claude-*`, `codex-*`) on SSH without copying local credentials to remote disk.

### Check E: Claude Code, Codex & ACP Server Framing
- `llms/claudeCode.ts`: `--disable-slash-commands --tools "" --setting-sources "" --strict-mcp-config`, empty cwd, tag neutralization, `isolatedAuth`, `1h -> 5m -> none` cache stepdown.
- `auth/codexAuth.ts`: single-flight refresh (`credentialFlightsByStoragePath`) + `15_000ms` deadline.
- `acp/transport.ts` (`StdioJsonRpcServer`): linear `MAX_ACP_FRAME_BYTES = 16 MiB`, per-session serialized prompts with concurrent cancel/permission/question, duplicate `attachExisting` -> `AcpSessionConflictError`.

## 2. Verification Matrix & Report

```bash
bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:desktop
bun test xerxes/test/acp*.test.ts xerxes/test/agy*.test.ts xerxes/test/cliAcpDelegation.test.ts xerxes/test/claudeCode*.test.ts xerxes/test/codex*.test.ts xerxes/test/desktopAcp*.test.tsx xerxes/test/desktopAgyParity.test.tsx xerxes/test/desktopClaudeLanguageAndAcpDiscovery.test.tsx
bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/tuiAcpCompatibility.test.tsx
```

Report **Critical**, **Warning**, and **Suggestion** findings with `file:line`, violated invariant, `PASS`/issues per Check A–E, and test results.
