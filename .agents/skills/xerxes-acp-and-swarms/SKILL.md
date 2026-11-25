---
name: xerxes-acp-and-swarms
description: Use when modifying external ACP bridges (Antigravity, Claude Code, Codex), Immortals Swarms, native subagents or worktrees, Goal Mode, or Superagent abilities.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, acp, antigravity, swarm, subagents, goal, superagent]
---

# Xerxes ACP Bridges, Immortals Swarms, Subagents, Goal & Superagent

Checklist for external ACP runtimes, Immortals swarms, subagents/worktrees, Goal Mode, and Superagent tiers. See `.agents/rules/acp-swarms-and-agents.md` for always-on safety constraints.

## When to use

- Editing `xerxes/src/{acp,swarm,agents}/`, `xerxes/src/daemon/{acpBridgeHost,agyBrainBridge,swarmHost,subagent*,goalCommand,superagent*}.ts`, `xerxes/src/auth/{agyRuntimeEnv,codexAuth}.ts`, `xerxes/src/desktop/main/agyCredentials.ts`, `xerxes/src/runtime/{goal*,subagentWorktrees}.ts`, or `xerxes/src/hub/supervisor/abilities/`.

## Workflow A: External ACP Bridges & Antigravity Parity

- **ACP Server (`xerxes/src/acp/`)**: Enforce `MAX_ACP_FRAME_BYTES = 16 MiB` (`AcpFrameByteCounter`), per-session prompt serialization with concurrent `cancel`/permission/question responses, explicit `null` id vs omitted notification id, and `AcpSessionConflictError` on duplicate `attachExisting`.
- **Antigravity thin-client guarantees (`acpBridgeHost.ts`, `server.ts`, `externalAgents.ts`)**:
  1. **Verbatim human-only turns**: Refuse non-human composed turns (`userAuthored !== true` -> `ANTIGRAVITY_COMPOSED_TURN_REFUSAL`). Never inject Xerxes workspace context, instructions, expanded commands, `[steer from ...]` wrappers, `[Note: ...]` image notices, or `@mention` text; `/btw` uses `plain: true`.
  2. **Zero forged actions**: Refuse `/goal` mutations (`ANTIGRAVITY_GOAL_REFUSAL`), `swarm.{post,kill,model}` (except coordinator `acp.set_model`), `terminal.action`/`schedule.cancel` on `agy-task:*` (`AGY_BOARD_ACTION_REFUSAL`), `artifacts.update` on `agy:*`, and Hub remote driving (`antigravity-local-only`).
  3. **Read-only `agyBrainBridge.ts`**: Never write/rename/delete under `~/.gemini/antigravity(-cli)/brain/`. Preserve tail caps (1 MiB inspect, 64 KiB activity, 2 MiB tasks/steps, 8 MiB board), `listAllowedPreviewRoots` containment, magic-byte/SVG sniffing, and swarm-worker filtering.
  4. **Capabilities & skills**: Omit `fs.writeTextFile` for AGY (`!isAntigravity`), never auto-approve `session/request_permission` for `interaction_*` (even in Turbo), validate `fs/*` via `resolveWorkspaceScopedPathReal` (8 MiB / UTF-8-no-NUL), and keep `discoverRuntimeSkillEntries` strictly isolated to the external runtime's roots.
  5. **Env & auth**: `buildAgySubprocessEnv` (`agyRuntimeEnv.ts`) strips `AGY_OUTER_HOST_ENV_KEYS`, `AGY_STRIPPED_XERXES_ENV_KEYS`, `XERXES_*` (except `XERXES_TEST_*`), and outer `GEMINI_HOME`, loading only `AGY_ALLOWED_ENV_KEYS` from `0600` `env.json`. `agyCredentials.ts` checks keyring existence only (never reads/exports/plants Google OAuth tokens). `acp.login` (`QUEUE_BYPASS_METHODS`, 5m TTL) drives `agy acp --login` on the daemon host with `DISPLAY`/`WAYLAND_DISPLAY`/`BROWSER` stripped on Linux/SSH.
- **Claude Code & Codex**: Keep `claudeCode.ts` model-only flags (`--disable-slash-commands --tools "" --setting-sources "" --strict-mcp-config`), tag neutralization, and `isolatedAuth`; keep `codexAuth.ts` single-flight refresh (`credentialFlightsByStoragePath`, 15s deadline).

## Workflow B: Immortals Swarms (`src/swarm/`, `daemon/swarmHost.ts`, `tools/swarmTools.ts`)

- **Board, channels & wakeup**: Reject `@#main` (`ValidationError`, use `@all`). Keep `formatSwarmNarratorFeedback` / `as_feedback: true` stripping `@` before members/groups/`all`/`coordinator`/`#channels` so only `@narrator` wakes. Preserve permanent killed names, unkillable `coordinator`, host-only `kill{name:"all"}` (`allowAll: true`), `added_by` lineage scoping, refusal of `model` switch while `working`, and `stopped` lazy-history restore.
- **Worktrees & patches**: `inherit` survives `kill`; `share` (`git worktree add --detach`) and `branch` (`git clone --local`) delete permanently on `kill`. Every git call uses `-c core.hooksPath=/dev/null`, strips `GIT_*`, sets `GIT_OPTIONAL_LOCKS=0`, 60s timeout, and serializes `share` per `git-common-dir`. Keep `assertSafePatchPaths` + `git apply --check --binary`.
- **RPC-only UI boundary**: UI/Desktop/VS Code/Mobile MUST NEVER import `src/swarm/` directly (`nativeSwarmUi.test.ts`); state travels via v35 RPCs (`swarm.*`) and `swarm_event`. Keep `xerxes/skills/immortals-{coordinator,monitor,narrator}/SKILL.md` synced.

## Workflow C: Subagents, Goal Mode & Superagent

- **Subagents**: Intersect caller & definition allow-lists with fail-closed sentinel `['\0']`; preserve `SUBAGENT_BLOCKED_TOOLS`, server-side `maxDepth`, `delegatedPermissionExceeds`, `SubagentConversationPersistence` mutual-exclusion claims, post-injection `consume()` + `holds()`, and fail-closed `respawnRecovered` (`providerRoute` 64-hex SHA-256 + `modelCallBindings`). Worktrees (`subagentWorktrees.ts`) stay git-only with `0600` UUID manifests, refuse dirty removal, and never auto-commit.
- **Goal Mode**: Separate durable `phase` (`active|paused|blocked|complete`) from process-local `activation` (`armed|disarmed`; resumed/forked sessions start `disarmed`). Enforce `foldGoalChanges` + CAS `{ goal_id, revision }`, human vs goal-round authority split, min caps (`GOAL_MIN_DURATION_MS = 60_000`, `GOAL_MIN_TOTAL_TOKENS = 1_000`), and `GoalTokenBudget.fork()`.
- **Superagent (A/B/C/D tiers)**: A SEE bounded read-only (<= 60 KiB / 384 KiB), B REMEMBER/SCHEDULE, C DO Hub controls only (no shell/sandbox, human-only approvals), D GATED `propose_risky_action` (live private owner approval on exact canonical args, fail-closed pre-mutation audit + hash/revision recheck, single-use). Mark observations `untrusted: true` and enforce both screenshot disclosure gates (`captureDisclosure` + `hubModelAuthorizeImages`).

## Verification commands

```bash
# ACP & Antigravity
bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:desktop
bun test xerxes/test/acp*.test.ts xerxes/test/agy*.test.ts xerxes/test/cliAcpDelegation.test.ts xerxes/test/desktopAcp*.test.tsx xerxes/test/desktopAgyParity.test.tsx xerxes/test/desktopClaudeLanguageAndAcpDiscovery.test.tsx xerxes/test/claudeCode*.test.ts xerxes/test/codex*.test.ts
bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/tuiAcpCompatibility.test.tsx

# Immortals Swarms
bun run --cwd xerxes check
bun test xerxes/test/nativeSwarm*.test.* xerxes/test/swarmClientSurfacesDesktopVscode.test.ts xerxes/test/swarmGraph.test.tsx xerxes/test/desktopSwarmInspectorFormat.test.tsx xerxes/test/swarmDaemonMemory.test.ts xerxes/test/swarmHubSurfaces.test.ts
bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/nativeSwarmUi.test.ts xerxes/src/ui/__tests__/swarmClientSurfacesUi.test.ts
bunx --bun --no-install vitest run --config xerxes/vitest.mobile.config.ts xerxes/src/mobile/__tests__/swarmClientSurfacesMobile.test.ts

# Subagents, Goal Mode & Superagent
bun test xerxes/test/nativeSubagent*.test.ts xerxes/test/isolatedNativeSubagent.test.ts xerxes/test/subagent*.test.ts xerxes/test/daemonSubagent*.test.ts xerxes/test/turnInterruptSubagents.test.ts xerxes/test/hybridPlan*.test.ts xerxes/test/goal*.test.ts xerxes/test/desktopNotifyPlanGoal.test.ts xerxes/test/superagent*.test.ts xerxes/test/desktopSuperagentControl.test.tsx
bunx --bun --no-install vitest run --config xerxes/vitest.mobile.config.ts xerxes/src/mobile/__tests__/superagentControl*.test.*
```
