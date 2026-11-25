# ACP, Immortals Swarms, Subagents, Goal Mode & Superagent Rules

Applies to `src/{acp,swarm,agents}/`, `src/daemon/{acpBridgeHost,agyBrainBridge,swarm*,subagent*,goalCommand,superagent*}.ts`, `src/auth/{agyRuntimeEnv,codexAuth}.ts`, `src/llms/claudeCode.ts`, `src/desktop/main/agyCredentials.ts`, `src/runtime/{goal*,subagentWorktrees}.ts`, `src/hub/supervisor/`.

## 1. External ACP & Antigravity Thin-Client Isolation

- **Human-only verbatim turns**: `daemon/server.ts` MUST refuse non-human turns (`userAuthored !== true`) on Antigravity (`ANTIGRAVITY_COMPOSED_TURN_REFUSAL`: goals, schedules/`/loop`, restart resume, board wakes, `/init`). Never inject workspace context, instructions, expanded commands, `[steer from ...]` wrappers, omitted-image notices, or `@mention` fallback text; `/btw` sets `plain: true`.
- **Zero forged actions / local-only**: Refuse `/goal` mutations (`ANTIGRAVITY_GOAL_REFUSAL`), `swarm.{post,kill,model}` (except coordinator via `acp.set_model`), `/swarm kill`, `terminal.action`/`schedule.cancel` on `agy-task:*` (`AGY_BOARD_ACTION_REFUSAL`), `artifacts.update` on `agy:*`, and Hub remote driving (`antigravity-local-only`).
- **Read-only `agyBrainBridge.ts`**: Never mutate `~/.gemini/antigravity(-cli)/brain/`. Keep tail caps (`1M`/`64K`/`2M`/`8M`), direct-child preview roots, magic-byte/SVG sniff, and swarm-worker filtering.
- **Protocol & skill isolation**: Omit `fs.writeTextFile` for AGY; never auto-approve `interaction_*` (even in Turbo); send PDFs as `resource_link`; scope `fs/*` via `resolveWorkspaceScopedPathReal` (8 MiB UTF-8 no NUL, no dangling symlinks); advertise `clientCapabilities` only when handled. On external ACP, `discoverRuntimeSkillEntries` reads ONLY that runtime's skills/commands (never `xerxes/skills/` or `~/.xerxes/skills/`), `/` completes only ACP entries, and auto-compaction/titling stay off.
- **Env, auth & providers**: `buildAgySubprocessEnv` strips `AGY_OUTER_HOST_ENV_KEYS`, `AGY_STRIPPED_XERXES_ENV_KEYS`, `XERXES_*` (except `XERXES_TEST_*`), and `GEMINI_HOME`, loading only `AGY_ALLOWED_ENV_KEYS` from `0600` `env.json`. `acp.login` runs `agy acp --login` on-host (stripping `DISPLAY`/`WAYLAND_DISPLAY`/`BROWSER` on Linux/SSH) without copying OAuth tokens. `agyCredentials.ts` checks keyring existence only (`adc`/`gemini_api_key` via stdin `bun -e` after 5m review; secret-free `AgyLocalStatus`). `claudeCode.ts` runs model-only (`--disable-slash-commands --tools "" --setting-sources "" --strict-mcp-config`, empty cwd, tag neutralization, `isolatedAuth`); `codexAuth.ts` uses single-flight refresh (`15s`). `StdioJsonRpcServer` caps frames at `16 MiB` and rejects duplicate `attachExisting`.

## 2. Immortals Swarms, Worktrees & Surface Boundary

- **RPC-only surfaces**: UI, Desktop, VS Code, and Mobile MUST NEVER import `src/swarm/` (`nativeSwarmUi.test.ts`); use v35 `swarm.*` RPCs and `swarm_event`.
- **Board & authority**: `@#main` is invalid (`ValidationError`; use `@all`). `as_feedback: true` / `formatSwarmNarratorFeedback` strips `@` before members/groups/`all`/`coordinator`/`#channels` (wakes only `@narrator`). Killed names are permanent; `coordinator` cannot be killed; `kill{name:"all"}` needs host-only `allowAll: true`; workers manage only self/`added_by` helpers; never switch `model` on `working` members; restored workers start `stopped` and revive on mention; reject interactive-approval worker tools.
- **Worktrees & patches**: `inherit` survives `kill`; `share` (`git worktree add --detach`) and `branch` (`git clone --local`/copy) delete on `kill`. Git calls set `-c core.hooksPath=/dev/null`, strip `GIT_*`, set `GIT_OPTIONAL_LOCKS=0`, 60s timeout, and serialize `share` per `git-common-dir`. `assertSafePatchPaths` blocks absolute paths, `..`, `.git`, conflict markers, and requires `git apply --check --binary` (no stage/commit).

## 3. Subagents, Presets & Worktrees

- Add new main-agent tools to **`xerxes/src/agents/default/agent.yaml`** `tools:`.
- `effectiveConfig` intersects allow-lists (`['\0']` fail-closed) and unions exclusions; `SUBAGENT_BLOCKED_TOOLS` blocks recursion unless `_allowSubagentDelegation === true`; child `maxDepth`/`permissionMode` never exceed parent.
- `SubagentConversationPersistence` enforces claims, CAS, and terminal eviction; `NativeSubagentTurnCoordinator` consumes post-injection and retains `listRetryTasks()` generations; `respawnRecovered` fails closed on changed/missing 64-hex `providerRoute` or `modelCallBindings`; `resolveOwnedSubagentRetry` checks ownership.
- `subagentWorktrees.ts` is git-only (no fallback/auto-commits), writes `0600` UUID manifests in non-symlink `<git-common-dir>/xerxes-agent-worktrees/`, refuses dirty removal, honors `setupPolicy: 'disabled'`, and verifies SHA-256 `reviewId` + `destinationState`. `createHybridPlacement` fails closed without a local model.

## 4. Goal Mode & Superagent

- **Goal Mode**: Separate durable `phase` (`active|paused|blocked|complete`) from process-local `activation` (`armed|disarmed`; resumed/forked start `disarmed`). Enforce `60s`/`1_000`-token floors, `foldGoalChanges` + CAS `{ goal_id, revision }`, authority split (`blocked` refused before round 3; `record_evidence` bound to verified tool call), zero goal tools for subagents, and `GoalTokenBudget.fork()` bound to `goalId`.
- **Superagent**: A (`observeWorkspace` <= 60/384 KiB), B (remember/schedule), C (Hub controls only; no shell/sandbox; human approvals), D (`propose_risky_action`: live private Hub owner approval on exact args, no Autopilot, rechecked after fail-closed audit, single-use, no retry on unknown outcome). Tag external data `untrusted: true`; `superagentCapture*` defaults off (inert loopback HTML, 4 widths x 900, 256 KiB PNG, `captureDisclosure` + `hubModelAuthorizeImages`).
