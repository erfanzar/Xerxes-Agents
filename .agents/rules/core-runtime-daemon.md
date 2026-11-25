# Core Runtime & Daemon Rules

Applies to `xerxes/src/{cli,xerxes,index}.ts` and `xerxes/src/{core,protocol,types,daemon,streaming,llms,auth,routing,executors,tools,lsp,mcp,security,session,context,memory,runtime,tasks,extensions,channels,cron,api-server,audit,bridge,cortex,workspace,images,local,operators}/`.

## 1. TypeScript, Bun & Cross-Platform Contracts

- Keep `exactOptionalPropertyTypes`, `noUncheckedIndexedAccess`, `verbatimModuleSyntax`, and `isolatedModules` enabled in `xerxes/tsconfig.json`. Never widen `key?: T` to `key?: T | undefined`; always use `...(x === undefined ? {} : { x })`.
- End local ESM imports with `.js`, use `import type` for types, and start every `.ts`/`.tsx` file with the 2-line Apache-2.0 header (`// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).` / `// Licensed under the Apache License, Version 2.0.`).
- Prefer Bun APIs (`Bun.file`, `Bun.write`, `Bun.spawn`, `Bun.serve`, `Bun.sqlite`, `Bun.YAML`, `Bun.Glob`, `Bun.CryptoHasher`). Call `timer.unref?.()` on housekeeping timers. Inject host ports (`env`, `platform`, `clock`, `fetchImplementation`, fs/process) instead of reading ambient globals in core logic.
- Windows lockstep: guard Unix socket `mkdir`/`rm` with `!isNamedPipePath(path)` (`\\\\.\\pipe\\xerxes-global-<sha256[:16]>`), spawn `.cmd`/`.bat` shims via `planSpawn()` + `windowsVerbatimArguments: true` (reject `%`/`\r`/`\n`/`\0`), and keep `core/hostPlatform.ts` aligned with `ui/lib/hostPlatform.ts` (`windowsSupport.test.ts`).

## 2. Daemon v35 Protocol, Queues & Slash Sync

- One global daemon per `XERXES_HOME`; never mutate process `cwd`. Preserve `DaemonWorkspaces` per-cwd isolation (`WORKSPACE_RELEASE_DELAY_MS = 60_000`) and `inFlightTurnRunners` across `reload()`.
- **Bidirectional `PROTOCOL.md` sync (`daemonProtocolContract.test.ts`)**:
  - Every RPC method added/changed in `daemon/server.ts` MUST match `xerxes/src/ui/PROTOCOL.md` (no undocumented methods, no phantom rows).
  - Document every emitted/broadcast wire event in `PROTOCOL.md`, removing `*(bridge only)*` once a non-`bridge/` producer emits it.
  - Use wire event names (`think_part`, not `thinking_part`) in `PRODUCTIVE_TURN_EVENTS`.
- **Queue tiers**: Never put session-mutating RPCs in `QUEUE_BYPASS_METHODS` or `CONCURRENT_DISPATCH_METHODS`; keep `QUEUE_BYPASS_METHODS` for read-only/control calls that must answer mid-turn.
- **Hub & Slash**: Update `HUB_SESSIONLESS_METHODS` and `security/hubIdentityVerifier.ts` (`hubMethodPolicy`) for sessionless Hub RPCs. Update `bridge/commands.ts` (`COMMAND_REGISTRY`), `daemon/server.ts` (`HANDLED_CANONICAL_COMMANDS` / `DAEMON_EXTENSION_COMMANDS`), and `bridge/slashRouter.ts` (`ROUTED_COMMANDS`) together, verifying zero collisions in `harnessLint.ts`.

## 3. Prompt-Cache Stability, Turn Loop & Persistence

- **Prompt-cache separation**: NEVER inject per-turn mutable state (memory diffs, live goal status, date changes, instruction diffs, approved lessons, context deltas) into `assembleContextLayers()`. Keep the system prompt byte-identical in `SessionPromptSnapshots` across rebuilds; put all volatile updates in `assembleTurnContext()` (`<turn-context>`).
- **Streaming loop (`streaming/loop.ts`)**: `groupToolDecisions` parallelizes ONLY consecutive `concurrencySafe` calls (`MAX_CONCURRENT_TOOL_CALLS = 8`) and never reorders across unsafe, denied, or cancelled calls. `normalizeToolOutput` runs AFTER `tool_result_persist` hooks (`neutralizeSystemReminders` + non-empty fallback). Cancellation must synthesize paired tool results.
- **Persistence & compaction**: Keep `DaemonTranscriptStore` summary fields before `\n  "messages": [` (head reads), enforce generation CAS + `.jsonl` crash journal, and refuse user-only dead turns (`transcriptHasHistory`). `ContextCompressor` preserves `protectFirst: 3`/`protectLast: 6`, caps `<swe-working-set>` at 1,600 chars, marks `xerxes_compaction_summary: true`, and runs `repairToolMessageSequence`.
- **Reflection & lessons**: Reflection runs strictly post-persistence & post-`turn_end`, never auto-applies proposals, and excludes subagents/aborted turns. `LessonStore.record` requires host-verified `tool_call`/`test` evidence; only `approved` lessons inject (`MAX_LESSONS_PER_TURN = 5`).

## 4. Tools, Providers & Fail-Closed Security

- **Tools**: Keep `DEFAULT_TOOL_CAPABILITIES` fail-closed; preserve >240-char descriptions and footgun clauses (`toolDescriptionContracts.test.ts`); `refineCapabilities` widens `concurrencySafe` only for read-only `exec_command` without `stdin`. Register in `tools/index.ts`, add main-agent tools to **`xerxes/src/agents/default/agent.yaml`**, and audit all 7 sets (`permissions.ts`, `dataBoundary.ts`, `runSafety.ts`, `sweWorkingSet.ts`, `leanPrompt.ts`, `toolRegistry.ts`, `subagentManager.ts`).
- **Fail-closed boundaries**: Always `WorkspacePathResolver.resolve()` + `recheck()` before writes and update `FileStateTracker`. Keep `tool_permission_check` (`PreToolUse`), `ALWAYS_APPROVAL_TOOLS` (even under `accept-all`), `DataBoundary` (`.xerxes/data-policy.yaml` can only tighten), `PolicyRouter` (fail closed to `trust: 'cloud'` / `no_allowed_route`), `CommandSandbox` (no unsandboxed fallback), `SandboxRouter` (no JS closures), `BrowserManager` (explicit CDP endpoint only), `GitScm` (direct argv, no shell), and `ProjectPlugins` (SHA-256 trust, tools-only, no symlinks) fail-closed.
- **Providers & secrets**: Withhold ambient API keys from non-vendor `base_url`s unless `XERXES_ALLOW_AMBIENT_KEY=1`; never auto-route bare `-codex` to ChatGPT subscription; never collapse `CredentialTamperedError` into logged-out; `calcCost` returns `undefined` (never `$0`) when unpublished; honor `localOnlyActive()`; redact keys in `toJSON()` and `AuditEmitter`.
