---
name: add-tool
description: Guided procedure for adding a native Bun tool with strict schema, >240-char description contract, fail-closed ToolCapabilities, default/agent.yaml sync, all 7 security/tracking sets, and offline tests.
tags: [tools, security, permissions, runtime]
---

# Add a Native Bun Tool to Xerxes

Follow this procedure for the tool in: `$ARGUMENTS`.

## 1. Classify Scope & Side Effects

Determine from `$ARGUMENTS`:
- Tool name and registration path: core (`registerCoreTools` in `xerxes/src/tools/index.ts`), opt-in `EXTRA_TOOL_GROUPS` (`math`, `text`, `system`, `memory`), or host-injected (`xerxes/src/cli.ts`).
- Side effects: read-only, file-writing, shell/process, network/open-world, or destructive; subagent/Supervisor eligibility.

## 2. Implement Tool (`xerxes/src/tools/<module>.ts`)

1. **Header & Strict TS**: Start with the 2-line Apache-2.0 header (`// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).` / `// Licensed under the Apache License, Version 2.0.`). Use `.js` local imports, `import type`, and `...(x === undefined ? {} : { x })` for `exactOptionalPropertyTypes`.
2. **Injected Host Ports**: Never read ambient env, credentials, network globals, or `process.platform` directly inside testable logic; inject explicit ports and `AbortSignal`. Never return fake success when a host capability is unavailable.
3. **Path, File & Spawn Safety**:
   - Validate all arguments at boundary.
   - Resolve workspace file paths via `WorkspacePathResolver` (`xerxes/src/tools/pathSafety.ts`) and call `recheck()` before writes (symlink TOCTOU defense).
   - Integrate `FileStateTracker` (`xerxes/src/tools/fileState.ts`) for read-before-edit guards on file-editing tools.
   - Route Windows `.cmd`/`.bat` spawns through `planSpawn()` (`xerxes/src/core/windowsSpawn.ts`).
4. **Description Contract (`xerxes/test/toolDescriptionContracts.test.ts`)**: Keep description **> 240 chars** stating exact argument semantics, units (`SECONDS` for web timeouts), schema defaults, and footgun warnings; keep concise `guidance`.
5. **Explicit Fail-Closed `ToolCapabilities`**: `DEFAULT_TOOL_CAPABILITIES` (`xerxes/src/executors/toolRegistry.ts`) fails closed (`concurrencySafe: false`, `defer: true`, `destructive: true`, `openWorld: true`, `readOnly: false`, `maxResultBytes: 32_768`). Declare exact capabilities; set `concurrencySafe: true` only when genuinely race-free. Note `register()` keeps the first handler on duplicate `(name, agentId)` (use `replace()` intentionally) and `agentMetaTools` / `skillManageTools` are mutually exclusive.

## 3. Registration & Default Agent Allow-List

1. Wire registration in `xerxes/src/tools/index.ts` (or opt-in group / host wiring).
2. **Critical Pitfall**: Registering in `ToolRegistry` does **not** expose the tool to the default agent. Add core tools to `xerxes/src/agents/default/agent.yaml` (`tools:`) AND `standardTools` in `hardcodedBuiltinDefinitions()` (`xerxes/src/agents/definitions.ts`). Audit sibling presets in `xerxes/src/agents/default/*.yaml` so read-only presets never gain write tools.

## 4. Audit All 7 Security, Permission & Tracking Sets

Update every applicable set:
1. `xerxes/src/streaming/permissions.ts`: `SAFE_TOOLS`, `WRITING_TOOLS`, `ALWAYS_APPROVAL_TOOLS`, `PLAN_ACCEPT_ALL_TOOLS`.
2. `xerxes/src/security/dataBoundary.ts`: `WRITE_TOOLS`, `WRITE_TARGET_KEYS`, `SHELL_TOOLS`.
3. `xerxes/src/runtime/runSafety.ts`: `FILE_WRITERS`.
4. `xerxes/src/context/sweWorkingSet.ts`: `FILE_MUTATING_TOOLS`.
5. `xerxes/src/runtime/leanPrompt.ts`: `LEAN_CORE_TOOL_NAMES`, `LEAN_SWE_TOOL_NAMES`.
6. `xerxes/src/executors/toolRegistry.ts`: `ALWAYS_LOADED_TOOL_NAMES` (only if bypassing `ToolSearchTool` deferral).
7. `xerxes/src/agents/subagentManager.ts`: `SUBAGENT_BLOCKED_TOOLS` (and `assertSupervisorToolCeiling` in `xerxes/src/hub/supervisor/tools.ts`).

## 5. Offline Tests & Verification Gate

Add `xerxes/test/<tool>.test.ts` (and update `toolDescriptionContracts.test.ts` if covered) with injected ports testing happy path, invalid args, explicit error propagation, `AbortSignal`, symlink/path escape, permission modes, and `DataBoundary`:

```bash
bun run --cwd xerxes check:runtime
bun test xerxes/test/toolRegistry.test.ts xerxes/test/toolDescriptionContracts.test.ts xerxes/test/streamingPermissions.test.ts xerxes/test/dataBoundary.test.ts xerxes/test/dataBoundaryLeak.test.ts xerxes/test/toolSurface.test.ts xerxes/test/agents.test.ts
```
Run `xerxes/test/<tool>.test.ts`, then finish with `bun run check && git diff --check`.
