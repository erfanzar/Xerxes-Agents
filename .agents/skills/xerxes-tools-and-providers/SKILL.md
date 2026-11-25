---
name: xerxes-tools-and-providers
description: Use when adding or modifying a native Bun tool, tool capability or permission set, LLM provider transport, model registry entry, or OAuth and profile credential flow in Xerxes.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, tools, providers, llm, auth, security]
---

# Xerxes Tools & LLM Providers

Two core-runtime workflows: adding native Bun tools and extending LLM providers or auth flows. Follow `.agents/rules/core-runtime-daemon.md` (`exactOptionalPropertyTypes` conditional spread `...(x === undefined ? {} : { x })`, `.js` local imports, Apache-2.0 header) and `.agents/rules/acp-swarms-and-agents.md`.

## When to use

- Adding or updating a built-in tool in `xerxes/src/tools/` or changing permission/capability sets.
- Adding or updating an LLM provider in `xerxes/src/llms/` or OAuth/readiness/usage logic in `xerxes/src/auth/` and `xerxes/src/bridge/profiles.ts`.

## Workflow A: Adding or Modifying a Native Bun Tool

1. **Module & registration**:
   - Implement in `xerxes/src/tools/<module>.ts` and wire into `xerxes/src/tools/index.ts` (`registerCoreTools`, or opt-in `EXTRA_TOOL_GROUPS` / `cli.ts`).
   - `ToolRegistry.register()` keeps the first handler on duplicate `(name, agentId)` (use `replace()` to override). `agentMetaTools` and `skillManageTools` are mutually exclusive (`skill_manage`).
2. **Description & `guidance` contract**:
   - Keep descriptions >240 chars with concise `guidance`, preserving units (`SECONDS` on web tools), schema defaults, and all 18 footgun warnings checked by `xerxes/test/toolDescriptionContracts.test.ts`.
3. **Explicit fail-closed `ToolCapabilities`**:
   - `DEFAULT_TOOL_CAPABILITIES` fails closed (`concurrencySafe: false, defer: true, destructive: true, openWorld: true, readOnly: false, maxResultBytes: 32_768`). Only mark `concurrencySafe: true` if order-independent and side-effect-free (`MAX_CONCURRENT_TOOL_CALLS = 8`).
4. **Path, file-state & host-port safety**:
   - Use `WorkspacePathResolver` (`xerxes/src/tools/pathSafety.ts`) with `resolve()` **and** `recheck()` before writes to stop symlink races.
   - Wire `FileStateTracker` (`xerxes/src/tools/fileState.ts`, 200 entries/session) for read-before-edit checks.
   - Inject host ports (`env`, `platform`, `clock`, `fetchImplementation`, fs/process) and return typed, actionable errors when unavailable — never invent fake success.
5. **Expose to default agent (`xerxes/src/agents/default/agent.yaml`)**:
   - Add the tool name to `agent.tools:` in `xerxes/src/agents/default/agent.yaml` (and relevant subagent YAMLs). Registering in `ToolRegistry` alone does **not** expose a tool to the default agent.
6. **Audit all 7 security & tracking sets**:
   1. `xerxes/src/streaming/permissions.ts`: `SAFE_TOOLS`, `WRITING_TOOLS`, `ALWAYS_APPROVAL_TOOLS`, `PLAN_ACCEPT_ALL_TOOLS`.
   2. `xerxes/src/security/dataBoundary.ts`: `WRITE_TOOLS`, `WRITE_TARGET_KEYS`, `SHELL_TOOLS`.
   3. `xerxes/src/runtime/runSafety.ts`: `FILE_WRITERS`.
   4. `xerxes/src/context/sweWorkingSet.ts`: `FILE_MUTATING_TOOLS`.
   5. `xerxes/src/runtime/leanPrompt.ts`: `LEAN_CORE_TOOL_NAMES` / `LEAN_SWE_TOOL_NAMES`.
   6. `xerxes/src/executors/toolRegistry.ts`: `ALWAYS_LOADED_TOOL_NAMES` (vs deferred `ToolSearchTool` loading).
   7. `xerxes/src/agents/subagentManager.ts`: `SUBAGENT_BLOCKED_TOOLS`.
7. **Offline tests**: Add injected-port tests in `xerxes/test/` covering success, validation errors, error propagation, and `AbortSignal` cancellation.

## Workflow B: Adding or Updating an LLM Provider or Auth Flow

1. **Registry (`xerxes/src/llms/providerRegistry.ts`)**: Add to `PROVIDERS` (plus `PROVIDER_ALIASES` / `MODELS_DEV_PROVIDER_NAMES` if aliased).
2. **Client transport (`xerxes/src/llms/client.ts`)**:
   - Wire `createLlmClient` (and `MULTI_API_PROVIDERS` if multi-protocol).
   - Enforce `localOnlyActive()`, `ambientApiKeyAllowed()` / `getApiKey` (never send ambient env keys to non-vendor `base_url` unless `XERXES_ALLOW_AMBIENT_KEY=1`), `withConnectTimeout` (15s default), and `AbortSignal`.
   - Keep `LLMConfig.toJSON()` redacting `api_key: null` and return `undefined` from `calcCost` when pricing is unpublished (never fake `$0`).
3. **OAuth, readiness, usage & profiles**:
   - Use `CredentialStorage` (`xerxes/src/auth/storage.ts`, AES-256-GCM `0600` + CAS): never collapse `CredentialTamperedError` into logged-out, handle `CredentialConflictError`, and deduplicate rotating refresh tokens via single-flight locks (see `codexAuth.ts` `credentialFlightsByStoragePath`).
   - Support per-UUID accounts via `ProviderAccountRegistry` (`provider-accounts/<uuid>`).
   - Wire read-only checks in `xerxes/src/daemon/providerReadiness.ts` (never refresh tokens or leak secrets during readiness checks), plus `auth/profileSessionOptions.ts`, `auth/command.ts`, and `xerxes/src/bridge/profiles.ts`.
   - Update `xerxes/src/auth/usageView.ts` while keeping it strictly `types: []` browser-pure (shared with Mobile PWA).

## Pitfalls

- Never bypass `ALWAYS_APPROVAL_TOOLS` or loosen `DataBoundary` via `.xerxes/data-policy.yaml`; `stdin` always disqualifies read-only `exec_command`.
- `resolveProvider` must never auto-route bare `-codex` models onto ChatGPT subscription (must be explicit `codex/...` or `/backend-api/codex`).
- When `@earendil-works/pi-ai` changes, run `bun run --cwd xerxes generate:image-catalog` to keep `check:image-catalog` green.

## Verification commands

```bash
bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:mobile
bun test xerxes/test/toolDescriptionContracts.test.ts xerxes/test/windowsSupport.test.ts xerxes/test/imageModels.test.ts
bun test xerxes/test/<your-tool-or-provider>.test.ts
```
