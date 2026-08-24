

===== Explore CLI & runtime bootstrap =====

## Xerxes CLI Entry & Runtime Bootstrap — Exploration Report

### 1. CLI dispatch (`xerxes/src/cli.ts`, ~1470 lines)

`cli.ts` is a top-level script, not a main() function: dispatch is an if/else chain at module scope over `Bun.argv.slice(2)`, preprocessed by `extractAgentOption` (`runtime/commandOptions.js`) which pulls out `--agent`. A `NON_ONESHOT_COMMANDS` set rejects `--agent` for commands that own their own runtime. Modes:

- **Meta/service**: `--help` (`renderHelp` over `HELP_GROUPS` data), `--version`, `skill` (`runBundledSkillCli`), `auth`, `doctor` (`runAllDoctorChecks`), `install`, `update`, `export` (`runExport` with manual `parseExportOptions`).
- **Daemon**: `runDaemon` builds `daemonRuntime()` (an `InMemoryDaemonRuntime` with a `runnerFactory` that rebuilds `ToolRegistry` + `AgentTurnRunner` per settings change) plus channel manager, and runs until SIGINT/SIGTERM via an idempotent `finish` promise. `telegram` wraps it by injecting a Telegram channel config.
- **ACP**: `runAcp` → `acpServer()` → `AcpAgentRunner` + `serveACPStdio`.
- **TUI**: `runTui()` requires a TTY, resolves the built OpenTUI entry via `resolveTuiEntry` (`runtime/distribution.js`), stamps `XERXES_DAEMON_BUILD_ID`, and spawns `Bun.spawn([process.execPath, entry])` under `withTerminalWatchdog` — the TUI is a *child process* talking to the daemon.
- **One-shot**: bare prompt or stdin → `runOneShot()` (permission mode `accept-all`, streams `runTurn` events to stdout, sets `process.exitCode = 1` on terminal provider error); `--resume` with a prompt → `runResumedOneShot`, without → TUI resume via `XERXES_TUI_RESUME`.

Typed command errors (`InstallCommandError`, `UpdateCommandError`, `AuthCommandError`) render through `reportCommandUsageError` — two clean stderr lines, never a stack.

### 2. Runtime bootstrap (`xerxes/src/runtime/`)

- `bootstrap.ts: bootstrap()` is a staged pipeline (environment → git_info → xerxes_md → project_agent_workspace → commands → tools → system_prompt) returning `BootstrapResult` with per-stage status/duration and an `asMarkdown()` diagnostic report. All I/O goes through an injectable `BootstrapHost`; byte budgets are explicit constants (`MAX_BOOTSTRAP_*`, e.g. 32 KiB instructions, 96 KiB `.agents` workspace). Failed optional probes become `skipped` stages, not failures. `buildBootstrapSystemPrompt` generates the "Tools available this turn" listing.
- **Profiles**: `promptProfiles.ts` (`PromptProfile` FULL/COMPACT/MINIMAL/NONE → immutable `PromptProfileConfig` with inclusion gates and caps).
- **Budgets**: `iterationBudget.ts` (`IterationBudget` with consume/refund/`BudgetExhausted`), plus `costTracker.ts`, `denialBudget.ts`, `circuitBreaker.ts`.
- **Diagnostics**: `doctor.ts` (severity-tagged `Diagnosis` checks: Bun, PATH, provider keys, XERXES_HOME), `editDiagnostics.ts`, `errorClassifier.ts`.
- `features.ts` composes plugins/skills/policy/sandbox behind `RuntimeFeaturesConfig`; `queryEngine.ts` is the embedded turn engine.

### 3. Core (`xerxes/src/core/`)

- `paths.ts`: `xerxesSubdir`/`agentsSubdir` with traversal-safe `resolveSubdir` (rejects absolute segments, `..`, and escapes).
- `config.ts` (~1515 lines): typed config classes (`ExecutorConfig`, `LLMConfig`, `SecurityConfig`…) built by a `FieldSpec`/`parseFields` mini-framework with aliases, ranges, and env provenance (`getConfigProvenance`).
- `errors.ts`: `XerxesError` hierarchy with frozen structured `details`.
- `basics.ts`: legacy decorator registries (`basicRegistry`, `REGISTRY`) — explicitly labeled legacy.
- `argumentValidation.ts` (runtime): schema-subset tool-arg validator that also *coerces* provider JSON strings and returns the repaired payload via `coerced`.

### 4. Patterns & risks

Patterns: host-port injection everywhere (`BootstrapHost`, `GitCommandRunner`, LLM/sandbox ports), frozen config objects, discriminated-union stage statuses, byte-budgeted prompt injection, build-ID handshake between CLI and daemon (`XERXES_DAEMON_BUILD_ID`).

Risks/notes: no TODO/FIXME markers in scope. `cli.ts` is a very large module running at import time — a throw anywhere kills the process uncaught. One-shot runs default to `accept-all` permissions (documented, but worth noting for piped use). `core/config.ts` and `core/basics.ts` self-describe as legacy surfaces parallel to the daemon config path.


===== Explore LLM providers & streaming loop =====

## Summary: `xerxes/src/llms/` + `xerxes/src/streaming/` (read-only exploration)

### 1. Provider registry
`xerxes/src/llms/providerRegistry.ts` holds the static `PROVIDERS` map (15 entries: `anthropic`, `openai`, `openai-codex`, `openrouter`, `claude-code`, `gemini`, `kimi`, `kimi-code`, `qwen`, `zhipu`, `deepseek`, `minimax`, `ollama`, `lmstudio`, `custom`) plus a `COSTS` per-million-token table. Each `ProviderConfig` carries one of three `ProviderTransport`s — `anthropic`, `claude-code`, or `openai` — so most vendors ride the OpenAI-compatible transport. Helpers: `resolveProvider`, `bareModel`, `getApiKey`, `getContextLimit`. Client factory `createLlmClient` (in `xerxes/src/llms/client.ts`, used by `cli.ts`) also honors plugin providers via `isPluginLlmProviderFactory` from `extensions/plugins.ts`.

### 2. Streaming turn
- **Event vocabulary** (`streaming/events.ts`): `StreamEvent` = `text`, `thinking`, `provider_retry`, `tool_start`, `permission_request`, `tool_end`, `usage_update`, `skill_suggestion`, `turn_done`; `turn_done` carries a closed `TurnStopReason` union (e.g. `completed`, `aborted`, `context_overflow`, `output_limit`).
- **Loop** (`streaming/loop.ts`, `runTurn` ~L311, 1667 lines): async generator over tool turns with retry loop, per-attempt accumulators reset each attempt so partial data from a failed attempt never leaks into the persisted assistant message, and `ToolRoundTextDeduper` suppressing replayed prefixes.
- **Tool-call parsing**: native deltas merged via `mergeToolDeltas`/`completedToolCalls` in `client.ts` (falling back to `deterministicToolCallId`); raw-text formats handled by a pure stateless parser registry in `streaming/parsers/index.ts` (`TOOL_CALL_PARSER_REGISTRY`: xml_tool_call, llama, mistral, qwen, qwen3_coder, deepseek_v3/v3.1, glm45/47, kimi_k2, longcat) with substring-based `detectToolCallFormat`.
- **Cancellation repair**: every `tool_use` block must get a `tool_result`; aborted/cancelled calls get `cancelledToolResult` (loop.ts ~L887, L1460) so the replayed transcript stays provider-valid.
- **Late steer**: `drainSteer` dependency injects `[steer from user]` at safe boundaries (L419); on abort the steer is persisted as `[steer from user saved for next turn]` (L762).
- **Terminal guarantee**: try/catch/finally yields exactly one `turn_done` even for errors outside the attempt handler (L1027–1078); the OpenAI client throws if a stream ends without a terminal event (client.ts ~L352).

### 3. Wire normalization
Adapters emit a neutral `LlmDelta`/`LlmCompletion` (`client.ts` L134–156): `OpenAiCompatibleClient` (SSE chat completions), `ResponsesApiClient` + `ResponsesEventTranslator` (`streaming/responsesApi.ts`, terminal-event guard, post-terminal events dropped), `AnthropicMessagesClient` (`llms/anthropic.ts`, content blocks + prompt caching via `streaming/promptCaching.ts`), `GeminiClient` (`normalizeFinishReason` maps `max_tokens`→`length`), `OllamaClient`. `streaming/sse.ts` is an incremental SSE parser (10 MB record cap); `thinkingParser.ts` splits thinking blocks from text.


===== Explore tools, executors, security =====

## Summary: tools, executors, security

### 1. Tool registry & execution
`xerxes/src/executors/toolRegistry.ts` is the single executor surface. `ToolRegistry` (implements `ToolExecutor`) maps tool name → per-`agentId` `RegisteredTool[]` (`definition` + `handler` + `capabilities`), with **agent-first lookup that falls back only to `agentId: 'default'`** — one agent's variant is never visible to another. `register()` warns on shadowing duplicates; `replace()`/`unregister()` handle intentional overrides. `execute()` checks `signal.aborted`, validates/coerces arguments via `validateToolArguments` (repaired payload is what the handler receives), then serializes results via `serializeToolResult`.

Key design: `ToolCapabilities` (`concurrencySafe`, `defer`, `interruptBehavior: 'block'|'cancel'`, `destructive`, `maxResultBytes`, `openWorld`, `readOnly`) with **fail-closed `DEFAULT_TOOL_CAPABILITIES`** (undeclared ⇒ destructive, unsafe-to-parallelize). Deferred schema loading (`definitionsForTranscript`, `revealedToolNames`) derives live tool schemas by scanning the transcript for `TOOL_SEARCH_LOADED_KEY` markers, so compaction never leaves the model a schema it can't see. `refineCapabilities` may only *widen* `concurrencySafe` for `exec_command` read-only invocations.

`xerxes/src/tools/index.ts` — `registerCoreTools(registry, CoreToolsOptions)` wires everything; privileged surfaces are strictly opt-in (throws `ConfigurationError` on conflicting skill-manage registration).

### 2. Permission/policy gates & sandbox
- `xerxes/src/security/policy.ts` — `ToolPolicy` (allow/deny/optional sets, deny always wins, case-insensitive) + `PolicyEngine` (per-agent policies shadow global; listeners notified but can't block enforcement). 
- `xerxes/src/streaming/permissions.ts` — the runtime gate: `PermissionMode` (`accept-all` default YOLO), `ALWAYS_APPROVAL_TOOLS` (`send_message`, `RemoteTriggerTool`, `ScheduleCronTool`) prompt in every mode with an explicit `bypassAlwaysApprove` escape hatch no in-tree caller sets; `SAFE_TOOLS` auto-approve; `PermissionBroker` port injects the interactive prompt.
- `xerxes/src/security/approvals.ts` — `ApprovalStore` with `once|session|always` scopes, atomic persistence of ALWAYS records; no global singleton.
- `xerxes/src/security/shellAnalysis.ts` — bounded segment-wise shell analyzer; unparseable constructs (substitution, heredocs, `eval`) return `unresolved`, never safe; env-assignment hijack prefixes (LD_PRELOAD, NODE_OPTIONS…) explicitly blocked.
- `xerxes/src/security/sandbox.ts` — `SandboxRouter` (off/warn/strict modes, elevated vs sandboxed tool sets) routing to a `SandboxBackend` port that receives **serializable requests, never closures**; `SandboxedToolExecutor` wraps any `ToolExecutor`. Backends in `security/sandboxBackends/` (docker, daytona, modal, singularity, ssh, subprocess) behind caller-owned `SandboxBackendRegistry`; Bun runtime only builds the subprocess backend from config — others must be explicitly injected.
- Also: `promptScanner.ts` (bounded-regex prompt-injection scanner for imported context), `urlSafety.ts` (scheme deny-list, SSRF checks), `pathSecurity.ts`, `redact.ts`, `credentialFiles.ts`.

### 3. Built-in inventory
File I/O (`fileTools`, `codingTools` opt-in), process (`processTools` — `exec_command` argv-only, no shell; background via `BackgroundCommandManager`), math/data/system/web tools default-on; opt-in: agent memory/meta, browser (`BrowserPort`), computer use (`ComputerUsePort`), media (`MediaToolPorts`: image/TTS/transcription/vision), Home Assistant, RL (`RLBackend`), send-message, skill-manage, Claude-compat tools (`claudeTools/`: subagent ops, MCP lookup, LSP adapter, notebook, remote).

### 4. Host-port pattern
Every privileged capability is an injected interface (`BrowserPort`, `ComputerUsePort`, `MediaToolPorts`, `RLBackend`, `SessionSearchPort`, `PermissionBroker`, `SandboxBackend`, channel dispatchers); no port ⇒ tool isn't registered or errors actionably.

### 5. Notable patterns & risks
Strengths: fail-closed defaults, deny-first evaluation, derived-not-stored deferred schemas, one-directional capability refinement, ReDoS-bounded threat patterns. Observations: `DEFAULT_PERMISSION_MODE` is `accept-all` (documented YOLO default — hosts must opt into stricter modes); no TODO/FIXME markers found; `exec_command`'s no-shell design pushes shell semantics to interpreter invocation, which the approval gate then only sees as e.g. `bash` (acknowledged in the tool description).


===== Explore sessions, memory, context =====

## Persistence & Memory Report (read-only)

**1. Session storage, search, replay, export**

- `xerxes/src/session/store.ts` defines the `SessionStore` interface with three implementations: `InMemorySessionStore`, `FileSessionStore` (JSON-per-session, atomic tmp+rename writes, workspace-scoped subdirectories, `safeSegment` path sanitization), and the preferred `SQLiteSessionStore` (Bun `bun:sqlite`, WAL, `PRAGMA busy_timeout`, additive `user_version` migrations — the comment at line ~210 warns "Never edit an applied entry"). Rows keep the full JSON record so unknown fields survive; `migrateSessionRecord` in `migrations.ts` upgrades on load.
- `models.ts` (`SessionRecord`, `TurnRecord`, `ToolCallRecord`, `CURRENT_SESSION_SCHEMA_VERSION = 1`) is the wire format; `daemonTranscript.ts` defines a second durable format (`DAEMON_SESSION_FORMAT = 'xerxes-daemon-session'`, schema v2).
- Search: `search.ts` has `SessionFTSIndex` (SQLite FTS5 over prompt+response, per-session delete-and-reindex), `SessionIndex` (coupled to the SQLite store in one transaction, optional `Embedder` for hybrid BM25+semantic scoring per `SearchHit`), and `linearSessionSearch` fallback. Malformed FTS syntax degrades to LIKE; real storage errors rethrow. `transcriptSearch.ts` searches daemon transcripts.
- Replay: `replay.ts` `ReplayView` — read-only projection with `getTimeline()` (turn_start/tool_call/turn_end/agent_transition events), agent filtering. `resumeRepair.ts` + `RESUME_REPLAY_SENTINEL` repair interrupted tool calls.
- Export: `runtime/sessionExport.ts` (`EXPORT_SCHEMA = 'xerxes.session.export.v1'`, formats `json|jsonl|md|lovely-pirate`, `SessionExportError`); `snapshots.ts`/`snapshotDiff.ts`/`branching.ts` cover snapshots and branch/diff.

**2. Four-tier memory & retrieval**

Tiers in `xerxes/src/memory/`: `ShortTermMemory` (bounded FIFO working set, capacity 20), `LongTermMemory` (SQLite/`RAGStorage`-backed, 10k cap, 365-day retention, `ownerId` tenancy, batched access-flush writes), `EntityMemory` (regex-based entity/relationship extraction, `MAX_ENTITY_CONTEXTS = 20`), and file-based `AgentMemory` (canonical IDENTITY/SOUL/USER/MEMORY/KNOWLEDGE/INSIGHTS/EXPERIENCES.md, global/project scopes). `ContextualMemory` composes short+long with promotion thresholds; `userMemory.ts`/`agentSelfMemory.ts` add per-user/self tiers. Retrieval: `retrieval.ts` `HybridRetriever` blends `HashEmbedder` cosine (0.55), BM25 (0.3), recency (0.15), with a WeakMap embedding cache. Storage ports: `SimpleStorage`, `NamespacedStorage`, `SQLiteStorage`, `RAGStorage` (`storage.ts`), `vectorStorage.ts`; external providers via explicit `plugins/` registry (mem0, honcho, etc.) — never auto-discovered.

**3. Prompt-injection bounds & compaction**

- Injection bounds: `agentMemory.ts` caps `MAX_MEMORY_FILE_PROMPT_BYTES = 4_000`, `MAX_MEMORY_INDEX_ENTRIES = 48`, `MAX_MEMORY_INDEX_BYTES = 8KB`, total `MAX_MEMORY_SECTION_BYTES = 32KB`; `contextFencing.ts` strips forged `<memory-context>` tags (`sanitizeMemoryContext`) and wraps recall in `MEMORY_CONTEXT_SYSTEM_NOTE`; recalled content passes `scanContextContent` (`security/promptScanner.ts`).
- Compaction (`xerxes/src/context/`): `ContextCompressor` (threshold 0.75 of window, protects first 3/last 6, prunes tool results, folds middle into a typed-flag summary — `COMPACTION_SUMMARY_MARKER` prevents quoted-marker confusion), `CompactionProvisioner` (ratios 0.35/0.5/0.75, synchronous `CompactionModelPort` injection — no provider construction), `compactionStrategies.ts` (six named strategies delegating to the provisioner), plus `toolPairRepair.ts`, `toolResultPruner.ts`, `headroom.ts` (4k-char previews), `SmartTokenCounter`.

**4. Patterns & risks**

Ports-and-adapters everywhere (embedders, model ports, storage, providers injected); immutable records + extra-field preservation; tenant isolation via namespacing/ownerId. Risks: `HashEmbedder` is a bag-of-words hash — weak "semantic" signal; FTS5 reindexes whole sessions per save (write amplification); `FileSessionStore.findSessionPaths` scans directories per lookup; no TODO/FIXME markers found in these trees.


===== Explore daemon, API, ACP, MCP, channels =====

# Protocol Surfaces & Channels — Xerxes Report

## 1. Daemon v35 protocol
`xerxes/src/daemon/server.ts` (`DaemonServer`, line ~653, file is 7235 lines): NDJSON newline-delimited JSON-RPC over a Unix socket, plus an optional WebSocket gateway (`websocketGateway.ts`, bearer-token auth, per-client byte/buffer caps, loopback default). Transports are abstracted behind the tiny `DaemonTransportConnection` interface (`transport.ts`). Unknown methods fail explicitly with `Unknown method: <name>` (server.ts:1864). Key methods (server.ts:1337–1862): `initialize`, `session.open/list/status/usage/title/compress/search/save/undo/delete/most_recent/active_list`, `turn.submit` (alias `prompt`), `turn.background`, `turn.cancel`, `turn.steer`, `cancel_all`, `subagent.retry`, `slash`, `commands.catalog`, `complete`, `set_mode`/`set_plan_mode`, `permission_response`, `question_response`, `fetch_models`, `reasoning_levels`, `provider_list/save/select/delete`, `runtime.status/update_status/reload`, `channel.list/enable/disable`, `terminal.list/inspect/control`, `browser.manage`, `daemon.wipe_memory/history`, `shutdown`. Slash commands dispatched via `HANDLED_CANONICAL_COMMANDS` (~70 commands, server.ts:269). Backpressure is enforced (`maxSocketFrameBytes`, `maxPendingSocketBytes`, 16 MiB frame cap) and dispatch is serialized per connection (server.ts:1286).

## 2. OpenAI-compatible API
`xerxes/src/api-server/`: `server.ts` exposes only `/health`, `/v1/models`, `/v1/chat/completions` (lines 291–297), with SSE streaming. `protocol.ts` (`parseChatCompletionRequest`, `ApiRequestError`) strictly validates the chat-completions subset (temperature 0–2, penalties ±2, tools, tool_choice, stream_options). Optional bearer auth (`/health` exempt), CORS opt-in, sliding-window rate limit (disabled by default), 16 MiB body cap, loopback bind default. `cortexCompletionService.ts` routes models whose id contains `cortex` to the multi-agent backend.

## 3. ACP and MCP
- **ACP** (`xerxes/src/acp/`): Agent Client Protocol server over stdio NDJSON, `protocolVersion '0.9'` (`server.ts`, `ServerCapabilities`: streaming/tools/permissions/fork). `runner.ts` (`AcpAgentRunner`) drives turns through `streaming/loop.ts` `runTurn` with an injected `LlmClient` and `ToolExecutor`; `permissions.ts` (`AcpPermissionBoard`, `routePermission`) brokers tool approvals; `session.ts` (`AcpSessionStore`) tracks sessions.
- **MCP** (`xerxes/src/mcp/`): both sides. Client side: `client.ts` (`MCPClient`), `http.ts` and stdio transports, `manager.ts` (`MCPManager` fleet lifecycle via `MCPClientPort`/`MCPClientFactory`), `reconnect.ts` (backoff with `scrubCredentials`), `oauth.ts`, `osv.ts` (vulnerability scanning). Server side: `server.ts` (`MCPToolServer`) exposes the native `ToolRegistry` as MCP tools over stdio JSON-RPC, sharing the 16 MiB frame cap. Daemon hot-reloads via `reload-mcp`.

## 4. Channel adapters
`xerxes/src/channels/`: uniform `Channel` interface (`base.ts`: `send/start/stop`, `InboundHandler`), normalized `ChannelMessage` (`types.ts`), `ChannelRegistry` failure isolation (`CHANNEL_LIFECYCLE_FAILURE_LIMIT = 100`), host-owned `ChannelManager` (no credentials, throws `ChannelNotConfiguredError`). Adapters: telegram, telegramPolling, discord(+gateway/applications), slack, matrix, mattermost, signal, whatsApp, blueBubbles, twilioSms, emailImap/smtpTransport, dingtalk, feishu, wecom, homeAssistant, genericWebhook. `telegram.ts` (`TelegramChannel extends WebhookChannel`): timing-safe secret-token check, sender allowlists, payload cap, `scanContextContent` prompt scanning, path/traceback redaction. `turnRouter.ts` bridges inbound messages into daemon turns (typing indicators, session reset policy, prompt scanning).

## 5. Patterns & risks
**Patterns:** everything is a narrow injected port (transports, LLM clients, MCP factories, channel adapters); boundary validation everywhere; consistent 16 MiB frame caps across daemon/WS/ACP/MCP; loopback-default binds; explicit errors instead of fake fallbacks. **Risks:** `daemon/server.ts` at 7.2k lines is a god-object (sessions, cron, providers, channels, subagents in one class); API rate limiting is in-memory only (won't survive restart/cluster); bearer auth on API and WS is single static token. No `TODO`/`FIXME` markers found in any of the five directories.


===== Explore cortex, agents, extensions, skills =====

## Xerxes multi-agent & extensions exploration

### 1. Multi-agent topology/orchestration (`xerxes/src/cortex/`)
Everything is port-injected — no LLM client is constructed inside the layer.
- **`CortexOrchestrator`** (`orchestrator.ts`) is a dependency-aware task runner with `CortexProcess` = `SEQUENTIAL`/`PARALLEL`, `failFast`, a finite `DEFAULT_MAX_PARALLEL = 4` cap (unbounded only via explicit `Infinity` opt-in), cancellation via `AbortSignal` (`CortexRunAbortedError`), and `validateTaskGraph` from `task.ts`. Run status is typed: `succeeded | partial | failed`.
- **`Cortex`** (`cortex.ts`, ~1000 lines) adds hierarchical mode via explicit ports — `CortexManagerPlanPort`, `CortexManagerReviewPort` (bounded `maxReviewAttempts`), `CortexManagerSummaryPort` — and consensus mode with `CortexConsensusSynthesizer` plus a `nativeConsensusSynthesis` fallback. Comment notes "No JSON parser or implicit manager LLM".
- **`CortexPlanner`** (`planner.ts`) parses an injected `PlanGenerator`'s XML output (via `xml.ts`) into an `ExecutionPlan` of typed `PlanStep`s with dependencies and complexity, executing in dependency layers.
- **`DynamicCortex` / `DynamicTaskBuilder`** (`dynamic.ts`) build one-off or chained prompt tasks; `taskCreator.ts` decomposes objectives via an injected `TaskCreator`. `agents/` holds `CortexAgent` with delegation port, rate-limit status, and `renderCortexAgentSystemPrompt`.

### 2. Agent spec conventions (`xerxes/src/agents/`)
- YAML specs (`agentSpec.ts`): `loadAgentSpec`/`loadAgentSpecData` resolve `version: 1` documents with `extend` inheritance (cycle-guarded, `INHERIT` sentinel), fields: `name`, `system_prompt`/`system_prompt_path` + `system_prompt_args` templating, `model`, `tools`, `allowed_tools`, `exclude_tools`, `subagents` map, `max_depth` (`Infinity` = unset), `isolation`, `when_to_use`. Tiny custom YAML parser in `yaml.ts`.
- Built-ins in `agents/default/`: `agent.yaml` (root, full tool set incl. `SpawnAgents`, memory tools) + `coder`, `researcher`, `planner`, `reviewer`, `tester`, `objective` subagents, each narrowing tools and excluding delegation (`exclude_tools: AgentTool, SpawnAgents…`) to block recursive fan-out.
- **`SubAgentManager`** (`subagentManager.ts`, 1775 lines) enforces `SUBAGENT_BLOCKED_TOOLS`, `DEFAULT_MAX_SPAWNED_AGENTS = 100`, retained-terminal cap 128, worktree isolation port, persistence (`subagentPersistence.ts`). Plus `AutoCompactAgent`, `CompactionAgent`, `ProfileAgent`, and a capability-based `AgentOrchestrator` with switch triggers.

### 3. Extensions (`xerxes/src/extensions/`)
- **Skills** (`skills.ts`): `SkillRegistry` discovers `SKILL.md` files with strict budgets (`MAX_SKILL_FILE_BYTES` 1 MiB, 32 MiB/pass, depth 32, index ≤16 KiB/128 entries), frontmatter validation (`FORBIDDEN_FRONTMATTER_KEYS`), prompt-injection scanning, workspace trust predicates.
- **Guard/Hub/Sync** (`skillsGuard.ts`, `skillsHub.ts`, `skillsSync.ts`): trusted-repo allowlist (`TRUSTED_REPOS = erfanzar/xerxes`), hash pinning, quarantine dir, path-escape rejection, lock file + audit log. `skillSources/` supports local/GitHub/agentskills.io/official origins.
- **Plugins** (`plugins.ts`): typed `PluginType` (tool/hook/provider/channel/search/speech), `PluginRegistry` with conflict errors, dependency resolution (`dependency.ts` semver-ish `VersionConstraint`, circular detection), dynamic provider factories validated via `isPluginLlmProviderFactory`.
- **Hooks** (`hooks.ts`): 9 hook points; mutation hooks thread values, `tool_permission_check` is fail-closed. **Slash plugins** (`slashPlugins.ts`) register commands in a registry isolated from built-ins.
- **Skill authoring** (`skillAuthoring/`): full pipeline — tracker → trigger → drafter/proposal → verifier → persist via `SkillProposalStorePort`; telemetry and observer ports.

### 4. Bundled skills inventory (`xerxes/skills/`)
49 directories, each a SKILL.md bundle, grouped roughly: dev workflow (`plan`, `systematic-debugging`, `software-development`, `deepscan`, `bug-bounty-hunter`, `github`, `evaluation`), delegation to external CLIs (`claude-code`, `codex`, `opencode`, `xerxes-agent`, `native-mcp`, `mcporter`), ML/training (`grpo-rl-training`, `huggingface-hub`, `pallas-kernel`, `training`, `inference`, `models`), media/creative (`ascii-art`, `ascii-video`, `manim-video`, `p5js`, `excalidraw`, `songwriting-and-ai-music`, `youtube-content`), integrations (`himalaya`, `imessage`, `apple-notes`, `apple-reminders`, `google-workspace`, `openhue`, `polymarket`, `webhook-subscriptions`), research (`arxiv`, `research`, `autoresearch`, `ocr-and-documents`), misc (`minecraft-modpack-server`, `pokemon-player`, `findmy`, `cloud`).

### 5. Patterns & risks
- **Patterns:** every external capability is an injected port (executor, plan/review/summary, trust, store); typed discriminated statuses; immutable/frozen specs; explicit finite caps with `Infinity` opt-in; fail-closed permission hooks.
- **Risks/notes:** no TODO/FIXME found in these dirs. `subagentManager.ts` at ~1.8k lines is a hotspot; the hand-rolled YAML parser (`agents/yaml.ts`) is a maintenance/security surface; `TRUSTED_REPOS` is a single hardcoded repo; `skillsHub` audit/lock assumes single-writer; unbounded fan-out possible via explicit `Infinity` opt-in (documented but worth monitoring). Read-only; nothing changed.


===== Explore the TUI layer =====

# Xerxes TUI Exploration Report

## 1. Architecture
React + OpenTUI (`@opentui/react`, jsxImportSource pragma) rendered natively in the terminal. Entry: `xerxes/src/ui/opentui/entry.tsx` (process/TTY lifecycle, early-input capture, graceful exit, memory monitor, `rendererSingleton.ts`). App root: `opentui/app.tsx` → `AppOpenTui` wraps `useMainApp(gw)` output in `GatewayProvider` → `AppLayout` (`opentui/appLayout.tsx`).

The daemon seam is `gatewayClient.ts` — `GatewayClient` speaks newline-delimited JSON-RPC 2.0 over a per-project Unix socket (`$XERXES_HOME/daemon/projects/<sha256(project)[:16]>.sock`), spawning the Bun daemon when unreachable. Contract is frozen in `PROTOCOL.md` (frame cap `MAX_GATEWAY_FRAME_BYTES` = 16 MiB; startup/RPC timeouts env-tunable; LRU session-key map, `MAX_SESSION_KEYS = 200`). `gatewayAdapter.ts` + `gatewayTypes.ts` normalize daemon events (snake_case ⇄ PascalCase asserted by contract tests).

State is **nanostores** atoms, not React context: `$uiState`/`$uiTheme` (`app/uiStore.ts`), `$overlayState` (`app/overlayStore.ts`), `turnStore.ts`, `$spawnHistory`, `$thinkingVisibility`, `$attachments`, `panelSizeStore.ts`.

## 2. Components/hooks and state flow
`app/useMainApp.ts` (~1300 lines) is the controller, composing hooks: `useComposerState`, `useInputHandlers`, `useSubmission`, `useSessionLifecycle`, `useConfigSync`, `useLongRunToolCharms`, plus `hooks/useVirtualHistory.ts` (windowed transcript + `lib/virtualHeights.ts` estimators) and `hooks/useCompletion.ts`. Daemon events flow through `app/createGatewayEventHandler.ts` into stores; `turnController.ts`/`turnStore.ts` track live turns. Views: `appLayout.tsx` (scrollbox transcript, `<textarea>` composer, chrome), `appChrome.tsx`, `messageLine.tsx`/`StreamingMarkdown`, panels: `agentPanel.tsx` (F6), `terminalPanel.tsx`, `diffPanel.tsx`, pickers (`modelPicker`, `sessionPicker`, `reasoningPicker`, `copyPicker`).

## 3. Slash commands, completion, overlays
Registry: `app/slash/registry.ts` (`SLASH_COMMANDS`, `findSlashCommand`) aggregating command modules (`core/session/ops/setup/debug/attach/credits/maintenance/agentRetry` — `agentRetry` shadows `/agents` last). Dispatch: `app/createSlashHandler.ts` with flight-number + sid **stale guards** so async results can't mutate a session you switched away from; unknown names resolve via daemon skill catalog with ambiguity reporting. Completion: `hooks/useCompletion.ts` (`slashCompletionsFromCatalog`, path/at-mention completion, `rankCompletionItems`) rendered by `completionMenu.tsx`; `domain/slash.ts` `completionToApplyOnSubmit`. Overlays are boolean flags in `$overlayState` (never replace the transcript; `$isBlocked` gates input; `OVERLAY_BLOCKS_BACKGROUND_HOTKEYS` is a compile-enforced per-overlay hotkey policy, `agents:false` so F6 toggles). Transcript survives cancel/error because pickers render on top of the mounted scrollbox.

## 4. New uncommitted pieces
- `lib/toolRun.ts` + `app/toolRunStore.ts`: fold runs of ≥4 (`TOOL_RUN_MIN`) consecutive successful tool-trail lines into one summary (tally, slowest call, duration); failures/in-flight never fold; per-run expand state survives virtualization (`toggleToolRun`, `toolRunExpanded`); consumed in `messageLine.tsx` (`groupToolRun`).
- `lib/agentGroups.ts`: order agent panel RUNNING→FAILED→DONE with recency + spawn-index tiebreak (`orderAgentRecords`, `agentHeading`); used by `agentPanel.tsx`.
- `lib/terminalGroups.ts`: same idea for terminals (`orderTerminals`, `terminalHeading`, null exitCode = failed); used by `terminalPanel.tsx`.
- `lib/diffFiles.ts`: file index over parsed diff (`indexDiffFiles`, `fileAtRow`) for jump-to-file in `diffPanel.tsx`.
- `__tests__/toolLeaders.test.ts` covers `toolLeaderDots` in `messageLine.tsx` (dot leaders filling to `contentColumnWidth(cols) − 7`).

## 5. Testing approach & risks
Vitest (`test:ui` = `vitest run --config vitest.ui.config.ts`), ~60 files in `__tests__/` plus co-located `lib/*.test.ts`; behavior-focused (lifecycle, overlay restoration, gateway resume, leader-width math). Type gate: `check:ui` (tsc). **Risks:** no TODOs/FIXMEs found; `gatewayClient.ts` is large (~1991 lines) with sync `execFileSync git` at construction; new grouping/collapse work is uncommitted and unverified by me (I ran no tests); overlay hotkey policy relies on manual `OVERLAY_FLAG_KEYS` list (compile-checked via `satisfies`, but flag ordering is by hand).


===== Explore tests, build, docs, playground =====

# Xerxes Repo Survey — Tests, Build, Docs, Tooling

## 1. Test organization & coverage themes

**`xerxes/test/`** (~270 files, `bun test ./test` via `test:runtime`) is flat, one file per subsystem. Theme clusters:
- **Streaming/loop:** `loop.test.ts`, `loopHardening`, `loopParallelTools`, `loopRetry`, `streamingLoopParity`, `streamingWireEvents`, `interrupt`, `turnInterruptSubagents`
- **Daemon/protocol:** `daemonServer`, `daemonInteractions`, `daemonAutoCompact`, `daemonTranscript`, `websocketGateway`, plus several `daemon*Parity` contract tests
- **Memory (~20 files):** `memoryCore`, `memoryEntity`, `vectorStorage`, `memoryStorageRobustness`, many `memory*Parity` tests
- **Security:** `security*`, `sandbox*`, `permissions`, `audit`, `redactionWindows`, `channelWebhookSecurity`
- **Providers/LLMs:** `anthropic`, `gemini`, `ollama`, `llmClient`, `responsesApi*`, `providerRegistry`
- **CLI:** `cliFlags`, `cliOneShot`, `cliResume`, `cliExport`, `cliTelegram`, `cliUpdate`, `doctor`
- **Channels:** `channelAdapters`, `discordGateway`, `telegramPolling`, `emailImap`, etc.
- **Skills/extensions:** `skills*`, `skillAuthoring*`, `bundledSkill*` (asset safety, CLI, scripts)
- **Playground:** `playgroundCli`, `playgroundEval`, `playgroundHardCatalog` + fixture `playgroundTransportFixture.ts`
- **Meta-tests:** `bunOnlyRepository.test.ts`, `installerParity`, `releasePackage`, `distribution`, `containerDistribution`, `maintenanceScripts`

Legacy naming survives: `*PythonParity` tests (`bridgePythonParity`, `sessionPythonParity`, `toolsPythonParity`, `core*Parity`) and two non-test spike scripts (`heartbeat-spike.ts`, `heartbeat-spike2.ts`) sitting in the test dir.

**TUI tests** (`xerxes/src/ui/__tests__/`, ~62 files) run under **vitest** (`xerxes/vitest.ui.config.ts`, `test:ui`), covering gateway client lifecycle, session tabs/pickers, composer/paste, streaming markdown, virtual history, theme, OpenTUI parity.

## 2. Build/bundling pipeline

- `xerxes/package.json` `build` = `build:runtime` + `build:ui`.
- `build:runtime`: `bun build src/cli.ts --target=bun --minify --outdir dist`, then **`xerxes/scripts/copyBundledSkills.ts`** copies `xerxes/skills/` → `dist/skills` and `src/agents/default` → `dist/default`, rejecting symlinks, non-regular files, and executable bits.
- `build:ui`: **`xerxes/scripts/buildTui.ts`** bundles `src/ui/opentui/entry.tsx` → `dist/ui/entry.js` (ESM, minified, react-devtools stubbed, shebang stripped, `createRequire` banner). React, react-reconciler, and `@opentui/*` stay **external** — the artifact requires `node_modules` at runtime (documented in comments); **`verifyTuiBuild.ts`** validates the output.
- Release tooling: `xerxes/scripts/releasePackage.ts` (1400 lines; `check`/`prepare`/`publish-dry-run`, package format v2, min Bun 1.3.12, non-redistributable skill exclusions) + `smokeReleasePackage.ts`, `smokeTuiGateway.ts`, `smokeTuiComplete.ts`, `realUseCheck.ts`, `swarmIntegration.ts`, `assertBunOnly.ts` (repo guard). Root `scripts/` has only `install.sh`/`install.ps1`.
- **tsconfigs:** `xerxes/tsconfig.json` (strict + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`, bundler resolution, excludes `src/ui`); `xerxes/tsconfig.ui.json` (nodenext, react-jsx, scoped to `src/ui`); `examples/tsconfig.json`.

## 3. Playground evaluation framework

`xerxes/playground/` (`cli.ts`, `harness.ts`, `evaluator.ts`, `isolation.ts`, `warmup.ts`, `hard.ts`, `hardTasks.ts`, `hardCli.ts`, `daemonTransport.ts`, `types.ts`, README.md) is the Bun-native replacement for a retired Python eval harness. It creates private per-run home/workspace dirs without touching `XERXES_HOME` or discovering credentials; the host must inject an `EvaluationSessionPort` transport (e.g. `DaemonEvaluationSessionPort`) and optionally an `EvaluationJudgePort`. Two suites: 8-task warmup and a 16-task typed hard catalog with watchdog-bounded behavioral graders. Entry points: `playground:warmup`, `playground:hard`.

## 4. Docs & examples

`docs/` holds ~20 Markdown guides (index, system-architecture, configuration-guide, testing-guide, deployment-guide, openclaw_parity, telegram-gateway, etc.) plus generated **`docs/_bun/`** HTML (site + `typescript-api/` per-module API pages) built by `bun ./src/docs/cli.ts` (`docs:build`); API docs from `src/maintenance/apiDocsGenerator.ts`. Sphinx-era `Makefile`/`make.bat` linger. `examples/` has 11 TypeScript demos (4 scenario files, cortex/deepsearch demos, interactive agent, textual_tui) + README + tsconfig, covered by `xerxes/test/rootExamples.test.ts`.

## 5. Risks / notes

- TUI bundle is **not self-contained** — needs node_modules at runtime (by design; `bun build --compile` is the single-file path).
- `docs/Makefile`/`make.bat` are Sphinx leftovers in a Bun-docs pipeline — possible stale references.
- Non-test `heartbeat-spike*.ts` files inside `xerxes/test/` may be picked up by test globs.
- Extensive `*PythonParity` naming implies a completed Python→Bun migration whose parity scaffolding may be prunable. No TODO/FIXME found in `xerxes/scripts` or `xerxes/playground`.

No files were modified.