# Xerxes Bun / TypeScript Rewrite

## Status

Started 2026-07-13. The target is a Bun-native TypeScript runtime with feature and wire-protocol parity for Xerxes' current public behavior. This is a replacement project, not a line-by-line transliteration and not a Python-to-Node bridge.

### 2026-07-15 production-readiness checkpoint

The migration is **complete and production-gated**. Xerxes now has a native Bun/TypeScript runtime,
daemon, API, channels, browser/CDP integration, CLI, release packaging, and a sole OpenTUI renderer.
The retired `src/python/`, Python test tree, Python packaging and CI configuration, virtual environment,
bytecode/caches, old Python distribution artifacts, legacy renderer, engine switch, and duplicate TUI
bundle have all been removed. A strict filesystem scan reports zero repository Python source, bytecode,
stub, or extension files (`.py`, `.pyc`, `.pyo`, `.pyi`, `.pyx`).

The current verification gate is:

1. `bun install --frozen-lockfile`;
2. `bun run check` (including the Bun-only repository guard);
3. `bun run test` — 987 runtime tests / 6,292 assertions and 421 OpenTUI tests;
4. `bun run build` and the generated OpenTUI bundle smoke check;
5. staged release-package integrity and archive smoke checks;
6. native CLI, gateway, completion, and real-use smoke checks; and
7. `git diff --check` plus a staged-index safety audit.

A real-provider regression matrix also exercises arithmetic without tools or durable-memory pollution,
exact file reads, missing-file recovery, bundled skill activation, safe and approval-gated direct argv,
two concurrent titled agents across runtime reloads, agent usage/file metadata, exact session usage,
resume whitespace, and streaming overlap suppression. The 28 production assertions pass with the
configured `kimi-for-coding` model. The harness fixes discovered by that matrix make unavailable tools
explicit, keep tool schemas and bootstrap guidance aligned, preserve subagents across reloads, persist
approvals safely, cancel disconnected turns, and report complete session usage.

The OpenTUI client now includes the Xerxes Derafsh Kaviani mark with a faster blue-gold-purple wave,
Grok-inspired transcript and tool presentation, and a responsive right-side agent rail that reports each
short-titled agent's state, parentage, rules, files, tool calls, API calls, and token usage. Narrow terminals
retain the transcript and expose the same information through the agents overlay. Browser automation
attaches only to an explicitly supplied Chromium CDP endpoint. Docker itself was not available locally,
so its build remains CI-verified; the Docker runtime layout and release package have native tests.

Historical checkpoints below are retained as contemporaneous migration evidence.

### 2026-07-13 historical active-cutover checkpoint

The native tree now contains 416 TypeScript runtime source files (103,533 lines), 216 Bun test files (32,879 lines), and 11 root TypeScript examples. The remaining non-cache Python inventory is 743 files / 170,427 lines: 405 runtime files, 333 test files, and five playground files. This is an active migration count, not an estimate of feature completeness.

This wave adds or hardens native parity for daemon command/session persistence, bridge metadata and cancellation, LLM request compatibility fields, session indexing failure isolation, streaming steer/subagent-event behavior, context, memory persistence/retrieval/profiles/injection bounds, operator host ports and PTY timing, GRPO training host ports, and a defensive safety-evaluation skill. It also replaces the Sphinx/docstring/header-maintenance flow, root examples, installer, README entrypoints, Bun CI/release paths, and several completed root scripts with native Bun implementations.

The documentation cutover now uses Markdown handwritten sources and generated TypeScript API pages only. The 215 retired RST/Sphinx API stubs have been removed; the native builder rejects the retired compatibility configuration and does not render RST input. Its focused regression suite verifies Markdown rendering, TypeScript API generation, owned-output cleanup, ignored RST sources, and configuration rejection.

At that checkpoint, the active tree passed `bun run check` and a serial `bun run test`: 845 Bun runtime tests / 5,368 assertions and 431 legacy UI tests, with one explicit skip and one todo. The then-current production checks passed for the runtime bundle, legacy-renderer bundle, and generated TUI bundle smoke (`1,143,255` bytes); `bun run xerxes --help` and `bun run xerxes doctor` also passed. Doctor reported only the expected local setup warnings when no provider key or global `xerxes` launcher was configured. The root `build` wrapper still had to be run only when its generated TUI artifact was safe to replace.

The latest runtime hardening records provider-requested tools outside the configured model-visible surface through an explicit `onUnconfiguredToolCalls` dependency. Cortex agents stop the turn and return a typed configuration error rather than executing or silently treating such a request as a successful text-only response. Root examples now typecheck with the workspace-pinned TypeScript compiler rather than resolving an ambient latest compiler, and the conversational example has an explicit self-referential assistant type.

### 2026-07-14 migration progress

The Python inventory is now 622 files / 151,509 lines across `src/python` and `tests`; `playground/` contains zero Python files. This is down from the prior 743-file / 170,427-line checkpoint after verified cleanup batches: fully native cron/logging source and dedicated tests; 48 byte-identical root-level Python test copies whose canonical namespaced tests remain; ACP/training leaf packages plus their dedicated Python tests; the fully native OpenAI-compatible API-server package and its dedicated tests; dependency-free `core.basics` / `core.config` leaves with their dedicated tests; a full native warm-up/hard playground with 16 typed Bun hard-task checkers; and nine individually proven runtime leaves (streaming debug, audit OTLP, memory fencing/vector storage, context result storage, session snapshot diff, and channel identity/reset/sticker state). ACP/training deletion was preceded by zero-external-import scans and 25 focused Bun tests with 139 expectations; API-server deletion has 16 focused Bun tests with 82 expectations; core leaves have focused native tests and zero remaining Python imports; playground deletion has targeted native CLI/catalog tests and zero-reference scans; the leaf sweep has 35 focused Bun tests with 278 expectations. These deletions do not establish whole-runtime cutover; remaining Python entrypoints are intentionally not used as validation gates.

Native distribution now makes both advertised executable contracts real. The Bun installer and staged release package ship `xerxes` and `xerxes-acp`; the ACP launcher forwards arguments into the native CLI. Native ACP now accepts `--write-registry`, `--permission-mode`, and `--project-dir`, matching the public Python ACP command surface where implemented. A resumed one-shot command (`xerxes --resume <id> <prompt>`) now stays native and persists its result, while bare resume remains interactive. The native daemon now exposes tested `/history`, `/snapshot`, `/snapshots`, `/rollback`, and read-only `/cron list` handlers rather than advertising them without execution. The Python-only CI/release workflows have been replaced by existing Bun workflows, and the Dockerfile now targets the Bun bundle; Docker was unavailable locally, so the image build remains a CI verification item. The full Bun check/test/build gate must be rerun after each subsequent implementation batch.

The migration is still incomplete. The next implementation priority is the daemon/bridge command-control plane and published executable behavior, followed by browser/computer-use, persistent channel gateways, bundled skill command rewiring, playground hard-task conversion, remaining LLM lifecycle parity, and removal of legacy Python packaging/CI/tooling only once their Bun replacements are executable.

Completed Python paths are being removed only after their native replacements and targeted tests pass: the old Sphinx configuration/dependencies, ten root examples, six maintenance scripts, the old swarm driver, and their direct Python tests. Python remains the oracle for every uncompleted subsystem. The repository is **not** cut over and must not be described as complete until the inventory reaches zero and the final Bun-only gate passes.

### Active OpenTUI visual redesign

The Bun OpenTUI client now uses a sparse, keyboard-first, dark terminal composition inspired only by the high-level interaction structure of the public Grok CLI/Grok Build: a slim workspace breadcrumb, quiet canvas, centered welcome card, low contextual tip, full-width framed composer, and restrained chrome. It uses original Xerxes copy and the supplied Xerxes Derafsh Kaviani Braille-pixel terminal glyph verbatim; it does not copy xAI, SpaceX, or Grok logos, artwork, or product text.

The implementation preserves native v35 gateway behavior, transcript virtualization, keyboard/mouse input, screen-width fallbacks, light-mode readability, screen-reader-safe plain text, and existing approval/clarification/session overlays. The centered welcome screen is startup-only, so active and resumed transcripts retain their existing virtualized rendering. Slash-only startup actions now remain transient: `[intro, slash]` keeps the welcome behind the provider-profile and model pickers, while the command echo appears normally once a real user or system result exists. It adds responsive framed/compact/narrow layouts, path-boundary-safe home shortening, explicit keyboard hints, and a charcoal/amber dark palette. The production builder now writes the entry-point artifact returned by `Bun.build` rather than mistakenly validating a stale output; it supports `XERXES_TUI_BUILD_OUT` for isolated build checks and asserts the startup welcome content. Component regressions cover the mark, responsive width decisions, metadata, status identity, home-path safety, startup slash-command state, and real artifact output. UI type checking, all UI tests, the sole OpenTUI build, generated-bundle smoke check, and an isolated live 180-column `bun run xerxes` terminal launch plus `/provider` prompt check pass.

### 2026-07-13 native expansion checkpoint

The current native tree contains 398 TypeScript runtime source files (99,007 lines), two Bun integration drivers (1,327 lines), and 178 Bun test files (26,599 lines). This checkpoint adds the following fully native, independently tested slices:

- Bridge server/session command surfaces; OpenAI-protocol and function-execution type contracts; agent compaction/profile/subagent support.
- Cortex core helpers, task agent, and the sequential, parallel, hierarchical, consensus, and planned topology engine.
- A bounded diagnostic streaming loop with explicit cancellation, terminal-error, no-retry, and unavailable-tool behavior.
- Bundled native skill modules for arXiv, Excalidraw, nearby search, Polymarket, YouTube transcripts, Google Workspace OAuth/API access, OCR/document extraction, and OOXML PowerPoint manipulation.
- Native maintenance CLIs for headers and docstrings; TypeScript JSDoc/API documentation generators; a deterministic native swarm driver; and an opt-in real-use checker.

Verified focused gates at this checkpoint include strict `bunx tsc --noEmit -p tsconfig.json`; 21 Cortex tests / 108 assertions; 16 bundled-skill tests / 109 assertions; 15 maintenance/docs/real-use tests / 117 assertions; a native swarm driver with 18 reported checks; and the real-use CLI with 14 passed local checks, three explicit external skips, and zero failures. These are additive migration checkpoints, not evidence of cutover.

At this historical checkpoint, Python was deliberately preserved as the behavioral oracle: 408 runtime files / 114,874 lines under `src/python/xerxes/`, and 768 Python files / 179,169 lines across the repository. Subsequent verified slices have removed completed Python paths; see the active cutover checkpoint above.

### Historical implementation status

At that earlier checkpoint, the executable Bun runtime lived in xerxes/ with strict type checking, a bundled CLI, and contract/integration coverage. Before the expansion checkpoint above, it contained 316 TypeScript source files (79,369 lines) and 153 TypeScript test files (21,463 lines). The last complete root verification before the newer additive slices completed 581 Bun runtime tests with 3,221 assertions, strict TypeScript checks for both packages, both production builds, and the generated legacy UI bundle smoke check; the legacy UI suite also passed 377 tests with one intentional skip. A new complete root gate was required after each expansion wave.

Implemented slices include provider routing plus OpenAI-compatible and direct Anthropic SSE; portable streaming, permission, tool, objective-guard, interaction-mode, interrupt, nudge, auxiliary-model contracts, core registries, and async stream buffering; v35 Unix-socket NDJSON daemon and native WebSocket transport with persisted Python-readable transcripts; OpenAI-compatible HTTP chat completions; the Xerxes embedded facade; native filesystem/process/data/system/math/web tools with a bounded output cache; policy/path/URL/prompt security; an allow-listed Bun subprocess sandbox; agent-spec loading and routing; MCP stdio, legacy SSE, Streamable HTTP, OSV package checks, native tool-registry integration, and stdio server; ACP stdio server/runner; session search/replay/branching/snapshots/export/workspace identities; skills/plugins/hooks, guarded skill hub, manifest sync, slash-plugin registry, and the skill-authoring pipeline; operator managers; four-tier memory foundations, an explicit persistence-compatible memory facade, external-memory provider registry, and fenced recalled-memory context; context compression, headroom, window accounting, and repository mapping; audit; console/structured logging; OAuth credentials; cron; Cortex orchestration; training utilities; setup wizard and doctor diagnostics; Bun distribution/install/update planning; and bridge command metadata.

At that time, the legacy TUI launched the Bun daemon for the supported v35 Unix path, with daemon-launch and prompt-turn smoke coverage. Its dev, start, bundle, legacy-renderer build, typecheck, and test lifecycle entrypoints ran through Bun rather than npm, Node, tsx, or esbuild commands. Native daemon controls covered implemented slash dispatch/completion, steering, approval/question replies, provider profile mutation/model discovery, reload/compaction, configured WebSocket startup, persisted-session export, and doctor diagnostics. The canonical command registry advertised only commands that had real daemon handlers; unavailable commands returned explicit unsupported results instead of simulated success. Persistent agent memory was a Bun-native global/project store with atomic writes, append locking, prompt injection, and the declared agent-memory tools. Safe static web tools, explicit Google/DuckDuckGo provider adapters, opt-in image/TTS/transcription/vision ports, workspace memory CRUD, and channel message dispatch were present.

Channels now have a configuration-driven lifecycle, inbound turn router, Telegram long polling, webhook hosting, durable identity resolution/hashing, reset policies, channel Markdown workspaces/import, installation-scoped OAuth, sticker-cache persistence, and an Email adapter with explicit injected IMAP/SMTP transports. Several provider-specific gateway transports remain relay-only or host-owned by design; they are not falsely represented as direct Bun implementations.

The latest native slices add a composed runtime-session store (transcript/history/cost tracking, explicit context/filesystem ports, bounded durable records); active bootstrap-system-prompt injection; prompt profiles and context builders; runtime feature composition; resumable bridge-session projections; deterministic stream tool IDs, SSE helpers, shared Anthropic prompt caching, and a direct Ollama `/api/chat` NDJSON client selected by the regular client factory. They also add a focused streaming wire-event codec, a host-port bridge slash router, SQLite vector storage, external-memory plugin adapters, local/GitHub/official skill sources, daemon skill creation, approvals/redaction and sandbox credential/file-sync helpers, and browser-provider contracts. The standalone wire-event codec, bridge slash router, memory plugins, and configurable runtime-prompt/feature APIs are intentionally not described as daemon-integrated where that wiring has not been completed.

The runtime is deliberately **not** declared cut over. Python still occupies 408 files and 114,874 lines under src/python/, and the repository currently contains 751 non-cache `.py` files / 172,412 lines. The user's requested no-Python end state has therefore not been reached. The largest remaining work is direct browser/desktop automation adapters; persistent provider-specific channel gateways (including Discord gateway, Matrix sync, Signal, and direct email sockets); full bridge/slash handler and command parity; container sandbox parity; production DNS-pinning and live-provider evaluation; full TUI/CLI feature parity; translation or replacement of the remaining Python behavioral tests and bundled Python skill scripts; the remaining real-use/playground paths; package metadata conversion; and finally deletion of every Python source path. Do not claim the rewrite complete until Phase 5's criteria are met.

## Historical baseline

- Python runtime: 408 files and 114,874 lines under `src/python/xerxes/`.
- Entire repository: 768 Python files and 179,169 Python lines, including Python tests, tooling, and bundled skill scripts.
- Tests: 301 Python test files, distributed across every runtime subsystem.
- Existing TypeScript UI at the historical baseline: owned the legacy renderer and had Bun lifecycle entrypoints; it spoke newline-delimited JSON-RPC to the Bun daemon on the supported v35 Unix path.
- Compatibility contracts to preserve first: CLI behavior, profile/session locations, daemon JSON-RPC protocol v35, WebSocket events, OpenAI-compatible HTTP API, MCP transports, and YAML agent/skill formats.

## Target layout

```text
xerxes/
├── package.json                 # Bun scripts and package metadata
├── tsconfig.json                # strict ESNext compiler settings
├── src/
│   ├── cli.ts                   # `xerxes` command modes
│   ├── index.ts                 # public SDK exports
│   ├── core/                    # errors, config, paths, validation
│   ├── types/                   # messages, tools, agent contracts
│   ├── llms/                    # provider registry and adapters
│   ├── streaming/               # async event-driven agent loop
│   ├── runtime/                 # query engine, budgets, pricing
│   ├── tools/                   # tool registry and implementations
│   ├── security/                # policy, approvals, sandbox routing
│   ├── session/                 # SQLite storage, FTS, replay
│   ├── daemon/                  # JSON-RPC Unix socket/WebSocket server
│   ├── api-server/              # OpenAI-compatible HTTP server
│   ├── mcp/                     # stdio, legacy SSE, and Streamable HTTP MCP
│   ├── channels/                # chat adapters
│   └── extensions/              # plugins, skills, hooks
└── test/                        # Bun tests, contract fixtures, integration tests
```

At that historical checkpoint, the terminal client remained a separate workspace. Its gateway client launched the Bun daemon for supported v35 Unix-socket flows; unported RPC controls were called out in compatibility notes rather than routed through Python.

## Architectural decisions

1. Use native async iterators end-to-end. Python's synchronous generator loop becomes an `AsyncGenerator<StreamEvent>`; tool execution and permission resolution remain sequential by default.
2. Treat every external format as a contract. Provider inputs, daemon frames, OpenAI API responses, persisted sessions, YAML agent specs, and tool schemas get fixtures before their producing subsystem is changed.
3. Use Bun primitives where they remove adapters: `Bun.serve`, `Bun.spawn`, `Bun.file`, `Bun.sqlite`, `Bun.password`, native WebSocket support, and `fetch`. Keep framework dependencies out of the core runtime.
4. Use explicit ports for provider, tool, storage, policy, and channel integrations. A port is an interface such as:

```ts
export interface LlmClient {
  stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<LlmDelta>
}

export interface ToolExecutor {
  execute(call: ToolCall, context: ToolContext, signal?: AbortSignal): Promise<ToolResult>
}
```

5. Do not preserve Python-only internal class hierarchies or compatibility shims. Preserve observable behavior; redesign internals around typed discriminated unions and composition.

## Historical phases

### 0. Freeze behavior and create parity fixtures

- Record protocol v35 request/response/event fixtures from the existing daemon.
- Export representative session JSON and agent YAML fixtures without credentials.
- Translate high-value unit tests for provider routing, message conversion, thinking parsing, permission gates, loop detection, and session persistence.
- Define feature scorecards by subsystem; no Python subsystem is removed merely because a TypeScript directory exists.

### 1. Protocol, configuration, and daemon foundation (substantially implemented)

- Create the Bun package, strict TypeScript settings, public types, v35 NDJSON JSON-RPC framing, daemon socket paths, and a replaceable daemon runtime/turn-runner seam.
- Preserve the then-existing legacy client unchanged while `initialize`, `session.*`, `turn.*`, and `runtime.status` are proven against protocol fixtures.
- Implement a versioned dual-reader for the daemon transcript JSON and structured session records before switching persistence.
- Switch the legacy gateway client to launch the Bun daemon for the supported v35 Unix path. **Implemented at that checkpoint:** Bun binary/entry overrides, live launch/status smoke tests, and prompt streaming. Unsupported daemon RPCs remained explicitly surfaced rather than falling back to Python.

### 2. Agent runtime, tools, and security (in progress)

- Port typed errors, provider registry, streaming event types, thinking parser, tool registry, permission broker, loop detector, and agent loop.
- Implement OpenAI-compatible SSE streaming first, then direct Anthropic request/stream conversion.
- Port provider detection, model routing, context limits, costs, Kimi Code headers, request retry semantics, and tool-pair cancellation repair exactly.
- Port the function registry/executor, policy engine, Bun subprocess/container sandbox router, and tools by capability group. **Implemented so far:** filesystem/process/data/system/math/web tools, a bounded read-only tool-output cache, persistent agent-memory tools, workspace-memory CRUD, send-message dispatch, opt-in media/voice/vision ports, Claude-compatible adapters, and operator foundations. Browser/computer-use, advanced AI/media, Home Assistant, and RL surfaces are explicit host-owned ports rather than fabricated local implementations; direct browser/desktop adapter parity remains open.

### 3. State, services, and integrations (in progress)

- Port SQLite session storage and FTS, transcript compaction, replay/branching/snapshots, profile storage, memory tiers, and audit events.
- Port the HTTP API and ACP server with the same OpenAI-compatible and JSON-RPC response shapes.
- Port browser, MCP, media, agent-delegation, plugins, skills, hooks, and all channel adapters after the core executor is stable. **Implemented so far:** MCP stdio plus legacy SSE and Streamable HTTP transports, OSV install vetting, native MCP-to-tool-registry integration, spawned-agent manager, plugin/skill hooks/hub/guard/sync, generic webhook plus primary and relay channel adapters, native WebSocket transport, channel OAuth/workspaces, and injectable media/voice/vision providers. Browser/computer-use and persistent gateway parity remain open.

### 4. Extensions and integrations (in progress)

- Port YAML agent loading, plugins, skills, hooks, authoring pipeline, cron, training, OAuth, bridge profiles, and all 14 channel adapters. **Implemented so far:** YAML agents, plugins/hooks/slash plugins, guarded skill installation/sync, cron, training, OAuth, profiles, Cortex, and partial channel coverage; all 14 adapters still require their own completed status and integration fixtures.
- Port Cortex orchestration after the core executor and spawned-agent protocol are stable.
- Replace Python-only docs/build/install flows with Bun equivalents.

### 5. Cutover

- Run the Bun and Python implementations against the same contract suite and live-provider opt-in evals.
- Make Bun the sole `xerxes` executable and remove the Python implementation in one intentional compatibility-breaking release.
- Update CI to run `bun test`, `bunx tsc --noEmit`, formatting, protocol integration, and opt-in provider evals.

## Translation map

| Python area | Bun/TypeScript destination | Migration dependency |
| --- | --- | --- |
| `core`, `types` | `core`, `types` | none |
| `daemon`, `bridge`, daemon config | `protocol`, `daemon` | core/types |
| `llms`, `streaming`, `runtime` | `llms`, `streaming`, `runtime` | protocol/core/types |
| `executors`, `tools`, `operators`, `security` | matching packages | runtime |
| `session`, `memory`, `context`, `audit` | matching packages | daemon/runtime |
| `api_server`, `acp` | matching packages | daemon/session/tools |
| `agents`, `extensions`, `mcp`, `cortex`, `cron` | matching packages | daemon/tools |
| `channels`, `auth`, `training` | matching packages | daemon/extensions |

## Historical initial runtime invariants

```ts
export type StreamEvent =
  | { type: 'text'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_start'; call: ToolCall }
  | { type: 'permission_request'; request: PermissionRequest }
  | { type: 'tool_end'; result: ToolResult }
  | { type: 'turn_done'; usage: TokenUsage }

export async function* runTurn(
  request: TurnRequest,
  dependencies: TurnDependencies,
): AsyncGenerator<StreamEvent> {
  // The provider delta stream is normalized before it reaches tools/UI.
  // Every requested tool receives exactly one result, including cancellation.
}
```

The event schema intentionally made every stream event serializable, so the in-process runtime, daemon, API, MCP, channels, and then-current TUI all consumed one vocabulary.

## Exit criteria per subsystem

1. The Bun implementation has strict type checking and unit tests.
2. Contract fixtures match the Python implementation where a public format is retained.
3. Integration tests cover cancellation, provider errors, permission denial, and persistence boundaries.
4. The matching Python tests have been translated or replaced with stronger contract coverage.
5. The subsystem scorecard is marked complete only when its consumers no longer import or launch Python.

## Risk controls

- Delete an individual Python path only after its native replacement, targeted TypeScript check/test, and consuming entrypoint validation pass. Preserve every uncompleted path as a behavioral oracle.
- Never reuse credentials or recorded private sessions in fixtures.
- Keep wire protocol v35 stable through the daemon cutover; a v36 change requires an explicit compatibility decision.
- Port expensive/remote adapters last and gate their live tests behind explicit credentials.
