# Xerxes — Project Overview & Product Design Record

## What is Xerxes?

Xerxes is a multi-agent orchestration framework for building, running, and serving LLM-powered agents. It treats **agents, tools, memory, and runtime policy as first-class pluggable components** — not as scripts glued together — and ships with both a programmatic Python API and an interactive Rust-based TUI.

Distribution name: `xerxes-agent` · Python module: `xerxes` · Version: **0.2.0** (Alpha).

## Why it exists

Most agent frameworks make a single opinionated choice — a specific LLM provider, a specific chat UI, a specific task graph — and then ask you to bend the rest of your system around that choice. Xerxes inverts that: the framework core (`Xerxes`, `AgentOrchestrator`, `BaseLLM`, `MemoryStore`) is deliberately minimal, and everything else — providers, tools, memory backends, channels, UIs — is a swappable component behind a protocol. The goal is that the *same agent definition* can run in a CLI, in a chat channel, in an API server, in a daemon, or embedded in a Python script, without modification.

## Audience

- **Application developers** building agents over mixed LLM providers (OpenAI, Anthropic, Gemini, local Ollama, etc.) who need a uniform execution loop with streaming, tool-calling, memory, and hand-off.
- **Platform teams** building internal agent deployments who need sandbox policies, audit events, cost tracking, session replay, and OpenAI-wire-protocol compatibility for existing clients.
- **Chat-product integrators** wiring agents into Slack / Telegram / Discord / Email / Matrix / WhatsApp / Signal / Feishu / etc. via a single unified channel model.

## Core ideas

### Agents are data, not code

An `Agent` is a Pydantic dataclass with an LLM model, an instruction prompt, a function list, a sampling config, and optional switch-triggers. You register it, you call `xerxes.run(prompt)`, and the framework handles the streaming loop, tool extraction, permission checks, tool execution, retry logic, and history management.

### Two orchestration tiers

- **Basic** (`Xerxes.run` / `Xerxes.thread_run` / `Xerxes.athread_run`): one prompt → one agent → stream events. The agent may switch to another registered agent via triggers. Fits most single-conversation workflows.
- **Cortex** (`Cortex.kickoff`): pre-declared `CortexTask` list executed under one of five process types — `SEQUENTIAL`, `PARALLEL`, `HIERARCHICAL` (manager delegates to workers), `CONSENSUS` (all agents contribute + synthesize), `PLANNED` (LLM planner builds a DAG). Fits multi-step workflows with explicit task graphs.

### LLMs are behind one interface

`BaseLLM` is an abstract class with 7 methods (`generate_completion`, `extract_content`, `process_streaming_response`, `stream_completion`, `astream_completion`, `parse_tool_calls`, `_initialize_client`). Eight providers ship in-tree: OpenAI, Anthropic, Gemini, Ollama, plus OpenAI-compat shims for DeepSeek / Qwen / Kimi / Zhipu / LMStudio / Custom. A central `registry.py` holds cost tables (59 models) and auto-detects provider from model name.

### Tools follow one convention

`AgentBaseFn` is a base class with a `static_call(**kwargs) -> dict` entry point. The class name becomes the function name exposed to the LLM; the parameters become its JSON schema. 150+ tools ship, grouped into 18 modules (coding, web, system, math, data, AI, memory, media, home-automation, RL, Claude-Code-compatible file-edit tools, and more).

### Runtime features are opt-in

`RuntimeFeaturesConfig.enabled` defaults to `False`. When turned on, you get: plugin + skill discovery, per-agent policy overrides, sandbox routing (OFF / WARN / STRICT with docker or subprocess backends), prompt profiles (FULL / COMPACT / MINIMAL / NONE), session storage (in-memory or file), audit emitter (JSONL or OTEL), cost tracker, and an operator-tool bundle (pty, browser, plans, user-prompt, sub-agents). Everything below this layer remains pure.

### Streaming is the default

Every LLM response is an async iterator over typed events (`TextChunk`, `ThinkingChunk`, `ToolStart`, `ToolEnd`, `PermissionRequest`, `TurnDone`). A thread-safe `StreamerBuffer` lets you consume the stream from a synchronous caller, and the same event stream drives the API server (SSE), the Rust TUI (JSON-RPC), and the daemon (WebSocket).

## Surface area

```
┌─────────────────────────────────────────────────────────────┐
│  Rust TUI   FastAPI /v1    Daemon (WS+Unix)   Channels×14  │
│     │            │               │                 │       │
│     ▼            ▼               ▼                 ▼       │
│  ┌─ bridge.py (JSON-RPC) ───┐  ┌─ channels/base.Channel ─┐ │
│  │  (stdin/stdout)          │  │  + IdentityResolver      │ │
│  └───────────┬──────────────┘  └────────────┬────────────┘ │
│              │                               │              │
│              ▼                               ▼              │
│        ┌────────────────── Xerxes ──────────────────┐       │
│        │  AgentOrchestrator + FunctionRegistry      │       │
│        │  StreamerBuffer + run_agent_loop           │       │
│        └──┬──────────┬────────────┬───────────┬─────┘       │
│           │          │            │           │             │
│           ▼          ▼            ▼           ▼             │
│       BaseLLM   MemoryStore  Extensions   RuntimeFeatures  │
│       (×12)    (SQLite/RAG)  (plugins/    (policy/sandbox/ │
│                              hooks/skills) audit/session)  │
└─────────────────────────────────────────────────────────────┘
```

## Non-goals

- **Xerxes is not an LLM.** It's infrastructure *around* LLMs.
- **Xerxes is not a fine-tuning framework.** No training loops, no gradient code.
- **Xerxes is not a RAG vendor.** It ships a practical SQLite-backed vector store and a hybrid retriever, but it doesn't compete with dedicated vector DBs — the `MemoryStorage` protocol lets you plug in whatever you want.
- **Xerxes is not prescriptive about UI.** The Rust TUI is one option; the FastAPI server is another; the channel adapters are a third. The framework runs headless by default.

## Current state (2026-04-18)

- **Status:** Development Status :: 3 - Alpha (pre-release).
- **Tests:** 1501 passing, ~64k LOC Python covered.
- **Recent work:**
  - Package renamed `xerxes_agent` → `xerxes` (commit 543fb74).
  - Removed bundled Chainlit UI package (decision: UI belongs outside core).
  - Critical security fixes: sandbox pickle-escape closed (child→parent IPC now JSON); math-tools `eval()` replaced with AST-whitelist evaluator.
  - Packaging, pytest, CI, Dockerfile, and pre-commit all repointed to the renamed module.
- **Known gaps:** monorepo-style release / version policy not settled; no formal RFC process yet; mid-tier docs (now being written in this pass) did not previously exist.

## Design principles (short version)

1. **Small blast radius.** The framework core (Xerxes + executors + streaming + types) is ~10k LOC and has no hard dependency on any provider, channel, UI, or runtime feature. Everything else is an optional dependency or a plugin.
2. **Opt-in complexity.** If you want `MemoryStore`, pass it. If you want audit, enable `RuntimeFeaturesConfig`. If you want cortex, import `Cortex`. Calling `xerxes.run(prompt)` with only an LLM is valid.
3. **Protocols over inheritance.** `SessionStore`, `MemoryStorage`, `Embedder`, `AuditCollector`, `Channel` are all protocols — you conform by shape, not by base class.
4. **Streaming as the source of truth.** Non-streaming responses are assembled from streaming events, not the other way round.
5. **Trust boundaries are explicit.** The parent process trusts its own pickle. The child does not. The sandbox router records every decision. Every tool call fires an audit event before and after.
6. **Hybrid runtime.** Python owns logic; Rust owns the rendering-sensitive TUI. They talk over newline-delimited JSON — trivially observable, trivially replayable.

## Where to go next

- [System architecture](system-architecture.md) — component diagrams, data flow, execution loops.
- [Codebase summary](codebase-summary.md) — per-package inventory + key dependencies.
- [API reference](api-reference.md) — HTTP endpoints exposed by `api_server`.
- [Configuration guide](configuration-guide.md) — `XerxesConfig`, env vars, feature flags.
- [Testing guide](testing-guide.md) — how tests are organized and run.
- [Deployment guide](deployment-guide.md) — Docker, CI, running as a daemon.
- [Code standards](code-standards.md) — idioms, patterns, style.
- [Design guidelines](design-guidelines.md) — Rust TUI structure and theming.
- [Changelog](changelog.md) — recent history.
