# Codebase Summary

A dense tour of the Xerxes repository: where each package lives, what it owns, and what it depends on.

## Top-level layout

```
Xerxes-Agents/
├── src/
│   ├── python/xerxes/         # Main Python framework (~64k LOC, 23 subpackages)
│   └── rust/xerxes-cli/       # Ratatui-based interactive TUI (~3.5k LOC Rust)
├── tests/                     # 1501 pytest tests
├── docs/                      # Sphinx API docs + these human-written docs
├── examples/                  # Runnable demo scripts
├── scripts/                   # Utility scripts
├── pyproject.toml             # Python package + hatch build + Rust hook
├── Cargo.toml                 # Rust workspace root
├── Dockerfile                 # Multi-stage container build
├── docker-compose.yml         # Local compose setup
├── pytest.ini                 # Test config
├── Makefile                   # Convenience targets
└── hatch_build.py             # Custom rust-build hatch hook
```

## Python package inventory

Size-ranked; lines include docstrings and blank lines.

| Package | Files | Lines | Role | Depends on |
|---------|------:|------:|------|-----------|
| `xerxes.tools` | 18 | 11,588 | Agent-facing tool catalogue — coding (read/write/edit/exec), web (DuckDuckGo, Google, browser), system (processes, files), data (json/csv/text), math (safe AST evaluator), AI (embed, summarize, classify), memory, media, home-automation, RL, and a Claude-Code-compatible tool surface (`claude_tools.py`). | `types`, `core`, `memory` |
| `xerxes.cortex` | 16 | 8,668 | Multi-agent orchestration. `orchestration/cortex.py` hosts the main `Cortex` class; `orchestration/planner.py` builds XML DAGs; `orchestration/task_creator.py` decomposes prompts; `orchestration/dynamic.py` adds ad-hoc prompt execution. `agents/agent.py` wraps an internal Xerxes instance per `CortexAgent`. | `xerxes`, `llms`, `memory`, `types`, `logging` |
| `xerxes.runtime` | 19 | 5,800 | Runtime feature layer (opt-in). `bootstrap.py` runs the 6-stage startup; `bridge.py` wires legacy tools → registry; `features.py` holds `RuntimeFeaturesConfig` / `AgentRuntimeOverrides`; `profiles.py` owns prompt profiles; `query_engine.py` drives multi-turn conversations with budget; `cost_tracker.py`/`transcript.py`/`history.py` persist turn data. | `llms`, `types`, `memory`, `audit`, `security`, `session`, `extensions` |
| `xerxes.memory` | 14 | 5,334 | Four memory types: `ShortTermMemory` (FIFO), `LongTermMemory` (persistent), `EntityMemory` (lightweight KG), `ContextualMemory`+`UserMemory` (hybrid + per-user). `storage.py` (SQLite/file/in-memory backends), `vector_storage.py` (JSON-encoded embeddings, no pickle), `retrieval.py` (hybrid BM25+cosine+recency), `turn_indexer.py` (hook-based auto-indexing). | `core`, `types` |
| `xerxes.llms` | 8 | 4,700 | `base.BaseLLM` abstract + concrete `openai.py`, `anthropic.py`, `gemini.py`, `ollama.py`, `compat.py` (OpenAI-wire-compat shims), `registry.py` (cost table, provider detection, context limits). | `types`, `core` |
| `xerxes.extensions` | 15 | 3,664 | `plugins.py` (`PluginRegistry`, `PluginMeta`, `PluginType`); `hooks.py` (`HookRunner`, 7 hook points); `skills.py` (markdown-based skills with YAML frontmatter); `dependency.py` (semver + cycle detection). | `types`, `core` |
| `xerxes.operators` | 10 | 3,209 | `state.OperatorState` — lazy-initialized composition root over `PTYSessionManager`, `BrowserManager`, `PlanStateManager`, `UserPromptManager`, `SpawnedAgentManager`. `@operator_tool` decorator generates agent-exposed tool wrappers. | `tools`, `extensions` |
| `xerxes.types` | 7 | 3,287 | Core types: `Agent`, `AgentBaseFn`, `AgentFunction`, `Response` (agent_types); `SystemMessage`/`UserMessage`/`AssistantMessage`/`ToolMessage`/`MessagesHistory` (messages); `RequestFunctionCall`/`FunctionCallInfo`/`Completion`/`StreamChunk`/`ExecutionResult`/`ExecutionStatus`/`AgentCapability`/`AgentSwitch` (function_execution_types); `Function`/`Tool`/`ToolChoice`/`FunctionCall`/`ToolCall` (tool_calls); OpenAI protocol shims (`converters`, `oai_protocols`). | (none — leaf) |
| `xerxes.channels` | 23 | 2,307 | `base.Channel` ABC + `identity.IdentityResolver` + `registry` + 14 adapters: slack, telegram, email_imap, discord, whatsapp, signal, matrix, wecom, feishu, dingtalk, mattermost, bluebubbles, sms, home_assistant. | `memory` (identity persistence) |
| `xerxes.session` | 7 | 2,129 | `models.py` (SessionRecord / TurnRecord / ToolCallRecord / AgentTransitionRecord — all JSON-serializable); `store.py` (InMemory / File backends); `replay.py` (ReplayView — timeline reconstruction); `summarizer.py` (session condensing); `search.py` (cross-session fuzzy search). | `audit` |
| `xerxes.api_server` | 7 | 2,120 | FastAPI server (`server.py`); OpenAI-compatible `/v1/chat/completions` (`routers.py`); streaming SSE (`completion_service.py`); Cortex multi-agent streaming (`cortex_completion_service.py`); Pydantic protocol models (`models.py`). | `xerxes`, `cortex`, `streaming`, `llms` |
| `xerxes.core` | 9 | 2,088 | `prompt_template.py` (PromptSection / PromptTemplate); `config.py` (`XerxesConfig` + env / file loading); `errors.py` (exception hierarchy); `streamer_buffer.py` (thread-safe queue); `multimodal.py` (PIL wrapper); `paths.py` (`XERXES_HOME` resolution); `utils.py`. | (none — leaf) |
| `xerxes.audit` | 5 | 1,902 | `events.py` (typed event hierarchy: TurnStart, ToolCallAttempt, ToolPolicyDecision, SandboxDecision, ToolLoopWarning, Error, …); `collector.py` (InMemory / JSONL / Composite); `emitter.py` (thread-safe `AuditEmitter` with `emit_*` helpers); `otel_exporter.py`. | `types` |
| `xerxes.agents` | 10 | 1,940 | Built-in agents: `_coder_agent` (code), `_planner_agent` (task decomp), `_researcher_agent` (search+summarize), `_data_analyst_agent` (stats), `profile_agent` (heuristic user profiler), `compaction_agent` (LLM summarizer), `auto_compact_agent` (threshold config), `subagent_manager` (thread-pool sub-agents with optional git-worktree isolation). `definitions.py` provides the file-loaded agent registry (user/project `.md` with YAML frontmatter). | `xerxes`, `types`, `llms`, `tools` |
| `xerxes.logging` | 3 | 1,544 | `structured.py` (structlog + OTEL + Prometheus — all optional; metrics: `xerxes_tool_calls_total`, `xerxes_turn_duration_seconds`, `xerxes_tokens_consumed_total`, `xerxes_cost_usd_total`, `xerxes_switches_total`, etc.); `console.py` (human-friendly logging helpers). | (none — all optional deps soft-imported) |
| `xerxes.mcp` | 5 | 1,487 | `types.py` (MCPServerConfig, MCPTool, MCPResource, MCPPrompt, MCPTransportType); `client.py` (MCPClient with stdio + SSE + streamable-HTTP); `manager.py` (MCPManager — cross-server tool routing). | `types` |
| `xerxes.streaming` | 5 | 1,402 | `events.py` (AgentState + chunk types + TurnDone); `loop.py` (`run_agent_loop` — multi-turn streaming agent loop); `messages.py` (NeutralMessage + provider converters); `permissions.py` (PermissionMode + `check_permission`). | `types` |
| `xerxes.bridge` | 4 | 1,356 | `__main__.py` (entrypoint for `python -m xerxes.bridge`); `server.py` (1120 LOC — `BridgeServer` class, JSON-RPC over stdin/stdout, profile load/save); `profiles.py` (API provider profile storage under `$XERXES_HOME/profiles/`). | `xerxes`, `llms`, `streaming` |
| `xerxes.daemon` | 9 | 1,304 | `__main__.py` (entrypoint for `xerxes wakeup`); `server.py` (DaemonServer); `service.py` (lifecycle); `task_runner.py` (ThreadPoolExecutor runner); `gateway.py` (WebSocket gateway); `socket_channel.py` (Unix domain socket for local `xerxes send`); `config.py`. | `xerxes`, `streaming` |
| `xerxes.security` | 6 | 1,240 | `sandbox.py` (SandboxMode: OFF/WARN/STRICT; SandboxConfig; SandboxRouter); `sandbox_backends/docker_backend.py` (Docker CLI, JSON result IPC); `sandbox_backends/subprocess_backend.py` (subprocess + `resource` memory limits, JSON child→parent IPC); `policy.py` (tool-allowlist engine). | `types`, `logging` |
| `xerxes.context` | 3 | 1,027 | `compaction_strategies.py` (SummarizationStrategy, SlidingWindowStrategy, PriorityBasedStrategy, TruncateStrategy); `token_counter.py` (SmartTokenCounter with provider-specific backends — tiktoken, anthropic SDK, google SDK, or char-based fallback). | `llms` |
| `xerxes._bin` | 2 | 75 | `_launcher.py` — Python wrapper that finds and execs the bundled Rust `xerxes` binary. Entry point for the `xerxes` CLI script. | (none) |
| `xerxes.executors` (top-level) | 1 | 1,374 | `FunctionRegistry` (name → callable + agent), `AgentOrchestrator` (multi-agent switch trigger table + routing + execution history). | `types`, `runtime.loop_detection`, `security` |
| `xerxes.xerxes` (top-level) | 1 | 2,849 | The `Xerxes` class — main user-facing framework entry point. Wires `AgentOrchestrator` + `FunctionExecutor` + `StreamerBuffer` + optional `MemoryStore` + optional `RuntimeFeaturesConfig` into a coherent run loop. | everything downstream |

## Rust workspace

```
src/rust/
├── Cargo.toml              # Workspace root
└── xerxes-cli/
    └── src/
        ├── main.rs         # Clap CLI + event loop + terminal setup
        ├── app.rs          # Application state (Mode + cells + scroll)
        ├── bridge.rs       # spawn & JSON-RPC with `python -m xerxes.bridge`
        ├── events.rs       # Request / Event protocol structs
        ├── render.rs       # Ratatui UI (738 lines — the biggest file)
        ├── input.rs        # Crossterm keyboard handling + slash-popup
        ├── slash.rs        # 20 slash commands (/provider, /cost, /help, …)
        ├── markdown.rs     # Markdown → ratatui Lines
        ├── diff.rs         # Unified diff display
        ├── theme.rs        # Color scheme
        └── spinner.rs      # Progress spinner
```

Entry point: `xerxes` CLI binary (installed by the wheel's entry point `xerxes = "xerxes._bin._launcher:launch"`).

## Key dependencies

Parsed from [pyproject.toml](../pyproject.toml) — runtime core (13 packages):

| Package | Version | Purpose | Runtime |
|---------|---------|---------|:-------:|
| `pydantic` | >=2.9.2,<3.0 | Config + model validation (XerxesConfig, Agent, Message types) | ✓ |
| `openai` | >=1.72.0 | OpenAI SDK (also powers OpenAI-compat shims) | ✓ |
| `httpx` | >=0.25.0 | HTTP client for Anthropic, Ollama, web_tools, API-server calls | ✓ |
| `psutil` | >=5.9.0 | System-info tool + process manager | ✓ |
| `pillow` | >=11.0.0 | Multimodal image handling | ✓ |
| `numpy` | >=1.24.0 | Vector ops for retrieval / embeddings | ✓ |
| `google-generativeai` | >=0.8,<0.9 | Gemini provider | ✓ |
| `fastapi` | >=0.116.1 | API server framework | ✓ |
| `uvicorn` | >=0.35.0 | ASGI server for FastAPI | ✓ |
| `jinja2` | >=3.1.6 | Prompt template rendering | ✓ |
| `requests` | >=2.31.0 | Synchronous HTTP (some tools) | ✓ |
| `beautifulsoup4` | >=4.12.0 | Web scraping (web_tools) | ✓ |
| `lxml` | >=5.0.0 | XML parsing (cortex planner, etc.) | ✓ |
| `feedparser` | >=6.0.0 | RSS/feed parsing (web tools) | ✓ |
| `python-dateutil` | >=2.8.0 | Date parsing in tools | ✓ |
| `pyyaml` | >=6.0 | Agent `.md` YAML frontmatter | ✓ |
| `ddgs` | >=9.5.4 | DuckDuckGo search engine | ✓ |
| `playwright` | >=1.54.0 | Browser automation (operators.browser) | ✓ |

Dev extras (`[dev]`): `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`, `ruff`, `black`, `mypy`, `pre-commit`, `ipykernel`, `notebook`.

Optional extras: `monitoring` (structlog, prometheus_client, opentelemetry, sentry_sdk, datadog); `vectors` (scikit-learn, sentence-transformers); `mcp` (mcp SDK).

## Test layout

```
tests/
├── conftest.py             # Fixtures
├── test_agent_types.py     # Type tests
├── test_basics.py          # Smoke tests
├── test_cortex_*.py        # Cortex orchestrator tests
├── test_llm_base.py        # LLM interface contract tests
├── test_memory_*.py        # Each memory class
├── test_*_tools.py         # Tool suites (math, data, coding, system, web, …)
├── test_runtime_*.py       # Runtime features
├── test_sandbox.py         # Sandbox routing
├── test_sandbox_backends.py  # Docker + subprocess backends
├── test_session_*.py       # Session store/replay/search
├── test_skill_*.py         # Skill discovery + lifecycle
└── test_storage.py         # Memory storage backends
```

See [testing-guide.md](testing-guide.md) for how to run them.

## Where new code goes

| Adding… | Put it in… |
|---------|-----------|
| A new LLM provider | `src/python/xerxes/llms/<name>.py` (inherit `BaseLLM`); register in `llms/registry.py` COSTS + PROVIDERS |
| A new tool | `src/python/xerxes/tools/<name>.py` (inherit `AgentBaseFn`, implement `static_call`) |
| A new chat channel | `src/python/xerxes/channels/adapters/<name>.py` (inherit `Channel`) |
| A new memory backend | Implement `MemoryStorage` protocol (no base class needed) in `src/python/xerxes/memory/` |
| A new runtime feature | `src/python/xerxes/runtime/` — add config field to `RuntimeFeaturesConfig`, wire into `RuntimeFeaturesState` |
| A new audit event type | `src/python/xerxes/audit/events.py` — subclass `AuditEvent`, add helper to `audit/emitter.py` |
| A new process type | `src/python/xerxes/cortex/core/enums.py` — extend `ProcessType`, add `_run_<name>` method to `Cortex` |
| A built-in agent | `src/python/xerxes/agents/<name>_agent.py` + register in `agents/__init__.py` |
