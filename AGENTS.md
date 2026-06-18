# AGENTS.md

Guidance for AI agents (Claude Code, Codex, OpenCode, etc.) working in this repository.
Read this before making any changes.

---

## Project Overview

**Xerxes** (`xerxes-agent` v0.2.0) is a multi-agent orchestration framework for building,
running, and serving LLM-powered agents. Pure-Python runtime with a `prompt_toolkit` TUI,
a JSON-RPC daemon, and an HTTP API server. It supports 12+ LLM providers, a large
Claude-Code-compatible tool surface, MCP integration, sub-agent spawning, sandboxed
execution, 14 messaging-channel adapters, and a four-tier memory system.

- **Author:** Erfan Zare Chavoshi (@erfanzar)
- **License:** Apache-2.0
- **Python:** 3.11+ (3.11 / 3.12 / 3.13)
- **Package root:** `src/python/xerxes/`
- **Scale:** ~100k LOC, 212 test files, 4306 collected tests

---

## Quick Reference — Commands

All Python invocations go through `uv`. Do not call bare `python` or `python3`.

```bash
# Install (editable, with dev deps)
uv pip install -e ".[dev]"

# Run the agent
uv run xerxes                          # interactive TUI
uv run xerxes -r <session_id>          # resume a session
uv run xerxes "explain this function"  # one-shot (pipes to stdout)
uv run xerxes telegram --token ...     # daemon with Telegram gateway

# Tests
uv run pytest                          # full suite (4306 tests)
uv run pytest tests/runtime/           # one directory
uv run pytest tests/llms/test_registry.py::test_detect_provider  # one test

# Lint & format
uv run ruff check src/python/xerxes/
uv run ruff format --check src/python/xerxes/
uv run ruff check --fix src/python/xerxes/<file>   # autofix one file
uv run ruff format src/python/xerxes/<file>        # format one file

# Type checking (run from src/python/)
cd src/python && uv run mypy xerxes --ignore-missing-imports

# Pre-commit (all hooks)
uv run pre-commit run --all-files

# Build docs (Sphinx)
cd docs && make html
```

**Line length:** 121. **Target:** py311.

**Ruff select rules:** `A, B, E, F, I, NPY, RUF, UP, W`
**Ruff ignores:** `F722, B008, UP015, A005, E501, B010`
**`__init__.py` ignores:** `E402, F401` (allows re-export ordering and unused imports in packages)

---

## Repository Layout

```md
src/python/xerxes/
├── __main__.py          # CLI entry — dispatches telegram / one-shot / TUI
├── xerxes.py            # Xerxes facade — ties LLM + agents + memory + runtime
├── executors.py         # AgentOrchestrator + FunctionExecutor (tool dispatch)
├── _compat_shims.py     # Backward-compat aliases (do not extend)
│
├── core/                # Config (Pydantic), errors (typed hierarchy), paths, utils
├── types/               # Data structures: Agent, messages, tool_calls, protocols
├── llms/                # BaseLLM ABC + providers (openai, anthropic, gemini, ollama)
├── streaming/           # The agent loop (loop.py), events, permissions, messages
├── runtime/             # QueryEngine, features, session, resilience, cost tracking
├── tools/               # ~30 tool modules (coding, browser, system, AI, media, ...)
├── agents/              # Built-in agent definitions (loaded from default/*.yaml)
├── operators/           # PTY, browser, subagents, plans, user-prompt managers
├── memory/              # 4-tier memory: short/long/entity/user + embedders + retrieval
├── context/             # Token counting, compaction, tool-result pruning
├── security/            # Policy engine, sandbox, prompt scanner, path/url safety
├── extensions/          # Plugins, skills, hooks, skill-authoring pipeline
├── daemon/              # JSON-RPC daemon server (socket + WebSocket + webhooks)
├── channels/            # 14 messaging adapters (telegram, slack, discord, ...)
├── api_server/          # FastAPI HTTP server (OpenAI-compatible completions)
├── mcp/                 # Model Context Protocol client + integration
├── acp/                 # Agent Client Protocol server (stdio/HTTP JSON-RPC)
├── bridge/              # Provider profile management + model fetching
├── session/             # Session store, FTS index, replay, branching, migrations
├── cortex/              # Higher-level orchestration (planner, tasks, universal agent)
├── tui/                 # prompt_toolkit terminal UI
├── audit/               # Audit event collector + emitter + OTLP exporter
├── cron/                # Scheduled job scheduler
├── training/            # Batch runner, trajectory compression, RL training
├── auth/                # OAuth storage
└── skills/              # Bundled skill directories (SKILL.md + scripts)
```

**Tests** mirror the package structure under `tests/`.
**Playground** (`playground/`) holds scored eval harnesses (`eval.py`, `eval_hard.py`).

---

## Architecture — How a Turn Works

### Entry flow

1. `xerxes` CLI (`__main__.py`) picks one of three modes:
   - **telegram** → starts `DaemonServer` with Telegram channel enabled
   - **one-shot** → spawns a daemon via `BridgeClient`, runs prompt with `accept-all` permissions,
     streams `TextPart` events to stdout, exits on `TurnEnd`
   - **interactive** → launches `XerxesTUI` (prompt_toolkit)

2. The **TUI and daemon** talk to the runtime via the async streaming loop
   (`streaming/loop.py`). The **API server** and **tests** use the synchronous
   `Xerxes.run()` / `QueryEngine.submit()` path.

### The core loop (`streaming/loop.py:run()`)

`run()` is a **synchronous generator** that drives one full turn:

1. Appends the user message to `state.messages`.
2. Detects the provider via `llms.registry.detect_provider(model)`.
3. Loops up to `MAX_TOOL_TURNS` (50) times:
   - Streams the LLM response through `_stream_llm()` (provider adapter).
   - `_ThinkingParser` incrementally splits `<think>...</think>` tags across chunk
     boundaries into `TextChunk` / `ThinkingChunk` events.
   - Records token usage and appends the assistant message.
   - If tool calls are present: gates each through the **permission system**
     (`streaming/permissions.py`), executes via `tool_executor`, feeds results back.
   - If no tool calls: breaks.
4. Yields stream events: `TextChunk`, `ThinkingChunk`, `ToolStart`, `PermissionRequest`,
   `ToolEnd`, `TurnDone`, `SkillSuggestion`.

Key design points:

- **Cancellation** is polled via `cancel_check()` between tools and at loop top.
  When cancelled mid-tool-batch, synthetic cancelled-results are backfilled for every
  un-run tool call so the next request doesn't 400.
- **Error recovery:** if the LLM stream errors mid-turn, the partial assistant text is
  preserved with an `[Error: ...]` marker and the turn closes cleanly with `TurnDone`.
- **Steer drain** and **agent-event drain** hooks allow mid-turn user injection and
  passive sub-agent event visibility.

### The facade (`xerxes.py:Xerxes`)

`Xerxes` owns:

- an `EnhancedAgentOrchestrator` (agent registry + switch triggers)
- an `EnhancedFunctionExecutor` (tool dispatch with retry, policy gating, sandbox)
- an optional `MemoryStore` (four-tier)
- `RuntimeFeaturesState` (plugins, skills, sandbox, audit, operators, authoring pipeline)
- a `LoopDetector`

It registers two built-in agent-switch triggers: **capability-based** (picks the agent
with the highest `performance_score` for a required capability) and **error-recovery**
(falls back to the current agent's `fallback_agent_id`).

### Tool execution (`executors.py`)

`EnhancedFunctionExecutor` handles:

- argument validation
- retry with exponential backoff
- `ToolPolicy` enforcement (`security/policy.py`) → raises `ToolPolicyViolation`
- sandbox dispatch (`security/sandbox.py` — docker or subprocess backend)
- loop detection (`runtime/loop_detection.py`)
- per-call audit event emission

`FunctionRegistry` stores tools as `(callable, agent_id)` tuples so lookups prefer the
current agent's version.

### QueryEngine (`runtime/query_engine.py`)

Per-session driver wrapping the streaming loop. Enforces `max_turns` (50),
`max_budget_tokens` (500k), and triggers transcript compaction at
`compact_after_turns` (20). Exposes both blocking `submit()` and streaming
`submit_stream()`.

---

## Key Subsystems

### LLM Providers (`llms/`)

- `BaseLLM` (ABC) + `LLMConfig` (dataclass) in `llms/base.py`.
- Providers: `OpenAILLM` (openai SDK), `AnthropicLLM` (httpx direct), `GeminiLLM`,
  `OllamaLLM` (httpx direct).
- **Registry** (`llms/registry.py`): `PROVIDERS` dict (12 providers), `COSTS` table
  (USD per 1M tokens), `detect_provider(model)` via prefix matching,
  `get_context_limit(model)`, `calc_cost(model, in_tok, out_tok)`.

**Provider quirks (do not "fix" these — they are intentional workarounds):**

- **Kimi Code** (`api.kimi.com/coding/v1`) requires a `claude-code/1.0.0` User-Agent
  or it returns 403. Headers are injected via `provider_default_headers()`. See
  `llms/registry.py:386-397`.
- **MiniMax** (`api.minimax.io/v1`) has no `GET /models` endpoint (returns 404).
  `bridge/profiles.py` catches the 404 for minimax/minimaxi and returns a hardcoded
  model list. See `_PROVIDERS_WITHOUT_MODELS` at `bridge/profiles.py:196`.
- Custom `base_url` endpoints default to `max_retries=0` (fail fast); official endpoints
  default to 2 retries. See `_request_max_retries()` at `streaming/loop.py:239`.

### Tools (`tools/`)

Tools are plain Python callables registered with the `FunctionRegistry`. The package
`__init__.py` re-exports ~150 tool symbols grouped into `TOOL_CATEGORIES` (15 categories:
file_system, execution, web, data, ai, math, system, memory, agent, workflow, notebook,
lsp, mcp, remote, browser, home_assistant, rl, media, meta).

Claude-Code-compatible tools live in `claude_tools.py` (FileEditTool, GrepTool, GlobTool,
AgentTool, SpawnAgents, TaskCreateTool, TodoWriteTool, MCPTool, etc.).

### Operators (`operators/`)

Session-attached managers that back the high-power tool surface:

- `PTYSessionManager` — interactive shell sessions
- `BrowserManager` — Playwright browser control
- `SpawnedAgentManager` — sub-agent lifecycle
- `UserPromptManager` — ask-user-question flow

Tool sets (`operators/config.py`):

- `SAFE_OPERATOR_TOOLS`: `ask_user`, `web.time`, `update_plan`
- `HIGH_POWER_OPERATOR_TOOLS`: `exec_command`, `write_stdin`, `apply_patch`,
  `spawn_agent`, `resume_agent`, `send_input`, `wait_agent`, `close_agent`,
  `view_image`, `web.search_query`, `web.image_query`, `web.open`, `web.click`,
  `web.find`, `web.screenshot`, `web.weather`, `web.finance`, `web.sports`

When `operator.power_tools_enabled=True`, high-power tools are moved out of the
`optional_tools` deny-set so they become reachable.

### Agents (`agents/`)

Built-in agents load from `agents/default/*.yaml` via `agentspec.py:load_agent_spec()`.
If YAML loading fails, a hardcoded fallback set (`_HARDCODED_BUILTIN_AGENTS`) is used.

`AgentDefinition` dataclass: `name`, `description`, `system_prompt`, `model`, `tools`,
`allowed_tools`, `exclude_tools`, `source`, `max_depth` (default 5), `isolation`.

### Memory (`memory/`)

Four tiers: `ShortTermMemory`, `LongTermMemory`, `EntityMemory`, `UserMemory`.
Storage backends: `SimpleStorage`, `SQLiteStorage`, `RAGStorage`, `SQLiteVectorStorage`.
Embedders: `HashEmbedder` (default, no deps), `OpenAIEmbedder`, `OllamaEmbedder`,
`SentenceTransformerEmbedder` (requires `xerxes[vectors]`).
`HybridRetriever` combines keyword + vector search with configurable `RetrievalWeights`.

### Security (`security/`)

- `PolicyEngine` + `ToolPolicy` — coarse allow/deny gate before permission/sandbox.
  When `allow` is non-empty it's an exclusive allow-list. `optional_tools` are
  gated-off-by-default tools.
- `SandboxRouter` — routes tool execution to docker or subprocess backend.
- `prompt_scanner`, `path_security`, `url_safety`, `approvals`, `redact`.

### Daemon (`daemon/`)

`DaemonServer` multiplexes three listening surfaces on one async loop:

- Unix socket (`SocketChannel`) — TUI connects here
- WebSocket gateway (`WebSocketGateway`) — remote clients
- Channel webhooks (`ChannelWebhookServer`) — Telegram/Slack/etc.

JSON-RPC protocol version: **35**. Old `task.*` methods return `MIGRATED_ERROR`; use
`session.open`, `turn.submit`, `turn.cancel`, `session.list`, `runtime.status`.

### Channels (`channels/`)

14 adapters under `channels/adapters/`: telegram, slack, discord, email_imap, matrix,
signal, whatsapp, dingtalk, feishu, wecom, mattermost, sms, home_assistant, bluebubbles.
Each implements the `Channel` ABC (`channels/base.py`).

### Skills & Extensions (`extensions/`)

- `SkillRegistry` discovers skills from `skill_dirs` + conventional `<workspace>/skills`.
- `Skill` / `SkillMetadata` parsed from `SKILL.md` frontmatter (`parse_skill_md()`).
- `HookRunner` dispatches lifecycle hooks at `HOOK_POINTS`.
- `SkillAuthoringPipeline` auto-authors skills from observed tool-use patterns.
- `PluginRegistry` discovers plugins from `plugin_dirs`.

### Session (`session/`)

`SessionStore` (SQLite) persists sessions. `FTSIndex` provides full-text search over
transcripts. Supports branching (`branching.py`), snapshots (`snapshots.py`), replay
(`replay.py`), and schema migrations (`migrations/`).

---

## Communication & Commits

- NEVER say "You're absolutely right!"
- NEVER credit yourself (the agent) in commit messages.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with 🤖 unless the exact text was explicitly
  approved by the user.
- When using `gh` to inspect issues or PRs, prefer `--json <fields>` or explicit narrow
  flags such as `--comments`. Avoid plain `gh issue view` / `gh pr view` — they can fail
  on this repo because GitHub classic project fields are deprecated.
- Commit messages follow **Commitizen** conventions (enforced by pre-commit
  `commitizen` hook at the `commit-msg` stage). Format: `type(scope): description`.

---

## Code Style

- **Imports at the top of the file.** No local imports except to break circular
  dependencies or guard optional deps. The codebase uses lazy imports inside functions
  for heavy modules (e.g. `from .streaming.loop import run` inside `QueryEngine.submit`)
  — follow this pattern only for circular-dependency or heavy-startup cases.
- No `TYPE_CHECKING` guards — fix cycles structurally via protocols. (Note:
  `executors.py` currently has one `if tp.TYPE_CHECKING` block; do not add more.)
- Prefer **top-level functions** over classes when code does not mutate shared state.
  Reduce deep inheritance hierarchies.
- Use **early returns** to reduce nesting.
- Document public APIs with concise **Google-style docstrings** (Args/Returns/Raises).
  Skip docstrings on trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer `logging` over `print` (except in scripts and debugging). Every module starts
  with `logger = logging.getLogger(__name__)`.
- Resolve environment-dependent defaults **once** and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`). Update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse.
  Start simple; abstract only under real pressure.
- **Delete dead code:** unused parameters, stale options, old experiments.
- **Top-level constants** for magic strings/numbers (e.g. `MAX_TOOL_TURNS = 50`,
  `DAEMON_PROTOCOL_VERSION = 35`).
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle (`async with` for LLM clients).

### Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed in this codebase).
- No abbreviations like `exe` — use `exec` or full words.

### Types & Data Structures

- **Pydantic** for configuration models (`core/config.py`: `XerxesConfig`,
  `ExecutorConfig`, `MemoryConfig`, `SecurityConfig`, `LLMConfig`).
- **Dataclasses** for runtime state and value objects (`RuntimeFeaturesConfig`,
  `QueryEngineConfig`, `LLMConfig` in `llms/base.py`, `AgentDefinition`,
  `ProviderConfig`).
- **StrEnum** over string keys (`LogLevel`, `EnvironmentType`, `LLMProvider`).
- **Frozen dataclasses** for immutable metadata (`ProviderConfig(frozen=True)`).
- **Enums** for typed values (`PolicyAction`, `AgentSwitchTrigger`, `ExecutionStatus`).
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.

### Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location (provider defaults in
  `llms/registry.py:PROVIDERS`; runtime defaults in `runtime/features.py`;
  query defaults in `runtime/query_engine.py:QueryEngineConfig`).
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass
  (e.g. `RuntimeFeaturesConfig` embeds `OperatorRuntimeConfig`, `ToolPolicy`,
  `LoopDetectionConfig`, `SandboxConfig`).

### API Design

- Accept only what necessary. Replace boolean flags with meaningful parameters
  (`num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior.
- Normalize inputs to a standard format **once at the boundary**, not throughout.
  (Example: `streaming/messages.py` normalizes to both Anthropic and OpenAI formats
  at the provider adapter boundary.)

### Error Handling

- **Let exceptions propagate by default.** The typed hierarchy lives in
  `core/errors.py`: `XerxesError` (base) → `AgentError`, `FunctionExecutionError`,
  `XerxesTimeoutError`, `ValidationError`, `RateLimitError`, `XerxesMemoryError`,
  `ClientError`, `ConfigurationError`, `AgentSpecError`.
- Only catch to add meaningful context and re-raise, or to intentionally alter
  control flow (see the LLM-stream error recovery in `streaming/loop.py:396-425`).
- NEVER swallow exceptions unless specifically requested.
- **Assert liberally;** prefer `raise ValueError` over silent fallbacks.
- Policy violations raise `ToolPolicyViolation` (`security/policy.py`).
- Loop detection raises `ToolLoopError` (`runtime/loop_detection.py`).

### Documentation

- **Sphinx** docs live in `docs/` with autodoc RST files under `docs/api_docs/`.
  Build with `cd docs && make html`.
- Keep docs content in sync with code. Use Markdown and Sphinx-style cross-references.
- Write docs that stand alone without conversational context.

### Deprecation

- **NO BACKWARD COMPATIBILITY.** Update all call sites instead.
- Only add compatibility shims if the user explicitly requests it.
- `_compat_shims.py` and `memory/compat.py` exist for legacy aliases — do not extend them.

### Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

### LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:

- Over-protective `try/except` and defensive `None` checks
- Tautological tests (type exists, constant has value)
- Verbose/redundant docstrings and unnecessary `__all__` in `__init__.py`
- Boolean dispatch instead of separate classes
- Environment variables instead of explicit parameters

---

## Planning

- Produce detailed plans with code snippets. Ask questions up front instead of guessing.
- When a request is too large for one pass, capture a plan in `.agents/projects/`
  before pausing.

---

## Provider Model Registry

Providers are auto-detected by model name prefix. You can also force a provider with
`provider/model` notation (e.g. `anthropic/claude-sonnet-4-5`).

| Provider  | Type      | Key env var         | Example models                     |
| --------- | --------- | ------------------- | ---------------------------------- |
| anthropic | anthropic | `ANTHROPIC_API_KEY` | claude-opus-4-6, claude-sonnet-4-6 |
| openai    | openai    | `OPENAI_API_KEY`    | gpt-4o, gpt-4.1, o3-mini           |
| gemini    | openai    | `GEMINI_API_KEY`    | gemini-2.5-pro, gemini-2.0-flash   |
| zhipu     | openai    | `ZHIPU_API_KEY`     | glm-5.2, glm-5.1, glm-4.6          |
| deepseek  | openai    | `DEEPSEEK_API_KEY`  | deepseek-chat, deepseek-reasoner   |
| qwen      | openai    | `DASHSCOPE_API_KEY` | qwen-max, qwen3-235b-a22b          |
| kimi      | openai    | `MOONSHOT_API_KEY`  | moonshot-v1-128k, kimi-latest      |
| kimi-code | openai    | `KIMI_CODE_API_KEY` | kimi-for-coding                    |
| minimax   | openai    | `MINIMAX_API_KEY`   | MiniMax-M2.7-highspeed, abab6.5    |
| ollama    | openai    | (none)              | llama3.3, mistral, qwen2.5-coder   |
| lmstudio  | openai    | (none)              | (dynamic)                          |
| custom    | openai    | `CUSTOM_API_KEY`    | (user-defined)                     |

To add a new provider: add a `ProviderConfig` entry to `PROVIDERS` in
`llms/registry.py`, add pricing to `COSTS`, and add a prefix rule to `_PREFIX_MAP`.

---

## File Header

Every Python file begins with the Apache-2.0 copyright header:

```python
# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

The `scripts/fix_license_headers.py` script can add/refresh this automatically.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`):

- **Lint job:** `ruff check` + `ruff format --check` + `mypy` on `src/python/xerxes/`
- **Security job:** `bandit` + `safety check` + `pip-audit` (all `continue-on-error: true`)

Pre-commit hooks (`.pre-commit-config.yaml`):
trailing-whitespace, end-of-file-fixer, check-yaml/json/toml, check-added-large-files
(>1000KB), ruff (--fix), ruff-format, black, mypy, isort (black profile), bandit
(excludes tests/), commitizen (commit-msg stage), prettier (yaml/json/markdown,
excludes docs/).

Test naming convention: `test_*.py` with test functions named `test_*` (enforced by
pre-commit `name-tests-test` with `--pytest-test-first`).
