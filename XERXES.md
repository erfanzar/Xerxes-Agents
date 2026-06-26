# XERXES.md — Xerxes-Agents Project Context

> **Version:** 0.2.6
> **Project root:** `/Users/erfan/Documents/Projects/Xerxes-Agents`
> **Author:** Erfan Zare Chavoshi (@erfanzar)
> **License:** Apache-2.0
> **Python:** 3.11+ (3.11 / 3.12 / 3.13)

## What This Project Is

Xerxes-Agents is a multi-agent orchestration framework for building, running, and serving LLM-powered agents. It is a **pure-Python runtime** (~110k LOC across 405 `.py` files) with three primary interfaces:

1. **Interactive TUI** (`prompt_toolkit`) — terminal-first coding assistant
2. **FastAPI server** — OpenAI-compatible `/v1/chat/completions` with SSE streaming
3. **JSON-RPC daemon** — WebSocket + Unix socket for persistent background processes

Key subsystems: 150+ built-in tools (Claude-Code-compatible surface), 12 LLM providers, 4-tier memory (short/long/entity/user), 14 messaging channel adapters, 5-process-type Cortex multi-agent orchestration, sandboxed execution, MCP client integration, and 60+ skill bundles.

## Repository Shape

```md
Xerxes-Agents/
├── src/python/xerxes/          # 405 .py files, ~110k LOC, 35 subpackages
│   ├── xerxes.py               # 2,345 LOC — main facade (Xerxes, orchestration, memory, loop)
│   ├── executors.py             # 1,332 LOC — tool dispatch, policy, sandbox, retry
│   ├── streaming/loop.py        # 1,768 LOC — synchronous generator core loop
│   ├── __main__.py              # CLI dispatcher (TUI / one-shot / daemon)
│   ├── daemon/                  # JSON-RPC server (WebSocket + socket + webhooks)
│   ├── tui/                     # prompt_toolkit terminal UI (~19 files)
│   ├── api_server/              # FastAPI OpenAI-compatible server
│   ├── cortex/                  # Multi-agent orchestration (5 process types)
│   ├── llms/                    # 12 provider adapters + registry
│   ├── tools/                   # 30+ tool modules (claude_tools, coding, web, system, ...)
│   ├── memory/                  # 4 tiers + embedders + 8 external providers
│   ├── security/                # Policy engine + sandbox backends + prompt scanner
│   ├── channels/                # 14 messaging adapters
│   ├── extensions/              # Skills, hooks, authoring pipeline
│   ├── operators/               # PTY, browser, subagents, user-prompt managers
│   ├── session/                 # SQLite persistence + FTS5 + branching
│   ├── types/                   # Data structures (OAI protocols, messages, tool_calls)
│   └── core/                    # Pydantic config, errors, paths, utils
├── tests/                       # 350 test files, mirrors src/ structure + integration/
├── examples/                    # 11 demo scripts (cortex, deepsearch, scenarios)
├── playground/                  # Eval harnesses (eval.py, eval_hard.py) — bills real tokens
├── docs/                        # Sphinx docs + Markdown guides (12 human-written, 160 RST stubs)
├── scripts/                     # Dev scripts (install.sh, fix_license_headers.py, format_and_generate_docs.py)
├── pyproject.toml               # hatchling, ruff, mypy, pytest, basedpyright config
├── uv.lock                      # UV lockfile
└── .github/workflows/           # ci.yml (lint + security, **no test job**), release.yml
```

## Build, Test, and Development Commands

All Python work goes through `uv`. Do not call bare `python` or `python3`.

```bash
# Install (editable, with all dev tools)
uv pip install -e ".[dev]"

# Full feature set
uv pip install -e ".[full]"

# Run the agent
uv run xerxes                          # interactive TUI
uv run xerxes -r <session_id>          # resume session
uv run xerxes "explain this function"  # one-shot

# Tests
uv run pytest                          # 4,503 tests (excludes eval by default)
uv run pytest tests/llms/              # one directory
uv run pytest -m eval                  # real-LLM eval harnesses (requires API keys)

# Lint & format
uv run ruff check src/python/xerxes/
uv run ruff format --check src/python/xerxes/
uv run ruff check --fix src/python/xerxes/<file>
uv run ruff format src/python/xerxes/<file>
uv run ruff check --select ARG,C901,SIM src/python/xerxes/  # advisory only

# Type checking
uv run mypy src/python/xerxes --ignore-missing-imports

# Pre-commit
uv run pre-commit run --all-files

# Docs (Sphinx)
make -C docs html

# License header fix
python scripts/fix_license_headers.py

# Package build
python -m build
```

### Critical CI/Test Gap

**`.github/workflows/ci.yml` has NO pytest job.** The CI runs lint + security but never executes the 4,500+ tests. This is a known gap. Also, the `.venv` in this repo currently has **only core dependencies** — `pytest-asyncio` and other dev extras are missing. If you run `pytest` without first doing `uv pip install -e ".[dev]"`, async tests will fail with "async def functions are not natively supported."

## Architecture — How a Turn Works

The core unit is a **turn** driven by `streaming/loop.py:run()` — a synchronous generator yielding `StreamEvent` objects:

```md
User message → state.messages
    → detect provider (llms/registry.py prefix matching)
    → loop up to MAX_TOOL_TURNS (50):
        → stream LLM via _stream_llm()
        → _ThinkingParser splits <thinking> tags across chunk boundaries
        → record token usage, append assistant message
        → if tool calls present:
            → gate each via PermissionMode (AUTO / ACCEPT_ALL / MANUAL / PLAN)
            → execute via tool_executor (FunctionRegistry)
            → feed results back into messages
        → if no tool calls: break
    → yield TextChunk / ThinkingChunk / ToolStart / ToolEnd / PermissionRequest / TurnDone
```

Key architectural points:

- **Facade** (`xerxes.py`): `Xerxes` owns `AgentOrchestrator`, `FunctionExecutor`, optional `MemoryStore`, `RuntimeFeaturesState`, and `LoopDetector`.
- **Tool execution** (`executors.py`): `EnhancedFunctionExecutor` handles validation → retry → `ToolPolicy` enforcement → sandbox dispatch → audit emission.
- **QueryEngine** (`runtime/query_engine.py`): Per-session driver enforcing `max_turns` (50), `max_budget_tokens` (500k), and compaction at `compact_after_turns` (20).
- **Security stack** (`security/`): `PolicyEngine` → `PermissionMode` → `SandboxConfig` → `SandboxBackend` (docker or subprocess).
- **Memory** (`memory/`): 4 tiers (`ShortTermMemory`, `LongTermMemory`, `EntityMemory`, `UserMemory`) with 5 storage backends (`Simple`, `SQLite`, `RAG`, `SQLiteVector`, `File`).
- **Providers** (`llms/`): `BaseLLM` ABC + 12 implementations. `detect_provider(model)` uses prefix matching. `COSTS` table tracks USD per 1M tokens.

## Code Conventions

- **Line length:** 121 (both ruff and black)
- **Python target:** 3.11 (`py311`)
- **Ruff select:** `A, B, E, F, I, NPY, RUF, UP, W` (ignores `F722, B008, UP015, A005, E501, B010`)
- **Imports:** relative inside `src/python/xerxes/`, absolute in tests/examples. `from __future__ import annotations` used for forward references.
- **Docstrings:** Google-style, mandatory for public classes/methods. Include `Args`, `Returns`, `Raises`, and runnable `Example` blocks. No inline comments that restate the code.
- **Naming:** `snake_case.py` modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, `_leading_underscore` private helpers. Tool classes are `PascalCase` verb-noun (`ReadFile`, `WriteFile`).
- **Type hints:** All public functions have full type hints. Prefer `X | None` over `Optional[X]`. `basedpyright` is intentionally neutered (most diagnostics disabled). `mypy` is loose (`ignore_missing_imports = true`).
- **`__all__`:** Used in `__init__.py` and regular modules to define public APIs.
- **`TYPE_CHECKING`:** Used pragmatically for circular-import typing despite AGENTS.md saying it's "avoided." Do not add new `TYPE_CHECKING` guards unless breaking a circular dependency.
- **File header:** Every Python file starts with the Apache-2.0 copyright header. Use `scripts/fix_license_headers.py` to normalize.
- **Pre-commit:** trailing-whitespace, end-of-file-fixer, check-yaml/json/toml, check-added-large-files (>1000KB), ruff --fix, ruff-format, black, mypy, isort (black profile), bandit (excludes tests/), commitizen (conventional commits), prettier (YAML/JSON/Markdown, excludes docs/).

## Agent Workflow Rules

1. **All changes must pass `ruff check src/python/xerxes/`** — zero tolerance on blocking rules.
2. **Never add `TYPE_CHECKING` guards** unless you are actively breaking a circular import. Fix cycles structurally via protocols instead.
3. **Top-level imports only.** No local imports except for circular-dependency guards or heavy optional deps (e.g., `playwright` inside `BrowserManager`).
4. **No `*_utils.py` modules** — use descriptive names (`text_cleaning.py`).
5. **No ad-hoc compatibility hacks** (`hasattr(m, "old_attr")`). Update all call sites consistently.
6. **Delete dead code** — unused parameters, stale options, old experiments.
7. **Separate computation from I/O.** Split compute from upload/write.
8. **Context managers for resource lifecycles.** `async with` for LLM clients.
9. **Let exceptions propagate.** Catch only to add context and re-raise, or to alter control flow intentionally. NEVER swallow exceptions.
10. **Commit messages:** follow Commitizen conventions (`type(scope): description`). The `commitizen` hook enforces this at the `commit-msg` stage.

## Risks & Caveats — Things That Will Bite You

### Pre-Existing Test Failures (88 failures, 4 root causes)

The test suite has **88 failures** that are NOT caused by your changes. They cluster into 4 root causes. Do not waste time "fixing" them individually:

| Cluster | Count       | Root Cause                                                                                                                                                                                                                   | Key Files                                                                                                                                                                   |
| ------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A**   | ~32         | `BaseLLM` has an abstract `close()` method; test fixtures (`ConcreteLLM`, `_FakeLLM`, `_ProviderLLM`) never implement it.                                                                                                    | `tests/llms/test_llm_base.py`, `tests/test_llm_base.py`, `tests/test_runtime_integration.py`, `tests/security/test_sandbox.py`, `tests/runtime/test_runtime_integration.py` |
| **B**   | ~8          | `structlog` API conflict: `logging/structured.py` passes `event="function_call"` inside a `log_data` dict and then calls `logger.info("msg", **log_data)`. The `event` kwarg collides with structlog's first positional arg. | `tests/logging/test_logging_config.py`                                                                                                                                      |
| **C**   | ~25+        | pytest-asyncio is in `Mode.STRICT` (no `asyncio_mode = "auto"` in `pyproject.toml`), but many async tests lack `@pytest.mark.asyncio`.                                                                                       | `tests/integration/test_swarm_integration.py`, `tests/test_openai_reasoning.py`, `tests/test_streaming_providers.py`, `tests/test_tool_reinvocation.py`                     |
| **D**   | ~1 + errors | Test pollution: security/sandbox and API-server tests **pass individually** but fail in the full suite due to global state left by earlier tests.                                                                            | `tests/security/test_sandbox.py`, `tests/test_api_server.py`                                                                                                                |

**Specific errors to recognize:**

- `TypeError: Can't instantiate abstract class ConcreteLLM without an implementation for abstract method 'close'` → Cluster A
- `TypeError: ... got multiple values for argument 'event'` → Cluster B
- `async def functions are not natively supported. You need to install a suitable plugin...` → Cluster C
- `AssertionError: ...` in security/API tests that pass with `pytest <file>` → Cluster D (test pollution)

**Verify:** Always run a failing test in isolation (`pytest <path/to/test_file.py>::<test_name>`) before debugging. If it passes in isolation, it's a pre-existing cluster D issue or cross-test pollution.

**Quick fix:** Adding `asyncio_mode = "auto"` to `[tool.pytest.ini_options]` in `pyproject.toml` would eliminate most Cluster C failures. This is a known one-line fix that has not been applied yet.

### Security & Reliability

- **5 pickle.loads locations** in `memory/storage.py`, `memory/vector_storage.py`, and sandbox backends — **untrusted deserialization risk.** Do not add new `pickle` usage. Prefer `json` or `msgpack`.
- **2 shell=True locations** in `tools/system_tools.py` and `tools/standalone.py` — **command injection risk.** A prior fix attempt (June 20, 2026) did not persist; the code still uses `shell=True` in some paths. When modifying these files, always use `shlex.split(command)` + `shell=False`.
- **API key storage in `bridge/server.py`** stores raw API keys without redaction. `handle_provider_list` spreads them verbatim to clients.
- **FastAPI server** has no CORS middleware, no rate limiting, and auth is disabled by default (`enable_authentication: False`).
- **Dockerfile** source path mismatch: copies `xerxes_agent/` but the actual layout is `src/python/xerxes/`. Version label says `0.0.18` but project is `0.2.6`.
- **Docker Compose** has hardcoded secrets (`xerxes_pass`, `admin`). PostgreSQL and Redis services are declared but **unused** by the runtime code.

### Documentation Drift

- **README says "prompt_toolkit TUI"** but `docs/system-architecture.md` and `docs/design-guidelines.md` describe a **Rust `ratatui` TUI.** There is no Rust code in the repo. The actual TUI is pure Python `prompt_toolkit` (`tui/`).
- **Stale module paths:** `docs/openclaw_parity.md` has 25+ references to `xerxes_agent.*` — the correct import is `xerxes`.
- **CHANGELOG** lives at `docs/changelog.md` instead of the repo root.
- **~160 Sphinx RST stubs** in `docs/api_docs/` are bare `automodule` with no narrative content.

### Performance & Scale

- **No vector index** — brute-force cosine similarity over JSON embeddings in SQLite. `LongTermMemory` and `RAGStorage` will degrade with >10k entries.
- **No async SQLite** — all DB I/O in FastAPI routes is synchronous (blocking).
- **No connection pooling** for SQLite — consider WAL mode and `busy_timeout`.

### Provider Quirks (Intentional — Do Not "Fix")

- **Kimi Code** (`api.kimi.com/coding/v1`) requires a `claude-code/1.0.0` User-Agent or it returns 403. Headers are injected via `provider_default_headers()` in `llms/registry.py`.
- **MiniMax** has no `GET /models` endpoint (returns 404). `bridge/profiles.py` catches this and returns a hardcoded list.
- Custom `base_url` endpoints default to `max_retries=0` (fail fast); official endpoints default to 2 retries.

## Skills & Common Workflows

The `.agents/skills/` directory contains repository-specific skills for repeated workflows:

| Skill                 | When to Use                                                 |
| --------------------- | ----------------------------------------------------------- |
| `add-llm-provider`    | Adding a new LLM provider to `llms/registry.py`             |
| `add-tool-module`     | Adding a new tool to `tools/` with registration boilerplate |
| `add-channel-adapter` | Adding a new messaging channel to `channels/adapters/`      |
| `add-agent-spec`      | Adding a new YAML agent spec to `agents/default/`           |
| `author-skill`        | Creating a new `SKILL.md` skill bundle                      |
| `fix-license-headers` | Normalizing Apache-2.0 headers across the repo              |

See `.agents/skills/<name>/SKILL.md` for step-by-step instructions, exact file paths, and code templates.

## Memory & Context Files

This project uses Xerxes persistent memory. Key project-scoped memory files:

- `tmp-files/bug_hunt_report_2026_06_22.md` — Full security/robustness bug hunt (80 claims, 44 confirmed)
- `tmp-files/bug_hunt_verification_2026_06_22.md` — Independent adversarial re-verification of bug claims
- `tmp-files/code_quality_audit_2026-06-18.md` — Code quality audit (duplication, dead code, performance)
- `tmp-files/AGENT_NOTES.md` — Structure analysis report
- `findings/` (project memory) — Adversarial bug audits (MCP, bridge, security)
- `audits/` (project memory) — Bug-claim verification reports
- `journal/` (project memory) — Daily work logs (2026-06-18 through 2026-06-22)
- `EXPERIENCES.md` (project memory) — Lessons learned (benchmarks, bug fixes, tool patterns)
- `MEMORY.md` (project memory) — Durable facts and decisions (CI gaps, MPS plugin findings, version bumps)
- `KNOWLEDGE.md` (project memory) — Mental models (streaming loop, security stack, memory architecture)

A future agent should read `tmp-files/bug_hunt_report_2026_06_22.md`, `tmp-files/code_quality_audit_2026-06-18.md`, and `MEMORY.md` before making large architectural changes.
