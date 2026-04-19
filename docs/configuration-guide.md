# Configuration Guide

Everything that can be tuned from outside the code: `XerxesConfig`, environment variables, runtime feature flags, per-agent overrides, and provider profiles.

## XerxesConfig

[core/config.py](../src/python/xerxes/core/config.py) defines a hierarchical Pydantic config that captures every externally-tunable parameter. It's **optional** — the framework runs fine without it — but if you want a single file to rule your deployment, this is it.

### Structure

```python
from xerxes.core.config import XerxesConfig

config = XerxesConfig(
    environment="PRODUCTION",
    debug=False,

    executor={
        "timeout": 30.0,
        "max_retries": 3,
        "max_concurrent": 10,
        "metrics_enabled": True,
    },
    memory={
        "max_short_term_entries": 100,
        "max_long_term_entries": 5000,
        "persistence_enabled": True,
        "persistence_path": "~/.xerxes/memory.db",
        "embedding_model": "sentence-transformers",
    },
    security={
        "input_validation": True,
        "output_sanitization": True,
        "rate_limit_per_minute": 60,
        "allowed_functions": [],       # empty list = allow all
        "blocked_functions": ["system_tools.ExecuteShell"],
    },
    llm={
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "streaming": True,
    },
    logging={
        "level": "INFO",
        "format": "json",
        "file": "~/.xerxes/xerxes.log",
    },
    observability={
        "tracing_enabled": True,
        "metrics_enabled": True,
        "profiling_enabled": False,
    },
    plugins={},               # plugin-specific config dicts
    features={
        "agent_switching": True,
        "function_chaining": True,
        "context_awareness": True,
        "auto_retry": True,
    },
)
```

### Loading sources

```python
# From file (JSON or YAML auto-detected by extension)
config = XerxesConfig.from_file("./xerxes.yaml")

# From environment variables (prefix XERXES_)
config = XerxesConfig.from_env()

# Merge
config = XerxesConfig.from_file("./base.yaml").merge(XerxesConfig.from_env())

# Current global
from xerxes.core.config import get_config, set_config
set_config(config)
current = get_config()
```

## Environment variables

### Core

| Variable | Purpose | Default |
|----------|---------|---------|
| `XERXES_HOME` | Root directory for profiles, memory DBs, sessions, daemon socket | `~/.xerxes` |
| `XERXES_DEBUG` | Enable debug logging (`True` / `False`) | `False` |
| `XERXES_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |
| `XERXES_LOG_FORMAT` | `json` or `human` | `human` |
| `XERXES_CONFIG_FILE` | Path to a YAML/JSON `XerxesConfig` | (none) |

All `XERXES_HOME`-derived paths (profiles, daemon socket, memory DB, logs) are resolved at import time via [core/paths.py](../src/python/xerxes/core/paths.py) — changing the env var mid-process won't take effect without reloading the module.

### LLM provider API keys

Recognized by [llms/registry.py](../src/python/xerxes/llms/registry.py):

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `KIMI_API_KEY` / `MOONSHOT_API_KEY` | Kimi (Moonshot) |
| `QWEN_API_KEY` / `DASHSCOPE_API_KEY` | Qwen (Alibaba) |
| `ZHIPU_API_KEY` | Zhipu |
| `OLLAMA_HOST` | Ollama base URL (default `http://localhost:11434`) |
| `LMSTUDIO_HOST` | LM Studio base URL |

Resolution order: value passed to the LLM constructor → `LLMConfig.api_key` → env var → `ProviderConfig.default_api_key`.

### Feature toggles

| Variable | Purpose |
|----------|---------|
| `XERXES_RUNTIME_ENABLED` | Master switch for `RuntimeFeaturesConfig` |
| `XERXES_SANDBOX_MODE` | `OFF` / `WARN` / `STRICT` |
| `XERXES_SANDBOX_BACKEND` | `docker` / `subprocess` |
| `XERXES_SANDBOX_TIMEOUT` | Seconds (default 30) |
| `XERXES_SANDBOX_MEMORY_MB` | Per-sandboxed-call memory cap (default 512) |
| `XERXES_PERMISSION_MODE` | `auto` / `accept-all` / `manual` |

### OpenTelemetry

| Variable | Purpose |
|----------|---------|
| `OTEL_SERVICE_NAME` | e.g. `xerxes-prod` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | `key1=value1,key2=value2` |
| `OTEL_RESOURCE_ATTRIBUTES` | `environment=prod,team=agents` |

### Sentry / Datadog / Prometheus

- `SENTRY_DSN` — enables Sentry error reporting via `sentry-sdk`.
- `DD_AGENT_HOST`, `DD_ENV`, `DD_SERVICE`, `DD_VERSION` — Datadog standard env vars.
- Prometheus is pull-based: expose `/metrics` via your own FastAPI integration using `prometheus-client`.

## RuntimeFeaturesConfig

The master switch for the opt-in feature layer.

```python
from xerxes.runtime import RuntimeFeaturesConfig
from xerxes.runtime.profiles import PromptProfile

features = RuntimeFeaturesConfig(
    enabled=True,
    plugin_dirs=["./plugins", "~/.xerxes/plugins"],
    skill_dirs=["./skills", "~/.xerxes/skills"],
    sandbox={
        "mode": "WARN",
        "backend_type": "subprocess",
        "sandboxed_tools": {"ExecuteShell", "ExecutePythonCode"},
        "sandbox_timeout": 30.0,
        "sandbox_memory_limit_mb": 512,
    },
    policy={
        "allowed_tools": None,          # None = all
        "blocked_tools": [],
        "loop_detection_enabled": True,
        "max_same_tool_in_window": 5,
    },
    prompt_profile=PromptProfile.FULL,
    session_store="./sessions",
    audit_sink="./audit.jsonl",
    operator_tools_enabled=False,
)

xerxes = Xerxes(llm=llm, runtime_features=features)
```

### Per-agent overrides

```python
from xerxes.runtime.features import AgentRuntimeOverrides

override = AgentRuntimeOverrides(
    policy={"blocked_tools": ["WebScraper"]},
    sandbox={"mode": "STRICT"},
    enabled_skills=["research"],
    prompt_profile=PromptProfile.COMPACT,
)

agent.runtime_overrides = override   # consumed by Xerxes.run
```

Any field left as `None` inherits from the global `RuntimeFeaturesConfig`.

## PromptProfile

Controls how much context goes into the system prompt. See [system-architecture.md#runtime-profiles](system-architecture.md#runtime-profiles) for the summary table.

For most workloads, the default `FULL` is correct. Drop to `COMPACT` for delegated sub-agents (saves tokens), `MINIMAL` for internal tool agents that only need the tool list, and `NONE` for OpenClaw-style control where the caller supplies the entire prompt.

## Per-agent sampling

`Agent` carries its own sampling params — **override them at the HTTP layer** (via `temperature` / `top_p` / etc. in the request) or at the Python layer (by mutating `agent.temperature`, `agent.max_tokens`, etc., before calling `xerxes.run`).

```python
agent = Agent(
    id="writer",
    instructions="You write haiku.",
    model="gpt-4o",
    temperature=0.9,
    top_p=0.95,
    max_tokens=256,
    presence_penalty=0.3,
    frequency_penalty=0.3,
    stop=["\n\n"],
    function_timeout=10.0,
    max_function_retries=2,
)
```

## Provider profiles

The Rust CLI and bridge read **provider profiles** from `$XERXES_HOME/profiles/*.json`. Each profile records:

```json
{
  "name": "openai-gpt4",
  "provider": "openai",
  "model": "gpt-4o",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "extra": {}
}
```

Active profile is selected via `xerxes --profile <name>` or from the TUI's `/provider` slash command. Profiles can be created interactively in the TUI's provider-setup wizard.

## Per-session `XERXES.md`

At startup, `runtime.bootstrap` loads two `XERXES.md` files if present:

1. `~/.xerxes/XERXES.md` — user-global context.
2. `./XERXES.md` (project root) — project-specific context.

Their content is injected into the system prompt (respecting the active `PromptProfile`). Typical use: project conventions, domain glossaries, repo layout hints.

Analogous to Claude Code's `CLAUDE.md` — structure is free-form markdown.

## Memory backend selection

```python
from xerxes.memory import MemoryStore, MemoryType
from xerxes.memory.storage import SQLiteStorage, FileStorage, SimpleStorage

store = MemoryStore(
    storage=SQLiteStorage(path="~/.xerxes/memory.db"),
    short_term_capacity=100,
    long_term_capacity=5000,
)

xerxes = Xerxes(llm=llm, memory_store=store)
```

Available storage backends:

- `SimpleStorage` — in-memory dict. Fast, transient, great for tests.
- `FileStorage` — pickle files on disk. Portable, slow for large sets.
- `SQLiteStorage` — default persistent backend. Fast enough for 100k-entry ranges.
- `RAGStorage` — wrapper that adds semantic-search capability via an `Embedder`.

For vector search specifically, use [memory.vector_storage.SQLiteVectorStorage](../src/python/xerxes/memory/vector_storage.py) — JSON-encoded embeddings, no pickle, portable across machines.

## Sandbox config

[security/sandbox.py](../src/python/xerxes/security/sandbox.py) defines `SandboxConfig`:

```python
from xerxes.security.sandbox import SandboxConfig, SandboxMode, SandboxBackendConfig

sandbox = SandboxConfig(
    mode=SandboxMode.STRICT,
    sandboxed_tools={"ExecuteShell", "ExecutePythonCode", "WebScraper"},
    elevated_tools={"ReadFile"},          # exempt, runs on host regardless
    sandbox_timeout=30.0,
    sandbox_memory_limit_mb=512,
    sandbox_network_access=False,
    backend_type="docker",                # or "subprocess"
    backend_config=SandboxBackendConfig(
        image="python:3.12-slim",
        mount_paths={"/host/data": "/container/data"},
        mount_readonly=True,
        env_vars={"DEBUG": "1"},
    ),
)
```

Modes:

- **OFF** — no sandboxing. Default.
- **WARN** — log every sandboxed tool call; still run on host.
- **STRICT** — enforce sandbox. If the backend is unavailable, the tool call raises `SandboxExecutionUnavailableError`.

Backends:

- **subprocess** — cheap, always available, no filesystem/network isolation, but process isolation + memory cap (Unix) via `resource.RLIMIT_AS`.
- **docker** — real container, full isolation, requires Docker daemon.

**Security note:** both backends serialize child→parent results as JSON, not pickle. This closes a `__reduce__` escape that existed historically. See [debug/260418-1007-fullhunt/findings.md](../debug/260418-1007-fullhunt/findings.md).

## Audit sinks

```python
from xerxes.audit import AuditEmitter
from xerxes.audit.collector import (
    InMemoryCollector,
    JSONLSinkCollector,
    CompositeCollector,
)

sinks = CompositeCollector([
    JSONLSinkCollector("./audit.jsonl"),
    InMemoryCollector(max_events=10_000),
])

emitter = AuditEmitter(sinks)
```

OTEL bridge: if `runtime.audit_otel_enabled = True` and `opentelemetry-sdk` is installed, events are also emitted as OTLP traces via `xerxes.audit.otel_exporter`.

## Session store

```python
from xerxes.session import FileSessionStore, InMemorySessionStore

store = FileSessionStore("~/.xerxes/sessions/")
# or transient:
store = InMemorySessionStore()
```

Wire into `RuntimeFeaturesConfig.session_store`. The runtime writes one `SessionRecord` per conversation; use `ReplayView` to inspect past runs.

## See also

- [deployment-guide.md](deployment-guide.md) — how these configs are used in Docker / daemon / CI.
- [api-reference.md](api-reference.md) — which of these affect HTTP behavior.
- [system-architecture.md](system-architecture.md) — where each of these plugs in.
