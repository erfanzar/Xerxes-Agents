# Deployment Guide

How to package, ship, and run Xerxes in production-adjacent environments.

## Packaging

Xerxes uses **hatchling** with a custom Rust build hook to produce a single wheel that bundles:

- The Python `xerxes` package
- A compiled Rust binary at `xerxes/_bin/xerxes`
- A Python launcher at `xerxes/_bin/_launcher.py` (exposed as the `xerxes` console script)

### Build a wheel

```bash
uv build --wheel
# → dist/xerxes_agent-0.2.0-py3-none-any.whl
```

Under the hood, [hatch_build.py](../hatch_build.py) bundles the TypeScript CLI before hatch packs the wheel.

### Install

```bash
# From local wheel
pip install dist/xerxes_agent-0.2.0-py3-none-any.whl

# Editable dev install
uv pip install -e .

# With optional extras
pip install "xerxes-agent[search,monitoring,vectors,mcp]"    # = [full]
pip install "xerxes-agent[search,vectors]"                   # = [research]
pip install "xerxes-agent[monitoring]"                       # = [enterprise]
```

Available extras (see [pyproject.toml](../pyproject.toml)):

| Extra | Adds |
|-------|------|
| `dev` | pytest, ruff, black, mypy, pre-commit, ipykernel, notebook |
| `search` | (currently empty — reserved) |
| `monitoring` | structlog, prometheus-client, opentelemetry-sdk, sentry-sdk, datadog, psutil |
| `vectors` | scikit-learn, sentence-transformers |
| `mcp` | mcp |
| `full` | search + monitoring + vectors + mcp |
| `research` | search + vectors |
| `enterprise` | monitoring |

### CLI entry point

After install, `xerxes` is on `$PATH` and launches the Rust TUI via the Python shim:

```bash
xerxes                              # launch TUI with default config
xerxes --model gpt-4o --profile openai
xerxes send "summarize the current branch"   # push to running daemon
xerxes wakeup --install             # install systemd/launchd unit (platform-dependent)
```

## Docker

[Dockerfile](../Dockerfile) builds a slim production image in two stages:

```
python:3.11-slim (builder)         python:3.11-slim (runtime)
│                                   │
├─ gcc, g++, make, libpq-dev        ├─ libpq5, curl
├─ pip wheel . → /wheels            ├─ pip install /wheels/*
│                                   ├─ COPY src/python/xerxes → ./xerxes
│                                   ├─ non-root user `xerxes`
│                                   ├─ HEALTHCHECK: `python -c "import xerxes"`
│                                   └─ CMD python -m xerxes
```

Build & run:

```bash
docker build -t xerxes:local .
docker run --rm -e OPENAI_API_KEY=sk-… -p 8000:8000 xerxes:local
```

### docker-compose

[docker-compose.yml](../docker-compose.yml) wires up a full development stack:

```yaml
services:
  xerxes:      # main app, port 8000
  postgres:    # optional persistence target
  redis:       # optional caching
  prometheus:  # metrics scrape
  grafana:     # metrics dashboards (admin/admin)
  jupyter:     # notebook environment
```

Bring it up:

```bash
export OPENAI_API_KEY=sk-…
docker compose up -d xerxes postgres redis
# then optionally:
docker compose up -d prometheus grafana
```

> Note: postgres/redis are placeholders for future backends. The current `MemoryStore` defaults to SQLite; postgres/redis config blocks in `pyproject.toml` are commented out.

## Daemon mode

[xerxes.daemon](../src/python/xerxes/daemon/) runs Xerxes as a long-lived background agent that accepts work over **WebSocket** (external) and **Unix socket** (local).

### Starting

```bash
# foreground
python -m xerxes.daemon

# via CLI wrapper
xerxes wakeup                    # start
xerxes wakeup --status           # check
xerxes wakeup --stop             # graceful shutdown
xerxes wakeup --install          # install systemd unit (Linux) / launchd plist (macOS)
xerxes wakeup --uninstall
```

### Config ([xerxes.daemon.config.DaemonConfig](../src/python/xerxes/daemon/config.py))

- `ws_host` — WebSocket bind host (default `127.0.0.1`)
- `ws_port` — WebSocket bind port (default `8765`)
- `socket_path` — Unix domain socket (default `$XERXES_HOME/daemon.sock`)
- `max_concurrent_tasks` — executor pool size
- `log_dir` — daemon log directory
- `auth_token` — shared secret for WebSocket clients (optional but recommended)

### Submitting work

Local:

```bash
xerxes send "refactor tests/test_auth.py"
xerxes send "summarize today's PRs" --wait
```

Remote (WebSocket):

```bash
websocat ws://host:8765 <<< '{"method":"task.create","params":{"prompt":"hello","token":"…"}}'
```

Events stream back per task in real time (`task.started`, `task.progress`, `task.tool`, `task.tool_done`, `task.completed`, `task.failed`).

## FastAPI server

For API-style deployments (drop-in OpenAI replacement), run [xerxes.api_server.XerxesAPIServer](../src/python/xerxes/api_server/server.py):

```python
from xerxes import Agent, OpenAILLM, Xerxes
from xerxes.api_server import XerxesAPIServer

llm = OpenAILLM(api_key=...)
agent = Agent(id="assistant", instructions="Be helpful.", model="gpt-4o")

xerxes = Xerxes(llm=llm)
xerxes.register_agent(agent)

XerxesAPIServer(
    agents={"assistant": agent},
    xerxes_instance=xerxes,
).run(host="0.0.0.0", port=8000)
```

Behind a reverse proxy (nginx / Caddy / Traefik), terminate TLS and set:

- `proxy_read_timeout` → at least the LLM streaming timeout (120s+ is safe).
- `proxy_buffering off` — required for SSE streaming to not buffer.

See [api-reference.md](api-reference.md) for the full endpoint spec.

## CI / CD

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every push:

```
┌─ lint ──────────────────────────────────────────┐
│ ruff check src/python/xerxes/                   │
│ black --check src/python/xerxes/                │
│ mypy src/python/xerxes/ --ignore-missing-imports│
└─────────────────────────────────────────────────┘

┌─ test (matrix) ────────────────────────────────┐
│ ubuntu-latest   × python 3.10/3.11/3.12/3.13   │
│ macos-latest    × python 3.10/3.11/3.12/3.13   │
│ windows-latest  × python 3.10/3.11/3.12        │
│                                                │
│ pytest tests/ --cov=src/python/xerxes          │
└────────────────────────────────────────────────┘

┌─ security ───────────────────────────────────────┐
│ bandit -r src/python/xerxes/                    │
│ safety check                                     │
│ pip-audit                                        │
└──────────────────────────────────────────────────┘
```

Pre-commit hooks (see [.pre-commit-config.yaml](../.pre-commit-config.yaml)) run a subset locally on commit.

## Environment variables

Set at deploy time — see [configuration-guide.md](configuration-guide.md) for the full list:

Required (one of):
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OLLAMA_HOST`

Recommended:
- `XERXES_HOME` → `/var/lib/xerxes` (for session + memory persistence)
- `XERXES_LOG_LEVEL` → `INFO`
- `XERXES_LOG_FORMAT` → `json`
- `XERXES_SANDBOX_MODE` → `STRICT` in untrusted environments

Observability:
- `OTEL_EXPORTER_OTLP_ENDPOINT` → your OTLP collector
- `SENTRY_DSN` → error reporting
- `DD_AGENT_HOST` + DD_SERVICE / DD_ENV → Datadog

## Persistence

By default, Xerxes writes to `$XERXES_HOME` (falls back to `~/.xerxes`):

```
$XERXES_HOME/
├── XERXES.md                 # global context
├── agents/                   # user agent definitions
├── skills/                   # user skills
├── plugins/                  # user plugins
├── profiles/                 # provider profiles (JSON)
├── sessions/                 # FileSessionStore entries
├── memory.db                 # default SQLite memory
├── daemon.sock               # Unix socket (when daemon is up)
└── xerxes.log                # structured log file (if enabled)
```

In Docker, mount this as a volume to persist across restarts:

```yaml
services:
  xerxes:
    volumes:
      - ./xerxes_home:/var/lib/xerxes
    environment:
      - XERXES_HOME=/var/lib/xerxes
```

## Resource sizing

Empirical baselines (per idle agent, model-independent):

| Scenario | RSS | Notes |
|----------|-----|-------|
| Bare `Xerxes` with one LLM | ~80 MB | No memory store, no runtime features |
| + SQLite MemoryStore | ~120 MB | Grows with memory size |
| + Runtime features + audit | ~150 MB | Session store + emitter |
| + Playwright browser | +400 MB per active page | Only when operator tools are enabled |
| API server (1 worker) | +50 MB | FastAPI + uvicorn |
| Daemon (WS+socket) | +30 MB | |

Scale by running multiple API-server processes behind a load balancer — the framework is stateless per-request (conversation state lives in the client-supplied `messages` history, or in the session store if enabled).

## Upgrade path

1. Bump `version` in [pyproject.toml](../pyproject.toml).
2. Update [docs/changelog.md](changelog.md).
3. `uv build --wheel` and verify the wheel contents (`unzip -l dist/xerxes_agent-*.whl`).
4. Tag and release.

Between **0.x** releases, expect API changes in any public symbol — the project is pre-1.0. Pin a specific version in production.

## Related

- [api-reference.md](api-reference.md) — HTTP surface.
- [configuration-guide.md](configuration-guide.md) — every tunable.
- [testing-guide.md](testing-guide.md) — running tests in CI.
- [system-architecture.md](system-architecture.md) — what each component does.
