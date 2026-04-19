# Testing Guide

How Xerxes tests are organized, how to run them, and how to add new ones.

## Overview

- **Framework:** pytest (async tests via `pytest-asyncio`, mocking via `pytest-mock`, coverage via `pytest-cov`).
- **Total:** 1501 tests at the time of writing.
- **Runtime:** ~23 seconds for the full suite on a recent MacBook.
- **Target coverage:** 80% (enforced by `--cov-fail-under=80` in [pytest.ini](../pytest.ini)).
- **Config:** [pytest.ini](../pytest.ini).

## Running tests

### Everything

```bash
PYTHONPATH=src/python .venv/bin/python -m pytest tests/
```

Or, if you have `xerxes-agent` installed in editable mode (`uv pip install -e .`), just:

```bash
pytest tests/
```

### One file, one class, one test

```bash
pytest tests/test_cortex_task.py
pytest tests/test_cortex_task.py::TestCortexTask
pytest tests/test_cortex_task.py::TestCortexTask::test_execute
```

### Without coverage (faster)

```bash
pytest tests/ --no-cov
```

### Stop at first failure

```bash
pytest tests/ -x --tb=short
```

### Filter by keyword

```bash
pytest tests/ -k "sandbox and not docker"
```

### Verbose event stream

```bash
pytest tests/test_runtime_integration.py -v -s
```

## Organization

Tests live in [tests/](../tests/) with a flat-ish structure — one `test_<module>.py` per target module. No parallel directory mirror of `src/python/xerxes/`.

```
tests/
├── conftest.py                   # shared fixtures
├── test_agent_types.py
├── test_ai_tools.py
├── test_api_server.py
├── test_audit.py / test_audit_events.py
├── test_basics.py                # smoke tests
├── test_channel_adapters.py
├── test_channels_framework.py
├── test_circuit_breaker.py
├── test_coding_tools.py
├── test_compaction_strategies.py
├── test_config.py
├── test_converters.py
├── test_cortex_memory.py
├── test_cortex_task.py
├── test_cortex_tool.py
├── test_cortex_utils.py
├── test_data_tools.py
├── test_dependency.py
├── test_embedders.py
├── test_entity_memory.py
├── test_errors.py
├── test_executors.py
├── test_executors_detailed.py
├── test_fallback.py
├── test_hooks.py
├── test_hybrid_retrieval.py
├── test_llm_base.py
├── test_logging_config.py
├── test_loggings.py
├── test_loop_detection.py
├── test_math_tools.py
├── test_mcp_integration.py
├── test_mcp_types.py
├── test_media_tools.py
├── test_memory.py / test_memory_debug.py / test_memory_detailed.py / test_memory_injection.py
├── test_messages.py
├── test_multimodal.py
├── test_openai_reasoning.py
├── test_operator_managers.py / test_operator_tools.py
├── test_otel_exporter.py
├── test_paths.py
├── test_plugins.py
├── test_policy.py
├── test_profile_agent.py / test_profile_decay.py
├── test_prompt_profiles.py
├── test_query_engine.py
├── test_rag_persistence.py
├── test_resilience.py
├── test_rl_tools.py
├── test_runtime_bridge.py / test_runtime_context.py / test_runtime_integration.py
├── test_sandbox.py / test_sandbox_backends.py
├── test_semantic_search.py
├── test_session.py / test_session_replay.py / test_session_search.py / test_session_summarizer.py
├── test_skill_authoring.py / test_skill_drafter.py / test_skill_lifecycle.py / test_skill_matcher.py / test_skill_pipeline.py / test_skill_telemetry.py / test_skill_verifier.py
├── test_skills.py
├── test_standalone_tools.py
├── test_storage.py
├── test_streamer_buffer.py
├── test_streaming_permissions.py
├── test_streaming_providers.py
├── test_system_tools.py
├── test_token_counter.py
├── test_tool_reinvocation.py
├── test_turn_indexer.py
├── test_user_memory.py / test_user_profile.py
├── test_utils.py
└── test_vector_storage.py
```

## Conventions

### Naming

- **File:** `test_<module>.py` where `<module>` matches a `xerxes.*` module. E.g. `test_sandbox.py` for `xerxes.security.sandbox`.
- **Class:** `Test<Subject>`, e.g. `TestSandboxRouter`.
- **Function:** `test_<behavior>`, e.g. `test_execute_timeout`, `test_returns_none_when_key_missing`.

Configured in [pytest.ini](../pytest.ini):

```
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Markers

Declared markers (from [pytest.ini](../pytest.ini)):

```
unit:         Pure unit test (no IO)
integration:  Crosses component boundaries (sandbox, runtime, etc.)
slow:         >1 second expected
asyncio:      Async test (usually auto-applied)
```

Apply with `@pytest.mark.integration` etc. To skip slow tests: `pytest -m "not slow"`.

### Async tests

`asyncio_mode = auto` — any `async def test_*` is automatically wrapped. No need for `@pytest.mark.asyncio` on every function.

```python
async def test_llm_streams():
    llm = FakeLLM()
    chunks = []
    async for chunk in llm.astream_completion(...):
        chunks.append(chunk)
    assert chunks[-1]["is_final"]
```

### Fixtures

Common fixtures live in [conftest.py](../tests/conftest.py). Keep local fixtures *in the test file* if they aren't reused.

### Mocking

Mock the **boundary**, not the internals. When a test needs to stub `subprocess.run`, `httpx.AsyncClient`, `docker info`, etc., use `monkeypatch.setattr` or `mock.patch` targeting the **fully-qualified `xerxes.` path**:

```python
# Right:
with mock.patch("xerxes.security.sandbox_backends.docker_backend.subprocess.run") as mock_run:
    mock_run.return_value = mock.Mock(returncode=0, stdout=b64(result), stderr="")
    ...

# Wrong (old module name, now broken):
with mock.patch("xerxes_agent.security.sandbox_backends.docker_backend.subprocess.run"):
    ...
```

The project was renamed `xerxes_agent` → `xerxes` recently. If you copy-paste from old test fixtures, update the module prefix.

### Integration-style tests

Tests that actually spawn subprocesses (e.g. [test_sandbox_backends.py](../tests/test_sandbox_backends.py) `TestSubprocessBackendExecution`) exercise the real subprocess IPC end-to-end. They're slower but verify the pickle-in / JSON-out invariant.

To run only integration tests:

```bash
pytest tests/ -m integration
```

### Plugin tests

[test_plugins.py](../tests/test_plugins.py) and [test_runtime_integration.py](../tests/test_runtime_integration.py) write tiny plugin Python files to `tmp_path` and discover them. Embedded plugin source uses the `xerxes.extensions.plugins` path:

```python
plugin_code = """
from xerxes.extensions.plugins import PluginMeta, PluginType

PLUGIN_META = PluginMeta(name="my_plugin", version="1.0", plugin_type=PluginType.TOOL)

def my_tool(x: str) -> str:
    return f"tool: {x}"

def register(registry):
    registry.register_tool("my_tool", my_tool, meta=PLUGIN_META)
"""
(tmp_path / "my_plugin.py").write_text(plugin_code)
```

## Coverage

Coverage is configured in [pytest.ini](../pytest.ini):

```
--cov=src/python/xerxes
--cov-report=term-missing
--cov-report=html      # → ./htmlcov/
--cov-report=xml       # → ./coverage.xml
--cov-fail-under=80
```

Open `htmlcov/index.html` in a browser for interactive drill-down. The XML report feeds CI coverage badges.

## Adding a test

1. Put it in `tests/test_<module>.py` where `<module>` matches the code under test.
2. Use `Test<Subject>` class containers when you have ≥2 related tests on the same subject.
3. Mock external I/O at the nearest `xerxes.`-qualified import, not inside helpers.
4. Prefer `async def` over threading for anything exercising the streaming path.
5. Keep fixtures local unless reused in ≥3 tests.
6. Run the full suite before submitting — failures elsewhere often mean you picked up a global mutation from your new test.

```bash
pytest tests/test_your_new_module.py -v
pytest tests/ --no-cov -q   # full sanity pass
```

## Recent test-ordering quirk

A single test (`tests/test_skill_matcher.py::TestSkillMatcher::test_best_returns_single`) occasionally flakes when run after a specific upstream test. It passes reliably in isolation. If you see it flake, re-run; it's a known test-isolation issue not a real bug.

## CI

GitHub Actions runs the full suite + ruff + black + mypy + bandit on every push — see [.github/workflows/ci.yml](../.github/workflows/ci.yml). The matrix covers Python 3.10, 3.11, 3.12, 3.13 across Ubuntu / Windows / macOS (3.13 skipped on Windows).

Tests do not require API keys. Anything that would hit a live LLM is mocked.

## Related

- [code-standards.md](code-standards.md) — conventions these tests enforce.
- [system-architecture.md](system-architecture.md) — what the integration tests exercise.
- [debug/260418-1007-fullhunt/findings.md](../debug/260418-1007-fullhunt/findings.md) — the debug session that caught the last round of packaging + test-path bugs.
