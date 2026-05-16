# Changelog

Selected highlights from the project's git history, grouped by major theme. For the full history, run `git log --oneline --no-merges`.

Project identity has shifted over time: originally `eLLM`, renamed to `AgentX`, then `Calute`, and now **`xerxes-agent`** (with Python module name `xerxes`).

---

## 0.2.0 — 2026-04-16 (current)

### Project rename and restructure

- **Renamed Python module** `xerxes_agent` → `xerxes` ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Distribution package stays `xerxes-agent` for PyPI compatibility.
- All build config (pyproject, hatch hook, pytest.ini, CI, Dockerfile, pre-commit) repointed to the new module.
- Orphan `src/python/xerxes_agent/` shim directory removed.
- Metric `xerxes_agent_switches_total` renamed to `xerxes_switches_total`.

### Security hardening

- **Fixed sandbox pickle-escape vulnerability.** Child-to-parent IPC in both `docker_backend` and `subprocess_backend` now uses JSON instead of pickle. The old symmetric pickle design was an arbitrary-code-execution vector via `__reduce__` (a malicious sandboxed tool could craft a payload that would execute `os.system(...)` in the parent process upon deserialization).
- **Fixed `math_tools.Calculator` eval-sandbox bypass.** Replaced `eval(expr, {"__builtins__": {}})` with an AST-whitelist evaluator that accepts only numeric literals, the arithmetic operators `+ - * / // % **`, unary `+/-`, the functions `sin cos tan log sqrt abs pow exp`, and the constants `pi` and `e`. The previous pattern was bypassable via `().__class__.__bases__[0].__subclasses__()`.
- Both fixes verified with negative proof-of-concept tests.

### Feature work

- CLI feature buildout and scrollable scrollback ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Agent orchestration, hand-off, and planning features ([a816780](https://github.com/erfanzar/Xerxes/commit/a816780)).
- Markdown rendering and CLI enhancements ([edfcfa9](https://github.com/erfanzar/Xerxes/commit/edfcfa9)).
- `Cmd-Enter` submit in the TUI ([7e88262](https://github.com/erfanzar/Xerxes/commit/7e88262)).
- Hardened tool routing and TUI behavior ([20e143b](https://github.com/erfanzar/Xerxes/commit/20e143b)).

### Removed

- **Chainlit UI package.** The `xerxes/ui/` subpackage wrapping Chainlit was removed; UI now belongs outside the core framework. Removed `chainlit` from optional dependencies, eliminated the `ui`/`research`/`full` extras' Chainlit dependencies, and stripped `create_ui` methods from `Xerxes`, `Cortex`, `CortexAgent`, `CortexTask`, `TaskCreator`, and `DynamicCortex`.
- Outdated `docs/api_docs/calute.rst` and other stale Sphinx references.

### Test fixes

- 15 previously-failing tests restored to passing. All referenced the old `xerxes_agent.*` module via `mock.patch` / `monkeypatch.setattr` strings that static-analysis couldn't catch. Rewritten to use `xerxes.*`.
- pytest coverage target corrected from nonexistent `src/python/xerxes_agent` to the real `src/python/xerxes`.

---

## 0.2.0 prerelease — early 2026

- **Modular runtime and TUI** ([88e9ac8](https://github.com/erfanzar/Xerxes/commit/88e9ac8)) — released as "Calute 0.2.0".
- Enhanced LLM functionality: reasoning-token extraction, sampling parameter overrides, compat shims for OpenAI-wire-compatible providers ([dcdbfdf](https://github.com/erfanzar/Xerxes/commit/dcdbfdf)).
- Comprehensive docstrings added across the codebase ([d2067be](https://github.com/erfanzar/Xerxes/commit/d2067be)).

---

## 0.1.x series

### 0.1.2 ([bda33b9](https://github.com/erfanzar/Xerxes/commit/bda33b9))

- Modified `create_ui` parameter signature (since removed).

### 0.1.1 ([659cf05](https://github.com/erfanzar/Xerxes/commit/659cf05))

- Added `create_ui` method across `Xerxes` / `Cortex` / `CortexAgent` / `CortexTask` (since removed in 0.2.0).
- Gradio-based UI components added ([6194f5f](https://github.com/erfanzar/Xerxes/commit/6194f5f)) — later replaced by Chainlit, now removed.

### 0.1.0 core

- **MCP (Model Context Protocol) integration** ([474d7f7](https://github.com/erfanzar/Xerxes/commit/474d7f7), [8d1afaf](https://github.com/erfanzar/Xerxes/commit/8d1afaf)) — stdio + SSE + streamable-HTTP transports; `MCPClient`, `MCPManager`, `MCPTool`, `MCPResource` public API.

---

## 0.0.x series (Calute era)

- **Architecture overhaul with new LLM abstraction layer** ([7779931](https://github.com/erfanzar/Xerxes/commit/7779931)) — introduced `BaseLLM` as the abstract provider interface.
- **Migrated from custom chat types to OpenAI proxy types** ([9b4fbf1](https://github.com/erfanzar/Xerxes/commit/9b4fbf1)); improved API server streaming.
- **API server (FastAPI)** added in 0.0.19 ([0535858](https://github.com/erfanzar/Xerxes/commit/0535858)).
- **EnhancedMemoryStore** and general helper assistant ([cf05bec](https://github.com/erfanzar/Xerxes/commit/cf05bec)).
- Comprehensive tools suite: web search, code execution, browser tools, data tools, math tools, etc. ([45edc91](https://github.com/erfanzar/Xerxes/commit/45edc91), [061a762](https://github.com/erfanzar/Xerxes/commit/061a762)).
- Advanced delegation engine + memory management system ([02255dc](https://github.com/erfanzar/Xerxes/commit/02255dc)).
- `reinvoke_after_function` option for re-prompting LLM after tool execution ([d54b9ec](https://github.com/erfanzar/Xerxes/commit/d54b9ec)).
- Python 3.13 support ([9909837](https://github.com/erfanzar/Xerxes/commit/9909837)).
- XerxesConfig hierarchical config system; `/context`, `/cost`, `/compact` slash commands.

---

## Historical (pre-Xerxes: eLLM / AgentX era)

Deep history predating the current multi-agent framework shape. Highlights:

- eLLM → AgentX rename with project restructure ([8ecb339](https://github.com/erfanzar/Xerxes/commit/8ecb339)).
- Early Gradio GUI experiments (later deprecated).
- Initial Ollama backend support ([38654fc](https://github.com/erfanzar/Xerxes/commit/38654fc)).
- Original RAG work, LLM quantization support, WebSocket server experiments.

These commits are preserved for archaeological purposes but don't reflect current API or design.

---

## How to read this

Commit SHAs link to the GitHub commit if the repository is public. Dates are approximate (inferred from tag history and commit clustering, not author dates).

For a precise diff between any two points in history:

```bash
git log --oneline <from>..<to>
git diff --stat <from>..<to>
```

For a live changelog of the most recent work:

```bash
git log --oneline --no-merges -20
```
