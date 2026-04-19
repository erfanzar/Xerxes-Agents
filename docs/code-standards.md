# Code Standards

Conventions followed across [src/python/xerxes/](../src/python/xerxes/) and [src/rust/](../src/rust/). Not aspirations — this is what the repo actually does.

## Formatting

- **Python line length:** 121 (ruff + black both agree — see [pyproject.toml:133-140](../pyproject.toml)).
- **Python target:** 3.11 for tool config; supported range `>=3.10,<3.14`.
- **Formatter:** `black --preview` + `ruff format` + `isort --profile=black`.
- **Linter:** ruff with rules `A, B, E, F, I, NPY, RUF, UP, W`; ignores `F722, B008, UP015, A005, E501`.
- **Rust:** standard `cargo fmt` defaults.

Run locally:

```bash
ruff check src/python/xerxes/
ruff format src/python/xerxes/
black src/python/xerxes/
```

Pre-commit runs ruff, black, isort, mypy (loose), and bandit — see [.pre-commit-config.yaml](../.pre-commit-config.yaml).

## Typing

- All public functions have full type hints.
- Prefer `X | None` over `Optional[X]` (Python 3.10+ syntax).
- Prefer `dict[str, int]` over `Dict[str, int]`.
- Use `from __future__ import annotations` in modules with forward references or circular type needs.
- Type checkers in use: **mypy** (CI / pre-commit, loose) and **basedpyright** (configured with many report-types set to `none` — treat as advisory, not blocking).

## Pydantic vs dataclasses

| Use Pydantic (BaseModel) when… | Use `@dataclass` when… |
|---|---|
| The object crosses a boundary (HTTP, disk, LLM API, config file) and needs validation | The object is purely in-process and used for ergonomic field access |
| You need JSON schema generation or coercion | You care about mutation speed or `__slots__` |
| The type is user-facing (`Agent`, `XerxesConfig`, `ChatCompletionRequest`) | The type is internal runtime state (`RequestFunctionCall`, `StreamChunk`, `TurnResult`) |

Hybrid is fine — messages (`SystemMessage` / `UserMessage` / `AssistantMessage`) are Pydantic with discriminated unions.

## Async vs sync

- **Public entry points come in both flavors:** `Xerxes.run` (sync), `Xerxes.create_response` (async), `Xerxes.thread_run` (background thread), `Xerxes.athread_run` (async-friendly background).
- **LLM providers are async under the hood.** Sync paths wrap the async machinery with `_AsyncIteratorFromSyncStream` or similar adapters.
- **Streaming is the canonical output.** Non-streaming paths assemble the full result from the stream — do not write a separate non-streaming code path.
- **Don't block the event loop:** long CPU/IO work goes through `ThreadPoolExecutor` or `asyncio.to_thread`.

## Errors

Every thrown exception is a subclass of `XerxesError` (see [src/python/xerxes/core/errors.py](../src/python/xerxes/core/errors.py)):

```
XerxesError (base: message + details dict)
├── AgentError
├── FunctionExecutionError (preserves original_error)
├── XerxesTimeoutError
├── ValidationError
├── RateLimitError
├── XerxesMemoryError
├── ClientError
└── ConfigurationError
```

**Do not** catch `Exception` broadly. If an operation can swallow errors (e.g. memory writes from a hook), log at DEBUG/WARNING and re-raise only for the truly fatal cases.

## Naming

- Modules: `snake_case.py`.
- Classes: `PascalCase`.
- Private helpers: leading underscore, never imported across package boundaries.
- Internal builtin agents: filename prefix `_` (`_coder_agent.py`, `_planner_agent.py`) — they're importable via `xerxes.agents` but the underscore signals "this is a pre-packaged agent definition, not a public API."
- LLM classes: `<Provider>LLM` (`OpenAILLM`, `AnthropicLLM`, `OllamaLLM`, …).
- Tool classes (subclasses of `AgentBaseFn`): `PascalCase` verb noun (`ReadFile`, `WriteFile`, `GoogleSearch`, `ExecutePythonCode`). The class name *is* the LLM-visible function name.

## Docstrings

Google style. Every public class and method has:

- One-line summary.
- Optional extended description.
- `Args:` block if the function takes arguments.
- `Returns:` block if not obvious.
- `Raises:` block for exceptions the caller should expect.
- `Example:` block for non-trivial APIs.

Docstring examples should be runnable. When the project was renamed from `xerxes_agent` → `xerxes`, docstring imports were rewritten in a single pass — keep them in sync with the actual module.

## Comments

Default to **no inline comment.** Add one only when the *why* is non-obvious: a hidden constraint, an invariant, a workaround for a specific bug, or behavior that would surprise a reader.

Do **not** write comments that merely restate what the next line of code does. Well-named identifiers are the documentation.

Do **not** write PR-scoped or task-scoped comments ("added for the X flow", "fixes issue #123") — those go in the commit message or PR description and rot as code moves.

## File size

- Human-written docs: ≤800 lines per file (configurable via `docs.maxLoc`).
- README: ≤300 lines.
- Source files: no hard cap, but modules over ~1500 lines are a smell. [xerxes.py](../src/python/xerxes/xerxes.py) is 2849 lines and has been on the list-of-things-to-split for a while; [tools/claude_tools.py](../src/python/xerxes/tools/claude_tools.py) at ~40k is deliberately a port of the Claude Code tool set kept in one file for diff-ability against upstream.

## Imports

- **Absolute** inside tests and examples: `from xerxes.tools.math_tools import Calculator`.
- **Relative** inside the package: `from ..types import Agent`.
- **Ordered** by ruff's isort integration: stdlib, third-party, first-party, relative.
- **`TYPE_CHECKING` guards** for circular-import-prone typing. Example:

  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from xerxes.cortex.core.enums import ProcessType
  ```

## Tools & functions

A tool is a class that inherits `AgentBaseFn` and implements `static_call(**kwargs) -> dict`:

```python
from xerxes.types import AgentBaseFn

class GreetUser(AgentBaseFn):
    """Greet a user by name.

    Args:
        name: The user's display name.
        formal: If True, use "Dear {name}"; else "Hi {name}".

    Returns:
        {"greeting": str}
    """
    @staticmethod
    def static_call(name: str, formal: bool = False, **context_variables) -> dict:
        if formal:
            return {"greeting": f"Dear {name}"}
        return {"greeting": f"Hi {name}"}
```

**The class name becomes the LLM-visible function name.** The argument schema is introspected from the signature + type hints + default values. `**context_variables` receives runtime injections (agent_id, memory_store, etc.) if the framework binds them.

Return values **must be JSON-serializable** if the tool runs in a sandbox (child→parent IPC is JSON, not pickle). Dicts, lists, strings, numbers, None are fine. Non-JSON types are coerced via `default=repr` — lossy, so avoid.

## LLM providers

A provider is a subclass of [BaseLLM](../src/python/xerxes/llms/base.py) with these overrides:

- `_initialize_client(self)` — set up whatever client object the provider needs.
- `generate_completion(self, prompt, model, temperature, max_tokens, top_p, stop, stream, **kwargs)` async.
- `extract_content(self, response)` — return the text from whatever shape the provider returned.
- `process_streaming_response(self, response, callback)` — iterate streaming deltas, call `callback(chunk)` per chunk.
- `stream_completion(self, response, agent)` / `astream_completion(self, response, agent)` — yield normalized `{"content", "buffered_content", "function_calls", "tool_calls", "is_final"}` dicts.
- `parse_tool_calls(self, raw_data)` — turn provider-specific tool-call data into a list of `FunctionCall`.

Register cost + context limit in [llms/registry.py](../src/python/xerxes/llms/registry.py) so that `calc_cost()`, `detect_provider()`, and `get_context_limit()` work.

For OpenAI-wire-compatible providers (DeepSeek, Qwen, Zhipu, Kimi, LMStudio, Custom), extend `OpenAICompatLLM` instead — `base_url` and `api_key` resolution happens automatically from the registry.

## Channels

A channel is a subclass of [channels.base.Channel](../src/python/xerxes/channels/base.py) with:

- `async start(on_inbound: InboundHandler)` — listen on the underlying transport; for each message, build a `ChannelMessage` and call `on_inbound(msg)`.
- `async send(message: ChannelMessage)` — deliver outbound.
- `async stop()` — graceful shutdown.

All platform-specific details (auth, payload parsing, rate limits) stay inside the adapter. The rest of the framework sees only `ChannelMessage`.

## Memory storage

Implement the `MemoryStorage` protocol (six methods — `save`, `load`, `delete`, `exists`, `list_keys`, `clear`). No base class inheritance needed; Python's structural subtyping takes care of it. See [memory/storage.py](../src/python/xerxes/memory/storage.py) for existing implementations.

## Security

- **Never pass user input to `eval` / `exec` / shell strings** without an AST-level whitelist or safe API. [tools/math_tools.py](../src/python/xerxes/tools/math_tools.py) demonstrates the AST-whitelist pattern for arithmetic.
- **`pickle.loads` only on bytes you produced yourself.** Anything that has crossed a process boundary into attacker-controlled code (the sandbox child, an untrusted file, a network payload) must be deserialized with a format that cannot execute code (JSON, msgpack without ext types, etc.).
- **Every tool call goes through `FunctionExecutor`,** which fires audit events before and after. Do not bypass it even for "safe" tools — the audit trail is load-bearing for debugging.
- **Subprocess invocation:** prefer `subprocess.run(list, shell=False)`. `shell=True` is used intentionally in [tools/standalone.py](../src/python/xerxes/tools/standalone.py) and [tools/system_tools.py](../src/python/xerxes/tools/system_tools.py) because the tool's *purpose* is to execute user-supplied shell strings, but these tools are gated behind policy.

## Testing conventions

See [testing-guide.md](testing-guide.md) for the whole story. Short version:

- `tests/test_<module>.py` → tests for `xerxes.<module>`.
- Async tests use `@pytest.mark.asyncio` or `asyncio_mode = auto` from [pytest.ini](../pytest.ini).
- Mock **boundaries** (the actual `httpx` call, the actual `subprocess.run`), not internal helpers.
- Use `monkeypatch.setattr("xerxes.module.thing", fake)` — always the `xerxes.` prefix, never the old `xerxes_agent.`.

## Rust

- **One workspace crate:** [src/rust/xerxes-cli/](../src/rust/xerxes-cli/).
- Edition: 2021, `cargo fmt` defaults.
- Async runtime: `tokio`.
- Terminal: `ratatui` + `crossterm`.
- CLI parser: `clap`.
- Keep render code in `render.rs` and state machine in `app.rs` — don't intermix.

## Git hygiene

- **Conventional commits** (informal): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. See [recent git log](changelog.md) for the tone.
- Prefer **one commit per logical unit** — multiple fix commits are fine when they land together.
- Do not amend published commits.
- Never skip hooks (`--no-verify`) unless explicitly authorized.

## When in doubt

Read the surrounding file. The repo is 64k LOC with consistent idioms — the local convention almost always wins over anything documented here.
