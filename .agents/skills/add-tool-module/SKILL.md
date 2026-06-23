---
name: add-tool-module
description: Scaffold a new tool module in the Xerxes tools/ directory with proper schema, registration, categorization, and optional dependency declarations.
version: 1.0.0
tags: [tools, registration, schema, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when adding a new tool or tool module to the Xerxes framework. A "tool" is any Python callable or class that the LLM agent can invoke via the `FunctionRegistry`.

Examples:
- A new data-processing utility (CSV, JSON, XML transforms)
- A new web service client (API wrapper, scraper, RSS feed parser)
- A new system integration (database, cache, message queue)
- A new AI/ML utility (embedding model, classifier, summarizer)

Do NOT use this for:
- Adding a new LLM provider (use `add-llm-provider` skill)
- Adding a new memory backend (use `add-memory-backend` skill)
- Adding a new channel adapter (use `add-channel-adapter` skill)

# How to use

## 1. Pick a module name and location

New tool modules live under `src/python/xerxes/tools/`. The filename should be descriptive and `snake_case`:

- Good: `finance_tools.py`, `health_tools.py`, `geo_tools.py`
- Bad: `utils.py`, `helpers.py`, `new_stuff.py`

## 2. Inspect the existing pattern

Read `src/python/xerxes/tools/math_tools.py` or `src/python/xerxes/tools/web_tools.py` for the standard structure. Every tool module follows this pattern:

```python
# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
# ... (Apache-2.0 header)

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MyTool:
    """One-line description of what the tool does.

    Google-style docstring with Args/Returns/Raises/Example blocks.
    """

    @staticmethod
    def get_schema() -> dict[str, Any]:
        """Return the JSON schema for this tool."""
        return {
            "name": "my_tool",
            "description": "What this tool does in one sentence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "What param1 means.",
                    },
                    "param2": {
                        "type": "integer",
                        "description": "What param2 means.",
                    },
                },
                "required": ["param1"],
            },
        }

    @staticmethod
    def static_call(param1: str, param2: int = 0) -> str:
        """Execute the tool."""
        try:
            result = f"Processed {param1} with {param2}"
            return result
        except Exception as e:
            logger.exception("my_tool failed")
            raise
```

**Rules:**
- The class name MUST be `PascalCase` verb-noun (e.g., `CalculateTax`, `SearchNews`, `ConvertCurrency`). This name is the LLM-visible function name.
- Every tool class MUST have a `get_schema()` static method returning a JSON schema dict.
- Every tool class MUST have a `static_call()` static method (or `__call__`) that executes the tool.
- Use `from __future__ import annotations` if you need forward references.
- Start every module with `logger = logging.getLogger(__name__)`.

## 3. Register the tool in `tools/__init__.py`

Open `src/python/xerxes/tools/__init__.py` and:

1. Import your new symbols at the top of the file (follow existing import order: stdlib → third-party → first-party → relative).
2. Add them to `__all__`.
3. Add them to `TOOL_CATEGORIES` under the appropriate category (or create a new category).

Example edits:

```python
# In the import section
from .my_tools import MyTool, MyOtherTool

# In __all__
__all__ = [
    # ... existing tools ...
    "MyTool",
    "MyOtherTool",
    # ...
]

# In TOOL_CATEGORIES
TOOL_CATEGORIES: dict[str, list[str]] = {
    # ... existing categories ...
    "my_category": ["MyTool", "MyOtherTool"],
}
```

**Rules:**
- `TOOL_CATEGORIES` is a dict mapping category name → list of tool class name strings.
- Categories are for UI grouping and permission filtering. Pick an existing category if possible (e.g., `data`, `web`, `math`, `system`).
- Only create a new category if the tool truly doesn't fit any existing one.

## 4. Declare optional dependencies (if any)

If your tool requires a third-party package not in `pyproject.toml` core dependencies, add it to `TOOL_REQUIREMENTS` in `tools/__init__.py`:

```python
TOOL_REQUIREMENTS: dict[str, list[str]] = {
    # ... existing entries ...
    "MyTool": ["requests>=2.31.0"],
    "MyOtherTool": ["beautifulsoup4>=4.12.0", "lxml>=5.0.0"],
}
```

**Rules:**
- Only add requirements for packages NOT already in core dependencies.
- Use the same version specifier format as `pyproject.toml`.
- The framework uses this for runtime dependency checking and optional-install hints.

## 5. Add tests

Create a test file in `tests/tools/` named `test_my_tools.py` (matching the module name):

```python
import pytest
from xerxes.tools.my_tools import MyTool


class TestMyTool:
    def test_get_schema_returns_valid_dict(self):
        schema = MyTool.get_schema()
        assert schema["name"] == "my_tool"
        assert "parameters" in schema

    def test_static_call_basic(self):
        result = MyTool.static_call(param1="hello", param2=42)
        assert "hello" in result
        assert "42" in result

    def test_static_call_missing_required_raises(self):
        with pytest.raises(TypeError):
            MyTool.static_call()  # param1 is required
```

**Rules:**
- Test class name: `Test<MyTool>`.
- At minimum: test `get_schema` returns a valid dict, test `static_call` with happy path, test error paths.
- Mock external HTTP calls and I/O; don't hit real APIs in tests.

## 6. Run lint and type check

```bash
uv run ruff check --fix src/python/xerxes/tools/my_tools.py
uv run ruff check --fix src/python/xerxes/tools/__init__.py
uv run mypy src/python/xerxes/tools/my_tools.py --ignore-missing-imports
```

## 7. Verify registration

Run this to confirm the tool is discoverable:

```bash
uv run python -c "
from xerxes.tools import TOOL_CATEGORIES, __all__
print('my_tool' in __all__)
print('MyTool' in TOOL_CATEGORIES.get('my_category', []))
"
```

## Common pitfalls

- **Missing `__all__` entry:** The tool won't be imported by `from xerxes.tools import *` and won't show up in the agent's tool list.
- **Missing `TOOL_CATEGORIES` entry:** The tool will be importable but not categorized, causing UI grouping issues.
- **Schema name mismatch:** `schema["name"]` must match the class name exactly in `snake_case` if the LLM uses it as the function name. The framework typically auto-derives the function name from the class name, but consistency matters.
- **Missing `get_schema()` or `static_call()`:** The `FunctionRegistry` will raise at registration time.
- **Blocking I/O in `static_call()`:** If the tool does HTTP requests or file I/O, consider making it async or using `asyncio.to_thread()` for long-running operations, especially if called from the async streaming loop. However, the current framework's `static_call` convention is synchronous.
- **Catching and swallowing exceptions:** Let exceptions propagate so the agent loop can surface them to the user. Only catch to add context and re-raise.
