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
"""Cross-cutting helpers: async/sync bridging, tool-schema reflection, token estimation.

* :func:`run_sync` executes a coroutine from sync code, transparently spawning
  a worker thread when a loop is already running.
* :class:`XerxesBase` is the strict Pydantic v2 base used by data classes
  across the codebase (``extra="forbid"``, ``validate_default=True``).
* :func:`function_to_json` builds the OpenAI-compatible tool schema for any
  Python callable by inspecting its signature, resolved type hints, and
  Google-style docstring.
* :func:`estimate_tokens` and :func:`estimate_messages_tokens` give the
  cheap heuristic counts used when no provider tokenizer is available.
"""

import asyncio
import inspect
import re
from collections.abc import Callable, Coroutine
from datetime import datetime
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Block until ``coro`` finishes, handling the nested-loop case.

    When called outside any event loop, runs the coroutine directly via
    :func:`asyncio.run`. When already inside a running loop, spawns a worker
    thread with its own loop so we don't deadlock.
    """
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class XerxesBase(BaseModel):
    """Strict Pydantic base: forbids extra fields, validates defaults, uses enum values."""

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)


def debug_print(debug: bool, *args: str) -> None:
    """Print ``args`` with an ANSI-coloured timestamp when ``debug`` is true."""
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target: dict, source: dict) -> None:
    """Concatenate string fields and recurse into nested dicts; mutates ``target`` in place."""
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    """Fold a streaming ``delta`` chunk (text and tool_calls) into the accumulator."""
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Cheap heuristic token estimate from character length (≥ 1 for non-empty text)."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Sum :func:`estimate_tokens` over ``messages`` content with a 4-token per-message overhead."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += estimate_tokens(str(content))
        total += 4
    return total


def get_callable_public_name(func: Callable[..., Any]) -> str:
    """Return the wire-facing name for ``func``: ``__xerxes_schema__['name']`` or ``__name__``."""
    schema = getattr(func, "__xerxes_schema__", None)
    if isinstance(schema, dict):
        name = schema.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return getattr(func, "__name__", str(func))


def function_to_json(func: Callable[..., Any]) -> dict:
    """Build an OpenAI-style ``{type:"function", function:{...}}`` schema for ``func``.

    Parameter types are mapped from resolved type hints; descriptions come
    from the function's Google-style ``Args:`` block. A ``__xerxes_schema__``
    override (set by hand-tuned tools) wins over reflection.

    Raises:
        ValueError: When ``func``'s signature cannot be inspected.
    """
    schema = getattr(func, "__xerxes_schema__", None)
    custom_name = None
    custom_description = None
    custom_parameters = None
    if isinstance(schema, dict):
        custom_name = schema.get("name")
        custom_description = schema.get("description")
        custom_parameters = schema.get("parameters")

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        tuple: "array",
        set: "array",
        bytes: "string",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {get_callable_public_name(func)}: {e!s}") from e
    docstring = inspect.getdoc(func) or ""
    try:
        resolved_hints = get_type_hints(func)
    except Exception:
        resolved_hints = {}

    param_descriptions: dict[str, str] = {}
    args_match = re.search(r"(?:^|\n)Args:\n(?P<body>.*?)(?:\n[A-Z][A-Za-z ]*:\n|\Z)", docstring, re.DOTALL)
    if args_match:
        current_name: str | None = None
        current_lines: list[str] = []
        for raw_line in args_match.group("body").splitlines():
            if not raw_line.strip():
                continue
            param_match = re.match(r"\s{4}(\w+)(?:\s*\([^)]+\))?\s*:\s*(.*)", raw_line)
            if param_match:
                if current_name is not None:
                    param_descriptions[current_name] = " ".join(part.strip() for part in current_lines if part.strip())
                current_name = param_match.group(1).strip()
                current_lines = [param_match.group(2).strip()]
                continue
            if current_name is not None and raw_line.startswith(" " * 8):
                current_lines.append(raw_line.strip())
        if current_name is not None:
            param_descriptions[current_name] = " ".join(part.strip() for part in current_lines if part.strip())

    if not param_descriptions:
        param_pattern = r"(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)"
        matches = re.findall(param_pattern, docstring, re.DOTALL | re.MULTILINE)
        for param_name, description in matches:
            param_descriptions[param_name.strip()] = description.strip()

    parameters = {}
    for param in signature.parameters.values():
        if param.name == "context_variables" or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        param_info: dict[str, Any] = {"type": "string"}

        annotation = resolved_hints.get(param.name, param.annotation)
        if annotation != inspect.Parameter.empty:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin in (Union, UnionType):
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1 and type(None) in args:
                    param_info["type"] = type_map.get(non_none_types[0], "string")
                else:
                    param_info = {"type": "union", "types": [type_map.get(arg, "string") for arg in args]}
            elif origin in (list, tuple, set):
                param_info["type"] = "array"
                if args:
                    param_info["items"] = {"type": type_map.get(args[0], "string")}
            elif origin is dict:
                param_info["type"] = "object"
            elif annotation in type_map:
                param_info["type"] = type_map[annotation]
            else:
                param_info["type"] = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]
        if param.default != inspect.Parameter.empty:
            pass

        parameters[param.name] = param_info

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect.Parameter.empty
        and param.name != "context_variables"
        and param.kind != inspect.Parameter.VAR_KEYWORD
    ]

    description_parts: list[str] = []
    if isinstance(custom_description, str) and custom_description.strip():
        description_parts.append(custom_description.strip())
    if docstring.strip():
        if not description_parts or docstring.strip() != description_parts[-1]:
            description_parts.append(docstring.strip())

    return {
        "type": "function",
        "function": {
            "name": custom_name or get_callable_public_name(func),
            "description": "\n\n".join(description_parts),
            "parameters": (
                custom_parameters
                or {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                }
            ),
        },
    }
