# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Utility functions for the Calute framework.

This module provides helper functions for debugging, data merging, and function
introspection. It includes utilities for converting Python functions to JSON schema
format, merging response chunks, and debug printing with timestamps.
"""

import asyncio
import inspect
import re
from collections.abc import Coroutine
from datetime import datetime
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")
"""Generic type variable for type-safe operations."""


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously, handling nested event loops.

    This function safely executes a coroutine from synchronous code,
    handling the case where an event loop is already running (e.g., in
    Jupyter notebooks or async frameworks).

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.

    Example:
        >>> async def fetch_data():
        ...     return "data"
        >>> result = run_sync(fetch_data())
        >>> print(result)
        data
    """
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class CaluteBase(BaseModel):
    """Base Pydantic model for Calute configuration classes.

    This class provides a strict configuration base that enforces data validation
    and prevents accidental attribute errors. All Calute configuration models
    should inherit from this class.

    Attributes:
        model_config: Pydantic configuration that forbids extra attributes,
            validates default values, and uses enum values directly.

    Example:
        >>> class MyConfig(CaluteBase):
        ...     name: str
        ...     count: int = 0
        >>> config = MyConfig(name="test")
        >>> config.name
        'test'
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)


def debug_print(debug: bool, *args: str) -> None:
    """Print debug messages with timestamp if debug mode is enabled.

    Args:
        debug: Whether debug mode is enabled.
        *args: Variable number of string arguments to print.

    Returns:
        None

    Example:
        >>> debug_print(True, "Processing", "function", "call")
        [2025-01-27 10:30:45] Processing function call
    """
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target: dict, source: dict) -> None:
    """Recursively merge fields from source dictionary into target dictionary.

    For string values, concatenates them. For dict values, merges recursively.

    Args:
        target: The target dictionary to merge into.
        source: The source dictionary to merge from.

    Returns:
        None (modifies target in place)

    Example:
        >>> target = {"text": "Hello", "nested": {"key": "value"}}
        >>> source = {"text": " World", "nested": {"key": "2"}}
        >>> merge_fields(target, source)
        >>> print(target)
        {"text": "Hello World", "nested": {"key": "value2"}}
    """
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    """Merge a streaming response chunk into the final response.

    Handles special processing for tool_calls and removes role field.

    Args:
        final_response: The accumulated response dictionary.
        delta: The new chunk to merge.

    Returns:
        None (modifies final_response in place)

    Note:
        This function is specifically designed for merging streaming API
        response chunks that may contain tool calls.

    Example:
        >>> response = {"content": "Hello", "tool_calls": [{"name": "func"}]}
        >>> delta = {"content": " World"}
        >>> merge_chunk(response, delta)
        >>> print(response["content"])
        Hello World
    """
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple character-based heuristic. For more accurate counting,
    consider using tiktoken or a provider-specific tokenizer.

    Args:
        text: The text to estimate tokens for.
        chars_per_token: Average characters per token (default: 4.0 for English).

    Returns:
        Estimated number of tokens.

    Example:
        >>> estimate_tokens("Hello, world!")
        4
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the total number of tokens in a list of messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.

    Returns:
        Estimated total number of tokens.

    Example:
        >>> msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
        >>> estimate_messages_tokens(msgs)
        5
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += estimate_tokens(str(content))
        total += 4
    return total


def get_callable_public_name(func: callable) -> str:
    """Return the public tool name exposed to the model/runtime.

    Calute tools may carry a ``__calute_schema__`` dict whose ``name`` key
    differs from the Python callable's ``__name__``. This allows compatibility
    aliases and dotted tool names without changing the underlying implementation
    name.

    Args:
        func: The callable whose public tool name should be resolved.

    Returns:
        The public name string. If the callable carries a ``__calute_schema__``
        dictionary with a non-empty ``name`` entry, that value is returned.
        Otherwise falls back to ``func.__name__`` or ``str(func)``.

    Example:
        >>> def my_tool():
        ...     pass
        >>> get_callable_public_name(my_tool)
        'my_tool'
    """
    schema = getattr(func, "__calute_schema__", None)
    if isinstance(schema, dict):
        name = schema.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return getattr(func, "__name__", str(func))


def function_to_json(func: callable) -> dict:
    """Convert a Python function into a JSON-serializable dictionary.

    Extracts the function's signature, including its name, description
    (from docstring), and parameters with their types and descriptions.
    This is used to generate function schemas for LLM tool calling.

    If the function has a `__calute_schema__` attribute, it will be used
    directly instead of extracting from the signature. This is useful for
    dynamically created functions like MCP tool wrappers.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format,
        compatible with OpenAI function calling schema.

    Raises:
        ValueError: If the function signature cannot be extracted.

    Example:
        >>> def greet(name: str, age: int = 0) -> str:
        ...     '''Greet a person.
        ...     name: Person's name
        ...     age: Person's age
        ...     '''
        ...     return f"Hello {name}"
        >>> schema = function_to_json(greet)
        >>> print(schema["function"]["name"])
        greet
    """
    schema = getattr(func, "__calute_schema__", None)
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

    param_descriptions = {}
    param_pattern = r"(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)"
    matches = re.findall(param_pattern, docstring, re.DOTALL | re.MULTILINE)
    for param_name, description in matches:
        param_descriptions[param_name.strip()] = description.strip()

    parameters = {}
    for param in signature.parameters.values():
        if param.name == "context_variables" or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        param_info = {"type": "string"}

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
            "parameters": custom_parameters
            or {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
