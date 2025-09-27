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

import inspect
import re
from datetime import datetime
from typing import Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict


class CaluteBase(BaseModel):
    r"""
    Forbids extra attributes, validates default values and use enum values.
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
    """
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func: callable) -> dict:
    """Convert a Python function into a JSON-serializable dictionary.

    Extracts the function's signature, including its name, description
    (from docstring), and parameters with their types and descriptions.
    This is used to generate function schemas for LLM tool calling.

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
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e!s}") from e
    docstring = func.__doc__ or ""
    param_descriptions = {}
    param_pattern = r"(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+?)(?=\n\s*\w+(?:\s*\([^)]+\))?\s*:|$)"
    matches = re.findall(param_pattern, docstring, re.DOTALL | re.MULTILINE)
    for param_name, description in matches:
        param_descriptions[param_name.strip()] = description.strip()

    parameters = {}
    for param in signature.parameters.values():
        param_info = {"type": "string"}

        if param.annotation != inspect.Parameter.empty:
            origin = get_origin(param.annotation)
            args = get_args(param.annotation)

            if origin is Union:
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1 and type(None) in args:
                    param_info["type"] = type_map.get(non_none_types[0], "string")
                else:
                    param_info = {"type": "union", "types": [type_map.get(arg, "string") for arg in args]}
            elif origin in (list, tuple, set):
                param_info["type"] = "array"
                if args:
                    param_info["items"] = {"type": type_map.get(args[0], "string")}
            elif param.annotation in type_map:
                param_info["type"] = type_map[param.annotation]
            else:
                param_info["type"] = (
                    param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation)
                )

        if param.name in param_descriptions:
            param_info["description"] = param_descriptions[param.name]
        if param.default != inspect.Parameter.empty:
            pass

        parameters[param.name] = param_info

    required = [param.name for param in signature.parameters.values() if param.default == inspect.Parameter.empty]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
