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
"""Pre-execution validation of tool call arguments against their JSON schema.

When the LLM emits a tool call, the arguments dict is validated against the
tool's declared ``input_schema`` BEFORE the tool runs. Invalid arguments
return a structured error message to the model instead of executing — this
catches common LLM mistakes (missing required params, wrong types, unknown
keys) early, avoids wasted tool executions, and gives the model actionable
feedback to correct and retry.

Example::

    schema = {"type": "object", "required": ["file_path"],
              "properties": {"file_path": {"type": "string"}}}
    result = validate_tool_arguments("ReadFile", {"file_path": "x.py"}, schema)
    assert result.ok  # passes

    result = validate_tool_arguments("ReadFile", {}, schema)
    assert not result.ok
    assert "file_path" in result.error
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a tool call's arguments.

    Attributes:
        ok: ``True`` when arguments match the schema.
        tool_name: The tool that was validated.
        error: Human-readable validation error message (empty when ``ok``).
        missing: List of missing required parameter names (empty when ``ok``).
    """

    ok: bool
    tool_name: str
    error: str = ""
    missing: tuple[str, ...] = ()


def _check_type(value: Any, expected: str) -> bool:
    """Check a single value against a JSON-schema type string."""

    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> ValidationResult:
    """Validate ``arguments`` against ``schema`` before tool execution.

    Performs lightweight checks without requiring ``jsonschema`` as a
    dependency: required fields present, type checking on declared
    properties, and enum value validation. Returns a
    :class:`ValidationResult` describing the first error found (if any).

    Args:
        tool_name: Tool name for error messages.
        arguments: The arguments dict the LLM emitted.
        schema: The tool's ``input_schema`` (JSON-schema dict). When
            ``None`` or empty, validation always passes.

    Returns:
        :class:`ValidationResult` with ``ok=True`` on success.
    """

    if not schema:
        return ValidationResult(ok=True, tool_name=tool_name)

    if not isinstance(arguments, dict):
        return ValidationResult(
            ok=False,
            tool_name=tool_name,
            error=f"{tool_name}: expected arguments to be an object, got {type(arguments).__name__}",
        )

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    missing = [r for r in required if r not in arguments]
    if missing:
        return ValidationResult(
            ok=False,
            tool_name=tool_name,
            error=f"{tool_name}: missing required parameter(s): {', '.join(missing)}",
            missing=tuple(missing),
        )

    for key, value in arguments.items():
        if key not in properties:
            if schema.get("additionalProperties", True) is False:
                return ValidationResult(
                    ok=False,
                    tool_name=tool_name,
                    error=f"{tool_name}: unknown parameter '{key}' (schema has additionalProperties=false)",
                )
            continue

        prop = properties[key]
        expected_type = prop.get("type")
        if expected_type and not _check_type(value, expected_type):
            return ValidationResult(
                ok=False,
                tool_name=tool_name,
                error=f"{tool_name}: parameter '{key}' expected {expected_type}, got {type(value).__name__}",
            )

        enum_values = prop.get("enum")
        if enum_values and value not in enum_values:
            return ValidationResult(
                ok=False,
                tool_name=tool_name,
                error=f"{tool_name}: parameter '{key}' must be one of {enum_values}, got {value!r}",
            )

    return ValidationResult(ok=True, tool_name=tool_name)


def validate_and_format_error(
    tool_name: str,
    arguments: dict[str, Any] | str,
    schema: dict[str, Any] | None,
) -> str | None:
    """Validate and return an error string, or ``None`` if valid.

    Convenience wrapper that also handles the case where ``arguments``
    arrives as a JSON string (common from some providers). Returns a
    user/model-facing error string, or ``None`` when arguments are valid.
    """

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"{tool_name}: arguments are not valid JSON: {arguments[:200]}"
    result = validate_tool_arguments(tool_name, arguments, schema)
    return result.error if not result.ok else None


__all__ = ["ValidationResult", "validate_and_format_error", "validate_tool_arguments"]
