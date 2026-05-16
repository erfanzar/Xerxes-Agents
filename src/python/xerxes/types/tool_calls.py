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
"""Typed tool / function-call models compatible with the OpenAI schema.

Provides :class:`Function` (a callable's name, description, and JSON-schema
parameters), :class:`Tool` wrapping a function as an OpenAI-style tool,
:class:`FunctionCall` (name + JSON-string arguments), and :class:`ToolCall`
(id-bearing invocation). The :class:`ToolChoice` enum mirrors the
``tool_choice`` API parameter.
"""

import json
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import field_validator

from ..core.utils import XerxesBase


class Function(XerxesBase):
    """Describes a callable function with its name, description, and parameter schema.

    Attributes:
        name: The function's identifier used in tool call requests.
        description: A human-readable description of what the function does.
        parameters: A JSON Schema dict describing required and optional parameters.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any]


class ToolTypes(StrEnum):
    """Discriminator for the type of tool being described.

    Attributes:
        function: A callable function with a JSON Schema.
    """

    function = "function"


class ToolChoice(StrEnum):
    """Controls how the LLM selects tools.

    Attributes:
        auto: Let the model decide.
        none: Force no tool usage.
        any: Force at least one tool.
    """

    auto = "auto"
    none = "none"
    any = "any"


class Tool(XerxesBase):
    """Wraps a function definition as an OpenAI-compatible tool.

    Attributes:
        type: Discriminator, currently always ``ToolTypes.function``.
        function: The function definition.
    """

    type: ToolTypes = ToolTypes.function
    function: Function

    def to_openai(self) -> dict[str, Any]:
        """Serialize this tool to an OpenAI ``tool`` object.

        Returns:
            An OpenAI-compatible ``{"type": "function", "function": {...}}`` dict.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_tool: dict[str, Any]) -> "Tool":
        """Construct a Tool from an OpenAI ``tool`` object.

        Args:
            openai_tool: An OpenAI ``{"type": "function", "function": {...}}`` dict.

        Returns:
            A ``Tool`` instance.
        """
        return cls.model_validate(openai_tool)


class FunctionCall(XerxesBase):
    """Represents a function call invocation with a name and arguments.

    Attributes:
        name: The name of the function being called.
        arguments: A JSON string of key-value arguments (or a dict, serialized on assignment).
    """

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, v: str | dict[str, Any]) -> str:
        """Ensure ``arguments`` is a valid JSON string.

        If a dict is provided, serialize it to JSON. If a string is provided,
        validate that it is parseable JSON.

        Args:
            v: Either a JSON string or a dict to be serialized.

        Returns:
            A JSON string.

        Raises:
            ValueError: If *v* is a non-empty string that is not valid JSON.
        """
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, str):
            if v:
                try:
                    json.loads(v)
                except json.JSONDecodeError:
                    raise ValueError(f"Arguments must be valid JSON: {v[:100]}") from None
            return v
        return v


class ToolCall(XerxesBase):
    """An OpenAI-compatible tool call with a unique ID and function reference.

    Attributes:
        id: A unique identifier for this tool call.
        type: Discriminator, currently always ``ToolTypes.function``.
        function: The ``FunctionCall`` specifying which function to invoke and with what arguments.
    """

    id: str = "null"
    type: ToolTypes = ToolTypes.function
    function: FunctionCall

    def to_openai(self) -> dict[str, Any]:
        """Serialize this tool call to an OpenAI ``tool_call`` object.

        Returns:
            An OpenAI-compatible ``{"id": ..., "type": "function", "function": {...}}`` dict.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """Construct a ToolCall from an OpenAI ``tool_call`` object.

        Args:
            tool_call: An OpenAI ``{"id": ..., "type": "function", "function": {...}}`` dict.

        Returns:
            A ``ToolCall`` instance.
        """
        return cls.model_validate(tool_call)


ToolType = TypeVar("ToolType", bound=Tool)
