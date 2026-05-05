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
"""Tool calls module for Xerxes.

Exports:
    - Function
    - ToolTypes
    - ToolChoice
    - Tool
    - FunctionCall
    - ToolCall
    - ToolType"""

import json
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import field_validator

from ..core.utils import XerxesBase


class Function(XerxesBase):
    """Function.

    Inherits from: XerxesBase

    Attributes:
        name (str): name.
        description (str): description.
        parameters (dict[str, Any]): parameters."""

    name: str
    description: str = ""
    parameters: dict[str, Any]


class ToolTypes(StrEnum):
    """Tool types.

    Inherits from: StrEnum
    """

    function = "function"


class ToolChoice(StrEnum):
    """Tool choice.

    Inherits from: StrEnum
    """

    auto = "auto"
    none = "none"
    any = "any"


class Tool(XerxesBase):
    """Tool.

    Inherits from: XerxesBase

    Attributes:
        type (ToolTypes): type.
        function (Function): function."""

    type: ToolTypes = ToolTypes.function
    function: Function

    def to_openai(self) -> dict[str, Any]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_tool: dict[str, Any]) -> "Tool":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            openai_tool (dict[str, Any]): IN: openai tool. OUT: Consumed during execution.
        Returns:
            'Tool': OUT: Result of the operation."""

        return cls.model_validate(openai_tool)


class FunctionCall(XerxesBase):
    """Function call.

    Inherits from: XerxesBase

    Attributes:
        name (str): name.
        arguments (str): arguments."""

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, v: str | dict[str, Any]) -> str:
        """Validate arguments.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            v (str | dict[str, Any]): IN: v. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
    """Tool call.

    Inherits from: XerxesBase

    Attributes:
        id (str): id.
        type (ToolTypes): type.
        function (FunctionCall): function."""

    id: str = "null"
    type: ToolTypes = ToolTypes.function
    function: FunctionCall

    def to_openai(self) -> dict[str, Any]:
        """To openai.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return self.model_dump()

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """From openai.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            tool_call (dict[str, Any]): IN: tool call. OUT: Consumed during execution.
        Returns:
            'ToolCall': OUT: Result of the operation."""

        return cls.model_validate(tool_call)


ToolType = TypeVar("ToolType", bound=Tool)
