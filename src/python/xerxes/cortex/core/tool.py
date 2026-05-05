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
"""Cortex tool wrapper for callable functions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...core.utils import function_to_json


@dataclass
class CortexTool:
    """Wraps a callable as a tool with a JSON schema for LLM function calling.

    Attributes:
        name (str): The tool name exposed to the LLM.
        description (str): A description of what the tool does.
        function (Callable): The actual Python function to invoke.
        parameters (dict): JSON Schema for the function parameters.
        auto_generate_schema (bool): Whether to auto-generate the schema from
            the function signature when *parameters* is empty.
    """

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    auto_generate_schema: bool = True

    def to_function_json(self) -> dict:
        """Serialize the tool to a function-calling JSON schema.

        Returns:
            dict: A JSON object compatible with OpenAI-style function calling.
                OUT: Contains ``type``, ``function.name``, ``function.description``,
                and ``function.parameters``.
        """

        if self.auto_generate_schema and not self.parameters:
            schema = function_to_json(self.function)

            schema["function"]["name"] = self.name
            schema["function"]["description"] = self.description
            return schema
        else:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": (
                        self.parameters
                        or {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }
                    ),
                },
            }

    @classmethod
    def from_function(
        cls,
        function: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> "CortexTool":
        """Create a ``CortexTool`` instance from a plain callable.

        Args:
            function (Callable): The function to wrap.
                IN: Any callable object that will be exposed as a tool.
                OUT: Stored as the tool's executable function.
            name (str | None): Optional override for the tool name.
                IN: If ``None``, ``function.__name__`` is used.
                OUT: Becomes the ``name`` attribute of the created tool.
            description (str | None): Optional override for the tool description.
                IN: If ``None``, ``function.__doc__`` or an empty string is used.
                OUT: Becomes the ``description`` attribute of the created tool.

        Returns:
            CortexTool: A fully configured tool instance.
                OUT: Ready for registration with an agent or LLM.
        """

        return cls(
            name=name or function.__name__,
            description=description or function.__doc__ or "",
            function=function,
            parameters={},
            auto_generate_schema=True,
        )
