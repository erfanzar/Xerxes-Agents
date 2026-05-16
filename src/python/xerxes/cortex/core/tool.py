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
    """Wrap a callable as an LLM-callable tool with a JSON Schema.

    Attributes:
        name: Tool name surfaced to the model.
        description: Free-form description shown in the tool catalogue.
        function: Python callable invoked when the model selects the tool.
        parameters: Explicit JSON Schema for the parameters; ignored when
            ``auto_generate_schema`` is ``True`` and the field is left
            empty (a schema is then derived from the signature).
        auto_generate_schema: When ``True``, derive a schema from the
            function signature via :func:`function_to_json` whenever
            ``parameters`` is empty.
    """

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    auto_generate_schema: bool = True

    def to_function_json(self) -> dict:
        """Return an OpenAI-style ``function`` JSON descriptor for the tool.

        Uses the auto-generated schema when ``auto_generate_schema`` is set
        and ``parameters`` is empty, otherwise falls back to the explicit
        ``parameters`` (or an empty object schema).
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
        """Build a :class:`CortexTool` from a plain Python callable.

        ``name`` defaults to ``function.__name__`` and ``description``
        defaults to ``function.__doc__``.
        """

        return cls(
            name=name or function.__name__,
            description=description or function.__doc__ or "",
            function=function,
            parameters={},
            auto_generate_schema=True,
        )
