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


"""Tool definition for Cortex agents"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..utils import function_to_json


@dataclass
class CortexTool:
    """Tool that can be used by agents"""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    auto_generate_schema: bool = True

    def to_function_json(self) -> dict:
        """
        Convert tool to OpenAI function JSON format.

        If auto_generate_schema is True and no parameters are provided,
        automatically generate the schema from the function signature.
        """
        if self.auto_generate_schema and not self.parameters and self.function:
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
                    "parameters": self.parameters
                    or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }

    @classmethod
    def from_function(
        cls,
        function: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> "CortexTool":
        """
        Create a CortexTool from a function, automatically extracting metadata.

        Args:
            function: The function to wrap as a tool
            name: Optional custom name (defaults to function name)
            description: Optional custom description (defaults to function docstring)

        Returns:
            CortexTool instance with auto-generated schema
        """
        return cls(
            name=name or function.__name__,
            description=description or function.__doc__ or "",
            function=function,
            parameters={},
            auto_generate_schema=True,
        )
