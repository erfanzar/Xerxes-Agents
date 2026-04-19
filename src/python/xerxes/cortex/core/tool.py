# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tool definition for Cortex agents.

This module provides the CortexTool class, which represents callable tools
that can be used by Cortex agents to perform specific actions. Tools wrap
Python functions and automatically generate OpenAI-compatible function
schemas for integration with LLM function calling capabilities.

Key features:
- Automatic JSON schema generation from function signatures
- OpenAI function calling format compatibility
- Factory method for easy tool creation from existing functions
- Customizable parameter schemas for complex tool definitions

Typical usage example:

    @CortexTool.from_function
    def search_database(query: str, limit: int = 10) -> list:
        '''Search the database for matching records.'''
        return db.search(query, limit)


    tool = CortexTool(
        name="calculate",
        description="Perform arithmetic calculations",
        function=calculate_fn,
        parameters={"type": "object", "properties": {...}}
    )
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...core.utils import function_to_json


@dataclass
class CortexTool:
    """Tool that can be used by Cortex agents.

    CortexTool wraps a Python callable function and provides metadata
    and schema information required for integration with LLM function
    calling APIs. It can automatically generate parameter schemas from
    function signatures or accept custom schema definitions.

    Attributes:
        name: The name of the tool as it will appear to the LLM.
        description: Human-readable description of what the tool does.
            This is used by the LLM to decide when to use the tool.
        function: The Python callable that implements the tool's functionality.
        parameters: Optional dictionary defining the JSON schema for tool
            parameters. If empty and auto_generate_schema is True, the schema
            is automatically generated from the function signature.
        auto_generate_schema: Whether to automatically generate the parameter
            schema from the function signature when parameters is empty.
            Defaults to True.

    Example:
        >>> def get_weather(city: str, units: str = "celsius") -> str:
        ...     '''Get current weather for a city.'''
        ...     return f"Weather in {city}: 22 {units}"
        >>> tool = CortexTool.from_function(get_weather)
        >>> tool.to_function_json()
        {'type': 'function', 'function': {'name': 'get_weather', ...}}
    """

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any] = field(default_factory=dict)
    auto_generate_schema: bool = True

    def to_function_json(self) -> dict:
        """Convert tool to OpenAI function JSON format.

        Generates a dictionary conforming to the OpenAI function calling
        schema format. If auto_generate_schema is True and no parameters
        are provided, automatically generates the schema from the function
        signature using type hints and docstrings.

        Returns:
            dict: A dictionary in OpenAI function format containing:
                - type: Always "function"
                - function: Dictionary with name, description, and parameters

        Note:
            When using auto_generate_schema, the function should have type
            hints for best results. The tool name and description from this
            instance will override any auto-generated values.
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
        """Create a CortexTool from a function, automatically extracting metadata.

        Factory method that wraps an existing Python function as a CortexTool.
        Automatically extracts the function name and docstring for tool metadata,
        and enables auto-generation of the parameter schema from the function's
        type hints and signature.

        Args:
            function: The Python callable to wrap as a tool. Should have type
                hints on parameters for best schema generation results.
            name: Optional custom name for the tool. If None, defaults to
                ``function.__name__``.
            description: Optional custom description of the tool's purpose.
                If None, defaults to the function's docstring, or an empty
                string if no docstring is available.

        Returns:
            A new CortexTool instance configured with ``auto_generate_schema=True``
            and empty parameters dict (schema will be generated on first call
            to ``to_function_json()``).

        Example:
            >>> def search_db(query: str, limit: int = 10) -> list:
            ...     '''Search the database for records.'''
            ...     return []
            >>> tool = CortexTool.from_function(search_db)
            >>> tool.name
            'search_db'
            >>> tool = CortexTool.from_function(search_db, name="db_search")
            >>> tool.name
            'db_search'
        """
        return cls(
            name=name or function.__name__,
            description=description or function.__doc__ or "",
            function=function,
            parameters={},
            auto_generate_schema=True,
        )
