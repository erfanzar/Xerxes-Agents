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
"""Tool pool module for Xerxes.

Exports:
    - ToolPool
    - assemble_tool_pool"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_registry import ExecutionRegistry, RegistryEntry


@dataclass(frozen=True)
class ToolPool:
    """Tool pool.

    Attributes:
        tools (tuple[RegistryEntry, ...]): tools.
        denied_tools (frozenset[str]): denied tools.
        categories (tuple[str, ...]): categories.
        safe_only (bool): safe only."""

    tools: tuple[RegistryEntry, ...] = ()
    denied_tools: frozenset[str] = frozenset()
    categories: tuple[str, ...] = ()
    safe_only: bool = False

    @property
    def tool_count(self) -> int:
        """Return Tool count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return len(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return Tool names.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            tuple[str, ...]: OUT: Result of the operation."""
        return tuple(t.name for t in self.tools)

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Retrieve the tool.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            RegistryEntry | None: OUT: Result of the operation."""

        for t in self.tools:
            if t.name == name:
                return t
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        """To schemas.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        schemas = []
        for entry in self.tools:
            if entry.schema:
                schemas.append(entry.schema)
            else:
                schemas.append(
                    {
                        "name": entry.name,
                        "description": entry.description or f"Execute {entry.name}",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                )
        return schemas

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = [
            "# Tool Pool",
            "",
            f"Tools: {self.tool_count}",
            f"Safe only: {self.safe_only}",
            f"Categories: {', '.join(self.categories) or 'all'}",
            f"Denied: {', '.join(sorted(self.denied_tools)) or 'none'}",
            "",
        ]
        for tool in self.tools:
            safe_tag = " [safe]" if tool.safe else ""
            cat_tag = f" ({tool.category})" if tool.category else ""
            lines.append(f"- **{tool.name}**{safe_tag}{cat_tag} — {tool.description}")
        return "\n".join(lines)


def assemble_tool_pool(
    registry: ExecutionRegistry | None = None,
    categories: list[str] | None = None,
    deny_tools: set[str] | None = None,
    deny_prefixes: list[str] | None = None,
    safe_only: bool = False,
    include_mcp: bool = True,
) -> ToolPool:
    """Assemble tool pool.

    Args:
        registry (ExecutionRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
        categories (list[str] | None, optional): IN: categories. Defaults to None. OUT: Consumed during execution.
        deny_tools (set[str] | None, optional): IN: deny tools. Defaults to None. OUT: Consumed during execution.
        deny_prefixes (list[str] | None, optional): IN: deny prefixes. Defaults to None. OUT: Consumed during execution.
        safe_only (bool, optional): IN: safe only. Defaults to False. OUT: Consumed during execution.
        include_mcp (bool, optional): IN: include mcp. Defaults to True. OUT: Consumed during execution.
    Returns:
        ToolPool: OUT: Result of the operation."""

    if registry is None:
        return ToolPool()

    deny = deny_tools or set()
    prefixes = deny_prefixes or []

    tools = registry.list_tools(safe_only=safe_only)

    if categories:
        tools = [t for t in tools if t.category in categories]

    if deny:
        tools = [t for t in tools if t.name not in deny]

    if prefixes:
        tools = [t for t in tools if not any(t.name.startswith(p) for p in prefixes)]

    if not include_mcp:
        tools = [t for t in tools if "mcp" not in t.source_hint.lower()]

    return ToolPool(
        tools=tuple(tools),
        denied_tools=frozenset(deny),
        categories=tuple(categories or []),
        safe_only=safe_only,
    )


__all__ = [
    "ToolPool",
    "assemble_tool_pool",
]
