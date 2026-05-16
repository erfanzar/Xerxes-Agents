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
"""Filtered, schema-ready view of an :class:`ExecutionRegistry` tool set.

:class:`ToolPool` is an immutable snapshot of the tools that should be
exposed to the LLM for one turn (or one sub-agent), already filtered by
category, deny-list, safety, and MCP-inclusion rules. :func:`assemble_tool_pool`
builds the pool from a populated registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_registry import ExecutionRegistry, RegistryEntry


@dataclass(frozen=True)
class ToolPool:
    """Immutable filtered set of tools exposed to one turn or sub-agent.

    Attributes:
        tools: Tuple of registry entries that survived filtering.
        denied_tools: Tool names that were explicitly excluded.
        categories: Categories the pool was filtered to (empty = all).
        safe_only: Whether only safe (no-permission-prompt) tools were kept.
    """

    tools: tuple[RegistryEntry, ...] = ()
    denied_tools: frozenset[str] = frozenset()
    categories: tuple[str, ...] = ()
    safe_only: bool = False

    @property
    def tool_count(self) -> int:
        """Number of tools surviving the filter chain."""
        return len(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Tuple of tool names in the pool, in registry order."""
        return tuple(t.name for t in self.tools)

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Look up an entry by exact name; ``None`` if not in the pool."""
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        """Return JSON tool schemas suitable for sending to the LLM."""

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
        """Render the pool as a Markdown overview (counts, filters, tool list)."""

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
    """Build a filtered :class:`ToolPool` from ``registry``.

    Args:
        registry: Populated :class:`ExecutionRegistry`; an empty pool is
            returned when ``None``.
        categories: When provided, restrict tools to these categories.
        deny_tools: Set of tool names to exclude entirely.
        deny_prefixes: Tools whose name starts with any of these prefixes
            are excluded (useful for blocking ``mcp__foo__bar`` etc.).
        safe_only: When ``True`` keep only tools flagged ``safe``.
        include_mcp: When ``False`` drop tools whose ``source_hint`` mentions
            ``"mcp"``.
    """

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
