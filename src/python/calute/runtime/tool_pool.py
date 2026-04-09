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


"""Permission-filtered tool pool assembly.

Assembles a curated set of tools for an agent session, applying
permission policies, category filters, and capability constraints.

Inspired by the claw-code ``ToolPool`` and ``assemble_tool_pool`` patterns.

Usage::

    from calute.runtime.tool_pool import ToolPool, assemble_tool_pool

    pool = assemble_tool_pool(
        categories=["file_system", "execution"],
        deny_tools={"ExecuteShell"},
        safe_only=False,
    )
    print(pool.as_markdown())
    schemas = pool.to_schemas()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_registry import ExecutionRegistry, RegistryEntry


@dataclass(frozen=True)
class ToolPool:
    """An assembled, filtered set of tools ready for agent use.

    Attributes:
        tools: Tuple of tool entries included in the pool.
        denied_tools: Tools that were explicitly denied.
        categories: Categories that were requested.
        safe_only: Whether only safe tools were included.
    """

    tools: tuple[RegistryEntry, ...] = ()
    denied_tools: frozenset[str] = frozenset()
    categories: tuple[str, ...] = ()
    safe_only: bool = False

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Look up a tool by name."""
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        """Convert to Anthropic-format tool schemas."""
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
        """Render as markdown."""
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
    """Assemble a filtered tool pool from the execution registry.

    Args:
        registry: The execution registry to pull tools from.
            If None, creates an empty pool.
        categories: Only include tools from these categories.
        deny_tools: Explicit set of tool names to exclude.
        deny_prefixes: Exclude tools whose names start with these prefixes.
        safe_only: Only include safe (read-only) tools.
        include_mcp: Include MCP-sourced tools.

    Returns:
        A :class:`ToolPool` with the filtered tools.
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
