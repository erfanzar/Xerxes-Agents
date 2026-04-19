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


"""Unified execution registry for commands and tools.

Inspired by the claw-code ``ExecutionRegistry``, this module provides a
single registry that:

1. Registers all available commands (slash commands, built-in actions).
2. Registers all available tools (agent functions, file ops, etc.).
3. Routes incoming prompts to the best-matching command or tool.
4. Executes matched entries and returns structured results.

The registry supports fuzzy matching, permission filtering, and
categorized listing.

Usage::

    from xerxes.runtime.execution_registry import ExecutionRegistry

    registry = ExecutionRegistry()
    registry.register_command("commit", handler=commit_handler, description="Create a git commit")
    registry.register_tool("Read", handler=read_handler, description="Read a file", safe=True)


    matches = registry.route("read the config file", limit=3)


    result = registry.execute_tool("Read", {"file_path": "/etc/hosts"})
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EntryKind(Enum):
    """Kind of registry entry."""

    COMMAND = "command"
    TOOL = "tool"


@dataclass
class RegistryEntry:
    """A single registered command or tool.

    Attributes:
        name: Canonical name (e.g. ``"Read"``, ``"commit"``).
        kind: Whether this is a command or tool.
        description: Human-readable description.
        handler: Callable that executes this entry.
        category: Optional category (e.g. ``"file_system"``, ``"git"``).
        safe: Whether this entry is safe to auto-approve (read-only).
        source_hint: Where this entry comes from (e.g. ``"xerxes.tools.standalone"``).
        schema: Optional JSON schema for tool input parameters.
    """

    name: str
    kind: EntryKind
    description: str
    handler: Callable[..., Any] | None = None
    category: str = ""
    safe: bool = False
    source_hint: str = ""
    schema: dict[str, Any] | None = None


@dataclass
class ExecutionResult:
    """Result of executing a registry entry.

    Attributes:
        name: Entry name that was executed.
        kind: Whether it was a command or tool.
        handled: Whether the execution was successful.
        result: The output/result string.
        duration_ms: Execution duration in milliseconds.
        error: Error message if execution failed.
    """

    name: str
    kind: EntryKind
    handled: bool
    result: str
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class RouteMatch:
    """A match from routing a prompt against the registry.

    Attributes:
        name: Matched entry name.
        kind: Command or tool.
        score: Match score (higher is better).
        source_hint: Where this entry comes from.
        description: Entry description.
    """

    name: str
    kind: EntryKind
    score: int
    source_hint: str = ""
    description: str = ""


class ExecutionRegistry:
    """Unified registry for commands and tools.

    Provides registration, routing, execution, and listing for all
    available commands and tools in the Xerxes runtime.
    """

    def __init__(self) -> None:
        self._commands: dict[str, RegistryEntry] = {}
        self._tools: dict[str, RegistryEntry] = {}

    def register_command(
        self,
        name: str,
        handler: Callable[..., Any] | None = None,
        description: str = "",
        category: str = "",
        source_hint: str = "",
    ) -> None:
        """Register a command (slash command or built-in action)."""
        self._commands[name.lower()] = RegistryEntry(
            name=name,
            kind=EntryKind.COMMAND,
            description=description,
            handler=handler,
            category=category,
            source_hint=source_hint,
        )

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any] | None = None,
        description: str = "",
        category: str = "",
        safe: bool = False,
        source_hint: str = "",
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool (agent function)."""
        self._tools[name] = RegistryEntry(
            name=name,
            kind=EntryKind.TOOL,
            description=description,
            handler=handler,
            category=category,
            safe=safe,
            source_hint=source_hint,
            schema=schema,
        )

    def register_from_agent_functions(self, functions: list[Any]) -> None:
        """Bulk-register tools from a list of AgentFunction objects.

        Expects objects with ``name``, ``description``, and optionally
        ``callable_func`` attributes.
        """
        for func in functions:
            name = getattr(func, "name", str(func))
            desc = getattr(func, "description", "")
            handler = getattr(func, "callable_func", None) or getattr(func, "handler", None)
            self.register_tool(name=name, handler=handler, description=desc)

    def route(self, prompt: str, limit: int = 5) -> list[RouteMatch]:
        """Route a prompt to the best-matching commands and tools.

        Uses token overlap scoring: splits the prompt into tokens and
        scores each entry by how many tokens match its name, description,
        or category.

        Args:
            prompt: The user's input string.
            limit: Maximum number of matches to return.

        Returns:
            List of :class:`RouteMatch` sorted by score (descending).
        """
        tokens = {t.lower() for t in prompt.replace("/", " ").replace("-", " ").replace("_", " ").split() if len(t) > 1}
        if not tokens:
            return []

        matches: list[RouteMatch] = []

        for entry in list(self._commands.values()) + list(self._tools.values()):
            score = self._score_entry(entry, tokens)
            if score > 0:
                matches.append(
                    RouteMatch(
                        name=entry.name,
                        kind=entry.kind,
                        score=score,
                        source_hint=entry.source_hint,
                        description=entry.description,
                    )
                )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    @staticmethod
    def _score_entry(entry: RegistryEntry, tokens: set[str]) -> int:
        """Score an entry against a set of query tokens."""
        score = 0
        name_lower = entry.name.lower()
        desc_lower = entry.description.lower()
        cat_lower = entry.category.lower()

        for token in tokens:
            if token in name_lower:
                score += 3
            if token in desc_lower:
                score += 1
            if token in cat_lower:
                score += 2

        if name_lower in tokens:
            score += 5

        return score

    def execute_command(self, name: str, **kwargs: Any) -> ExecutionResult:
        """Execute a registered command by name."""
        entry = self._commands.get(name.lower())
        if entry is None:
            return ExecutionResult(
                name=name,
                kind=EntryKind.COMMAND,
                handled=False,
                result=f"Unknown command: {name}",
            )
        return self._execute(entry, **kwargs)

    def execute_tool(self, name: str, inputs: dict[str, Any] | None = None) -> ExecutionResult:
        """Execute a registered tool by name."""
        entry = self._tools.get(name)
        if entry is None:
            return ExecutionResult(
                name=name,
                kind=EntryKind.TOOL,
                handled=False,
                result=f"Unknown tool: {name}",
            )
        return self._execute(entry, **(inputs or {}))

    def _execute(self, entry: RegistryEntry, **kwargs: Any) -> ExecutionResult:
        """Execute a registry entry."""
        if entry.handler is None:
            return ExecutionResult(
                name=entry.name,
                kind=entry.kind,
                handled=False,
                result=f"No handler registered for {entry.kind.value}: {entry.name}",
            )

        t0 = time.monotonic()
        try:
            result = entry.handler(**kwargs)
            duration_ms = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                name=entry.name,
                kind=entry.kind,
                handled=True,
                result=str(result) if result is not None else "",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.error("Error executing %s %s: %s", entry.kind.value, entry.name, e)
            return ExecutionResult(
                name=entry.name,
                kind=entry.kind,
                handled=False,
                result="",
                duration_ms=duration_ms,
                error=str(e),
            )

    def get_command(self, name: str) -> RegistryEntry | None:
        """Look up a command by name."""
        return self._commands.get(name.lower())

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_commands(self, category: str | None = None) -> list[RegistryEntry]:
        """List all commands, optionally filtered by category."""
        entries = list(self._commands.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: e.name)

    def list_tools(self, category: str | None = None, safe_only: bool = False) -> list[RegistryEntry]:
        """List all tools, optionally filtered."""
        entries = list(self._tools.values())
        if category:
            entries = [e for e in entries if e.category == category]
        if safe_only:
            entries = [e for e in entries if e.safe]
        return sorted(entries, key=lambda e: e.name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in Anthropic tool format."""
        schemas = []
        for entry in self._tools.values():
            if entry.schema:
                schemas.append(entry.schema)
            else:
                schemas.append(
                    {
                        "name": entry.name,
                        "description": entry.description or f"Execute {entry.name}",
                        "input_schema": {
                            "type": "object",
                            "properties": {},
                        },
                    }
                )
        return schemas

    @property
    def command_count(self) -> int:
        return len(self._commands)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def summary(self) -> str:
        """Return a markdown summary of the registry."""
        lines = [
            "# Execution Registry",
            "",
            f"Commands: {self.command_count}",
            f"Tools: {self.tool_count}",
            "",
        ]
        if self._commands:
            lines.append("## Commands")
            for entry in self.list_commands():
                lines.append(f"- `/{entry.name}` — {entry.description}")
            lines.append("")
        if self._tools:
            lines.append("## Tools")
            for entry in self.list_tools():
                safe_tag = " [safe]" if entry.safe else ""
                lines.append(f"- `{entry.name}`{safe_tag} — {entry.description}")
        return "\n".join(lines)


__all__ = [
    "EntryKind",
    "ExecutionRegistry",
    "ExecutionResult",
    "RegistryEntry",
    "RouteMatch",
]
