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
"""Execution registry module for Xerxes.

Exports:
    - logger
    - EntryKind
    - RegistryEntry
    - ExecutionResult
    - RouteMatch
    - ExecutionRegistry"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EntryKind(Enum):
    """Entry kind.

    Inherits from: Enum
    """

    COMMAND = "command"
    TOOL = "tool"


@dataclass
class RegistryEntry:
    """Registry entry.

    Attributes:
        name (str): name.
        kind (EntryKind): kind.
        description (str): description.
        handler (Callable[..., Any] | None): handler.
        category (str): category.
        safe (bool): safe.
        source_hint (str): source hint.
        schema (dict[str, Any] | None): schema."""

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
    """Execution result.

    Attributes:
        name (str): name.
        kind (EntryKind): kind.
        handled (bool): handled.
        result (str): result.
        duration_ms (float): duration ms.
        error (str): error."""

    name: str
    kind: EntryKind
    handled: bool
    result: str
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class RouteMatch:
    """Route match.

    Attributes:
        name (str): name.
        kind (EntryKind): kind.
        score (int): score.
        source_hint (str): source hint.
        description (str): description."""

    name: str
    kind: EntryKind
    score: int
    source_hint: str = ""
    description: str = ""


class ExecutionRegistry:
    """Execution registry."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""
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
        """Register command.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            handler (Callable[..., Any] | None, optional): IN: handler. Defaults to None. OUT: Consumed during execution.
            description (str, optional): IN: description. Defaults to ''. OUT: Consumed during execution.
            category (str, optional): IN: category. Defaults to ''. OUT: Consumed during execution.
            source_hint (str, optional): IN: source hint. Defaults to ''. OUT: Consumed during execution."""

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
        """Register tool.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            handler (Callable[..., Any] | None, optional): IN: handler. Defaults to None. OUT: Consumed during execution.
            description (str, optional): IN: description. Defaults to ''. OUT: Consumed during execution.
            category (str, optional): IN: category. Defaults to ''. OUT: Consumed during execution.
            safe (bool, optional): IN: safe. Defaults to False. OUT: Consumed during execution.
            source_hint (str, optional): IN: source hint. Defaults to ''. OUT: Consumed during execution.
            schema (dict[str, Any] | None, optional): IN: schema. Defaults to None. OUT: Consumed during execution."""

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
        """Register from agent functions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            functions (list[Any]): IN: functions. OUT: Consumed during execution."""

        for func in functions:
            name = getattr(func, "name", str(func))
            desc = getattr(func, "description", "")
            handler = getattr(func, "callable_func", None) or getattr(func, "handler", None)
            self.register_tool(name=name, handler=handler, description=desc)

    def route(self, prompt: str, limit: int = 5) -> list[RouteMatch]:
        """Route.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 5. OUT: Consumed during execution.
        Returns:
            list[RouteMatch]: OUT: Result of the operation."""

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
        """Internal helper to score entry.

        Args:
            entry (RegistryEntry): IN: entry. OUT: Consumed during execution.
            tokens (set[str]): IN: tokens. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

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
        """Execute command.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            ExecutionResult: OUT: Result of the operation."""

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
        """Execute tool.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            inputs (dict[str, Any] | None, optional): IN: inputs. Defaults to None. OUT: Consumed during execution.
        Returns:
            ExecutionResult: OUT: Result of the operation."""

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
        """Internal helper to execute.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            entry (RegistryEntry): IN: entry. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            ExecutionResult: OUT: Result of the operation."""

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
        """Retrieve the command.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            RegistryEntry | None: OUT: Result of the operation."""

        return self._commands.get(name.lower())

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Retrieve the tool.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            RegistryEntry | None: OUT: Result of the operation."""

        return self._tools.get(name)

    def list_commands(self, category: str | None = None) -> list[RegistryEntry]:
        """List commands.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            category (str | None, optional): IN: category. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[RegistryEntry]: OUT: Result of the operation."""

        entries = list(self._commands.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: e.name)

    def list_tools(self, category: str | None = None, safe_only: bool = False) -> list[RegistryEntry]:
        """List tools.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            category (str | None, optional): IN: category. Defaults to None. OUT: Consumed during execution.
            safe_only (bool, optional): IN: safe only. Defaults to False. OUT: Consumed during execution.
        Returns:
            list[RegistryEntry]: OUT: Result of the operation."""

        entries = list(self._tools.values())
        if category:
            entries = [e for e in entries if e.category == category]
        if safe_only:
            entries = [e for e in entries if e.safe]
        return sorted(entries, key=lambda e: e.name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Tool schemas.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

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
        """Return Command count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return len(self._commands)

    @property
    def tool_count(self) -> int:
        """Return Tool count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return len(self._tools)

    def summary(self) -> str:
        """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

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
