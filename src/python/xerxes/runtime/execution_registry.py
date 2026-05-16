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
"""Unified registry of slash commands and tools for the runtime.

The :class:`ExecutionRegistry` is the single source of truth that the TUI,
bridge, and daemon use to look up command handlers (``/help``, ``/clear``,
``/model``, ...) and tool implementations (``Read``, ``WriteFile``, ...).
:meth:`ExecutionRegistry.route` scores prompts against entry names,
descriptions, and categories so the planner can suggest relevant tools, and
:meth:`ExecutionRegistry.tool_schemas` emits the JSON schemas advertised to
the LLM.
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
    """Discriminates between slash commands and tool definitions.

    Attributes:
        COMMAND: A user-facing slash command (``/help``, ``/model``).
        TOOL: An LLM-callable tool (``Read``, ``ExecuteShell``).
    """

    COMMAND = "command"
    TOOL = "tool"


@dataclass
class RegistryEntry:
    """One registered command or tool definition.

    Attributes:
        name: Canonical entry name (case-sensitive for tools, lowercased
            on lookup for commands).
        kind: Whether this entry is a :class:`EntryKind.COMMAND` or
            :class:`EntryKind.TOOL`.
        description: Human-readable summary shown in ``/help`` and routing.
        handler: Callable implementing the entry; entries may register
            without a handler when only metadata is wanted.
        category: Optional grouping label used by listing and routing.
        safe: ``True`` when the tool requires no permission prompts.
        source_hint: Free-form provenance hint (``"builtin"``, ``"plugin:foo"``).
        schema: Optional pre-built tool schema; auto-generated when ``None``.
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
    """Outcome of a single :meth:`ExecutionRegistry._execute` invocation.

    Attributes:
        name: Entry name that was invoked.
        kind: Whether the executed entry was a command or a tool.
        handled: ``True`` when a handler ran (even if it raised). ``False``
            when no handler was registered or the entry was unknown.
        result: ``str(...)`` of the handler's return value (empty on error).
        duration_ms: Wall-clock execution time in milliseconds.
        error: Stringified exception when the handler raised.
    """

    name: str
    kind: EntryKind
    handled: bool
    result: str
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class RouteMatch:
    """One scored hit returned by :meth:`ExecutionRegistry.route`.

    Attributes:
        name: Name of the matched entry.
        kind: Kind of the matched entry.
        score: Heuristic match score (higher is better; see
            :meth:`ExecutionRegistry._score_entry`).
        source_hint: Pass-through of the entry's ``source_hint``.
        description: Pass-through of the entry's description.
    """

    name: str
    kind: EntryKind
    score: int
    source_hint: str = ""
    description: str = ""


class ExecutionRegistry:
    """Holds the slash commands and LLM-callable tools available to a session."""

    def __init__(self) -> None:
        """Create an empty registry with separate command and tool dicts."""
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
        """Register or overwrite a slash command entry.

        Command lookups are case-insensitive: the key is stored as
        ``name.lower()`` while the original casing is preserved on the entry.
        Passing ``handler=None`` registers a stub so ``/help`` can still list
        the command.
        """

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
        """Register or overwrite a tool entry.

        Args:
            name: Tool name as advertised to the LLM (case-sensitive).
            handler: Callable invoked when the LLM calls this tool.
            description: Tool description shown to the LLM.
            category: Optional grouping label.
            safe: When ``True`` the tool may run without a permission prompt.
            source_hint: Free-form provenance hint.
            schema: Optional pre-built JSON schema; auto-generated otherwise.
        """

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
        """Register every ``AgentBaseFn`` (or duck-compatible) object as a tool.

        Reads each object's ``name``, ``description`` and ``callable_func``/
        ``handler`` attribute. Designed to ingest the tool list produced by
        :func:`xerxes.runtime.tool_pool.assemble_tool_pool`.
        """

        for func in functions:
            name = getattr(func, "name", str(func))
            desc = getattr(func, "description", "")
            handler = getattr(func, "callable_func", None) or getattr(func, "handler", None)
            self.register_tool(name=name, handler=handler, description=desc)

    def route(self, prompt: str, limit: int = 5) -> list[RouteMatch]:
        """Return the top ``limit`` entries that look relevant to ``prompt``.

        Tokenises ``prompt`` on whitespace/``/``/``-``/``_`` and runs
        :meth:`_score_entry` against every registered command and tool,
        returning the highest-scoring matches in descending order.
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
        """Compute a token-overlap score for ``entry`` against ``tokens``.

        Name hits weigh more than description hits; an exact name-in-tokens
        match gets a bonus. The returned integer is unbounded but typically
        small (0-20).
        """

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
        """Look up a slash command (case-insensitive) and run its handler.

        Returns an :class:`ExecutionResult` with ``handled=False`` when the
        command is unknown.
        """

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
        """Look up a tool by exact name and invoke its handler with ``inputs``.

        ``inputs`` is splatted into ``**kwargs``; pass an empty dict (or
        ``None``) to call a no-arg tool.
        """

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
        """Run ``entry.handler`` with ``**kwargs`` and package the outcome.

        Captures handler exceptions, records wall-clock duration, and stringifies
        the return value for the :class:`ExecutionResult`.
        """

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
        """Return the registered command entry for ``name``, case-insensitively."""
        return self._commands.get(name.lower())

    def get_tool(self, name: str) -> RegistryEntry | None:
        """Return the registered tool entry for the exact ``name``."""
        return self._tools.get(name)

    def list_commands(self, category: str | None = None) -> list[RegistryEntry]:
        """Return commands sorted by name, optionally filtered by category."""

        entries = list(self._commands.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: e.name)

    def list_tools(self, category: str | None = None, safe_only: bool = False) -> list[RegistryEntry]:
        """Return tools sorted by name, optionally filtered by category/safe-ness."""

        entries = list(self._tools.values())
        if category:
            entries = [e for e in entries if e.category == category]
        if safe_only:
            entries = [e for e in entries if e.safe]
        return sorted(entries, key=lambda e: e.name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return JSON tool schemas for every registered tool.

        Entries without an explicit ``schema`` get a minimal auto-generated
        schema with an empty ``input_schema.properties``.
        """

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
        """Number of registered slash commands."""
        return len(self._commands)

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def summary(self) -> str:
        """Render a Markdown overview of registered commands and tools."""

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
