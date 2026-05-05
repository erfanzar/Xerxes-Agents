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
"""History module for Xerxes.

Exports:
    - HistoryEvent
    - HistoryLog"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HistoryEvent:
    """History event.

    Attributes:
        kind (str): kind.
        title (str): title.
        detail (str): detail.
        timestamp (str): timestamp.
        metadata (dict[str, Any]): metadata."""

    kind: str
    title: str
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryLog:
    """History log.

    Attributes:
        events (list[HistoryEvent]): events."""

    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, kind: str, title: str, detail: str = "", **metadata: Any) -> HistoryEvent:
        """Add.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            kind (str): IN: kind. OUT: Consumed during execution.
            title (str): IN: title. OUT: Consumed during execution.
            detail (str, optional): IN: detail. Defaults to ''. OUT: Consumed during execution.
            **metadata: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            HistoryEvent: OUT: Result of the operation."""

        event = HistoryEvent(kind=kind, title=title, detail=detail, metadata=metadata)
        self.events.append(event)
        return event

    def add_tool_call(self, name: str, result_preview: str = "", duration_ms: float = 0.0) -> HistoryEvent:
        """Add tool call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            result_preview (str, optional): IN: result preview. Defaults to ''. OUT: Consumed during execution.
            duration_ms (float, optional): IN: duration ms. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            HistoryEvent: OUT: Result of the operation."""

        return self.add(
            kind="tool_call",
            title=name,
            detail=result_preview[:200],
            duration_ms=duration_ms,
        )

    def add_error(self, message: str, source: str = "") -> HistoryEvent:
        """Add error.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            message (str): IN: message. OUT: Consumed during execution.
            source (str, optional): IN: source. Defaults to ''. OUT: Consumed during execution.
        Returns:
            HistoryEvent: OUT: Result of the operation."""

        return self.add(kind="error", title=message, detail=source)

    def add_turn(self, model: str, in_tokens: int = 0, out_tokens: int = 0) -> HistoryEvent:
        """Add turn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model (str): IN: model. OUT: Consumed during execution.
            in_tokens (int, optional): IN: in tokens. Defaults to 0. OUT: Consumed during execution.
            out_tokens (int, optional): IN: out tokens. Defaults to 0. OUT: Consumed during execution.
        Returns:
            HistoryEvent: OUT: Result of the operation."""

        return self.add(
            kind="turn",
            title=f"Turn completed ({model})",
            detail=f"in={in_tokens}, out={out_tokens}",
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )

    def add_permission(self, tool_name: str, granted: bool) -> HistoryEvent:
        """Add permission.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            granted (bool): IN: granted. OUT: Consumed during execution.
        Returns:
            HistoryEvent: OUT: Result of the operation."""

        status = "granted" if granted else "denied"
        return self.add(
            kind=f"permission_{status}",
            title=f"{tool_name}: {status}",
        )

    def filter_by_kind(self, kind: str) -> list[HistoryEvent]:
        """Filter by kind.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            kind (str): IN: kind. OUT: Consumed during execution.
        Returns:
            list[HistoryEvent]: OUT: Result of the operation."""

        return [e for e in self.events if e.kind == kind]

    def last(self, n: int = 10) -> list[HistoryEvent]:
        """Last.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            n (int, optional): IN: n. Defaults to 10. OUT: Consumed during execution.
        Returns:
            list[HistoryEvent]: OUT: Result of the operation."""

        return self.events[-n:]

    @property
    def event_count(self) -> int:
        """Return Event count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return len(self.events)

    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""
        self.events.clear()

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = ["# Session History", "", f"Events: {self.event_count}", ""]
        for event in self.events:
            ts = event.timestamp[:19]
            detail = f" — {event.detail}" if event.detail else ""
            lines.append(f"- [{ts}] **{event.kind}**: {event.title}{detail}")
        return "\n".join(lines)

    def as_dicts(self) -> list[dict[str, Any]]:
        """As dicts.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        return [
            {
                "kind": e.kind,
                "title": e.title,
                "detail": e.detail,
                "timestamp": e.timestamp,
                **e.metadata,
            }
            for e in self.events
        ]


__all__ = [
    "HistoryEvent",
    "HistoryLog",
]
