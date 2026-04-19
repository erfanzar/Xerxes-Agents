# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Session history log for tracking discrete events.

Records a timeline of events during an agent session: tool calls,
permission decisions, model switches, errors, etc.

Inspired by the claw-code ``HistoryLog`` pattern.

Usage::

    from xerxes.runtime.history import HistoryLog

    log = HistoryLog()
    log.add("tool_call", "Read /etc/hosts", duration_ms=12.5)
    log.add("permission_denied", "Edit /etc/passwd")
    log.add("model_switch", "gpt-4o → claude-opus-4-6")
    print(log.as_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HistoryEvent:
    """A single event in the session history.

    Attributes:
        kind: Event kind (e.g. ``"tool_call"``, ``"error"``, ``"turn"``).
        title: Short title for the event.
        detail: Detailed description.
        timestamp: ISO 8601 timestamp.
        metadata: Optional structured metadata.
    """

    kind: str
    title: str
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryLog:
    """Ordered log of session events.

    Provides append, filtering, and markdown rendering.
    """

    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, kind: str, title: str, detail: str = "", **metadata: Any) -> HistoryEvent:
        """Record a new event.

        Args:
            kind: Event category.
            title: Short description.
            detail: Extended description.
            **metadata: Arbitrary key-value metadata.

        Returns:
            The created event.
        """
        event = HistoryEvent(kind=kind, title=title, detail=detail, metadata=metadata)
        self.events.append(event)
        return event

    def add_tool_call(self, name: str, result_preview: str = "", duration_ms: float = 0.0) -> HistoryEvent:
        """Convenience: record a tool call event."""
        return self.add(
            kind="tool_call",
            title=name,
            detail=result_preview[:200],
            duration_ms=duration_ms,
        )

    def add_error(self, message: str, source: str = "") -> HistoryEvent:
        """Convenience: record an error event."""
        return self.add(kind="error", title=message, detail=source)

    def add_turn(self, model: str, in_tokens: int = 0, out_tokens: int = 0) -> HistoryEvent:
        """Convenience: record an LLM turn completion."""
        return self.add(
            kind="turn",
            title=f"Turn completed ({model})",
            detail=f"in={in_tokens}, out={out_tokens}",
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )

    def add_permission(self, tool_name: str, granted: bool) -> HistoryEvent:
        """Convenience: record a permission decision."""
        status = "granted" if granted else "denied"
        return self.add(
            kind=f"permission_{status}",
            title=f"{tool_name}: {status}",
        )

    def filter_by_kind(self, kind: str) -> list[HistoryEvent]:
        """Return events of a specific kind."""
        return [e for e in self.events if e.kind == kind]

    def last(self, n: int = 10) -> list[HistoryEvent]:
        """Return the last N events."""
        return self.events[-n:]

    @property
    def event_count(self) -> int:
        return len(self.events)

    def clear(self) -> None:
        self.events.clear()

    def as_markdown(self) -> str:
        """Render the history as markdown."""
        lines = ["# Session History", "", f"Events: {self.event_count}", ""]
        for event in self.events:
            ts = event.timestamp[:19]
            detail = f" — {event.detail}" if event.detail else ""
            lines.append(f"- [{ts}] **{event.kind}**: {event.title}{detail}")
        return "\n".join(lines)

    def as_dicts(self) -> list[dict[str, Any]]:
        """Serialize events for JSON storage."""
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
