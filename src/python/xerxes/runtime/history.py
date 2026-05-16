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
"""Lightweight chronological event log attached to every session.

:class:`HistoryEvent` is one immutable record; :class:`HistoryLog` is the
ordered append-only buffer the query engine writes to as turns, tool calls,
permission prompts, errors, and compaction events happen. The log feeds
``/history``, audit replays, and session persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HistoryEvent:
    """One immutable history record.

    Attributes:
        kind: Short category tag (``"turn"``, ``"tool_call"``, ``"error"``,
            ``"permission_granted"``, ``"permission_denied"``, ...).
        title: Short headline describing the event.
        detail: Longer body text (often truncated to keep payloads small).
        timestamp: ISO-8601 timestamp captured at construction.
        metadata: Free-form extras stored alongside the record.
    """

    kind: str
    title: str
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryLog:
    """Append-only ordered list of :class:`HistoryEvent` records.

    Attributes:
        events: All events recorded so far, in insertion order.
    """

    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, kind: str, title: str, detail: str = "", **metadata: Any) -> HistoryEvent:
        """Append a new :class:`HistoryEvent` and return it."""

        event = HistoryEvent(kind=kind, title=title, detail=detail, metadata=metadata)
        self.events.append(event)
        return event

    def add_tool_call(self, name: str, result_preview: str = "", duration_ms: float = 0.0) -> HistoryEvent:
        """Record a tool invocation; ``result_preview`` is truncated to 200 chars."""

        return self.add(
            kind="tool_call",
            title=name,
            detail=result_preview[:200],
            duration_ms=duration_ms,
        )

    def add_error(self, message: str, source: str = "") -> HistoryEvent:
        """Record an error event keyed by ``message`` with optional ``source``."""
        return self.add(kind="error", title=message, detail=source)

    def add_turn(self, model: str, in_tokens: int = 0, out_tokens: int = 0) -> HistoryEvent:
        """Record a completed turn with model and token counts."""

        return self.add(
            kind="turn",
            title=f"Turn completed ({model})",
            detail=f"in={in_tokens}, out={out_tokens}",
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )

    def add_permission(self, tool_name: str, granted: bool) -> HistoryEvent:
        """Record the result of a permission prompt for ``tool_name``."""

        status = "granted" if granted else "denied"
        return self.add(
            kind=f"permission_{status}",
            title=f"{tool_name}: {status}",
        )

    def filter_by_kind(self, kind: str) -> list[HistoryEvent]:
        """Return every event whose ``kind`` matches exactly."""
        return [e for e in self.events if e.kind == kind]

    def last(self, n: int = 10) -> list[HistoryEvent]:
        """Return the most recent ``n`` events in insertion order."""
        return self.events[-n:]

    @property
    def event_count(self) -> int:
        """Number of recorded events."""
        return len(self.events)

    def clear(self) -> None:
        """Drop every recorded event."""
        self.events.clear()

    def as_markdown(self) -> str:
        """Render the log as a Markdown bullet list, one line per event."""

        lines = ["# Session History", "", f"Events: {self.event_count}", ""]
        for event in self.events:
            ts = event.timestamp[:19]
            detail = f" — {event.detail}" if event.detail else ""
            lines.append(f"- [{ts}] **{event.kind}**: {event.title}{detail}")
        return "\n".join(lines)

    def as_dicts(self) -> list[dict[str, Any]]:
        """Serialise every event as a JSON-friendly dict for persistence."""

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
