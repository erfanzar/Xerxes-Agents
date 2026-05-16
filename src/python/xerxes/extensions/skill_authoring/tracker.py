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
"""Tool call tracking and candidate assembly for skill authoring.

``ToolSequenceTracker`` records every tool invocation during an agent turn.
``SkillCandidate`` aggregates those events into a structure suitable for
``SkillDrafter``.
"""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCallEvent:
    """A single observed tool invocation during an agent turn.

    Attributes:
        tool_name: Tool identifier.
        arguments: Arguments passed to the tool.
        status: Result status (typically ``"success"`` or ``"failure"``).
        duration_ms: Elapsed time in milliseconds.
        error_type: Exception type, when the call failed.
        error_message: Exception message, when the call failed.
        timestamp: Wall-clock time of the event.
        retry_of: Index of the original call if this event is a retry.
    """

    tool_name: str
    arguments: dict[str, tp.Any] = field(default_factory=dict)
    status: str = "success"
    duration_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_of: int | None = None


@dataclass
class SkillCandidate:
    """Aggregated data from one agent turn, ready for skill drafting.

    Attributes:
        agent_id: Agent identifier.
        turn_id: Turn identifier.
        events: Observed tool call events.
        user_prompt: Original user message text.
        final_response: Agent's final textual response.
        completed_at: Wall-clock time the turn finished.
    """

    agent_id: str | None = None
    turn_id: str | None = None
    events: list[ToolCallEvent] = field(default_factory=list)
    user_prompt: str = ""
    final_response: str = ""
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def successful_events(self) -> list[ToolCallEvent]:
        """Return only events whose status is ``"success"``."""

        return [e for e in self.events if e.status == "success"]

    @property
    def unique_tools(self) -> list[str]:
        """Return tool names in order of first appearance, deduplicated."""

        seen: set[str] = set()
        out: list[str] = []
        for e in self.events:
            if e.tool_name not in seen:
                seen.add(e.tool_name)
                out.append(e.tool_name)
        return out

    @property
    def retries(self) -> int:
        """Return the count of events marked as retries."""

        return sum(1 for e in self.events if e.retry_of is not None)

    @property
    def total_duration_ms(self) -> float:
        """Return the sum of every event's ``duration_ms``."""

        return sum(e.duration_ms for e in self.events)

    def signature(self) -> str:
        """Return the sequence as a ``>``-joined string, e.g. ``"tool_a>tool_b"``."""

        return ">".join(e.tool_name for e in self.events)


class ToolSequenceTracker:
    """Accumulate tool call events across a single agent turn."""

    def __init__(self) -> None:
        """Initialize empty tracking state."""

        self._events: list[ToolCallEvent] = []
        self._agent_id: str | None = None
        self._turn_id: str | None = None
        self._user_prompt: str = ""
        self._call_start: float | None = None
        self._signatures: dict[str, int] = {}

    def begin_turn(
        self,
        agent_id: str | None = None,
        turn_id: str | None = None,
        user_prompt: str = "",
    ) -> None:
        """Clear internal state and record identifiers for a new turn."""

        self._events = []
        self._agent_id = agent_id
        self._turn_id = turn_id
        self._user_prompt = user_prompt
        self._call_start = None
        self._signatures = {}

    def record_call(
        self,
        tool_name: str,
        arguments: dict[str, tp.Any] | None = None,
        status: str = "success",
        duration_ms: float | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ToolCallEvent:
        """Record a single tool invocation as a ``ToolCallEvent``.

        Args:
            tool_name: Tool identifier.
            arguments: Arguments passed to the tool.
            status: Result status.
            duration_ms: Elapsed milliseconds; auto-computed from ``mark_call_start``
                if ``None``.
            error_type: Exception type on failure.
            error_message: Exception message on failure.

        Returns:
            The newly recorded event.
        """

        args = dict(arguments or {})
        sig = f"{tool_name}::{sorted(args.items())}"
        retry_of = self._signatures.get(sig)
        self._signatures[sig] = len(self._events)
        if duration_ms is None and self._call_start is not None:
            duration_ms = (time.perf_counter() - self._call_start) * 1000.0
            self._call_start = None
        elif duration_ms is None:
            duration_ms = 0.0
        ev = ToolCallEvent(
            tool_name=tool_name,
            arguments=args,
            status=status,
            duration_ms=float(duration_ms),
            error_type=error_type,
            error_message=error_message,
            retry_of=retry_of,
        )
        self._events.append(ev)
        return ev

    def mark_call_start(self) -> None:
        """Record ``time.perf_counter()`` so the next ``record_call`` can auto-time."""

        self._call_start = time.perf_counter()

    def end_turn(
        self,
        final_response: str = "",
    ) -> SkillCandidate:
        """Finalize the turn and return a ``SkillCandidate`` snapshot."""

        candidate = SkillCandidate(
            agent_id=self._agent_id,
            turn_id=self._turn_id,
            events=list(self._events),
            user_prompt=self._user_prompt,
            final_response=final_response,
        )
        self._events = []
        self._agent_id = None
        self._turn_id = None
        self._user_prompt = ""
        self._call_start = None
        self._signatures = {}
        return candidate

    @property
    def call_count(self) -> int:
        """Return the number of events recorded so far in this turn."""

        return len(self._events)

    @property
    def events(self) -> list[ToolCallEvent]:
        """Return a defensive copy of the events recorded this turn."""

        return list(self._events)
