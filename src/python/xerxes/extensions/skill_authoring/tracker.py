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
    """Single observed tool invocation.

    Attributes:
        tool_name (str): IN: Tool identifier. OUT: Stored.
        arguments (dict[str, tp.Any]): IN: Arguments dict. OUT: Stored.
        status (str): IN: Result status. OUT: Stored.
        duration_ms (float): IN: Elapsed milliseconds. OUT: Stored.
        error_type (str | None): IN: Exception type on failure. OUT: Stored.
        error_message (str | None): IN: Exception message on failure. OUT:
            Stored.
        timestamp (datetime): IN: Wall-clock time. OUT: Defaults to ``now``.
        retry_of (int | None): IN: Index of original call if this is a retry.
            OUT: Stored.
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
        agent_id (str | None): IN: Agent identifier. OUT: Stored.
        turn_id (str | None): IN: Turn identifier. OUT: Stored.
        events (list[ToolCallEvent]): IN: Observed calls. OUT: Stored.
        user_prompt (str): IN: Original user message. OUT: Stored.
        final_response (str): IN: Agent's final text. OUT: Stored.
        completed_at (datetime): IN: Completion time. OUT: Defaults to
            ``now``.
    """

    agent_id: str | None = None
    turn_id: str | None = None
    events: list[ToolCallEvent] = field(default_factory=list)
    user_prompt: str = ""
    final_response: str = ""
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def successful_events(self) -> list[ToolCallEvent]:
        """Return only events with ``status == "success"``.

        Returns:
            list[ToolCallEvent]: OUT: Filtered events.
        """

        return [e for e in self.events if e.status == "success"]

    @property
    def unique_tools(self) -> list[str]:
        """Return deduplicated tool names in order of first appearance.

        Returns:
            list[str]: OUT: Tool name list.
        """

        seen: set[str] = set()
        out: list[str] = []
        for e in self.events:
            if e.tool_name not in seen:
                seen.add(e.tool_name)
                out.append(e.tool_name)
        return out

    @property
    def retries(self) -> int:
        """Count how many events are retries.

        Returns:
            int: OUT: Number of events with ``retry_of`` set.
        """

        return sum(1 for e in self.events if e.retry_of is not None)

    @property
    def total_duration_ms(self) -> float:
        """Sum of all event durations.

        Returns:
            float: OUT: Total milliseconds.
        """

        return sum(e.duration_ms for e in self.events)

    def signature(self) -> str:
        """Return a ``>``-joined string of all tool names in order.

        Returns:
            str: OUT: Sequence signature like ``"tool_a>tool_b"``.
        """

        return ">".join(e.tool_name for e in self.events)


class ToolSequenceTracker:
    """Accumulates tool call events across a single agent turn."""

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
        """Reset state for a new turn.

        Args:
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
            user_prompt (str): IN: User message text. OUT: Stored.

        Returns:
            None: OUT: Internal buffers are cleared.
        """

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
        """Record a single tool invocation.

        Args:
            tool_name (str): IN: Tool identifier. OUT: Stored.
            arguments (dict[str, tp.Any] | None): IN: Arguments. OUT: Stored.
            status (str): IN: Result status. OUT: Stored.
            duration_ms (float | None): IN: Elapsed ms; auto-computed from
                ``mark_call_start`` if omitted. OUT: Stored.
            error_type (str | None): IN: Exception type. OUT: Stored.
            error_message (str | None): IN: Exception message. OUT: Stored.

        Returns:
            ToolCallEvent: OUT: Created event instance.
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
        """Record the start time for the next tool call.

        Returns:
            None: OUT: ``_call_start`` is set to ``time.perf_counter()``.
        """

        self._call_start = time.perf_counter()

    def end_turn(
        self,
        final_response: str = "",
    ) -> SkillCandidate:
        """Assemble the tracked events into a ``SkillCandidate``.

        Args:
            final_response (str): IN: Agent's final text response. OUT: Stored
                on the candidate.

        Returns:
            SkillCandidate: OUT: Aggregated turn data.
        """

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
        """Return the number of events recorded this turn.

        Returns:
            int: OUT: Length of ``_events``.
        """

        return len(self._events)

    @property
    def events(self) -> list[ToolCallEvent]:
        """Return a snapshot of recorded events.

        Returns:
            list[ToolCallEvent]: OUT: Copy of ``_events``.
        """

        return list(self._events)
