# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-turn tool-call sequence tracker for skill authoring.

The tracker observes every tool invocation during an agent turn,
recording call name, normalised arguments, duration, status, and any
retries. At the end of the turn the recorded events become a
:class:`SkillCandidate` that the trigger heuristic and drafter consume.
"""

from __future__ import annotations

import time
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCallEvent:
    """One recorded tool invocation within a sequence.

    Attributes:
        tool_name: Registered tool name.
        arguments: Argument dict (normalised; secrets caller's responsibility).
        status: ``"success"`` / ``"error"`` / ``"timeout"`` / ``"blocked"``.
        duration_ms: Wall-clock execution time.
        error_type: Exception class name when ``status != "success"``.
        error_message: Exception message when applicable.
        timestamp: When the call started.
        retry_of: When this is a retry, the index of the original call.
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
    """A finished tool sequence proposed for skill canonicalisation.

    Aggregates the per-turn :class:`ToolCallEvent` history into a
    digestible structure that triggers and drafters can reason about.

    Attributes:
        agent_id: Agent that produced the sequence.
        turn_id: Optional correlator for the originating turn.
        events: Ordered tool calls recorded during the turn.
        successful_events: Subset that completed successfully.
        unique_tools: Distinct tool names invoked.
        retries: Count of events flagged as retries.
        total_duration_ms: Sum of all call durations.
        user_prompt: The user prompt that started the turn (best-effort).
        final_response: The final agent response (best-effort).
        completed_at: When the sequence finished.
    """

    agent_id: str | None = None
    turn_id: str | None = None
    events: list[ToolCallEvent] = field(default_factory=list)
    user_prompt: str = ""
    final_response: str = ""
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def successful_events(self) -> list[ToolCallEvent]:
        """Subset of :attr:`events` with ``status == "success"``."""
        return [e for e in self.events if e.status == "success"]

    @property
    def unique_tools(self) -> list[str]:
        """Distinct tool names in invocation order."""
        seen: set[str] = set()
        out: list[str] = []
        for e in self.events:
            if e.tool_name not in seen:
                seen.add(e.tool_name)
                out.append(e.tool_name)
        return out

    @property
    def retries(self) -> int:
        """Number of recorded retry events."""
        return sum(1 for e in self.events if e.retry_of is not None)

    @property
    def total_duration_ms(self) -> float:
        """Sum of all call durations in milliseconds."""
        return sum(e.duration_ms for e in self.events)

    def signature(self) -> str:
        """Stable string signature of the tool-name sequence.

        Used by the trigger to detect "novel" sequences (i.e. ones not
        already covered by an existing skill).
        """
        return ">".join(e.tool_name for e in self.events)


class ToolSequenceTracker:
    """Stateful per-turn tracker.

    Lifecycle: ``begin_turn() → record_call() … → end_turn()``. Each
    instance is single-threaded; the runtime should create a fresh
    tracker per turn or call :meth:`begin_turn` to reset.

    Example:
        >>> t = ToolSequenceTracker()
        >>> t.begin_turn(agent_id="coder", user_prompt="set up CI")
        >>> t.record_call("Read", {"path": ".github/workflows/ci.yml"})
        >>> t.record_call("Write", {"path": ".github/workflows/ci.yml", "content": "..."})
        >>> candidate = t.end_turn(final_response="CI is configured")
        >>> assert len(candidate.events) == 2
    """

    def __init__(self) -> None:
        """Initialise with no active turn."""
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
        """Start a new tracked turn (resets all state)."""
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
        """Append a single tool-call event.

        Detects retries by hashing ``(tool_name, arguments)`` against
        previously seen signatures within the current turn.

        Args:
            tool_name: Tool that was invoked.
            arguments: Argument dict.
            status: ``"success"`` / ``"error"`` / ``"timeout"`` / ``"blocked"``.
            duration_ms: Wall-clock time. When ``None``, computed via
                :meth:`mark_call_start` if a start timestamp is set.
            error_type: Exception class name (for non-success).
            error_message: Exception message (for non-success).

        Returns:
            The :class:`ToolCallEvent` that was recorded.
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
        """Stamp ``time.perf_counter()`` for the next :meth:`record_call`.

        Lets callers measure elapsed time without computing it themselves.
        Cleared on each ``record_call`` regardless of usage.
        """
        self._call_start = time.perf_counter()

    def end_turn(
        self,
        final_response: str = "",
    ) -> SkillCandidate:
        """Finalise the turn and return the :class:`SkillCandidate`."""
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
        """Number of events recorded so far in the active turn."""
        return len(self._events)

    @property
    def events(self) -> list[ToolCallEvent]:
        """Read-only view of the active turn's events (defensive copy)."""
        return list(self._events)
