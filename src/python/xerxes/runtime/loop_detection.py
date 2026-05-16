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
"""Detect repetitive tool-call patterns that suggest the agent is stuck.

:class:`LoopDetector` records every tool invocation a turn makes and watches
for three failure modes:

    * Same tool + arguments called many times in a row (``same_call``).
    * Two tools alternating like ping-pong (``pingpong``).
    * The turn-wide tool-call ceiling being reached (``max_calls``).

Each detection produces a :class:`LoopEvent` that registered listeners (and
:class:`RuntimeFeaturesState`'s audit hook) can react to. :class:`ToolLoopError`
is provided for callers that want to abort the turn instead of logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoopSeverity(Enum):
    """Severity level reported by :class:`LoopDetector`.

    Attributes:
        OK: No loop pattern detected.
        WARNING: Pattern emerging; surface to the user but keep running.
        CRITICAL: Pattern entrenched; the caller should abort the turn.
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LoopDetectionConfig:
    """Thresholds tuning :class:`LoopDetector` behaviour.

    Attributes:
        same_call_warning: Consecutive identical calls before WARNING.
        same_call_critical: Consecutive identical calls before CRITICAL.
        pingpong_warning: Alternations between two tools before WARNING.
        pingpong_critical: Alternations between two tools before CRITICAL.
        max_tool_calls_per_turn: Hard ceiling on total tool calls per turn;
            crossing it always emits CRITICAL.
        enabled: Master switch; when ``False`` every call returns ``OK``.
    """

    same_call_warning: int = 3
    same_call_critical: int = 5
    pingpong_warning: int = 4
    pingpong_critical: int = 6
    max_tool_calls_per_turn: int = 25
    enabled: bool = True


@dataclass
class LoopEvent:
    """One emit from :class:`LoopDetector`, OK or otherwise.

    Attributes:
        severity: Resulting :class:`LoopSeverity`.
        pattern: Identifier of the matched pattern (``"same_call"``,
            ``"pingpong"``, ``"max_calls"``, ``"none"``).
        tool_name: Most relevant tool name for the event.
        details: Human-readable explanation suitable for logs / audit.
        call_count: Number of repetitions that triggered the event.
    """

    severity: LoopSeverity
    pattern: str
    tool_name: str
    details: str
    call_count: int = 0


@dataclass
class _CallRecord:
    """Internal record of one tool invocation.

    Attributes:
        tool_name: Name of the tool.
        args_hash: Stable hash of the call's arguments.
    """

    tool_name: str
    args_hash: str


class LoopDetector:
    """Tracks per-turn tool calls and flags suspicious repetition patterns."""

    def __init__(self, config: LoopDetectionConfig | None = None):
        """Create a detector tuned by ``config`` (default thresholds when ``None``)."""
        self.config = config or LoopDetectionConfig()
        self._history: list[_CallRecord] = []
        self._listeners: list = []

    @property
    def call_count(self) -> int:
        """Number of calls recorded since the last :meth:`reset`."""
        return len(self._history)

    def add_listener(self, callback) -> None:
        """Register ``callback`` to receive every non-OK :class:`LoopEvent`."""
        self._listeners.append(callback)

    def reset(self) -> None:
        """Drop the recorded call history (typically called at turn start)."""
        self._history.clear()

    def record_call(self, tool_name: str, arguments: dict | str | None = None) -> LoopEvent:
        """Append a new call and evaluate every loop pattern.

        The returned event is always non-``None``; its ``severity`` is ``OK``
        when nothing suspicious was found. Non-OK events are also dispatched
        to every registered listener via :meth:`_emit`.
        """

        if not self.config.enabled:
            return LoopEvent(severity=LoopSeverity.OK, pattern="disabled", tool_name=tool_name, details="")

        args_hash = self._hash_args(arguments)
        record = _CallRecord(tool_name=tool_name, args_hash=args_hash)
        self._history.append(record)

        if len(self._history) >= self.config.max_tool_calls_per_turn:
            event = LoopEvent(
                severity=LoopSeverity.CRITICAL,
                pattern="max_calls",
                tool_name=tool_name,
                details=f"Reached max tool calls per turn ({self.config.max_tool_calls_per_turn})",
                call_count=len(self._history),
            )
            self._emit(event)
            return event

        same_event = self._check_same_call(record)
        if same_event.severity != LoopSeverity.OK:
            self._emit(same_event)
            return same_event

        pp_event = self._check_pingpong()
        if pp_event.severity != LoopSeverity.OK:
            self._emit(pp_event)
            return pp_event

        return LoopEvent(severity=LoopSeverity.OK, pattern="none", tool_name=tool_name, details="")

    def _check_same_call(self, current: _CallRecord) -> LoopEvent:
        """Count consecutive identical calls and emit warning/critical if needed."""

        count = 0
        for rec in reversed(self._history):
            if rec.tool_name == current.tool_name and rec.args_hash == current.args_hash:
                count += 1
            else:
                break

        if count >= self.config.same_call_critical:
            return LoopEvent(
                severity=LoopSeverity.CRITICAL,
                pattern="same_call",
                tool_name=current.tool_name,
                details=f"Same tool+args called {count} times consecutively",
                call_count=count,
            )
        if count >= self.config.same_call_warning:
            return LoopEvent(
                severity=LoopSeverity.WARNING,
                pattern="same_call",
                tool_name=current.tool_name,
                details=f"Same tool+args called {count} times consecutively",
                call_count=count,
            )
        return LoopEvent(severity=LoopSeverity.OK, pattern="same_call", tool_name=current.tool_name, details="")

    def _check_pingpong(self) -> LoopEvent:
        """Detect two-tool ping-pong alternation in the recent call tail."""

        if len(self._history) < 4:
            return LoopEvent(severity=LoopSeverity.OK, pattern="pingpong", tool_name="", details="")

        names = [r.tool_name for r in self._history]
        if len(set(names[-4:])) > 2:
            return LoopEvent(severity=LoopSeverity.OK, pattern="pingpong", tool_name="", details="")

        alternation = 0
        for i in range(len(names) - 1, 0, -1):
            if names[i] != names[i - 1]:
                alternation += 1
            else:
                break

        if alternation >= self.config.pingpong_critical:
            return LoopEvent(
                severity=LoopSeverity.CRITICAL,
                pattern="pingpong",
                tool_name=names[-1],
                details=f"Ping-pong pattern detected ({alternation} alternations)",
                call_count=alternation,
            )
        if alternation >= self.config.pingpong_warning:
            return LoopEvent(
                severity=LoopSeverity.WARNING,
                pattern="pingpong",
                tool_name=names[-1],
                details=f"Ping-pong pattern detected ({alternation} alternations)",
                call_count=alternation,
            )
        return LoopEvent(severity=LoopSeverity.OK, pattern="pingpong", tool_name="", details="")

    def _emit(self, event: LoopEvent) -> None:
        """Log ``event`` and fan it out to every registered listener."""

        logger.log(
            logging.WARNING if event.severity == LoopSeverity.WARNING else logging.ERROR,
            "Loop detection [%s] %s: %s",
            event.severity.value,
            event.pattern,
            event.details,
        )
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.warning("Loop detection listener error", exc_info=True)

    @staticmethod
    def _hash_args(arguments: dict | str | None) -> str:
        """Return a stable MD5 hash of ``arguments`` for equality comparisons."""

        if arguments is None:
            return "empty"
        if isinstance(arguments, str):
            raw = arguments
        else:
            raw = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()


class ToolLoopError(Exception):
    """Raised when callers want to abort a turn because of a critical loop event."""

    def __init__(self, event: LoopEvent) -> None:
        """Wrap ``event`` and synthesise a descriptive exception message."""
        self.event = event
        super().__init__(f"Tool loop detected ({event.pattern}): {event.details}")
