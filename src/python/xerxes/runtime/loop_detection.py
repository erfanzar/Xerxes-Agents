# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Tool-loop detection for Xerxes.

Detects repetitive tool call patterns and prevents infinite loops:

1. **Same-call repetition**: The same tool with the same arguments is
   called N times in a row without meaningful progress.
2. **Ping-pong detection**: Two tools alternate back and forth (A→B→A→B).
3. **Total iteration cap**: Hard limit on total tool calls per session turn.

Each detector emits observable events/log entries and can be configured
with warning and critical thresholds.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoopSeverity(Enum):
    """Severity level for a loop detection event.

    Attributes:
        OK: No loop pattern detected; tool call proceeds normally.
        WARNING: A potential loop pattern has been detected but the call
            is still allowed. A log entry is emitted for observability.
        CRITICAL: A confirmed loop pattern has been detected and the
            call should be blocked to prevent infinite execution.
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LoopDetectionConfig:
    """Configuration for loop detection thresholds.

    Attributes:
        same_call_warning: Consecutive identical calls before warning.
        same_call_critical: Consecutive identical calls before blocking.
        pingpong_warning: Alternation count before warning.
        pingpong_critical: Alternation count before blocking.
        max_tool_calls_per_turn: Hard cap on total tool calls in one turn.
        enabled: Master switch for loop detection.
    """

    same_call_warning: int = 3
    same_call_critical: int = 5
    pingpong_warning: int = 4
    pingpong_critical: int = 6
    max_tool_calls_per_turn: int = 25
    enabled: bool = True


@dataclass
class LoopEvent:
    """Emitted when a loop pattern is detected or when a call is checked.

    Carries the severity, pattern type, and descriptive details for
    every tool call check performed by the :class:`LoopDetector`.

    Attributes:
        severity: The severity level of the detection result.
        pattern: The type of loop pattern detected. One of
            ``"same_call"``, ``"pingpong"``, ``"max_calls"``,
            ``"none"``, or ``"disabled"``.
        tool_name: Name of the tool that triggered the event.
        details: Human-readable description of the detection result.
        call_count: The relevant repetition or alternation count that
            triggered the event. Defaults to ``0`` for OK events.
    """

    severity: LoopSeverity
    pattern: str
    tool_name: str
    details: str
    call_count: int = 0


@dataclass
class _CallRecord:
    """Internal record for a single tool call.

    Stores the tool name and an MD5 hash of the serialised arguments
    for efficient equality comparison in loop pattern detection.

    Attributes:
        tool_name: Name of the tool that was invoked.
        args_hash: MD5 hex digest of the serialised call arguments.
    """

    tool_name: str
    args_hash: str


class LoopDetector:
    """Stateful loop detector for a single agent turn.

    Create a new LoopDetector for each turn (or session). Call
    ``record_call`` after every tool invocation. The detector returns
    a ``LoopEvent`` with severity OK, WARNING, or CRITICAL.

    Example:
        >>> detector = LoopDetector()
        >>> event = detector.record_call("search", {"q": "hello"})
        >>> event.severity
        <LoopSeverity.OK: 'ok'>
    """

    def __init__(self, config: LoopDetectionConfig | None = None):
        """Initialise the loop detector with the given configuration.

        Args:
            config: Loop detection thresholds and settings. When
                ``None``, uses default :class:`LoopDetectionConfig`
                values.
        """
        self.config = config or LoopDetectionConfig()
        self._history: list[_CallRecord] = []
        self._listeners: list = []

    @property
    def call_count(self) -> int:
        """Return the total number of tool calls recorded so far.

        Returns:
            The count of tool calls stored in the internal history.
        """
        return len(self._history)

    def add_listener(self, callback) -> None:
        """Register a callable invoked with each emitted LoopEvent.

        Listeners receive every WARNING and CRITICAL event. Exceptions
        raised by listeners are caught and logged without interrupting
        detection.

        Args:
            callback: A callable that accepts a single :class:`LoopEvent`
                argument.
        """
        self._listeners.append(callback)

    def reset(self) -> None:
        """Reset detector state for a new turn.

        Clears the call history so the detector can be reused across
        turns without creating a new instance. Registered listeners
        are preserved.
        """
        self._history.clear()

    def record_call(self, tool_name: str, arguments: dict | str | None = None) -> LoopEvent:
        """Record a tool call and check for loop patterns.

        Appends the call to the internal history and runs all detection
        checks in order: max-calls cap, same-call repetition, and
        ping-pong alternation. The first non-OK result is returned
        immediately.

        Args:
            tool_name: Name of the tool being invoked.
            arguments: The call arguments as a dict, a JSON string, or
                ``None``. Arguments are hashed for comparison; the raw
                values are not stored.

        Returns:
            A :class:`LoopEvent` describing the detection result.
            Severity ``CRITICAL`` means the call should be blocked;
            ``WARNING`` means it should be logged but allowed;
            ``OK`` means no pattern was detected.
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
        """Count consecutive identical calls at the tail of history.

        Walks backwards through the history counting records that share
        the same tool name and arguments hash as *current*. Returns
        a WARNING or CRITICAL event if the count exceeds the configured
        thresholds.

        Args:
            current: The call record to check for repetition.

        Returns:
            A :class:`LoopEvent` with the appropriate severity level.
        """
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
        """Detect A-B-A-B alternation pattern in tool call history.

        Examines the tail of the call history for an alternating
        two-tool pattern (e.g. ``search -> read -> search -> read``).
        Requires at least 4 history entries and no more than 2 distinct
        tool names in the last 4 calls to trigger.

        Returns:
            A :class:`LoopEvent` with WARNING or CRITICAL severity if
            alternation count exceeds the configured thresholds, or OK
            if no ping-pong pattern is detected.
        """
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
        """Log the event and notify all registered listeners.

        Logs at WARNING level for warning-severity events and at ERROR
        level for critical-severity events. Each registered listener
        is called with the event; exceptions in listeners are caught
        and logged without re-raising.

        Args:
            event: The loop detection event to emit.
        """
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
        """Return an MD5 hex digest of the serialised arguments for comparison.

        Args:
            arguments: The tool call arguments to hash. Accepts a dict
                (serialised via ``json.dumps`` with sorted keys), a raw
                JSON string, or ``None`` (which returns ``"empty"``).

        Returns:
            A 32-character hexadecimal MD5 digest string, or the
            literal ``"empty"`` when *arguments* is ``None``.
        """
        if arguments is None:
            return "empty"
        if isinstance(arguments, str):
            raw = arguments
        else:
            raw = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()


class ToolLoopError(Exception):
    """Raised when a tool call is blocked due to loop detection.

    Contains the :class:`LoopEvent` that triggered the block, allowing
    callers to inspect the pattern type, severity, and details.

    Attributes:
        event: The :class:`LoopEvent` that caused the tool call to be
            blocked.
    """

    def __init__(self, event: LoopEvent) -> None:
        """Initialise with the LoopEvent that triggered the block.

        Args:
            event: The critical-severity loop event that caused the
                tool call to be blocked.
        """
        self.event = event
        super().__init__(f"Tool loop detected ({event.pattern}): {event.details}")
