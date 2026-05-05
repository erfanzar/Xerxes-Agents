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
"""Loop detection module for Xerxes.

Exports:
    - logger
    - LoopSeverity
    - LoopDetectionConfig
    - LoopEvent
    - LoopDetector
    - ToolLoopError"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoopSeverity(Enum):
    """Loop severity.

    Inherits from: Enum
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LoopDetectionConfig:
    """Loop detection config.

    Attributes:
        same_call_warning (int): same call warning.
        same_call_critical (int): same call critical.
        pingpong_warning (int): pingpong warning.
        pingpong_critical (int): pingpong critical.
        max_tool_calls_per_turn (int): max tool calls per turn.
        enabled (bool): enabled."""

    same_call_warning: int = 3
    same_call_critical: int = 5
    pingpong_warning: int = 4
    pingpong_critical: int = 6
    max_tool_calls_per_turn: int = 25
    enabled: bool = True


@dataclass
class LoopEvent:
    """Loop event.

    Attributes:
        severity (LoopSeverity): severity.
        pattern (str): pattern.
        tool_name (str): tool name.
        details (str): details.
        call_count (int): call count."""

    severity: LoopSeverity
    pattern: str
    tool_name: str
    details: str
    call_count: int = 0


@dataclass
class _CallRecord:
    """Call record.

    Attributes:
        tool_name (str): tool name.
        args_hash (str): args hash."""

    tool_name: str
    args_hash: str


class LoopDetector:
    """Loop detector."""

    def __init__(self, config: LoopDetectionConfig | None = None):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LoopDetectionConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        self.config = config or LoopDetectionConfig()
        self._history: list[_CallRecord] = []
        self._listeners: list = []

    @property
    def call_count(self) -> int:
        """Return Call count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return len(self._history)

    def add_listener(self, callback) -> None:
        """Add listener.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            callback (Any): IN: callback. OUT: Consumed during execution."""

        self._listeners.append(callback)

    def reset(self) -> None:
        """Reset.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._history.clear()

    def record_call(self, tool_name: str, arguments: dict | str | None = None) -> LoopEvent:
        """Record call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            arguments (dict | str | None, optional): IN: arguments. Defaults to None. OUT: Consumed during execution.
        Returns:
            LoopEvent: OUT: Result of the operation."""

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
        """Internal helper to check same call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            current (_CallRecord): IN: current. OUT: Consumed during execution.
        Returns:
            LoopEvent: OUT: Result of the operation."""

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
        """Internal helper to check pingpong.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            LoopEvent: OUT: Result of the operation."""

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
        """Internal helper to emit.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            event (LoopEvent): IN: event. OUT: Consumed during execution."""

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
        """Internal helper to hash args.

        Args:
            arguments (dict | str | None): IN: arguments. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if arguments is None:
            return "empty"
        if isinstance(arguments, str):
            raw = arguments
        else:
            raw = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()


class ToolLoopError(Exception):
    """Tool loop error.

    Inherits from: Exception
    """

    def __init__(self, event: LoopEvent) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            event (LoopEvent): IN: event. OUT: Consumed during execution."""

        self.event = event
        super().__init__(f"Tool loop detected ({event.pattern}): {event.details}")
