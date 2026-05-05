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
"""Circuit breaker module for Xerxes.

Exports:
    - CircuitState
    - CircuitBreakerConfig
    - CircuitOpenError
    - CircuitBreakerRegistry"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    """Circuit state.

    Inherits from: Enum
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker config.

    Attributes:
        failure_threshold (int): failure threshold.
        cooldown_seconds (float): cooldown seconds.
        rolling_window_seconds (float): rolling window seconds.
        success_threshold (int): success threshold."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    rolling_window_seconds: float = 60.0
    success_threshold: int = 1


class CircuitOpenError(Exception):
    """Circuit open error.

    Inherits from: Exception
    """

    def __init__(self, key: str, opened_at: float) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            opened_at (float): IN: opened at. OUT: Consumed during execution."""

        self.key = key
        self.opened_at = opened_at
        super().__init__(f"Circuit '{key}' is OPEN since {opened_at:.0f}")


@dataclass
class _BreakerState:
    """Breaker state.

    Attributes:
        state (CircuitState): state.
        failures (list[float]): failures.
        consecutive_successes (int): consecutive successes.
        opened_at (float): opened at."""

    state: CircuitState = CircuitState.CLOSED
    failures: list[float] = field(default_factory=list)
    consecutive_successes: int = 0
    opened_at: float = 0.0


class CircuitBreakerRegistry:
    """Circuit breaker registry."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (CircuitBreakerConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution."""

        self.config = config or CircuitBreakerConfig()
        self._states: dict[str, _BreakerState] = {}
        self._lock = threading.Lock()

    def _entry(self, key: str) -> _BreakerState:
        """Internal helper to entry.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            _BreakerState: OUT: Result of the operation."""

        s = self._states.get(key)
        if s is None:
            s = _BreakerState()
            self._states[key] = s
        return s

    def should_allow(self, key: str, *, now: float | None = None) -> bool:
        """Determine whether allow.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        now = time.monotonic() if now is None else now
        with self._lock:
            s = self._entry(key)
            if s.state == CircuitState.CLOSED:
                return True
            if s.state == CircuitState.OPEN:
                if now - s.opened_at >= self.config.cooldown_seconds:
                    s.state = CircuitState.HALF_OPEN
                    s.consecutive_successes = 0
                    return True
                return False
            return True

    def record_success(self, key: str, *, now: float | None = None) -> None:
        """Record success.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution."""

        now = time.monotonic() if now is None else now
        with self._lock:
            s = self._entry(key)
            if s.state == CircuitState.HALF_OPEN:
                s.consecutive_successes += 1
                if s.consecutive_successes >= self.config.success_threshold:
                    s.state = CircuitState.CLOSED
                    s.failures.clear()
                    s.consecutive_successes = 0
            elif s.state == CircuitState.CLOSED:
                s.failures.clear()

    def record_failure(self, key: str, *, now: float | None = None) -> bool:
        """Record failure.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        now = time.monotonic() if now is None else now
        with self._lock:
            s = self._entry(key)
            cutoff = now - self.config.rolling_window_seconds
            s.failures = [t for t in s.failures if t >= cutoff]
            s.failures.append(now)
            if s.state == CircuitState.HALF_OPEN:
                s.state = CircuitState.OPEN
                s.opened_at = now
                s.consecutive_successes = 0
                return True
            if s.state == CircuitState.CLOSED and len(s.failures) >= self.config.failure_threshold:
                s.state = CircuitState.OPEN
                s.opened_at = now
                return True
            return False

    def state_of(self, key: str) -> CircuitState:
        """State of.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            CircuitState: OUT: Result of the operation."""

        with self._lock:
            return self._entry(key).state

    def reset(self, key: str | None = None) -> None:
        """Reset.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str | None, optional): IN: key. Defaults to None. OUT: Consumed during execution."""

        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def call(self, key: str, fn, *args, **kwargs):
        """Call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            fn (Any): IN: fn. OUT: Consumed during execution.
            *args: IN: Additional positional arguments. OUT: Passed through to downstream calls.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        if not self.should_allow(key):
            raise CircuitOpenError(key, opened_at=self._entry(key).opened_at)
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure(key)
            raise
        self.record_success(key)
        return result


__all__ = [
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
]
