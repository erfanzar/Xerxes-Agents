# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-tool circuit breaker for the Xerxes runtime.

Implements a classic three-state breaker (CLOSED → OPEN → HALF_OPEN)
so that a misbehaving tool is short-circuited until it has had a
chance to recover. Threadsafe; entirely in-memory.

States:

- **CLOSED**: normal — calls flow through, failures are counted.
- **OPEN**: too many failures within the window → calls are rejected
  immediately for the cooldown period.
- **HALF_OPEN**: cooldown elapsed → exactly one trial call is allowed.
  On success the breaker closes; on failure it re-opens with a fresh
  cooldown.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    """Three states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Tunable thresholds for a circuit breaker.

    Attributes:
        failure_threshold: Consecutive failures that trip the breaker.
        cooldown_seconds: Time the breaker stays OPEN before allowing
            a HALF_OPEN trial.
        rolling_window_seconds: Failures older than this are forgotten.
        success_threshold: Successes (in HALF_OPEN) needed to close.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    rolling_window_seconds: float = 60.0
    success_threshold: int = 1


class CircuitOpenError(Exception):
    """Raised when a breaker rejects a call because it is OPEN."""

    def __init__(self, key: str, opened_at: float) -> None:
        """Record which breaker rejected the call and when it opened."""
        self.key = key
        self.opened_at = opened_at
        super().__init__(f"Circuit '{key}' is OPEN since {opened_at:.0f}")


@dataclass
class _BreakerState:
    """Per-key mutable state used internally by :class:`CircuitBreakerRegistry`."""

    state: CircuitState = CircuitState.CLOSED
    failures: list[float] = field(default_factory=list)
    consecutive_successes: int = 0
    opened_at: float = 0.0


class CircuitBreakerRegistry:
    """Per-tool/per-agent circuit breakers.

    Each unique ``key`` (typically the tool name, optionally suffixed
    with an agent_id for per-agent isolation) gets its own state.

    Use the :meth:`should_allow` / :meth:`record_success` /
    :meth:`record_failure` triple from inside the executor, OR wrap a
    callable with :meth:`call`.

    Example:
        >>> br = CircuitBreakerRegistry()
        >>> for _ in range(5):
        ...     br.record_failure("flaky_tool")
        >>> br.should_allow("flaky_tool")
        False
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Create the registry with the supplied config (defaults when omitted)."""
        self.config = config or CircuitBreakerConfig()
        self._states: dict[str, _BreakerState] = {}
        self._lock = threading.Lock()

    def _entry(self, key: str) -> _BreakerState:
        """Return the per-key :class:`_BreakerState`, creating a fresh one on miss."""
        s = self._states.get(key)
        if s is None:
            s = _BreakerState()
            self._states[key] = s
        return s

    def should_allow(self, key: str, *, now: float | None = None) -> bool:
        """Return ``True`` when a call to *key* may proceed.

        Side-effect: when the cooldown for an OPEN breaker has elapsed,
        the state transitions to HALF_OPEN and this call returns
        ``True`` (allowing the trial).

        Args:
            key: Breaker identity (e.g. tool name).
            now: Optional override for the current time (for tests).
        """
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
        """Record a successful call against *key*.

        In HALF_OPEN, ``success_threshold`` consecutive successes close
        the breaker. In CLOSED, this clears any rolling-window failures.
        """
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
        """Record a failed call. Returns ``True`` when the breaker tripped now."""
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
        """Return the current state of *key*'s breaker."""
        with self._lock:
            return self._entry(key).state

    def reset(self, key: str | None = None) -> None:
        """Reset a single breaker, or all when *key* is ``None``."""
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def call(self, key: str, fn, *args, **kwargs):
        """Invoke ``fn(*args, **kwargs)`` guarded by the breaker.

        Raises :class:`CircuitOpenError` when *key* is currently OPEN.
        Records success or failure based on whether the callable raises.
        """
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
