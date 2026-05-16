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
"""Per-key circuit breakers protecting flaky downstream calls.

:class:`CircuitBreakerRegistry` keeps independent breakers keyed by a
caller-chosen identifier (typically ``"<provider>:<model>"``). Each breaker
follows the standard CLOSED → OPEN → HALF_OPEN state machine driven by
:meth:`CircuitBreakerRegistry.record_success` / :meth:`CircuitBreakerRegistry.record_failure`,
with thresholds taken from :class:`CircuitBreakerConfig`. :class:`CircuitOpenError`
is raised by :meth:`CircuitBreakerRegistry.call` when the breaker rejects an
invocation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    """Standard circuit-breaker state machine values.

    Attributes:
        CLOSED: Traffic flows normally and failures count toward tripping.
        OPEN: All calls are rejected until ``cooldown_seconds`` elapse.
        HALF_OPEN: Trial state; one probe call decides whether to close
            again or reopen on failure.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Thresholds and timers shared by every breaker in the registry.

    Attributes:
        failure_threshold: Failures within ``rolling_window_seconds`` that
            move a CLOSED breaker to OPEN.
        cooldown_seconds: Wait time before an OPEN breaker is allowed to
            transition to HALF_OPEN.
        rolling_window_seconds: Window over which failures accumulate.
        success_threshold: Consecutive HALF_OPEN successes required before
            the breaker transitions back to CLOSED.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    rolling_window_seconds: float = 60.0
    success_threshold: int = 1


class CircuitOpenError(Exception):
    """Raised by :meth:`CircuitBreakerRegistry.call` when the breaker is OPEN."""

    def __init__(self, key: str, opened_at: float) -> None:
        """Capture which key tripped and when it opened (``time.monotonic``)."""
        self.key = key
        self.opened_at = opened_at
        super().__init__(f"Circuit '{key}' is OPEN since {opened_at:.0f}")


@dataclass
class _BreakerState:
    """Per-key state held inside :class:`CircuitBreakerRegistry`.

    Attributes:
        state: Current :class:`CircuitState`.
        failures: Monotonic timestamps of failures inside the rolling window.
        consecutive_successes: HALF_OPEN successes accumulated so far.
        opened_at: Monotonic timestamp at which the breaker last opened.
    """

    state: CircuitState = CircuitState.CLOSED
    failures: list[float] = field(default_factory=list)
    consecutive_successes: int = 0
    opened_at: float = 0.0


class CircuitBreakerRegistry:
    """Thread-safe map of breaker key → :class:`_BreakerState`."""

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialise with shared ``config`` (default :class:`CircuitBreakerConfig` if ``None``)."""
        self.config = config or CircuitBreakerConfig()
        self._states: dict[str, _BreakerState] = {}
        self._lock = threading.Lock()

    def _entry(self, key: str) -> _BreakerState:
        """Return (lazily creating) the state record for ``key``. Caller holds lock."""

        s = self._states.get(key)
        if s is None:
            s = _BreakerState()
            self._states[key] = s
        return s

    def should_allow(self, key: str, *, now: float | None = None) -> bool:
        """Decide whether a call against ``key`` should proceed.

        Transitions an OPEN breaker to HALF_OPEN once ``cooldown_seconds``
        have elapsed and returns ``True`` so the caller can probe.
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
        """Note a successful call against ``key``.

        HALF_OPEN breakers transition to CLOSED once enough consecutive
        successes accumulate; CLOSED breakers reset their failure window.
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
        """Note a failed call against ``key`` and return whether it tripped the breaker.

        A HALF_OPEN failure always re-opens; a CLOSED breaker opens once the
        failure count in the rolling window reaches ``failure_threshold``.
        """

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
        """Return the current :class:`CircuitState` for ``key``."""
        with self._lock:
            return self._entry(key).state

    def reset(self, key: str | None = None) -> None:
        """Drop accumulated state for ``key`` (or all keys when ``None``)."""
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def call(self, key: str, fn, *args, **kwargs):
        """Invoke ``fn(*args, **kwargs)`` through the breaker for ``key``.

        Raises:
            CircuitOpenError: The breaker is currently OPEN and the cooldown
                hasn't elapsed.

        Exceptions from ``fn`` are recorded as failures and re-raised so the
        caller can handle them normally.
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
