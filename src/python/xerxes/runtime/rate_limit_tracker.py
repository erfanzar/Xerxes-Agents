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
"""Track per-(provider, model) rate-limit state from response headers.

Parses ``x-ratelimit-*`` and ``retry-after`` headers, retains the latest
snapshot per (provider, model) pair, and exposes :meth:`RateLimitTracker.should_throttle`
plus :meth:`RateLimitTracker.delay_ms` so callers can pre-emptively pause
before they get a 429. Defaults align with the OpenAI / Anthropic header
conventions and the constant tuples on :class:`RateLimitTracker` make it
straightforward to add new dialects.
"""

from __future__ import annotations

import threading
import time
import typing as tp
from dataclasses import dataclass, field


def _parse_int(value: tp.Any) -> int | None:
    """Parse ``value`` as an ``int`` or return ``None`` for non-numeric input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: tp.Any) -> float | None:
    """Parse ``value`` as a ``float`` or return ``None`` for non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RateLimitState:
    """Latest snapshot of the rate-limit budget for a given provider+model.

    Attributes:
        limit_requests: Total requests-per-minute budget reported by the provider.
        remaining_requests: Requests still available in the current window.
        limit_tokens: Total tokens-per-minute budget.
        remaining_tokens: Tokens still available in the current window.
        reset_at: Epoch seconds when the request bucket resets, if known.
        retry_after: Explicit retry hint in seconds from the most recent 429.
        last_updated: Wall-clock time when this snapshot was last refreshed.
    """

    limit_requests: int | None = None
    remaining_requests: int | None = None
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_at: float | None = None
    retry_after: float | None = None
    last_updated: float = field(default_factory=time.time)


class RateLimitTracker:
    """Track and reason about LLM-API rate limits per (provider, model)."""

    # Standard header names — lowercase, with fallbacks for common dialects.
    HEADER_LIMIT_REQS = ("x-ratelimit-limit-requests", "x-ratelimit-limit")
    HEADER_REMAINING_REQS = ("x-ratelimit-remaining-requests", "x-ratelimit-remaining")
    HEADER_LIMIT_TOKENS = ("x-ratelimit-limit-tokens",)
    HEADER_REMAINING_TOKENS = ("x-ratelimit-remaining-tokens",)
    HEADER_RESET = ("x-ratelimit-reset-requests", "x-ratelimit-reset")
    HEADER_RETRY_AFTER = ("retry-after",)

    def __init__(self, *, throttle_ratio: float = 0.05) -> None:
        """Construct a new tracker with a configurable throttle threshold.

        Args:
            throttle_ratio: Fraction of the limit below which
                :meth:`should_throttle` flips to ``True``. Must be in
                ``(0.0, 1.0)``; defaults to a conservative ``0.05`` (5%).

        Raises:
            ValueError: ``throttle_ratio`` is outside ``(0, 1)``.
        """
        if not 0.0 < throttle_ratio < 1.0:
            raise ValueError("throttle_ratio must be in (0.0, 1.0)")
        self._throttle_ratio = throttle_ratio
        self._states: dict[tuple[str, str], RateLimitState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _pluck(headers: dict[str, str], names: tuple[str, ...]) -> str | None:
        """Case-insensitive lookup that tries each name in order."""
        lower = {k.lower(): v for k, v in headers.items()}
        for n in names:
            v = lower.get(n)
            if v is not None:
                return v
        return None

    def update(self, provider: str, model: str, headers: dict[str, str]) -> RateLimitState:
        """Merge a response's rate-limit headers into the tracker.

        Args:
            provider: Provider identifier, e.g. ``"openai"`` or ``"anthropic"``.
            model: Model identifier as reported by the API.
            headers: Raw response headers; lookups are case-insensitive.

        Returns:
            The updated :class:`RateLimitState` for the ``(provider, model)`` key.
            Missing headers are left untouched (a partial update never wipes
            an earlier observation).
        """
        now = time.time()
        with self._lock:
            state = self._states.setdefault((provider, model), RateLimitState())
            v = self._pluck(headers, self.HEADER_LIMIT_REQS)
            if (parsed := _parse_int(v)) is not None:
                state.limit_requests = parsed
            v = self._pluck(headers, self.HEADER_REMAINING_REQS)
            if (parsed := _parse_int(v)) is not None:
                state.remaining_requests = parsed
            v = self._pluck(headers, self.HEADER_LIMIT_TOKENS)
            if (parsed := _parse_int(v)) is not None:
                state.limit_tokens = parsed
            v = self._pluck(headers, self.HEADER_REMAINING_TOKENS)
            if (parsed := _parse_int(v)) is not None:
                state.remaining_tokens = parsed
            v = self._pluck(headers, self.HEADER_RESET)
            if (parsed := _parse_float(v)) is not None:
                state.reset_at = now + parsed
            v = self._pluck(headers, self.HEADER_RETRY_AFTER)
            if (parsed := _parse_float(v)) is not None:
                state.retry_after = parsed
            state.last_updated = now
            return state

    def state(self, provider: str, model: str) -> RateLimitState | None:
        """Return the latest snapshot for the key, or ``None`` if never seen."""
        with self._lock:
            return self._states.get((provider, model))

    def should_throttle(self, provider: str, model: str) -> bool:
        """Return ``True`` when the request or token budget is dangerously low.

        Also returns ``True`` while a recent ``retry-after`` window is still
        active.
        """
        st = self.state(provider, model)
        if st is None:
            return False
        if st.retry_after and st.last_updated + st.retry_after > time.time():
            return True
        # Check both ratios — whichever signals first wins.
        for limit, remaining in ((st.limit_requests, st.remaining_requests), (st.limit_tokens, st.remaining_tokens)):
            if limit and remaining is not None and limit > 0:
                ratio = remaining / limit
                if ratio < self._throttle_ratio:
                    return True
        return False

    def delay_ms(self, provider: str, model: str) -> int:
        """Return a suggested pre-call delay in milliseconds.

        Returns ``0`` when no throttling is needed. Honours a still-active
        ``retry_after`` first; otherwise sleeps up to the bucket reset (capped
        at 60 seconds) when :meth:`should_throttle` is true, falling back to a
        small conservative default when no reset hint is available.
        """
        st = self.state(provider, model)
        if st is None:
            return 0
        if st.retry_after:
            remaining_window = st.last_updated + st.retry_after - time.time()
            if remaining_window > 0:
                return int(remaining_window * 1000)
        if not self.should_throttle(provider, model):
            return 0
        # If budget is very low, sleep until reset (capped at 60 s).
        if st.reset_at:
            until_reset = max(0.0, st.reset_at - time.time())
            return int(min(60.0, until_reset) * 1000)
        return 250  # conservative default

    def clear(self) -> None:
        """Drop every tracked ``(provider, model)`` snapshot."""
        with self._lock:
            self._states.clear()


__all__ = ["RateLimitState", "RateLimitTracker"]
