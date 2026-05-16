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
"""Resilience primitives: jittered backoff and TTL'd tool-result caching.

:class:`BackoffPolicy` computes per-attempt sleep durations with four jitter
modes (none, full, equal, decorrelated). :class:`ToolResultCache` is a small
LRU + TTL cache keyed by ``(tool_name, hash_args(args))`` that lets the
streaming loop avoid repeating expensive idempotent tool calls within a turn.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import typing as tp
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum


class JitterMode(Enum):
    """Jitter strategy applied to :class:`BackoffPolicy` delays.

    Attributes:
        NONE: Deterministic exponential backoff with no jitter.
        FULL: Uniform random in ``[0, exp)`` per AWS Architecture blog.
        EQUAL: Half deterministic, half random.
        DECORRELATED: Random walk using the previous delay as a seed.
    """

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class BackoffPolicy:
    """Exponential backoff schedule with configurable jitter.

    Attributes:
        base_delay: Initial sleep in seconds for attempt 0.
        max_delay: Cap applied to the exponential growth.
        multiplier: Per-attempt growth factor (typically ``2.0``).
        mode: Active :class:`JitterMode`.
        rng: Optional ``random.Random`` for deterministic tests.
    """

    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    mode: JitterMode = JitterMode.FULL
    rng: random.Random | None = None

    def delay(self, attempt: int, *, last_delay: float = 0.0) -> float:
        """Return the sleep duration for ``attempt`` under the active jitter mode.

        ``last_delay`` is only consulted in :attr:`JitterMode.DECORRELATED`.
        """

        rng = self.rng or random
        attempt = max(0, attempt)
        cap = max(self.base_delay, self.max_delay)
        exp = min(cap, self.base_delay * (self.multiplier**attempt))
        if self.mode == JitterMode.NONE:
            return exp
        if self.mode == JitterMode.FULL:
            return rng.uniform(0.0, exp)
        if self.mode == JitterMode.EQUAL:
            return exp / 2.0 + rng.uniform(0.0, exp / 2.0)
        if self.mode == JitterMode.DECORRELATED:
            seed = max(self.base_delay, last_delay)
            return min(cap, rng.uniform(self.base_delay, seed * self.multiplier))
        return exp

    def sleep_iter(self, max_attempts: int) -> tp.Iterator[float]:
        """Yield ``max_attempts`` successive sleep durations from :meth:`delay`."""

        last = 0.0
        for i in range(max_attempts):
            d = self.delay(i, last_delay=last)
            last = d
            yield d


def hash_args(args: tp.Any) -> str:
    """Return a stable MD5 hex digest of ``args`` for cache-key purposes.

    Strings hash directly; everything else is JSON-encoded with
    ``sort_keys=True`` so dict insertion order is irrelevant.
    """

    if args is None:
        return "null"
    if isinstance(args, str):
        raw = args
    else:
        raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class _CacheEntry:
    """Internal cache record.

    Attributes:
        value: Cached tool result.
        inserted_at: Monotonic timestamp at which the entry was stored.
    """

    value: tp.Any
    inserted_at: float


class ToolResultCache:
    """LRU + TTL cache of tool results keyed by ``(name, args_hash)``."""

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 256) -> None:
        """Create a cache with the given TTL and maximum number of entries."""

        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Cumulative cache hits since construction."""
        return self._hits

    @property
    def misses(self) -> int:
        """Cumulative cache misses since construction."""
        return self._misses

    def __len__(self) -> int:
        """Return the current number of stored entries."""
        return len(self._entries)

    def get(self, tool_name: str, args: tp.Any, *, now: float | None = None) -> tuple[bool, tp.Any]:
        """Look up the cached result for ``(tool_name, args)``.

        Returns ``(True, value)`` on hit or ``(False, None)`` on miss/expired.
        Hits refresh LRU ordering.
        """

        now = time.monotonic() if now is None else now
        key = (tool_name, hash_args(args))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return False, None
            if now - entry.inserted_at > self.ttl_seconds:
                del self._entries[key]
                self._misses += 1
                return False, None
            self._entries.move_to_end(key)
            self._hits += 1
            return True, entry.value

    def set(self, tool_name: str, args: tp.Any, value: tp.Any, *, now: float | None = None) -> None:
        """Store ``value`` against ``(tool_name, args)``, evicting LRU entries past the cap."""

        now = time.monotonic() if now is None else now
        key = (tool_name, hash_args(args))
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, inserted_at=now)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get_or_set(
        self,
        tool_name: str,
        args: tp.Any,
        producer: tp.Callable[[], tp.Any],
        *,
        now: float | None = None,
    ) -> tuple[bool, tp.Any]:
        """Return ``(hit, value)``; on miss, calls ``producer()`` and caches the result."""

        hit, value = self.get(tool_name, args, now=now)
        if hit:
            return True, value
        value = producer()
        self.set(tool_name, args, value, now=now)
        return False, value

    def invalidate(self, tool_name: str | None = None) -> int:
        """Drop cached entries; restrict to ``tool_name`` when given, all otherwise.

        Returns the number of entries removed.
        """

        with self._lock:
            if tool_name is None:
                n = len(self._entries)
                self._entries.clear()
                return n
            removed = 0
            for key in list(self._entries.keys()):
                if key[0] == tool_name:
                    del self._entries[key]
                    removed += 1
            return removed


__all__ = [
    "BackoffPolicy",
    "JitterMode",
    "ToolResultCache",
    "hash_args",
]
