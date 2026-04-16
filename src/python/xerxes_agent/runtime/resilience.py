# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Resilience helpers: jittered backoff + tool result dedup cache.

Two small utilities the executor wires in:

- :class:`BackoffPolicy` — exponential backoff with full / equal /
  decorrelated jitter for retry loops.
- :class:`ToolResultCache` — TTL-bounded LRU cache keyed by
  ``(tool_name, args_hash)``; used to skip identical tool calls fired
  in quick succession (e.g. inside a loop).
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
    """Jitter algorithm. ``FULL`` is the AWS-recommended default."""

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class BackoffPolicy:
    """Exponential backoff with optional jitter.

    Attributes:
        base_delay: First retry delay in seconds.
        max_delay: Cap on the computed delay.
        multiplier: Exponential growth factor.
        mode: Which jitter algorithm to use.
        rng: Optional RNG for deterministic tests.
    """

    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    mode: JitterMode = JitterMode.FULL
    rng: random.Random | None = None

    def delay(self, attempt: int, *, last_delay: float = 0.0) -> float:
        """Compute the next sleep duration.

        Args:
            attempt: Zero-indexed retry attempt number.
            last_delay: Previous delay (used by DECORRELATED mode).

        Returns:
            Seconds to sleep before the next retry.
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
        """Yield delays for ``max_attempts`` retries.

        Use as ``for d in policy.sleep_iter(5): time.sleep(d); ...``.
        """
        last = 0.0
        for i in range(max_attempts):
            d = self.delay(i, last_delay=last)
            last = d
            yield d


def hash_args(args: tp.Any) -> str:
    """Stable MD5 hash of an argument bundle (dict / str / None)."""
    if args is None:
        return "null"
    if isinstance(args, str):
        raw = args
    else:
        raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class _CacheEntry:
    """Single stored value alongside the monotonic timestamp of insertion."""

    value: tp.Any
    inserted_at: float


class ToolResultCache:
    """TTL-bounded LRU cache for tool results.

    Skips identical ``(tool_name, args)`` invocations made within the
    TTL window. Pure in-memory; threadsafe via a single lock.

    Example:
        >>> cache = ToolResultCache(ttl_seconds=30, max_entries=128)
        >>> hit, val = cache.get_or_set("Read", {"path": "x"}, lambda: "abc")
        >>> assert val == "abc"  # computed
        >>> hit, val = cache.get_or_set("Read", {"path": "x"}, lambda: "boom")
        >>> assert val == "abc"  # cached
    """

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 256) -> None:
        """Configure the TTL (seconds) and the maximum LRU capacity."""
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Number of lookups that returned a fresh cached value."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of lookups that missed the cache or found an expired entry."""
        return self._misses

    def __len__(self) -> int:
        """Number of entries currently held in the cache."""
        return len(self._entries)

    def get(self, tool_name: str, args: tp.Any, *, now: float | None = None) -> tuple[bool, tp.Any]:
        """Return ``(hit, value_or_None)`` for a cached entry."""
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
        """Insert (or refresh) a cache entry, evicting LRU if needed."""
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
        """Return cached value or compute via *producer* and cache it."""
        hit, value = self.get(tool_name, args, now=now)
        if hit:
            return True, value
        value = producer()
        self.set(tool_name, args, value, now=now)
        return False, value

    def invalidate(self, tool_name: str | None = None) -> int:
        """Drop cache entries; ``None`` clears everything."""
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
