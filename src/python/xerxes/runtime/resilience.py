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
"""Resilience module for Xerxes.

Exports:
    - JitterMode
    - BackoffPolicy
    - hash_args
    - ToolResultCache"""

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
    """Jitter mode.

    Inherits from: Enum
    """

    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class BackoffPolicy:
    """Backoff policy.

    Attributes:
        base_delay (float): base delay.
        max_delay (float): max delay.
        multiplier (float): multiplier.
        mode (JitterMode): mode.
        rng (random.Random | None): rng."""

    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    mode: JitterMode = JitterMode.FULL
    rng: random.Random | None = None

    def delay(self, attempt: int, *, last_delay: float = 0.0) -> float:
        """Delay.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            attempt (int): IN: attempt. OUT: Consumed during execution.
            last_delay (float, optional): IN: last delay. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            float: OUT: Result of the operation."""

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
        """Sleep iter.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_attempts (int): IN: max attempts. OUT: Consumed during execution.
        Returns:
            tp.Iterator[float]: OUT: Result of the operation."""

        last = 0.0
        for i in range(max_attempts):
            d = self.delay(i, last_delay=last)
            last = d
            yield d


def hash_args(args: tp.Any) -> str:
    """Hash args.

    Args:
        args (tp.Any): IN: args. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    if args is None:
        return "null"
    if isinstance(args, str):
        raw = args
    else:
        raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class _CacheEntry:
    """Cache entry.

    Attributes:
        value (tp.Any): value.
        inserted_at (float): inserted at."""

    value: tp.Any
    inserted_at: float


class ToolResultCache:
    """Tool result cache."""

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 256) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            ttl_seconds (float, optional): IN: ttl seconds. Defaults to 30.0. OUT: Consumed during execution.
            max_entries (int, optional): IN: max entries. Defaults to 256. OUT: Consumed during execution."""

        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Return Hits.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return self._hits

    @property
    def misses(self) -> int:
        """Return Misses.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return self._misses

    def __len__(self) -> int:
        """Dunder method for len.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return len(self._entries)

    def get(self, tool_name: str, args: tp.Any, *, now: float | None = None) -> tuple[bool, tp.Any]:
        """Get.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            args (tp.Any): IN: args. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            tuple[bool, tp.Any]: OUT: Result of the operation."""

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
        """Set.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            args (tp.Any): IN: args. OUT: Consumed during execution.
            value (tp.Any): IN: value. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution."""

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
        """Retrieve the or set.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            args (tp.Any): IN: args. OUT: Consumed during execution.
            producer (tp.Callable[[], tp.Any]): IN: producer. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            tuple[bool, tp.Any]: OUT: Result of the operation."""

        hit, value = self.get(tool_name, args, now=now)
        if hit:
            return True, value
        value = producer()
        self.set(tool_name, args, value, now=now)
        return False, value

    def invalidate(self, tool_name: str | None = None) -> int:
        """Invalidate.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str | None, optional): IN: tool name. Defaults to None. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

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
