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
"""Content-hash cache for read-only tool outputs.

Wraps a tool executor so that identical read-only tool calls (ReadFile,
GrepTool, GlobTool, etc.) return cached results instead of re-executing.
Cache keys incorporate file modification times so the cache automatically
invalidates when a file changes on disk.

Example::

    cache = ToolOutputCache(max_entries=200)
    cached_executor = cache.wrap(real_executor)
    # First call executes; second identical call hits cache.
    cached_executor("ReadFile", {"file_path": "x.py"})
    cached_executor("ReadFile", {"file_path": "x.py"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHEABLE_TOOLS: frozenset[str] = frozenset(
    {
        "ReadFile",
        "GlobTool",
        "GrepTool",
        "ListDir",
    }
)

_DEFAULT_MAX_ENTRIES = 200
_DEFAULT_TTL_SECONDS = 300.0


def _extract_file_paths(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Extract file-path arguments from a tool call for mtime-based invalidation."""

    paths: list[str] = []
    for key in ("file_path", "path", "directory", "dir"):
        val = tool_input.get(key)
        if isinstance(val, str):
            paths.append(val)
    pattern = tool_input.get("pattern")
    if isinstance(pattern, str) and "/" in pattern:
        paths.append(str(Path(pattern).parent))
    return paths


def _compute_key(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Return a stable content-hash key for ``(tool_name, tool_input)``."""

    payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _file_signature(file_paths: list[str]) -> str:
    """Return a concatenated mtime+size string for the given paths."""

    sigs: list[str] = []
    for fp in file_paths:
        try:
            st = os.stat(fp)
            sigs.append(f"{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            sigs.append("missing")
    return "|".join(sigs)


class ToolOutputCache:
    """LRU cache with TTL and mtime-based invalidation for tool outputs.

    Only tools in :data:`_CACHEABLE_TOOLS` are cached; all others pass
    through uncached. Entries are invalidated when the underlying file's
    modification time changes, or when the TTL expires.

    Attributes:
        max_entries: Maximum number of cached results (LRU eviction).
        ttl_seconds: Time-to-live for each cache entry.
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[str, float, str]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Return the composite cache key: content-hash + file-mtime signature."""

        content_hash = _compute_key(tool_name, tool_input)
        file_paths = _extract_file_paths(tool_name, tool_input)
        file_sig = _file_signature(file_paths) if file_paths else "no-files"
        return f"{content_hash}:{file_sig}"

    def get(self, tool_name: str, tool_input: dict[str, Any]) -> str | None:
        """Return cached result or ``None`` on miss/expiry."""

        if tool_name not in _CACHEABLE_TOOLS:
            return None
        key = self._make_key(tool_name, tool_input)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        result, timestamp, _ = entry
        if time.monotonic() - timestamp > self.ttl_seconds:
            self._store.pop(key, None)
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return result

    def put(self, tool_name: str, tool_input: dict[str, Any], result: str) -> None:
        """Store a result in the cache (no-op for non-cacheable tools)."""

        if tool_name not in _CACHEABLE_TOOLS:
            return
        key = self._make_key(tool_name, tool_input)
        self._store[key] = (result, time.monotonic(), tool_name)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def wrap(self, executor: Callable[[str, dict[str, Any]], str]) -> Callable[[str, dict[str, Any]], str]:
        """Return a wrapped executor that checks the cache before calling ``executor``."""

        def cached_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
            cached = self.get(tool_name, tool_input)
            if cached is not None:
                logger.debug("Tool cache HIT: %s", tool_name)
                return cached
            result = executor(tool_name, tool_input)
            self.put(tool_name, tool_input, result)
            return result

        return cached_executor

    def invalidate(self, tool_name: str | None = None, file_path: str | None = None) -> None:
        """Drop cache entries matching ``tool_name`` and/or ``file_path``.

        When both are ``None``, clears the entire cache. This should be
        called after file-writing operations to prevent stale reads.
        """

        if tool_name is None and file_path is None:
            self._store.clear()
            return
        keys_to_drop = []
        for key, (_result, _ts, cached_tool) in self._store.items():
            if tool_name is not None and cached_tool != tool_name:
                continue
            keys_to_drop.append(key)
        for k in keys_to_drop:
            self._store.pop(k, None)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> dict[str, int | float]:
        return {"hits": self._hits, "misses": self._misses, "size": self.size, "hit_rate": self.hit_rate}


__all__ = ["ToolOutputCache"]
