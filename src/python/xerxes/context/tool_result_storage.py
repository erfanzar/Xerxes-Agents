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
"""Three-level overflow store for tool outputs that exceed the inline limit.

Reference format: ``[tool-result-ref:<id>:<bytes>:<sha>]``. Storage
layers (fastest first):

1. In-memory LRU cache (default 32 entries).
2. JSON files under ``<base_dir>/<session_id>/``.
3. The placeholder embedded in the assistant transcript; the full
   payload remains fetchable via :meth:`ToolResultStorage.fetch`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

DEFAULT_INLINE_LIMIT_CHARS = 16_000  # anything larger overflows to disk
DEFAULT_LRU_SIZE = 32


def _sha1_short(s: str) -> str:
    """Return the first 12 hex chars of ``sha1(s)``."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


class ToolResultStorage:
    """LRU cache + on-disk JSON store for oversize tool outputs.

    Call :meth:`maybe_store` for every tool result; if the serialised
    content exceeds ``inline_limit`` it is written to disk and a small
    placeholder string is returned. :meth:`fetch` resolves a reference
    back to its payload (cache-first, disk fallback).
    """

    REF_PREFIX = "[tool-result-ref:"
    REF_SUFFIX = "]"

    def __init__(
        self,
        base_dir: str | Path,
        *,
        session_id: str = "default",
        inline_limit: int = DEFAULT_INLINE_LIMIT_CHARS,
        lru_size: int = DEFAULT_LRU_SIZE,
    ) -> None:
        """Open (and create) the per-session storage directory."""
        self._dir = Path(base_dir) / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._inline_limit = int(inline_limit)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lru_size = int(lru_size)
        self._lock = threading.Lock()

    @property
    def inline_limit(self) -> int:
        """Max char length kept inline before overflowing to disk."""
        return self._inline_limit

    def _ref(self, ref_id: str, byte_size: int, digest: str) -> str:
        """Format a ``[tool-result-ref:<id>:<bytes>:<sha>]`` placeholder."""
        return f"{self.REF_PREFIX}{ref_id}:{byte_size}:{digest}{self.REF_SUFFIX}"

    @classmethod
    def is_ref(cls, s: Any) -> bool:
        """Return ``True`` when ``s`` looks like a tool-result reference."""
        return isinstance(s, str) and s.startswith(cls.REF_PREFIX) and s.endswith(cls.REF_SUFFIX)

    @classmethod
    def parse_ref(cls, s: str) -> str | None:
        """Return the ``ref_id`` portion of a reference, or ``None``."""
        if not cls.is_ref(s):
            return None
        body = s[len(cls.REF_PREFIX) : -len(cls.REF_SUFFIX)]
        return body.split(":", 1)[0]

    def maybe_store(self, tool_name: str, content: Any) -> Any:
        """Return ``content`` unchanged or a reference placeholder.

        Non-string content is JSON-serialised for the size check. The
        return value is the original content when it fits inline, or a
        ``[tool-result-ref:...]`` placeholder when it doesn't.
        """

        if isinstance(content, str):
            payload = content
        else:
            try:
                payload = json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                payload = str(content)
        if len(payload) <= self._inline_limit:
            return content
        digest = _sha1_short(payload)
        ref_id = f"{tool_name}_{digest}"
        path = self._dir / f"{ref_id}.json"
        # Don't bother rewriting if the digest matches a prior payload.
        if not path.exists():
            path.write_text(payload, encoding="utf-8")
        with self._lock:
            self._cache[ref_id] = content
            self._cache.move_to_end(ref_id)
            while len(self._cache) > self._lru_size:
                self._cache.popitem(last=False)
        return self._ref(ref_id, len(payload), digest)

    def fetch(self, ref_or_id: str) -> Any | None:
        """Resolve a reference or raw id back to the stored payload."""

        if not isinstance(ref_or_id, str):
            return None
        ref_id = self.parse_ref(ref_or_id) or ref_or_id
        with self._lock:
            if ref_id in self._cache:
                self._cache.move_to_end(ref_id)
                return self._cache[ref_id]
        path = self._dir / f"{ref_id}.json"
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        with self._lock:
            self._cache[ref_id] = value
            self._cache.move_to_end(ref_id)
            while len(self._cache) > self._lru_size:
                self._cache.popitem(last=False)
        return value

    def list_refs(self) -> list[str]:
        """Return all stored reference ids for this session, sorted."""
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.iterdir() if p.suffix == ".json")

    def prune(self, keep: int = 100) -> int:
        """Delete all but the ``keep`` most recently modified entries.

        Returns the number of files actually removed.
        """

        files = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for p in files[keep:]:
            try:
                os.unlink(p)
                removed += 1
            except FileNotFoundError:
                pass
        return removed


__all__ = ["DEFAULT_INLINE_LIMIT_CHARS", "ToolResultStorage"]
