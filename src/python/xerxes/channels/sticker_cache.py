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
"""Per-platform sticker / emoji cache.

Telegram, Discord, and Slack all expose stickers or custom emoji through
their APIs. Downloading them on every send is wasteful and rate-limited,
so this module keeps a bounded LRU of ``(platform, sticker_id) → local
file path`` plus a JSON sidecar on disk so the cache survives restarts.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StickerRecord:
    """One cached sticker file.

    Attributes:
        platform: Channel name the sticker belongs to.
        sticker_id: Platform-specific sticker / emoji identifier.
        local_path: Filesystem path of the downloaded file.
        fetched_at: Unix timestamp when the file was downloaded.
    """

    platform: str
    sticker_id: str
    local_path: str
    fetched_at: float


class StickerCache:
    """Thread-safe LRU sticker cache with a JSON on-disk index.

    Concurrent ``get``/``put`` calls are safe; the disk index is rewritten on
    every ``put`` so a process crash never loses more than the in-flight
    operation. The LRU evicts the oldest entries when ``lru_size`` is
    exceeded; the corresponding files on disk are *not* deleted (operators
    can sweep them out of band if needed).
    """

    def __init__(self, base_dir: Path, *, lru_size: int = 256) -> None:
        """Build the cache rooted at ``base_dir``.

        Creates ``base_dir`` if missing and loads any existing JSON index so
        previously cached stickers are immediately available.

        Args:
            base_dir: Directory holding sticker files and the ``_index.json``
                sidecar.
            lru_size: Maximum number of records kept in memory; older
                entries are evicted on overflow.
        """
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base / "_index.json"
        self._lru: OrderedDict[tuple[str, str], StickerRecord] = OrderedDict()
        self._lru_size = lru_size
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self) -> None:
        """Hydrate ``self._lru`` from ``_index.json``; tolerate corruption."""
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for entry in data:
            try:
                rec = StickerRecord(**entry)
            except TypeError:
                continue
            self._lru[(rec.platform, rec.sticker_id)] = rec

    def _save_index(self) -> None:
        """Atomically rewrite ``_index.json`` from the current LRU snapshot."""
        data = [
            {"platform": r.platform, "sticker_id": r.sticker_id, "local_path": r.local_path, "fetched_at": r.fetched_at}
            for r in self._lru.values()
        ]
        self._index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---------------------------- public surface

    def get(self, platform: str, sticker_id: str) -> StickerRecord | None:
        """Look up a cached sticker and mark it as most recently used.

        Args:
            platform: Channel name.
            sticker_id: Platform-specific sticker id.

        Returns:
            The cached record if present, otherwise ``None``.
        """
        with self._lock:
            rec = self._lru.get((platform, sticker_id))
            if rec is not None:
                self._lru.move_to_end((platform, sticker_id))
            return rec

    def put(self, platform: str, sticker_id: str, local_path: Path) -> StickerRecord:
        """Insert or replace a cache entry and persist the index.

        Evicts the oldest entries when the cache exceeds ``lru_size``.

        Args:
            platform: Channel name.
            sticker_id: Platform-specific sticker id.
            local_path: Filesystem path of the downloaded sticker file.

        Returns:
            The newly created record.
        """
        rec = StickerRecord(
            platform=platform,
            sticker_id=sticker_id,
            local_path=str(local_path),
            fetched_at=time.time(),
        )
        with self._lock:
            self._lru[(platform, sticker_id)] = rec
            self._lru.move_to_end((platform, sticker_id))
            while len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
            self._save_index()
        return rec

    def size(self) -> int:
        """Return the number of records currently in the cache."""
        with self._lock:
            return len(self._lru)

    def clear(self) -> None:
        """Drop every entry from memory and the on-disk index.

        Does not delete the underlying sticker files.
        """
        with self._lock:
            self._lru.clear()
            self._save_index()


__all__ = ["StickerCache", "StickerRecord"]
