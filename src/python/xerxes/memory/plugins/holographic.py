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
"""Holographic fact-store provider (local).

Uses a tiny SQLite-backed store with trust scoring + temporal decay
so it works without external dependencies. ``HOLOGRAPHIC_DB_PATH``
controls the database location."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..._compat_shims import xerxes_subdir_safe
from ._base import ExternalMemoryProviderBase


class HolographicProvider(ExternalMemoryProviderBase):
    """Local fact-store provider backed by a tiny SQLite table.

    Requires no API key — uses ``sqlite3`` from the stdlib so it is
    always ``is_available``. ``HOLOGRAPHIC_DB_PATH`` overrides the
    default location."""

    name = "holographic"
    namespace_label = "holo"
    required_env = ()

    def _db_path(self) -> Path:
        """Return the SQLite database path, falling back to a per-user xerxes dir."""
        return Path(os.environ.get("HOLOGRAPHIC_DB_PATH", xerxes_subdir_safe("holographic", "facts.db")))

    def is_available(self) -> bool:
        """Always True — backed by stdlib ``sqlite3``."""
        return True

    def initialize(self) -> None:
        """Create the ``facts`` table on first use."""
        super().initialize()
        path = self._db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "content TEXT NOT NULL, "
                "trust REAL NOT NULL DEFAULT 1.0, "
                "created_at REAL NOT NULL, "
                "tags TEXT NOT NULL DEFAULT '[]')"
            )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Execute the action against the local ``facts`` SQLite table."""
        import json

        path = self._db_path()
        with sqlite3.connect(path) as conn:
            if action == "add":
                cur = conn.execute(
                    "INSERT INTO facts (content, trust, created_at, tags) VALUES (?, ?, ?, ?)",
                    (
                        arguments["content"],
                        float(arguments.get("trust", 1.0)),
                        time.time(),
                        json.dumps(arguments.get("tags", [])),
                    ),
                )
                return {"id": str(cur.lastrowid), "content": arguments["content"]}
            if action == "search":
                q = f"%{arguments['query']}%"
                cur = conn.execute(
                    "SELECT id, content, trust, created_at FROM facts WHERE content LIKE ? ORDER BY trust DESC LIMIT ?",
                    (q, int(arguments.get("limit", 20))),
                )
                return [{"id": str(r[0]), "content": r[1], "trust": r[2], "created_at": r[3]} for r in cur]
            if action == "list":
                cur = conn.execute(
                    "SELECT id, content, trust, created_at FROM facts ORDER BY id DESC LIMIT ?",
                    (int(arguments.get("limit", 20)),),
                )
                return [{"id": str(r[0]), "content": r[1], "trust": r[2], "created_at": r[3]} for r in cur]
            if action == "remove":
                conn.execute("DELETE FROM facts WHERE id = ?", (int(arguments["entry_id"]),))
                return {"removed": True}
        raise ValueError(f"unknown holographic action: {action}")


PROVIDER = HolographicProvider()

__all__ = ["PROVIDER", "HolographicProvider"]
