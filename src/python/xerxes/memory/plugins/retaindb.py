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
"""RetainDB cloud memory provider (with crash-safe write-behind queue).

Uses a local SQLite queue to durably stage memory writes so that
they're not lost if the process crashes before flush. A background
flush task would ship them to retaindb.com; the test/stub mode just
keeps them in the queue."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from .._compat_shims import xerxes_subdir_safe
from ._base import ExternalMemoryProviderBase


class RetainDBProvider(ExternalMemoryProviderBase):
    """RetainDB provider with a crash-safe SQLite write-behind queue.

    Memory writes are durably staged in a local SQLite ``pending`` table
    before they would be shipped to retaindb.com; the stub mode used in
    tests keeps them in the queue. Requires ``RETAINDB_API_KEY``."""

    name = "retaindb"
    namespace_label = "retain"
    required_env = ("RETAINDB_API_KEY",)

    def _queue_path(self) -> Path:
        """Return the SQLite queue path under the per-user xerxes dir."""
        return xerxes_subdir_safe("retaindb", "queue.sqlite")

    def initialize(self) -> None:
        """Create the ``pending`` and ``facts`` tables on first use."""
        super().initialize()
        path = self._queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pending (id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT, payload TEXT, queued_at REAL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, created_at REAL)"
            )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Persist or query facts locally; ``add`` also enqueues a pending upload."""
        import json

        path = self._queue_path()
        with sqlite3.connect(path) as conn:
            if action == "add":
                cur = conn.execute(
                    "INSERT INTO facts (content, created_at) VALUES (?, ?)",
                    (arguments["content"], time.time()),
                )
                # Also enqueue the upload.
                conn.execute(
                    "INSERT INTO pending (action, payload, queued_at) VALUES (?, ?, ?)",
                    ("add", json.dumps(arguments), time.time()),
                )
                return {"id": str(cur.lastrowid), "queued": True}
            if action == "search":
                q = f"%{arguments['query']}%"
                cur = conn.execute(
                    "SELECT id, content FROM facts WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (q, int(arguments.get("limit", 20))),
                )
                return [{"id": str(r[0]), "content": r[1]} for r in cur]
            if action == "list":
                cur = conn.execute(
                    "SELECT id, content FROM facts ORDER BY id DESC LIMIT ?",
                    (int(arguments.get("limit", 20)),),
                )
                return [{"id": str(r[0]), "content": r[1]} for r in cur]
            if action == "remove":
                conn.execute("DELETE FROM facts WHERE id = ?", (int(arguments["entry_id"]),))
                return {"removed": True}
        raise ValueError(f"unknown retaindb action: {action}")


PROVIDER = RetainDBProvider()

__all__ = ["PROVIDER", "RetainDBProvider"]
