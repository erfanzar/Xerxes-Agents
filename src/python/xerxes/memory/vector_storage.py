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
"""Vector storage module for Xerxes.

Exports:
    - logger
    - SQLiteVectorStorage"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import typing as tp
from pathlib import Path

from .embedders import Embedder, cosine_similarity, get_default_embedder
from .storage import MemoryStorage

logger = logging.getLogger(__name__)


class SQLiteVectorStorage(MemoryStorage):
    """Sqlite vector storage.

    Inherits from: MemoryStorage
    """

    def __init__(
        self,
        db_path: str = ".xerxes_memory/vectors.db",
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            db_path (str, optional): IN: db path. Defaults to '.xerxes_memory/vectors.db'. OUT: Consumed during execution.
            embedder (Embedder | None, optional): IN: embedder. Defaults to None. OUT: Consumed during execution."""

        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_default_embedder()
        self._init_db()

    def _init_db(self) -> None:
        """Internal helper to init db.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at)")
            conn.commit()

    def save(self, key: str, data: tp.Any) -> bool:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (tp.Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        try:
            payload = pickle.dumps(data)
        except Exception:
            logger.warning("Failed to pickle data for key=%s", key)
            return False
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = json.dumps(data, default=str, sort_keys=True)
        else:
            text = ""
        try:
            vec = self.embedder.embed(text) if text else [0.0] * max(1, self.embedder.dim or 1)
        except Exception:
            logger.warning("Embedder failed for key=%s; storing zero vector", key, exc_info=True)
            vec = [0.0] * max(1, self.embedder.dim or 1)
        emb_blob = json.dumps(vec)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO vectors (key, data, embedding) VALUES (?, ?, ?)",
                    (key, payload, emb_blob),
                )
                conn.commit()
            return True
        except Exception:
            logger.warning("SQLiteVectorStorage save failed for key=%s", key, exc_info=True)
            return False

    def load(self, key: str) -> tp.Any | None:
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            tp.Any | None: OUT: Result of the operation."""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT data FROM vectors WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return pickle.loads(row[0])
        except Exception:
            logger.warning("Failed to unpickle data for key=%s", key)
            return None

    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def exists(self, key: str) -> bool:
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM vectors WHERE key = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        with sqlite3.connect(self.db_path) as conn:
            if pattern:
                cur = conn.execute(
                    "SELECT key FROM vectors WHERE key LIKE ? ORDER BY created_at DESC",
                    (f"%{pattern}%",),
                )
            else:
                cur = conn.execute("SELECT key FROM vectors ORDER BY created_at DESC")
            return [r[0] for r in cur.fetchall()]

    def clear(self) -> int:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        with sqlite3.connect(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            conn.execute("DELETE FROM vectors")
            conn.commit()
            return int(n)

    def supports_semantic_search(self) -> bool:
        """Supports semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        return True

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, tp.Any]]:
        """Semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            threshold (float, optional): IN: threshold. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            list[tuple[str, float, tp.Any]]: OUT: Result of the operation."""

        if not query:
            return []
        try:
            qvec = self.embedder.embed(query)
        except Exception:
            logger.warning("Embedder failed for query; returning []", exc_info=True)
            return []
        results: list[tuple[str, float, tp.Any]] = []
        with sqlite3.connect(self.db_path) as conn:
            for key, data_blob, emb_blob in conn.execute("SELECT key, data, embedding FROM vectors"):
                try:
                    vec = json.loads(emb_blob)
                except Exception:
                    continue
                sim = cosine_similarity(qvec, vec)
                if sim < threshold:
                    continue
                try:
                    data = pickle.loads(data_blob)
                except Exception:
                    continue
                results.append((key, sim, data))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
