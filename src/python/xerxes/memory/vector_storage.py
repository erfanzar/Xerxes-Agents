# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""SQLite-backed vector storage for semantic memory.

A dependency-free vector storage backend that stores both the data
payload and its embedding in a single SQLite database. Cosine similarity
is computed in Python at query time (linear scan).

This is the **default** vector backend for Xerxes — it has no external
dependencies, persists across restarts, and scales to ~100k vectors with
acceptable latency. For larger collections, swap in Chroma/Qdrant via
the same :class:`~xerxes.memory.storage.MemoryStorage` protocol.
"""

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
    """SQLite-backed vector storage with built-in cosine search.

    Stores each entry as a row in a single ``vectors`` table:
    ``(key TEXT PRIMARY KEY, data BLOB, embedding BLOB)``. Pickled data
    plus a JSON-encoded vector keeps the schema self-describing and
    portable. Semantic search runs an in-Python cosine scan over all
    rows — fine up to tens of thousands of entries.

    For datasets larger than ~100k entries, prefer a dedicated vector
    DB (Chroma, Qdrant) plugged in via the :class:`MemoryStorage`
    protocol.

    Attributes:
        db_path: Path to the SQLite database file.
        embedder: The :class:`Embedder` used to compute vectors.
    """

    def __init__(
        self,
        db_path: str = ".xerxes_memory/vectors.db",
        embedder: Embedder | None = None,
    ) -> None:
        """Initialise the vector store.

        Args:
            db_path: Path to the SQLite database. Parent directories are
                created automatically.
            embedder: Embedder used for ``save`` and ``semantic_search``.
                Defaults to :func:`get_default_embedder`.
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_default_embedder()
        self._init_db()

    def _init_db(self) -> None:
        """Create the ``vectors`` table on first use."""
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
        """Embed (when text-like) and store the entry.

        For ``str`` and ``dict`` payloads, an embedding is computed via
        the configured :attr:`embedder`. For other types, a zero vector
        is stored (the entry is still retrievable by key but won't appear
        in semantic search).

        Args:
            key: Unique entry identifier.
            data: Anything pickleable; strings/dicts are also embedded.

        Returns:
            ``True`` on success.
        """
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
        """Retrieve the data payload for ``key``."""
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
        """Remove an entry by key. Returns ``True`` if a row was removed."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def exists(self, key: str) -> bool:
        """Return ``True`` when a row with *key* is present in the store."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM vectors WHERE key = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys (most recent first), optionally filtered by a LIKE substring."""
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
        """Delete every row from the vectors table and return how many were removed."""
        with sqlite3.connect(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            conn.execute("DELETE FROM vectors")
            conn.commit()
            return int(n)

    def supports_semantic_search(self) -> bool:
        """Capability flag: this backend provides cosine-based semantic search."""
        return True

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, tp.Any]]:
        """Cosine-similarity scan over all stored vectors.

        Linear in the number of stored entries; expect millisecond
        latency up to a few thousand rows and second-scale latency at
        ~100k. Above that, prefer a dedicated ANN index.

        Args:
            query: Free-text query (encoded with the same embedder).
            limit: Max number of results to return.
            threshold: Minimum cosine similarity to include.

        Returns:
            ``(key, similarity, data)`` tuples sorted by descending
            similarity.
        """
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
