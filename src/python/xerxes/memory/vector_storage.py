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
"""Single-file SQLite store that pairs each row with a dense embedding.

``SQLiteVectorStorage`` is a self-contained alternative to
``RAGStorage`` for use cases where the vectors live in the same
database as the data and a brute-force cosine scan is fast enough
(thousands of rows). Embeddings are stored as JSON text for portability."""

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
    """SQLite backend with built-in embeddings and cosine-similarity search."""

    def __init__(
        self,
        db_path: str = ".xerxes_memory/vectors.db",
        embedder: Embedder | None = None,
    ) -> None:
        """Ensure the parent directory exists and initialise the ``vectors`` table.

        Args:
            db_path: SQLite file path; ``~`` is expanded.
            embedder: Embedder used to vectorise saved text/dicts;
                defaults to the process-wide ``get_default_embedder``."""

        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_default_embedder()
        self._init_db()

    def _init_db(self) -> None:
        """Create the ``vectors`` table and its ``created_at`` index if absent."""

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
        """Persist ``data`` plus an embedding of its textual form.

        Strings are embedded directly; dicts are JSON-serialised first.
        For other types a zero vector is stored so the row still exists
        but does not match similarity queries."""

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
        """Unpickle the row at ``key`` or return ``None`` if missing/corrupt."""

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
        """Remove the row at ``key`` and return True iff one was removed."""

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def exists(self, key: str) -> bool:
        """Return True when a row exists for ``key``."""

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM vectors WHERE key = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return keys (newest first), optionally filtered by ``LIKE %pattern%``."""

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
        """Truncate the table and return the number of rows that were removed."""

        with sqlite3.connect(self.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            conn.execute("DELETE FROM vectors")
            conn.commit()
            return int(n)

    def supports_semantic_search(self) -> bool:
        """``SQLiteVectorStorage`` always supports semantic search."""

        return True

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, tp.Any]]:
        """Brute-force cosine scan over every stored embedding.

        Returns ``(key, similarity, data)`` triples sorted by descending
        similarity, dropping anything below ``threshold`` and capping
        the result at ``limit``."""

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
