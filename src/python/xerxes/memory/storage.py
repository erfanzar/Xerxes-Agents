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
"""Pluggable key/value backends used by the memory tiers.

Defines the ``MemoryStorage`` ABC and four concrete implementations:
``SimpleStorage`` (in-process dict), ``FileStorage`` (pickle files
under a directory), ``SQLiteStorage`` (SQLite with a ``WRITE_MEMORY``
env guard), and ``RAGStorage`` (decorator that adds embedding-based
semantic search on top of any other backend)."""

import hashlib
import json
import logging
import pickle
import sqlite3
import typing as tp
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStorage(ABC):
    """Abstract key/value backend for memory persistence.

    All concrete tiers (``ShortTermMemory``, ``LongTermMemory``,
    ``EntityMemory``, ``UserProfileStore``) interact only through this
    interface, so the storage layer can be swapped without touching
    memory logic. Semantic search is opt-in via
    ``supports_semantic_search``/``semantic_search``."""

    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """Persist ``data`` under ``key``; return True on success."""

        pass

    @abstractmethod
    def load(self, key: str) -> Any | None:
        """Return the data stored at ``key``, or ``None`` if absent."""

        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete the row at ``key``; return True if a row was removed."""

        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True when ``key`` has stored data."""

        pass

    @abstractmethod
    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return every key, optionally filtered by substring ``pattern``."""

        pass

    @abstractmethod
    def clear(self) -> int:
        """Wipe every row; return the number of rows removed."""

        pass

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, Any]]:
        """Return ``(key, similarity, data)`` triples; default is a no-op stub."""

        return []

    def supports_semantic_search(self) -> bool:
        """Return True when ``semantic_search`` does meaningful work (default False)."""

        return False


class SimpleStorage(MemoryStorage):
    """Ephemeral in-process backend backed by a plain dict.

    Used in tests and as the default backend for ``RAGStorage`` when no
    durable store is supplied."""

    def __init__(self):
        """Initialise the in-memory dict."""

        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> bool:
        """Store ``data`` under ``key`` in the dict."""

        self._data[key] = data
        return True

    def load(self, key: str) -> Any | None:
        """Return the dict entry for ``key`` or ``None``."""

        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """Pop ``key`` from the dict, returning True iff it existed."""

        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Return True when ``key`` is a member of the dict."""

        return key in self._data

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return dict keys, optionally filtered by substring."""

        keys = list(self._data.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Empty the dict and return how many rows were dropped."""

        count = len(self._data)
        self._data.clear()
        return count


class FileStorage(MemoryStorage):
    """Disk-backed pickle store with an MD5-hashed filename index.

    Each row is pickled to ``<storage_dir>/<md5>.pkl`` and the
    key-to-filename map is kept in ``_index.json`` for fast lookup
    and listing."""

    def __init__(self, storage_dir: str = ".xerxes_memory") -> None:
        """Ensure ``storage_dir`` exists and load the on-disk key index."""

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Load the JSON key->filename index from disk, or return empty."""

        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Write the in-memory key->filename index back to disk."""

        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _get_file_path(self, key: str) -> Path:
        """Return the deterministic pickle path for ``key`` (MD5-named)."""

        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.storage_dir / f"{key_hash}.pkl"

    def save(self, key: str, data: Any) -> bool:
        """Pickle ``data`` to disk and update the index; swallow IO errors."""

        try:
            file_path = self._get_file_path(key)
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            self._index[key] = str(file_path.name)
            self._save_index()
            return True
        except Exception:
            return False

    def load(self, key: str) -> Any | None:
        """Unpickle the row stored at ``key`` or return ``None``."""

        if key not in self._index:
            return None
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def delete(self, key: str) -> bool:
        """Unlink the pickle file for ``key`` and remove it from the index."""

        if key not in self._index:
            return False
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            file_path.unlink()
        del self._index[key]
        self._save_index()
        return True

    def exists(self, key: str) -> bool:
        """Return True when ``key`` is present in the index."""

        return key in self._index

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return index keys, optionally filtered by substring."""

        keys = list(self._index.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Delete every row tracked by the index and return the count."""

        count = 0
        for key in list(self._index.keys()):
            if self.delete(key):
                count += 1
        return count


class SQLiteStorage(MemoryStorage):
    """SQLite-backed store gated by the ``WRITE_MEMORY`` env flag.

    When ``WRITE_MEMORY=1`` is set the store writes pickled blobs to a
    SQLite file at ``db_path``; otherwise it transparently degrades to
    an in-process dict so the rest of the memory subsystem keeps
    functioning during read-only sessions and tests."""

    def __init__(self, db_path: str = ".xerxes_memory/memory.db") -> None:
        """Open (or fake) the database depending on ``WRITE_MEMORY``."""

        import os

        self.write_enabled = os.environ.get("WRITE_MEMORY", "0") == "1"

        self.db_path = Path(db_path)
        if self.write_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self._memory_storage: dict[str, Any] = {}

    def _init_db(self) -> None:
        """Create the ``memory`` table and ``created_at`` index if missing."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON memory(created_at)
            """
            )
            conn.commit()

    def save(self, key: str, data: Any) -> bool:
        """Insert-or-replace the row; route to the dict fallback when not write-enabled."""

        if not self.write_enabled:
            self._memory_storage[key] = data
            return True

        try:
            serialized = pickle.dumps(data)
            now = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory (key, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (key, serialized, now, now),
                )
                conn.commit()
            return True
        except Exception:
            return False

    def load(self, key: str) -> Any | None:
        """Unpickle the row at ``key``; return ``None`` if absent."""

        if not self.write_enabled:
            return self._memory_storage.get(key)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
        return None

    def delete(self, key: str) -> bool:
        """Delete the row; return True iff one was removed."""

        if not self.write_enabled:
            if key in self._memory_storage:
                del self._memory_storage[key]
                return True
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Return True when a row exists for ``key``."""

        if not self.write_enabled:
            return key in self._memory_storage

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM memory WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return keys (newest first), optionally filtered by substring."""

        if not self.write_enabled:
            keys = list(self._memory_storage.keys())
            if pattern:
                keys = [k for k in keys if pattern in k]
            return keys

        with sqlite3.connect(self.db_path) as conn:
            if pattern:
                cursor = conn.execute(
                    "SELECT key FROM memory WHERE key LIKE ? ORDER BY created_at DESC", (f"%{pattern}%",)
                )
            else:
                cursor = conn.execute("SELECT key FROM memory ORDER BY created_at DESC")
            return [row[0] for row in cursor.fetchall()]

    def clear(self) -> int:
        """Truncate the table and return the row count that was removed."""

        if not self.write_enabled:
            count = len(self._memory_storage)
            self._memory_storage.clear()
            return count

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM memory")
            conn.commit()
            return count


class RAGStorage(MemoryStorage):
    """Decorator backend that augments any ``MemoryStorage`` with embeddings + similarity search.

    On every write, an embedding of the saved text/dict is computed
    and persisted alongside the data (under the ``EMBEDDING_KEY_PREFIX``
    namespace) so that semantic search survives across processes.
    Falls back through sentence-transformers → OpenAI → a hashed TF-IDF
    approximation depending on what is available."""

    EMBEDDING_KEY_PREFIX = "_emb_"

    def __init__(
        self,
        backend: MemoryStorage | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedder: tp.Any = None,
    ) -> None:
        """Wrap ``backend`` and pick an embedding strategy.

        Args:
            backend: Underlying key/value store; defaults to
                ``SimpleStorage`` when ``None``.
            embedding_model: Identifier hint (``"tfidf"``,
                ``"text-embedding-..."``, or a sentence-transformers
                model name).
            embedding_api_key: Optional API key override for OpenAI.
            embedder: Pre-built embedder object (takes precedence)."""

        self.backend = backend or SimpleStorage()
        self.embeddings: dict[str, list[float]] = {}
        self._embedding_model_name = embedding_model
        self._embedding_api_key = embedding_api_key
        self._embedder = embedder
        self._tfidf_vectorizer = None
        self._tfidf_corpus: list[str] = []
        self._tfidf_keys: list[str] = []
        self._embedding_type = self._resolve_embedding_type(embedding_model)
        self._restore_embeddings()

    def _resolve_embedding_type(self, model: str | None) -> str:
        """Pick the active embedding type (tfidf / openai / sentence_transformers)."""

        if model == "tfidf":
            return "tfidf"

        if model and model.startswith("text-embedding"):
            return "openai"

        if model:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(model)
                return "sentence_transformers"
            except ImportError:
                pass

        if model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                return "sentence_transformers"
            except ImportError:
                pass

        return "tfidf"

    def _compute_embedding(self, text: str) -> list[float]:
        """Dispatch to the configured embedding strategy with graceful fallback to TF-IDF."""

        if self._embedder is not None and hasattr(self._embedder, "embed"):
            try:
                return list(self._embedder.embed(text))
            except Exception:
                logger.warning("Embedder.embed failed; falling back to TF-IDF", exc_info=True)
                return self._compute_tfidf_embedding(text)

        if self._embedding_type == "sentence_transformers":
            return self._compute_sentence_transformer_embedding(text)
        elif self._embedding_type == "openai":
            return self._compute_openai_embedding(text)
        else:
            return self._compute_tfidf_embedding(text)

    def _compute_tfidf_embedding(self, text: str) -> list[float]:
        """Cheap hashing-based TF-IDF stand-in (256-d L2-normalised)."""

        words = text.lower().split()
        if not words:
            return [0.0] * 128

        word_freq: dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        total = len(words)
        dim = 256
        vec = [0.0] * dim
        for word, count in word_freq.items():
            tf = count / total
            idx = hash(word) % dim
            vec[idx] += tf

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _compute_sentence_transformer_embedding(self, text: str) -> list[float]:
        """Load (lazily) and call a sentence-transformers model."""

        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._embedding_model_name or "all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_name)

        embedding = self._embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _compute_openai_embedding(self, text: str) -> list[float]:
        """Call the OpenAI embeddings API; fall back to TF-IDF on any failure."""

        import os

        try:
            from openai import OpenAI
        except ImportError:
            return self._compute_tfidf_embedding(text)

        api_key = self._embedding_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._compute_tfidf_embedding(text)

        try:
            client = OpenAI(api_key=api_key)
            model = self._embedding_model_name or "text-embedding-3-small"
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception:
            return self._compute_tfidf_embedding(text)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity; returns 0 when either vector has zero norm."""

        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def save(self, key: str, data: Any) -> bool:
        """Persist ``data`` and an embedding alongside it under a parallel key.

        Embedding keys (those starting with ``EMBEDDING_KEY_PREFIX``) are
        passed straight through without further embedding."""

        if key.startswith(self.EMBEDDING_KEY_PREFIX):
            return self.backend.save(key, data)

        success = self.backend.save(key, data)
        if success and isinstance(data, str | dict):
            text = str(data) if not isinstance(data, str) else data
            try:
                vec = self._compute_embedding(text)
                self.embeddings[key] = vec
                try:
                    self.backend.save(self.EMBEDDING_KEY_PREFIX + key, vec)
                except Exception:
                    logger.debug("Embedding persistence failed for %s", key, exc_info=True)
            except Exception:
                logger.warning("Failed to compute embedding for key=%s", key, exc_info=True)
        return success

    def load(self, key: str) -> Any | None:
        """Delegate to the wrapped backend's ``load``."""

        return self.backend.load(key)

    def delete(self, key: str) -> bool:
        """Delete the data row and its sibling embedding row."""

        self.embeddings.pop(key, None)
        try:
            self.backend.delete(self.EMBEDDING_KEY_PREFIX + key)
        except Exception:
            pass
        return self.backend.delete(key)

    def _restore_embeddings(self) -> None:
        """Reload embeddings persisted under the ``EMBEDDING_KEY_PREFIX`` namespace."""

        if self.backend is None:
            return
        try:
            keys = self.backend.list_keys(self.EMBEDDING_KEY_PREFIX)
        except Exception:
            logger.debug("Backend list_keys for embeddings failed", exc_info=True)
            return
        prefix_len = len(self.EMBEDDING_KEY_PREFIX)
        restored = 0
        for full_key in keys:
            if not full_key.startswith(self.EMBEDDING_KEY_PREFIX):
                continue
            data_key = full_key[prefix_len:]
            try:
                vec = self.backend.load(full_key)
            except Exception:
                continue
            if isinstance(vec, list) and vec and isinstance(vec[0], int | float):
                self.embeddings[data_key] = [float(v) for v in vec]
                restored += 1
        if restored:
            logger.debug("Restored %d embeddings from backend", restored)

    def exists(self, key: str) -> bool:
        """Delegate to the wrapped backend's ``exists``."""

        return self.backend.exists(key)

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """Return data keys only, hiding the internal embedding-prefixed rows."""

        all_keys = self.backend.list_keys(pattern)
        return [k for k in all_keys if not k.startswith(self.EMBEDDING_KEY_PREFIX)]

    def clear(self) -> int:
        """Drop every embedding and forward ``clear`` to the wrapped backend."""

        self.embeddings.clear()
        try:
            for k in list(self.backend.list_keys(self.EMBEDDING_KEY_PREFIX)):
                self.backend.delete(k)
        except Exception:
            pass
        return self.backend.clear()

    def supports_semantic_search(self) -> bool:
        """``RAGStorage`` always supports semantic search."""

        return True

    def semantic_search(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """Alias for ``search_similar`` to satisfy the ``MemoryStorage`` protocol."""

        return self.search_similar(query, limit=limit, threshold=threshold)

    def search_similar(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """Cosine-similarity scan of the embedding cache against ``query``.

        Returns ``(key, similarity, data)`` triples sorted highest-first
        and filtered by ``threshold``."""

        query_embedding = self._compute_embedding(query)
        results = []

        for key, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                data = self.backend.load(key)
                if data:
                    results.append((key, similarity, data))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
