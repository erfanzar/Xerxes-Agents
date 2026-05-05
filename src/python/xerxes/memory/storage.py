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
"""Storage module for Xerxes.

Exports:
    - logger
    - MemoryStorage
    - SimpleStorage
    - FileStorage
    - SQLiteStorage
    - RAGStorage"""

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
    """Memory storage.

    Inherits from: ABC
    """

    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        pass

    @abstractmethod
    def load(self, key: str) -> Any | None:
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Any | None: OUT: Result of the operation."""

        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        pass

    @abstractmethod
    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        pass

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float, Any]]:
        """Semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            threshold (float, optional): IN: threshold. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            list[tuple[str, float, Any]]: OUT: Result of the operation."""

        return []

    def supports_semantic_search(self) -> bool:
        """Supports semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        return False


class SimpleStorage(MemoryStorage):
    """Simple storage.

    Inherits from: MemoryStorage
    """

    def __init__(self):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> bool:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        self._data[key] = data
        return True

    def load(self, key: str) -> Any | None:
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Any | None: OUT: Result of the operation."""

        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        return key in self._data

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        keys = list(self._data.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        count = len(self._data)
        self._data.clear()
        return count


class FileStorage(MemoryStorage):
    """File storage.

    Inherits from: MemoryStorage
    """

    def __init__(self, storage_dir: str = ".xerxes_memory") -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            storage_dir (str, optional): IN: storage dir. Defaults to '.xerxes_memory'. OUT: Consumed during execution."""

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Internal helper to load index.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Internal helper to save index.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _get_file_path(self, key: str) -> Path:
        """Internal helper to get file path.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Path: OUT: Result of the operation."""

        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.storage_dir / f"{key_hash}.pkl"

    def save(self, key: str, data: Any) -> bool:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Any | None: OUT: Result of the operation."""

        if key not in self._index:
            return None
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if key not in self._index:
            return False
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            file_path.unlink()
        del self._index[key]
        self._save_index()
        return True

    def exists(self, key: str) -> bool:
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        return key in self._index

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        keys = list(self._index.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        count = 0
        for key in list(self._index.keys()):
            if self.delete(key):
                count += 1
        return count


class SQLiteStorage(MemoryStorage):
    """Sqlite storage.

    Inherits from: MemoryStorage
    """

    def __init__(self, db_path: str = ".xerxes_memory/memory.db") -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            db_path (str, optional): IN: db path. Defaults to '.xerxes_memory/memory.db'. OUT: Consumed during execution."""

        import os

        self.write_enabled = os.environ.get("WRITE_MEMORY", "0") == "1"

        self.db_path = Path(db_path)
        if self.write_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self._memory_storage: dict[str, Any] = {}

    def _init_db(self) -> None:
        """Internal helper to init db.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Any | None: OUT: Result of the operation."""

        if not self.write_enabled:
            return self._memory_storage.get(key)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
        return None

    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if not self.write_enabled:
            return key in self._memory_storage

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM memory WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

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
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

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
    """Ragstorage.

    Inherits from: MemoryStorage
    """

    EMBEDDING_KEY_PREFIX = "_emb_"

    def __init__(
        self,
        backend: MemoryStorage | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedder: tp.Any = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            backend (MemoryStorage | None, optional): IN: backend. Defaults to None. OUT: Consumed during execution.
            embedding_model (str | None, optional): IN: embedding model. Defaults to None. OUT: Consumed during execution.
            embedding_api_key (str | None, optional): IN: embedding api key. Defaults to None. OUT: Consumed during execution.
            embedder (tp.Any, optional): IN: embedder. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to resolve embedding type.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model (str | None): IN: model. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Internal helper to compute embedding.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            list[float]: OUT: Result of the operation."""

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
        """Internal helper to compute tfidf embedding.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            list[float]: OUT: Result of the operation."""

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
        """Internal helper to compute sentence transformer embedding.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            list[float]: OUT: Result of the operation."""

        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._embedding_model_name or "all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_name)

        embedding = self._embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _compute_openai_embedding(self, text: str) -> list[float]:
        """Internal helper to compute openai embedding.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            list[float]: OUT: Result of the operation."""

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
        """Internal helper to cosine similarity.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            vec1 (list[float]): IN: vec1. OUT: Consumed during execution.
            vec2 (list[float]): IN: vec2. OUT: Consumed during execution.
        Returns:
            float: OUT: Result of the operation."""

        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def save(self, key: str, data: Any) -> bool:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
            data (Any): IN: data. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Load.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            Any | None: OUT: Result of the operation."""

        return self.backend.load(key)

    def delete(self, key: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        self.embeddings.pop(key, None)
        try:
            self.backend.delete(self.EMBEDDING_KEY_PREFIX + key)
        except Exception:
            pass
        return self.backend.delete(key)

    def _restore_embeddings(self) -> None:
        """Internal helper to restore embeddings.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Exists.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            key (str): IN: key. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        return self.backend.exists(key)

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            pattern (str | None, optional): IN: pattern. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        all_keys = self.backend.list_keys(pattern)
        return [k for k in all_keys if not k.startswith(self.EMBEDDING_KEY_PREFIX)]

    def clear(self) -> int:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        self.embeddings.clear()
        try:
            for k in list(self.backend.list_keys(self.EMBEDDING_KEY_PREFIX)):
                self.backend.delete(k)
        except Exception:
            pass
        return self.backend.clear()

    def supports_semantic_search(self) -> bool:
        """Supports semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        return True

    def semantic_search(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """Semantic search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            threshold (float, optional): IN: threshold. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            list[tuple[str, float, Any]]: OUT: Result of the operation."""

        return self.search_similar(query, limit=limit, threshold=threshold)

    def search_similar(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """Search for similar.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            threshold (float, optional): IN: threshold. Defaults to 0.0. OUT: Consumed during execution.
        Returns:
            list[tuple[str, float, Any]]: OUT: Result of the operation."""

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
