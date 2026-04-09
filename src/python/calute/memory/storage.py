# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Storage backends for Calute memory system."""

import hashlib
import json
import pickle
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends.

    Defines the common key-value interface that all Calute storage
    implementations must provide. Concrete subclasses include
    :class:`SimpleStorage` (in-memory), :class:`FileStorage`
    (pickle files), :class:`SQLiteStorage` (database), and
    :class:`RAGStorage` (vector-enhanced wrapper).

    Subclasses must implement :meth:`save`, :meth:`load`, :meth:`delete`,
    :meth:`exists`, :meth:`list_keys`, and :meth:`clear`.
    """

    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """
        Save data with a key.

        Args:
            key: Unique identifier for the data
            data: Data to store

        Returns:
            True if save was successful, False otherwise
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Any | None:
        """
        Load data by key.

        Args:
            key: Unique identifier to retrieve

        Returns:
            Stored data if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data by key.

        Args:
            key: Unique identifier to delete

        Returns:
            True if deletion was successful, False if key not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def list_keys(self, pattern: str | None = None) -> list[str]:
        """
        List all stored keys, optionally filtered by pattern.

        Args:
            pattern: Optional substring pattern to filter keys

        Returns:
            List of matching key strings
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all stored data.

        Returns:
            Number of items cleared
        """
        pass


class SimpleStorage(MemoryStorage):
    """Simple in-memory storage (non-persistent).

    Provides fast dictionary-backed key-value storage that exists only in
    memory. All data is lost when the process terminates. Suitable for
    testing, development, and short-lived applications.

    Attributes:
        _data: Internal dictionary holding all stored key-value pairs.
    """

    def __init__(self):
        """Initialize empty in-memory storage."""
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> bool:
        """
        Save data in memory.

        Args:
            key: Unique identifier for the data
            data: Data to store

        Returns:
            Always returns True as in-memory saves don't fail
        """
        self._data[key] = data
        return True

    def load(self, key: str) -> Any | None:
        """
        Load data from memory.

        Args:
            key: Unique identifier to retrieve

        Returns:
            Stored data if found, None otherwise
        """
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """
        Delete data from memory.

        Args:
            key: Unique identifier to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in memory.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._data

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """
        List all keys, optionally filtered by pattern.

        Args:
            pattern: Optional substring pattern to filter keys

        Returns:
            List of matching key strings
        """
        keys = list(self._data.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """
        Clear all data from memory.

        Returns:
            Number of items cleared
        """
        count = len(self._data)
        self._data.clear()
        return count


class FileStorage(MemoryStorage):
    """File-based persistent storage using pickle serialisation.

    Each key-value pair is stored as a separate ``.pkl`` file named by
    the MD5 hash of its key. An ``_index.json`` file tracks the
    key-to-filename mapping. Suitable for moderate-sized datasets that
    need persistence across process restarts.

    Attributes:
        storage_dir: :class:`~pathlib.Path` to the directory holding
            pickle files and the index.
        _index: Dictionary mapping string keys to their pickle filenames.

    Example:
        >>> from calute.memory.storage import FileStorage
        >>> fs = FileStorage("/tmp/my_memory")
        >>> fs.save("key1", {"data": 42})
        True
        >>> fs.load("key1")
        {'data': 42}
    """

    def __init__(self, storage_dir: str = ".calute_memory") -> None:
        """Initialize file storage and create the storage directory.

        Args:
            storage_dir: Directory path for storing pickle files and the
                key index. Created (with parents) if it does not exist.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Load the key-to-filename index from disk.

        Returns:
            Dictionary mapping string keys to their pickle filenames.
            Returns an empty dict if the index file does not exist.
        """
        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Persist the key-to-filename index as JSON to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _get_file_path(self, key: str) -> Path:
        """
        Get file path for a key using MD5 hash.

        Args:
            key: The key to generate a file path for

        Returns:
            Path object for the pickle file
        """
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.storage_dir / f"{key_hash}.pkl"

    def save(self, key: str, data: Any) -> bool:
        """
        Save data to a pickle file.

        Args:
            key: Unique identifier for the data
            data: Data to store (must be pickle-serializable)

        Returns:
            True if save was successful, False on error
        """
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
        """
        Load data from a pickle file.

        Args:
            key: Unique identifier to retrieve

        Returns:
            Stored data if found, None otherwise
        """
        if key not in self._index:
            return None
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def delete(self, key: str) -> bool:
        """
        Delete a pickle file and its index entry.

        Args:
            key: Unique identifier to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        if key not in self._index:
            return False
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            file_path.unlink()
        del self._index[key]
        self._save_index()
        return True

    def exists(self, key: str) -> bool:
        """
        Check if key exists in the index.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._index

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """
        List all stored keys, optionally filtered by pattern.

        Args:
            pattern: Optional substring pattern to filter keys

        Returns:
            List of matching key strings
        """
        keys = list(self._index.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """
        Clear all pickle files and reset index.

        Returns:
            Number of items cleared
        """
        count = 0
        for key in list(self._index.keys()):
            if self.delete(key):
                count += 1
        return count


class SQLiteStorage(MemoryStorage):
    """SQLite-based persistent storage with ACID guarantees.

    Uses a local SQLite database for reliable persistent key-value
    storage. Data is serialised with :mod:`pickle` and stored as BLOBs.
    When the ``WRITE_MEMORY`` environment variable is not set to ``"1"``,
    the backend transparently falls back to a plain in-memory dictionary
    so that read-only or ephemeral sessions incur no disk I/O.

    Attributes:
        write_enabled: ``True`` when the ``WRITE_MEMORY`` environment
            variable is ``"1"``, enabling actual SQLite persistence.
        db_path: :class:`~pathlib.Path` to the SQLite database file.
        _memory_storage: In-memory fallback dictionary used when
            ``write_enabled`` is ``False``.

    Example:
        >>> import os
        >>> os.environ["WRITE_MEMORY"] = "1"
        >>> from calute.memory.storage import SQLiteStorage
        >>> store = SQLiteStorage("/tmp/mem.db")
        >>> store.save("k", {"v": 1})
        True
    """

    def __init__(self, db_path: str = ".calute_memory/memory.db") -> None:
        """Initialize SQLite storage with optional write persistence.

        When ``WRITE_MEMORY=1`` is set in the environment, the database
        file and its parent directory are created if they do not exist,
        and the schema is initialised. Otherwise, an in-memory dictionary
        is used as a transparent fallback.

        Args:
            db_path: File path for the SQLite database. Parent directories
                are created automatically when write is enabled.
        """
        import os

        self.write_enabled = os.environ.get("WRITE_MEMORY", "0") == "1"

        self.db_path = Path(db_path)
        if self.write_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self._memory_storage = {}

    def _init_db(self) -> None:
        """Initialize the database schema with the ``memory`` table and indexes.

        Creates the ``memory`` table (if it does not already exist) with
        columns for ``key`` (primary key), ``data`` (BLOB), ``created_at``,
        and ``updated_at``. Also creates an index on ``created_at`` for
        efficient ordering.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON memory(created_at)
            """)
            conn.commit()

    def save(self, key: str, data: Any) -> bool:
        """
        Save data to database or in-memory storage.

        Args:
            key: Unique identifier for the data
            data: Data to store (must be pickle-serializable)

        Returns:
            True if save was successful, False on error
        """
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
        """
        Load data from database or in-memory storage.

        Args:
            key: Unique identifier to retrieve

        Returns:
            Stored data if found, None otherwise
        """
        if not self.write_enabled:
            return self._memory_storage.get(key)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
        return None

    def delete(self, key: str) -> bool:
        """
        Delete from database or in-memory storage.

        Args:
            key: Unique identifier to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
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
        """
        Check if key exists in storage.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists, False otherwise
        """
        if not self.write_enabled:
            return key in self._memory_storage

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM memory WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """
        List all stored keys, optionally filtered by pattern.

        Args:
            pattern: Optional substring pattern to filter keys

        Returns:
            List of matching key strings, ordered by creation date (newest first)
        """
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
        """
        Clear all data from storage.

        Returns:
            Number of items cleared
        """
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
    """RAG-enhanced storage with vector similarity search capabilities.

    Wraps another :class:`MemoryStorage` backend and augments it with
    dense vector embeddings for semantic similarity search via
    :meth:`search_similar`. Supports multiple embedding backends:

    - **TF-IDF** (default) -- no external dependencies, hash-based
      256-dimensional vectors.
    - **sentence-transformers** -- dense embeddings from a local model
      (requires the ``sentence-transformers`` package).
    - **OpenAI** -- remote embeddings via the OpenAI API (requires the
      ``openai`` package and a valid API key).

    Attributes:
        backend: The underlying :class:`MemoryStorage` that handles
            actual data persistence.
        embeddings: Dictionary mapping storage keys to their computed
            embedding vectors.

    Example:
        >>> from calute.memory.storage import RAGStorage, SimpleStorage
        >>> rag = RAGStorage(SimpleStorage(), embedding_model="tfidf")
        >>> rag.save("doc1", "The cat sat on the mat")
        True
        >>> results = rag.search_similar("feline sitting", limit=5)
    """

    def __init__(
        self,
        backend: MemoryStorage | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
    ) -> None:
        """Initialize RAG storage with an embedding backend.

        Args:
            backend: Underlying :class:`MemoryStorage` for data persistence.
                Defaults to a new :class:`SimpleStorage` instance.
            embedding_model: Embedding backend selector. Accepted values:

                - ``None`` -- auto-detect best available (prefers
                  sentence-transformers, falls back to TF-IDF).
                - ``"tfidf"`` -- hash-based TF-IDF embeddings (no extra
                  dependencies).
                - A ``sentence-transformers`` model name (e.g.
                  ``"all-MiniLM-L6-v2"``).
                - An OpenAI model name starting with ``"text-embedding"``
                  (e.g. ``"text-embedding-3-small"``).
            embedding_api_key: API key for OpenAI embeddings. Falls back
                to the ``OPENAI_API_KEY`` environment variable when not
                provided.
        """
        self.backend = backend or SimpleStorage()
        self.embeddings: dict[str, list[float]] = {}
        self._embedding_model_name = embedding_model
        self._embedding_api_key = embedding_api_key
        self._embedder = None
        self._tfidf_vectorizer = None
        self._tfidf_corpus: list[str] = []
        self._tfidf_keys: list[str] = []
        self._embedding_type = self._resolve_embedding_type(embedding_model)

    def _resolve_embedding_type(self, model: str | None) -> str:
        """Resolve which embedding backend to use.

        Selects the appropriate embedding backend based on the requested model name.
        When model is None, auto-detects the best available backend (prefers
        sentence-transformers, falls back to tfidf if not installed).

        Args:
            model: Model name hint. ``"tfidf"`` forces TF-IDF; names starting with
                ``"text-embedding"`` select OpenAI; any other string attempts to
                load a sentence-transformers model; None triggers auto-detection.

        Returns:
            One of ``"tfidf"``, ``"openai"``, or ``"sentence_transformers"``.
        """
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
        """
        Compute text embedding using the configured backend.

        Args:
            text: Text to compute embedding for

        Returns:
            Embedding vector as list of floats
        """
        if self._embedding_type == "sentence_transformers":
            return self._compute_sentence_transformer_embedding(text)
        elif self._embedding_type == "openai":
            return self._compute_openai_embedding(text)
        else:
            return self._compute_tfidf_embedding(text)

    def _compute_tfidf_embedding(self, text: str) -> list[float]:
        """Compute TF-IDF based embedding using word frequencies.

        Builds a fixed-size 256-dimensional vector by hashing each word to a
        position, accumulating term-frequency weights, and L2-normalising the
        result. Requires no external dependencies.

        Args:
            text: Input text to encode.

        Returns:
            Normalised 256-dimensional float vector, or a 128-dimensional zero
            vector for empty input.
        """
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
        """Compute a dense embedding using a sentence-transformers model.

        Lazily loads the model on first call if it has not already been
        initialised. Falls back to ``"all-MiniLM-L6-v2"`` when no explicit
        model name is configured.

        Args:
            text: Input text to encode.

        Returns:
            Dense float embedding vector from the sentence-transformers model.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._embedding_model_name or "all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_name)

        embedding = self._embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _compute_openai_embedding(self, text: str) -> list[float]:
        """Compute an embedding using the OpenAI embeddings API.

        Falls back to TF-IDF if the ``openai`` package is not installed or no
        API key is available (via ``embedding_api_key`` or the
        ``OPENAI_API_KEY`` environment variable).

        Args:
            text: Input text to encode.

        Returns:
            Dense float embedding vector, or a TF-IDF fallback vector.
        """
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
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def save(self, key: str, data: Any) -> bool:
        """
        Save data with computed embedding.

        Args:
            key: Unique identifier for the data
            data: Data to store (embeddings computed for str/dict types)

        Returns:
            True if save was successful, False otherwise
        """
        success = self.backend.save(key, data)
        if success and isinstance(data, str | dict):
            text = str(data) if not isinstance(data, str) else data
            self.embeddings[key] = self._compute_embedding(text)
        return success

    def load(self, key: str) -> Any | None:
        """
        Load data from backend storage.

        Args:
            key: Unique identifier to retrieve

        Returns:
            Stored data if found, None otherwise
        """
        return self.backend.load(key)

    def delete(self, key: str) -> bool:
        """
        Delete data and its embedding.

        Args:
            key: Unique identifier to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        self.embeddings.pop(key, None)
        return self.backend.delete(key)

    def exists(self, key: str) -> bool:
        """
        Check if key exists in backend storage.

        Args:
            key: Unique identifier to check

        Returns:
            True if key exists, False otherwise
        """
        return self.backend.exists(key)

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """
        List keys from backend storage.

        Args:
            pattern: Optional substring pattern to filter keys

        Returns:
            List of matching key strings
        """
        return self.backend.list_keys(pattern)

    def clear(self) -> int:
        """
        Clear all data and embeddings.

        Returns:
            Number of items cleared
        """
        self.embeddings.clear()
        return self.backend.clear()

    def search_similar(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """
        Search for items similar to the query.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of tuples (key, similarity_score, data) sorted by similarity
        """
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
