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


"""Storage backends for Calute memory system"""

import hashlib
import json
import pickle
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryStorage(ABC):
    """
    Abstract base class for memory storage backends.

    Provides a common interface for different storage implementations
    including in-memory, file-based, and database storage.
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
    """
    Simple in-memory storage (non-persistent).

    Provides fast key-value storage that exists only in memory.
    Data is lost when the process terminates. Suitable for
    testing and short-lived applications.
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
    """
    File-based persistent storage using pickle.

    Stores each key-value pair as a separate pickle file, with an
    index file tracking the key-to-file mapping. Suitable for
    moderate-sized datasets that need persistence across restarts.
    """

    def __init__(self, storage_dir: str = ".calute_memory"):
        """
        Initialize file storage.

        Args:
            storage_dir: Directory path for storing pickle files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """
        Load the index mapping keys to files.

        Returns:
            Dictionary mapping keys to their file names
        """
        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the index to disk."""
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
    """
    SQLite-based persistent storage.

    Uses SQLite database for reliable persistent storage with ACID
    properties. Falls back to in-memory storage when WRITE_MEMORY
    environment variable is not set to "1".
    """

    def __init__(self, db_path: str = ".calute_memory/memory.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file
        """
        import os

        self.write_enabled = os.environ.get("WRITE_MEMORY", "0") == "1"

        self.db_path = Path(db_path)
        if self.write_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self._memory_storage = {}

    def _init_db(self):
        """Initialize database schema with memory table and indexes."""
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
    """
    RAG storage with vector similarity search capabilities.

    Wraps another storage backend and adds vector embedding support
    for semantic similarity search. Supports multiple embedding backends:
    - TF-IDF (default, no external dependencies)
    - sentence-transformers (if installed)
    - OpenAI embeddings (if openai is installed and api_key provided)
    """

    def __init__(
        self,
        backend: MemoryStorage | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
    ):
        """
        Initialize RAG storage.

        Args:
            backend: Underlying storage backend (defaults to SimpleStorage)
            embedding_model: Embedding model to use. Options:
                - None: auto-detect best available (sentence-transformers > tfidf)
                - "tfidf": TF-IDF based embeddings (no extra deps)
                - A sentence-transformers model name (e.g. "all-MiniLM-L6-v2")
                - An OpenAI model name (e.g. "text-embedding-3-small")
            embedding_api_key: API key for OpenAI embeddings (if using OpenAI model)
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
        """Resolve which embedding backend to use."""
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

        # Auto-detect: prefer sentence-transformers, fall back to tfidf
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

        Builds a vocabulary from all stored texts and computes
        term-frequency vectors for semantic comparison.
        """
        words = text.lower().split()
        if not words:
            return [0.0] * 128

        word_freq: dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        total = len(words)
        # Use consistent hashing to map words to vector positions
        dim = 256
        vec = [0.0] * dim
        for word, count in word_freq.items():
            tf = count / total
            # Hash word to a position in the vector
            idx = hash(word) % dim
            vec[idx] += tf

        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _compute_sentence_transformer_embedding(self, text: str) -> list[float]:
        """Compute embedding using sentence-transformers model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._embedding_model_name or "all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_name)

        embedding = self._embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _compute_openai_embedding(self, text: str) -> list[float]:
        """Compute embedding using OpenAI API."""
        import os

        try:
            from openai import OpenAI
        except ImportError:
            # Fall back to tfidf if openai not installed
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
