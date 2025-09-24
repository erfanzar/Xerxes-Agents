
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
    """Abstract base class for memory storage backends"""

    @abstractmethod
    def save(self, key: str, data: Any) -> bool:
        """Save data with a key"""
        pass

    @abstractmethod
    def load(self, key: str) -> Any | None:
        """Load data by key"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all stored keys, optionally filtered by pattern"""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all stored data, return number of items cleared"""
        pass


class SimpleStorage(MemoryStorage):
    """Simple in-memory storage (non-persistent)"""

    def __init__(self):
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> bool:
        """Save data in memory"""
        self._data[key] = data
        return True

    def load(self, key: str) -> Any | None:
        """Load data from memory"""
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        """Delete data from memory"""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._data

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered"""
        keys = list(self._data.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Clear all data"""
        count = len(self._data)
        self._data.clear()
        return count


class FileStorage(MemoryStorage):
    """File-based persistent storage using pickle"""

    def __init__(self, storage_dir: str = ".calute_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        """Load the index mapping keys to files"""
        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the index"""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.storage_dir / f"{key_hash}.pkl"

    def save(self, key: str, data: Any) -> bool:
        """Save data to file"""
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
        """Load data from file"""
        if key not in self._index:
            return None
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def delete(self, key: str) -> bool:
        """Delete file"""
        if key not in self._index:
            return False
        file_path = self.storage_dir / self._index[key]
        if file_path.exists():
            file_path.unlink()
        del self._index[key]
        self._save_index()
        return True

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._index

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all stored keys"""
        keys = list(self._index.keys())
        if pattern:
            keys = [k for k in keys if pattern in k]
        return keys

    def clear(self) -> int:
        """Clear all files"""
        count = 0
        for key in list(self._index.keys()):
            if self.delete(key):
                count += 1
        return count


class SQLiteStorage(MemoryStorage):
    """SQLite-based persistent storage"""

    def __init__(self, db_path: str = ".calute_memory/memory.db"):
        import os

        # Check WRITE_MEMORY environment variable
        self.write_enabled = os.environ.get('WRITE_MEMORY', '0') == '1'

        self.db_path = Path(db_path)
        if self.write_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            # Use in-memory storage when write is disabled
            self._memory_storage = {}

    def _init_db(self):
        """Initialize database schema"""
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
        """Save data to database or in-memory storage"""
        if not self.write_enabled:
            # Use in-memory storage
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
        """Load data from database or in-memory storage"""
        if not self.write_enabled:
            # Use in-memory storage
            return self._memory_storage.get(key)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
        return None

    def delete(self, key: str) -> bool:
        """Delete from database or in-memory storage"""
        if not self.write_enabled:
            # Use in-memory storage
            if key in self._memory_storage:
                del self._memory_storage[key]
                return True
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.write_enabled:
            # Use in-memory storage
            return key in self._memory_storage

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM memory WHERE key = ? LIMIT 1", (key,))
            return cursor.fetchone() is not None

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all stored keys"""
        if not self.write_enabled:
            # Use in-memory storage
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
        """Clear all data"""
        if not self.write_enabled:
            # Use in-memory storage
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
    """RAG storage with vector similarity search capabilities"""

    def __init__(self, backend: MemoryStorage | None = None):
        self.backend = backend or SimpleStorage()
        self.embeddings: dict[str, list[float]] = {}

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute text embedding (placeholder - use real embeddings in production)"""

        import hashlib

        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = [b / 255.0 for b in hash_bytes[:128]]
        return embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity"""
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def save(self, key: str, data: Any) -> bool:
        """Save with embedding"""
        success = self.backend.save(key, data)
        if success and isinstance(data, str | dict):
            text = str(data) if not isinstance(data, str) else data
            self.embeddings[key] = self._compute_embedding(text)
        return success

    def load(self, key: str) -> Any | None:
        """Load data"""
        return self.backend.load(key)

    def delete(self, key: str) -> bool:
        """Delete data and embedding"""
        self.embeddings.pop(key, None)
        return self.backend.delete(key)

    def exists(self, key: str) -> bool:
        """Check existence"""
        return self.backend.exists(key)

    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List keys"""
        return self.backend.list_keys(pattern)

    def clear(self) -> int:
        """Clear all"""
        self.embeddings.clear()
        return self.backend.clear()

    def search_similar(self, query: str, limit: int = 10, threshold: float = 0.0) -> list[tuple[str, float, Any]]:
        """Search for similar items"""
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
