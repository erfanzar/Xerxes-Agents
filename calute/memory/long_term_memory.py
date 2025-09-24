
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

"""Long-term memory implementation with persistence and semantic search"""

from datetime import datetime, timedelta
from typing import Any

from .base import Memory, MemoryItem
from .storage import RAGStorage, SQLiteStorage


class LongTermMemory(Memory):
    """
    Long-term memory with persistence and semantic search.
    Stores important information for extended periods.
    """

    def __init__(
        self,
        storage=None,
        enable_embeddings: bool = True,
        db_path: str | None = None,
        max_items: int = 10000,
        retention_days: int = 365,
    ):
        """
        Initialize long-term memory.

        Args:
            storage: Storage backend (defaults to SQLiteStorage)
            enable_embeddings: Enable semantic search
            db_path: Database path for SQLite storage
            max_items: Maximum items to store
            retention_days: Days to retain memories before expiry
        """
        if storage is None:
            if db_path:
                base_storage = SQLiteStorage(db_path)
            else:
                base_storage = SQLiteStorage()

            storage = RAGStorage(base_storage) if enable_embeddings else base_storage

        super().__init__(storage=storage, max_items=max_items, enable_embeddings=enable_embeddings)
        self.retention_days = retention_days
        self._load_from_storage()

    def _load_from_storage(self):
        """Load existing memories from storage"""
        if not self.storage:
            return

        for key in self.storage.list_keys("ltm_"):
            data = self.storage.load(key)
            if data:
                item = MemoryItem.from_dict(data)
                self._items.append(item)
                self._index[item.memory_id] = item

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        importance: float = 0.5,
        **kwargs,
    ) -> MemoryItem:
        """
        Save to long-term memory with importance scoring.

        Args:
            content: Content to save
            metadata: Additional metadata
            agent_id: Agent identifier
            user_id: User identifier
            conversation_id: Conversation identifier
            importance: Importance score (0-1)
            **kwargs: Additional fields

        Returns:
            Created memory item
        """
        metadata = metadata or {}
        metadata["importance"] = importance
        metadata.update(kwargs)

        item = MemoryItem(
            content=content,
            memory_type="long_term",
            metadata=metadata,
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        if self.max_items and len(self._items) >= self.max_items:
            self._cleanup_old_memories()

        self._items.append(item)
        self._index[item.memory_id] = item

        if self.storage:
            self.storage.save(f"ltm_{item.memory_id}", item.to_dict())

        return item

    def search(
        self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, use_semantic: bool = True, **kwargs
    ) -> list[MemoryItem]:
        """
        Search long-term memory using semantic similarity or keyword matching.

        Args:
            query: Search query
            limit: Maximum results
            filters: Filter criteria
            use_semantic: Use semantic search if available

        Returns:
            List of matching memory items
        """

        if use_semantic and isinstance(self.storage, RAGStorage):
            results = self.storage.search_similar(query, limit=limit * 2)
            memories = []

            for key, similarity, data in results:
                if key.startswith("ltm_"):
                    item = MemoryItem.from_dict(data)
                    item.relevance_score = similarity

                    if filters:
                        if not self._matches_filters(item, filters):
                            continue

                    item.access_count += 1
                    item.last_accessed = datetime.now()
                    memories.append(item)

                    if len(memories) >= limit:
                        break

            return memories

        query_lower = query.lower()
        matches = []

        for item in self._items:
            if filters and not self._matches_filters(item, filters):
                continue

            relevance = self._calculate_relevance(item.content, query_lower)

            age_days = (datetime.now() - item.timestamp).days
            recency_score = max(0, 1 - (age_days / self.retention_days))
            importance = item.metadata.get("importance", 0.5)

            item.relevance_score = relevance * 0.5 + recency_score * 0.3 + importance * 0.2

            if item.relevance_score > 0:
                item.access_count += 1
                item.last_accessed = datetime.now()
                matches.append(item)

        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        return matches[:limit]

    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Retrieve specific memories"""
        if memory_id:
            item = self._index.get(memory_id)
            if item:
                item.access_count += 1
                item.last_accessed = datetime.now()

                if self.storage:
                    self.storage.save(f"ltm_{memory_id}", item.to_dict())
            return item

        results = []
        for item in self._items:
            if filters and not self._matches_filters(item, filters):
                continue

            item.access_count += 1
            item.last_accessed = datetime.now()
            results.append(item)

            if len(results) >= limit:
                break

        return results

    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory item"""
        if memory_id not in self._index:
            return False

        item = self._index[memory_id]
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        if self.storage:
            self.storage.save(f"ltm_{memory_id}", item.to_dict())

        return True

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete memory items"""
        count = 0

        if memory_id:
            if memory_id in self._index:
                item = self._index[memory_id]
                self._items.remove(item)
                del self._index[memory_id]
                if self.storage:
                    self.storage.delete(f"ltm_{memory_id}")
                count = 1
        elif filters:
            to_remove = []
            for item in self._items:
                if self._matches_filters(item, filters):
                    to_remove.append(item)

            for item in to_remove:
                self._items.remove(item)
                del self._index[item.memory_id]
                if self.storage:
                    self.storage.delete(f"ltm_{item.memory_id}")
                count += 1

        return count

    def clear(self) -> None:
        """Clear all long-term memories"""
        if self.storage:
            for key in self.storage.list_keys("ltm_"):
                self.storage.delete(key)

        self._items.clear()
        self._index.clear()

    def _cleanup_old_memories(self):
        """Remove expired or low-importance memories"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        to_remove = []

        for item in self._items:
            if item.timestamp < cutoff_date:
                to_remove.append(item)

            elif item.metadata.get("importance", 0.5) < 0.3 and item.access_count < 2:
                to_remove.append(item)

        if len(to_remove) < len(self._items) * 0.2:
            self._items.sort(
                key=lambda x: (
                    x.metadata.get("importance", 0.5) * 0.3
                    + (x.access_count / 100) * 0.3
                    + (1 - (datetime.now() - x.timestamp).days / self.retention_days) * 0.4
                )
            )
            to_remove = list(self._items[: int(len(self._items) * 0.2)])

        for item in to_remove:
            self._items.remove(item)
            del self._index[item.memory_id]
            if self.storage:
                self.storage.delete(f"ltm_{item.memory_id}")

    def _matches_filters(self, item: MemoryItem, filters: dict[str, Any]) -> bool:
        """Check if item matches all filters"""
        for key, value in filters.items():
            if hasattr(item, key):
                if getattr(item, key) != value:
                    return False
            elif key in item.metadata:
                if item.metadata[key] != value:
                    return False
        return True

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score"""
        content_lower = content.lower()
        if query in content_lower:
            return 1.0

        query_words = query.split()
        if query_words:
            matching = sum(1 for word in query_words if word in content_lower)
            return matching / len(query_words)

        return 0.0

    def consolidate(self) -> str:
        """
        Consolidate memories into a coherent summary.
        Useful for generating context or reports.
        """
        if not self._items:
            return "No long-term memories available."

        grouped = {}
        for item in self._items:
            key = item.conversation_id or item.agent_id or "general"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)

        summary = ["Long-term memory summary:"]

        for key, items in grouped.items():
            items.sort(key=lambda x: (x.metadata.get("importance", 0.5), x.timestamp), reverse=True)

            summary.append(f"\n{key.title()}:")
            for item in items[:5]:
                summary.append(f"  • {item.content[:150]}")

        return "\n".join(summary)
