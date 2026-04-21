# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Long-term memory implementation with persistence and semantic search."""

from datetime import datetime, timedelta
from typing import Any

from .base import Memory, MemoryItem
from .storage import RAGStorage, SQLiteStorage


class LongTermMemory(Memory):
    """Long-term memory with persistence and semantic search.

    Designed for storing important information over extended periods.
    Supports both keyword-based and semantic (vector similarity) search
    depending on the storage backend. Automatically cleans up expired or
    low-importance memories when the item limit is reached.

    Attributes:
        retention_days: Number of days a memory item is retained before it
            becomes eligible for automatic cleanup.
        storage: The underlying :class:`MemoryStorage` backend used for
            persistence. May be a :class:`SQLiteStorage`,
            :class:`RAGStorage`, or any compatible implementation.

    Example:
        >>> from xerxes.memory import LongTermMemory
        >>> ltm = LongTermMemory(retention_days=90, max_items=500)
        >>> item = ltm.save("Project deadline is March 15", importance=0.9)
        >>> results = ltm.search("deadline")
    """

    def __init__(
        self,
        storage: Any | None = None,
        enable_embeddings: bool = True,
        db_path: str | None = None,
        max_items: int = 10000,
        retention_days: int = 365,
    ) -> None:
        """Initialize long-term memory with persistence and optional embeddings.

        When no ``storage`` is provided, a default backend is constructed:
        :class:`SQLiteStorage` (at ``db_path`` or the default path) optionally
        wrapped in :class:`RAGStorage` when ``enable_embeddings`` is ``True``.

        On initialisation, any previously persisted items (keys prefixed with
        ``ltm_``) are loaded from the storage backend into memory.

        Args:
            storage: Pre-configured :class:`MemoryStorage` backend. When
                ``None``, a new SQLite-backed storage is created.
            enable_embeddings: Whether to wrap the base storage in
                :class:`RAGStorage` for semantic search capability. Only
                effective when ``storage`` is ``None``.
            db_path: File path for the SQLite database. Only used when
                ``storage`` is ``None``.
            max_items: Maximum number of items to retain. When exceeded,
                :meth:`_cleanup_old_memories` is invoked.
            retention_days: Number of days after which a memory is eligible
                for automatic removal during cleanup.
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

    def _load_from_storage(self) -> None:
        """Load existing memory items from the storage backend on initialisation.

        Scans all keys with the ``ltm_`` prefix, deserialises each entry
        via :meth:`MemoryItem.from_dict`, and populates both
        :attr:`_items` and :attr:`_index`.
        """
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
        """Save a new item to long-term memory with importance scoring.

        Creates a :class:`MemoryItem` with ``memory_type="long_term"``,
        stores it in both the in-memory index and the persistent storage
        backend (if configured). If the item limit has been reached,
        :meth:`_cleanup_old_memories` is called first to free space.

        Args:
            content: Text content to store.
            metadata: Optional key-value metadata. An ``"importance"`` key
                is added automatically from the ``importance`` parameter.
            agent_id: Identifier of the agent that created this memory.
            user_id: Identifier of the user associated with this memory.
            conversation_id: Identifier of the conversation context.
            importance: Importance weight (0.0--1.0) used for ranking and
                cleanup decisions.
            **kwargs: Extra key-value pairs merged into ``metadata``.

        Returns:
            The newly created :class:`MemoryItem`.
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
        """Search long-term memory using semantic similarity or keyword matching.

        When the storage backend is a :class:`RAGStorage` instance and
        ``use_semantic`` is ``True``, performs vector-similarity search.
        Otherwise, falls back to keyword matching with a composite scoring
        formula that blends text relevance (50 %), recency (30 %), and
        importance (20 %).

        Matching items have their ``access_count`` incremented and
        ``last_accessed`` timestamp updated as a side-effect.

        Args:
            query: Natural-language or keyword search query string.
            limit: Maximum number of results to return.
            filters: Optional key-value criteria for narrowing results.
                Checked against both item attributes and metadata.
            use_semantic: When ``True`` and a :class:`RAGStorage` backend
                is available, performs vector-based semantic search.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            List of :class:`MemoryItem` instances sorted by descending
            relevance score, with at most ``limit`` entries.
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
        """
        Retrieve specific memories by ID or filter criteria.

        When memory_id is provided, returns the specific item and updates its
        access count. Otherwise, filters through memories and returns matches.

        Args:
            memory_id: Specific memory ID to retrieve
            filters: Filter criteria to match against memory attributes and metadata
            limit: Maximum number of items to return when using filters

        Returns:
            Single MemoryItem if memory_id provided, list of MemoryItem if filters used,
            or None if memory_id not found
        """
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
        """
        Update a memory item with new values.

        Persists changes to storage backend if configured.

        Args:
            memory_id: ID of the memory item to update
            updates: Dictionary of field names and new values to apply

        Returns:
            True if the update was successful, False if memory_id not found
        """
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
        """
        Delete memory items by ID or filter criteria.

        Removes items from both memory and storage backend if configured.

        Args:
            memory_id: Specific memory ID to delete
            filters: Filter criteria to match items for deletion

        Returns:
            Number of items deleted
        """
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
        """
        Clear all long-term memories.

        Removes all items from memory and storage backend. This operation
        permanently deletes all stored memories.
        """
        if self.storage:
            for key in self.storage.list_keys("ltm_"):
                self.storage.delete(key)

        self._items.clear()
        self._index.clear()

    def _cleanup_old_memories(self) -> None:
        """Remove expired or low-importance memories to free capacity.

        Applies a three-stage cleanup strategy:

        1. Remove items older than :attr:`retention_days`.
        2. Remove items with importance < 0.3 **and** access count < 2.
        3. If the above two stages did not free at least 20 % of current
           items, sort all items by a composite score (importance 30 %,
           normalised access count 30 %, recency 40 %) and remove the
           bottom 20 %.

        Removed items are also deleted from the storage backend if one is
        configured.
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        to_remove = []

        for item in self._items:
            if item.timestamp < cutoff_date:
                to_remove.append(item)

            elif item.metadata.get("importance", 0.5) < 0.3 and item.access_count < 2:
                to_remove.append(item)

        if len(to_remove) < len(self._items) * 0.2:
            self._items = sorted(
                self._items,
                key=lambda x: (
                    x.metadata.get("importance", 0.5) * 0.3
                    + (x.access_count / 100) * 0.3
                    + (1 - (datetime.now() - x.timestamp).days / self.retention_days) * 0.4
                ),
            )
            to_remove = list(self._items[: int(len(self._items) * 0.2)])

        for item in to_remove:
            self._items.remove(item)
            del self._index[item.memory_id]
            if self.storage:
                self.storage.delete(f"ltm_{item.memory_id}")

    def _matches_filters(self, item: MemoryItem, filters: dict[str, Any]) -> bool:
        """
        Check if item matches all filter criteria.

        Checks both direct attributes and metadata fields.
        Supports callable filter values for custom comparisons
        (e.g., ``{"importance": lambda x: x >= 0.8}``).

        Args:
            item: Memory item to check
            filters: Dictionary of field names to required values or callables

        Returns:
            True if item matches all filters, False otherwise
        """
        for key, value in filters.items():
            if hasattr(item, key):
                actual = getattr(item, key)
            elif key in item.metadata:
                actual = item.metadata[key]
            else:
                return False

            if callable(value):
                if not value(actual):
                    return False
            elif actual != value:
                return False
        return True

    def _calculate_relevance(self, content: str, query: str) -> float:
        """
        Calculate keyword-based relevance score.

        Uses exact match and word overlap to compute relevance.

        Args:
            content: Content string to search within
            query: Query string (should be lowercase)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        content_lower = content.lower()
        if query in content_lower:
            return 1.0

        query_words = query.split()
        if query_words:
            matching = sum(1 for word in query_words if word in content_lower)
            return matching / len(query_words)

        return 0.0

    def consolidate(self, merge_similar: bool = True, similarity_threshold: float = 0.8) -> str:
        """
        Consolidate memories by merging similar entries and producing a summary.

        Groups memories by conversation or agent, merges similar entries
        to reduce redundancy, removes low-value items, and produces a
        human-readable summary. When merge_similar is True, entries with
        high word overlap are combined into single entries.

        Args:
            merge_similar: Whether to merge entries with similar content
            similarity_threshold: Word overlap ratio to consider entries similar (0-1)

        Returns:
            Formatted string summary of consolidated long-term memory contents
        """
        if not self._items:
            return "No long-term memories available."

        if merge_similar:
            self._merge_similar_memories(similarity_threshold)

        grouped: dict[str, list[MemoryItem]] = {}
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
                importance = item.metadata.get("importance", 0.5)
                access_info = f"(importance: {importance:.1f}, accessed: {item.access_count}x)"
                summary.append(f"  - {item.content[:150]} {access_info}")

        return "\n".join(summary)

    def _merge_similar_memories(self, threshold: float = 0.8):
        """Merge memories with similar content to reduce redundancy.

        Compares word sets between items and merges those exceeding the
        similarity threshold. The merged item retains the higher importance
        and combined access count.

        Args:
            threshold: Minimum word overlap ratio to trigger a merge (0-1)
        """
        if len(self._items) < 2:
            return

        merged_ids: set[str] = set()

        for i, item_a in enumerate(self._items):
            if item_a.memory_id in merged_ids:
                continue
            words_a = set(item_a.content.lower().split())
            if not words_a:
                continue

            for j in range(i + 1, len(self._items)):
                item_b = self._items[j]
                if item_b.memory_id in merged_ids:
                    continue
                words_b = set(item_b.content.lower().split())
                if not words_b:
                    continue

                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap >= threshold:
                    keep = item_a if len(item_a.content) >= len(item_b.content) else item_b
                    discard = item_b if keep is item_a else item_a

                    keep.access_count += discard.access_count
                    keep_importance = keep.metadata.get("importance", 0.5)
                    discard_importance = discard.metadata.get("importance", 0.5)
                    keep.metadata["importance"] = max(keep_importance, discard_importance)

                    merged_ids.add(discard.memory_id)

        if merged_ids:
            for mid in merged_ids:
                if mid in self._index:
                    item = self._index[mid]
                    self._items.remove(item)
                    del self._index[mid]
                    if self.storage:
                        self.storage.delete(f"ltm_{mid}")
