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
"""Persistent long-term memory tier with semantic and lexical search.

``LongTermMemory`` writes to ``SQLiteStorage`` (optionally wrapped in
``RAGStorage`` for embeddings) and provides relevance/recency/importance
blended scoring. It enforces a soft item cap via aged-out and low-value
cleanup, and supports consolidation (similar-item merging + a textual
summary)."""

from datetime import datetime, timedelta
from typing import Any

from .base import Memory, MemoryItem
from .storage import RAGStorage, SQLiteStorage


class LongTermMemory(Memory):
    """Durable memory tier scored by relevance, recency, and importance."""

    def __init__(
        self,
        storage: Any | None = None,
        enable_embeddings: bool = True,
        db_path: str | None = None,
        max_items: int = 10000,
        retention_days: int = 365,
    ) -> None:
        """Construct the tier and hydrate previously persisted items.

        Args:
            storage: Pre-built ``MemoryStorage``; when ``None``, a new
                ``SQLiteStorage`` (optionally wrapped in ``RAGStorage``)
                is created.
            enable_embeddings: When True, wrap the default storage in
                ``RAGStorage`` so semantic search is available.
            db_path: SQLite path override; ignored if ``storage`` given.
            max_items: Soft cap that triggers cleanup on save.
            retention_days: Age threshold for the cleanup heuristic."""

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
        """Replay every ``ltm_*`` row from storage back into the in-memory index."""

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
        """Persist a new long-term item, triggering cleanup when at capacity.

        Args:
            content: Text payload to remember.
            metadata: Annotations; receives ``importance`` plus any
                extra ``**kwargs``.
            agent_id: Producing agent identifier.
            user_id: Owning user identifier.
            conversation_id: Originating conversation identifier.
            importance: Float in ``[0, 1]`` used for scoring + eviction.
            **kwargs: Folded into ``metadata`` as freeform annotations."""

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
        """Search the tier, preferring embedding-based recall when available.

        With ``use_semantic`` and a ``RAGStorage`` backend, query is
        embedded and similarity-ranked. Otherwise items are scored as
        ``0.5*relevance + 0.3*recency + 0.2*importance`` and sorted.
        Both paths bump ``access_count`` and ``last_accessed`` on each
        returned item.

        Args:
            query: Free-form search string.
            limit: Maximum results to return.
            filters: Field-level filters applied after scoring.
            use_semantic: Enable vector search when storage supports it.
            **kwargs: Accepted for protocol compatibility; unused."""

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
        """Fetch by id (incrementing access counters) or scan for matching items."""

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
        """Apply field updates and re-persist the item to storage."""

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
        """Delete a single item by id or every item matching ``filters``.

        Returns the number of items removed (from both the in-memory
        index and the durable backing store)."""

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
        """Delete every ``ltm_*`` entry from storage and wipe the in-memory index."""

        if self.storage:
            for key in self.storage.list_keys("ltm_"):
                self.storage.delete(key)

        self._items.clear()
        self._index.clear()

    def _cleanup_old_memories(self) -> None:
        """Evict aged-out and low-value items to make room under ``max_items``.

        Items older than ``retention_days`` or with low importance and
        few accesses are removed first. If that yields too few removals,
        the bottom 20% by composite score is dropped."""

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
        """Return True when ``item`` matches every key in ``filters``.

        Filter values can be plain values (equality test) or callables
        (predicate). Metadata fields are searched when no matching
        attribute exists on the item itself."""

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
        """Simple lexical relevance: substring match scores 1.0, else token overlap ratio."""

        content_lower = content.lower()
        if query in content_lower:
            return 1.0

        query_words = query.split()
        if query_words:
            matching = sum(1 for word in query_words if word in content_lower)
            return matching / len(query_words)

        return 0.0

    def consolidate(self, merge_similar: bool = True, similarity_threshold: float = 0.8) -> str:
        """Optionally merge near-duplicate items and produce a grouped summary.

        Items are grouped by ``conversation_id`` (falling back to
        ``agent_id`` then ``"general"``), sorted by importance and
        recency, and the top five per group are summarised.

        Args:
            merge_similar: When True, run ``_merge_similar_memories``
                first.
            similarity_threshold: Token-overlap ratio above which two
                items are merged."""

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
        """Collapse pairs of items whose token-overlap exceeds ``threshold``.

        The shorter item is discarded; the longer keeps the union of
        access counts and the maximum importance score. Run during
        ``consolidate``."""

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
