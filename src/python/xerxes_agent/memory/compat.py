# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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

"""Backward-compatible memory API for legacy Xerxes callers.

Provides :class:`MemoryType`, :class:`MemoryEntry`, and :class:`MemoryStore`
which mirror the older public surface so that existing code continues to work
without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from .base import MemoryItem
from .contextual_memory import ContextualMemory
from .storage import SQLiteStorage


class MemoryType(StrEnum):
    """Legacy memory type enum.

    Each value names a category of memory that the :class:`MemoryStore`
    maintains separately.
    """

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry(MemoryItem):
    """Legacy memory entry shape expected by the older API/tests.

    Extends :class:`~xerxes_agent.memory.base.MemoryItem` with additional fields
    that the legacy public API exposes: ``context``, ``importance_score``,
    and ``tags``. These are synced into ``metadata`` during ``__post_init__``
    so that the underlying storage machinery sees them.

    Attributes:
        memory_type: Memory category (default: SHORT_TERM).
        context: Arbitrary context dictionary attached to this entry.
        importance_score: Importance weight used for promotion decisions (0-1).
        tags: Searchable string labels for this entry.
    """

    memory_type: str | MemoryType = MemoryType.SHORT_TERM
    context: dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalise memory_type and sync legacy fields into metadata."""
        if isinstance(self.memory_type, MemoryType):
            self.memory_type = self.memory_type.value

        self.metadata = dict(self.metadata)
        if self.context:
            self.metadata["context"] = dict(self.context)
        if self.tags:
            self.metadata["tags"] = list(self.tags)
        self.metadata["importance"] = self.importance_score

    def to_dict(self) -> dict[str, Any]:
        """Serialise entry to a dictionary including legacy fields.

        Returns:
            Dictionary representation extending the base :meth:`MemoryItem.to_dict`
            output with ``context``, ``importance_score``, and ``tags`` keys.
        """
        data = super().to_dict()
        data["memory_type"] = self.memory_type
        data["context"] = dict(self.context)
        data["importance_score"] = self.importance_score
        data["tags"] = list(self.tags)
        return data


class MemoryStore(ContextualMemory):
    """Compatibility wrapper preserving the legacy memory store surface.

    Maintains per-type buckets (``self.memories``) alongside the underlying
    :class:`~xerxes_agent.memory.contextual_memory.ContextualMemory` stores so that
    legacy callers using :meth:`add_memory` / :meth:`retrieve_memories` etc.
    continue to work correctly.
    """

    def __init__(
        self,
        max_short_term: int = 100,
        max_working: int = 10,
        max_long_term: int = 10000,
        enable_vector_search: bool = False,
        embedding_dimension: int = 768,
        enable_persistence: bool = False,
        persistence_path: str | None = None,
        cache_size: int = 100,
        memory_type: MemoryType | None = None,
    ) -> None:
        """Initialise the legacy-compatible memory store.

        Args:
            max_short_term: Maximum entries in the short-term bucket.
            max_working: Maximum entries in the working-memory bucket.
            max_long_term: Maximum entries in the long-term bucket.
            enable_vector_search: Reserved; vector search is not currently used.
            embedding_dimension: Reserved embedding dimension hint.
            enable_persistence: Whether to enable SQLite persistence.
            persistence_path: File path for the SQLite database.
            cache_size: Reserved cache size hint.
            memory_type: Default :class:`MemoryType` for new entries.
        """
        import os

        write_memory = os.environ.get("WRITE_MEMORY", "0") == "1"
        storage = None
        if enable_persistence and persistence_path and write_memory:
            storage = SQLiteStorage(persistence_path)

        super().__init__(
            short_term_capacity=max_short_term,
            long_term_storage=storage,
            promotion_threshold=3,
            importance_threshold=0.7,
        )

        self.max_short_term = max_short_term
        self.max_working = max_working
        self.max_long_term = max_long_term
        self.enable_vector_search = enable_vector_search
        self.embedding_dimension = embedding_dimension
        self.cache_size = cache_size
        self.default_memory_type = memory_type or MemoryType.SHORT_TERM
        self.memories: dict[MemoryType, list[MemoryEntry]] = {memory_kind: [] for memory_kind in MemoryType}

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: str,
        context: dict | None = None,
        importance_score: float = 0.5,
        tags: list | None = None,
        **kwargs,
    ) -> MemoryEntry:
        """Create and store a new memory entry.

        Adds the entry to the appropriate per-type bucket, syncs it into the
        underlying ContextualMemory stores, and enforces capacity limits.

        Args:
            content: Text content to store.
            memory_type: Which bucket to place the entry in.
            agent_id: Identifier of the creating agent.
            context: Optional context dictionary attached to the entry.
            importance_score: Importance weight (0-1) influencing promotion.
            tags: Optional list of string tags for filtering.
            **kwargs: Additional fields passed to :class:`MemoryEntry`.

        Returns:
            The newly created :class:`MemoryEntry`.
        """
        entry = MemoryEntry(
            content=content,
            timestamp=kwargs.pop("timestamp", datetime.now()),
            memory_type=memory_type,
            agent_id=agent_id,
            context=context or {},
            importance_score=importance_score,
            tags=list(tags or []),
            **kwargs,
        )
        self.memories[memory_type].append(entry)
        self._sync_underlying_stores(entry, memory_type)
        self._enforce_limit(memory_type)
        return entry

    def retrieve_memories(
        self,
        memory_types: list[MemoryType] | None = None,
        agent_id: str | None = None,
        tags: list | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        query_embedding: object = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve entries matching the given criteria across memory types.

        Args:
            memory_types: Explicit list of types to search; defaults to all.
            agent_id: If provided, only return entries for this agent.
            tags: If provided, only return entries containing at least one tag.
            limit: Maximum number of entries to return.
            min_importance: Minimum importance_score threshold (inclusive).
            query_embedding: Accepted for API compatibility but not used.
            memory_type: Single type shortcut (used when memory_types is None).

        Returns:
            List of matching :class:`MemoryEntry` instances, newest first.
        """
        del query_embedding

        selected_types = memory_types or ([memory_type] if memory_type is not None else list(MemoryType))
        results: list[MemoryEntry] = []

        for selected_type in selected_types:
            for entry in self.memories[selected_type]:
                if agent_id and entry.agent_id != agent_id:
                    continue
                if tags and not any(tag in entry.tags for tag in tags):
                    continue
                if entry.importance_score < min_importance:
                    continue
                results.append(entry)

        results.sort(key=lambda item: item.timestamp, reverse=True)
        return results[:limit]

    def retrieve_recent(self, minutes_ago: int = 60) -> list[MemoryEntry]:
        """Return all entries created within the last *minutes_ago* minutes.

        Args:
            minutes_ago: Look-back window in minutes.

        Returns:
            List of :class:`MemoryEntry` instances, newest first.
        """
        cutoff = datetime.now() - timedelta(minutes=minutes_ago)
        recent = [entry for entries in self.memories.values() for entry in entries if entry.timestamp >= cutoff]
        recent.sort(key=lambda item: item.timestamp, reverse=True)
        return recent

    def clear_memories(
        self,
        memory_type: MemoryType | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Clear stored entries, optionally scoped to a type and/or agent.

        Args:
            memory_type: If provided, only clear this type's bucket.
            agent_id: If provided, only remove entries belonging to this agent.
        """
        if memory_type is None:
            if agent_id is None:
                for selected_type in MemoryType:
                    self.memories[selected_type] = []
            else:
                for selected_type in MemoryType:
                    self.memories[selected_type] = [
                        entry for entry in self.memories[selected_type] if entry.agent_id != agent_id
                    ]
        else:
            if agent_id is None:
                self.memories[memory_type] = []
            else:
                self.memories[memory_type] = [
                    entry for entry in self.memories[memory_type] if entry.agent_id != agent_id
                ]
        self._rebuild_underlying_stores()

    def consolidate_memories(
        self,
        agent_id: str | None = None,
        merge_similar: bool = True,
        threshold: float = 0.7,
    ) -> str:
        """Promote high-importance memories to long-term and return a summary.

        Iterates over short-term, working, and episodic buckets and moves any
        entry with ``importance_score >= threshold`` into the long-term bucket.
        Then generates a text summary of the most important and recent entries.

        Args:
            agent_id: If provided, only consider entries for this agent.
            merge_similar: Accepted for API compatibility; merging is not performed.
            threshold: Minimum importance score for promotion and inclusion in summary.

        Returns:
            Formatted text summary, or an empty string if no entries match.
        """
        del merge_similar

        promoted: list[MemoryEntry] = []
        for source_type in (MemoryType.SHORT_TERM, MemoryType.WORKING, MemoryType.EPISODIC):
            retained: list[MemoryEntry] = []
            for entry in self.memories[source_type]:
                if entry.importance_score >= threshold and (agent_id is None or entry.agent_id == agent_id):
                    self.memories[MemoryType.LONG_TERM].append(entry)
                    promoted.append(entry)
                else:
                    retained.append(entry)
            self.memories[source_type] = retained

        self._rebuild_underlying_stores()

        relevant = self.retrieve_memories(agent_id=agent_id, limit=20)
        if not relevant:
            return ""

        summary_parts: list[str] = []
        important = [memory for memory in relevant if memory.importance_score >= threshold]
        recent = relevant[:5]

        if important:
            summary_parts.append("Important facts:")
            for memory in important[:5]:
                summary_parts.append(f"- [{memory.importance_score:.1f}] {memory.content}")

        if recent:
            summary_parts.append("\nRecent context:")
            for memory in recent:
                if memory not in important:
                    summary_parts.append(f"- {memory.content}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> dict:
        """Return aggregate statistics including the legacy total_memories count.

        Returns:
            Dictionary extending the base statistics with ``total_memories``
            (sum across all type buckets) and ``cache_hit_rate`` (always 0.0).
        """
        stats = super().get_statistics()
        stats["total_memories"] = sum(len(entries) for entries in self.memories.values())
        stats["cache_hit_rate"] = 0.0
        return stats

    def _enforce_limit(self, memory_type: MemoryType) -> None:
        """Trim the named bucket if it exceeds its configured capacity.

        Args:
            memory_type: The bucket to check and trim.
        """
        if memory_type == MemoryType.SHORT_TERM:
            limit = self.max_short_term
        elif memory_type == MemoryType.WORKING:
            limit = self.max_working
        elif memory_type == MemoryType.LONG_TERM:
            limit = self.max_long_term
        else:
            limit = None

        if limit is not None and len(self.memories[memory_type]) > limit:
            overflow = len(self.memories[memory_type]) - limit
            del self.memories[memory_type][0:overflow]

        self._rebuild_underlying_stores()

    def _sync_underlying_stores(self, entry: MemoryEntry, memory_type: MemoryType) -> None:
        """Mirror a single entry into the appropriate ContextualMemory sub-stores.

        Long-term, semantic, and procedural entries go into ``long_term``;
        all others go into ``short_term``.

        Args:
            entry: The entry to mirror.
            memory_type: The source bucket for this entry.
        """
        if memory_type in {MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL}:
            self.long_term._items.append(entry)
            self.long_term._index[entry.memory_id] = entry
        else:
            self.short_term._items.append(entry)
            self.short_term._index[entry.memory_id] = entry
        self._items.append(entry)
        self._index[entry.memory_id] = entry

    def _rebuild_underlying_stores(self) -> None:
        """Rebuild the ContextualMemory sub-stores from the canonical buckets.

        Clears all sub-store state and re-syncs every entry from
        ``self.memories``.  Called after any bulk-deletion operation.
        """
        self.short_term._items = []
        self.short_term._index = {}
        self.long_term._items = []
        self.long_term._index = {}
        self._items = []
        self._index = {}

        for memory_type, entries in self.memories.items():
            for entry in entries:
                self._sync_underlying_stores(entry, memory_type)
