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
"""Legacy aliases for the multi-tier memory API.

Older Xerxes callers (and external integrations) expected a single
``MemoryStore`` wrapping a typed bucket dictionary plus a
``MemoryEntry`` dataclass with ``context``/``importance_score``/``tags``
fields. This module preserves those shapes while delegating to the
modern ``ContextualMemory`` + ``MemoryItem`` core."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from .base import MemoryItem
from .contextual_memory import ContextualMemory
from .storage import SQLiteStorage


class MemoryType(StrEnum):
    """Enumerated memory categories used by the legacy ``MemoryStore`` API.

    The five values cover the original taxonomy: short-term scratch,
    consolidated long-term, episodic (event-bound), semantic (fact-bound),
    working (in-progress reasoning), and procedural (how-to)."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry(MemoryItem):
    """Legacy ``MemoryItem`` extension carrying typed memory metadata.

    Attributes:
        memory_type: Tier label (string or ``MemoryType``); coerced to
            string during ``__post_init__``.
        context: Structured situational metadata mirrored into
            ``MemoryItem.metadata["context"]``.
        importance_score: Float in ``[0, 1]`` driving promotion and
            eviction decisions.
        tags: Free-form labels, mirrored into ``metadata["tags"]``."""

    memory_type: str | MemoryType = MemoryType.SHORT_TERM
    context: dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalise ``memory_type`` and mirror context/tags/importance into metadata."""

        if isinstance(self.memory_type, MemoryType):
            self.memory_type = self.memory_type.value

        self.metadata = dict(self.metadata)
        if self.context:
            self.metadata["context"] = dict(self.context)
        if self.tags:
            self.metadata["tags"] = list(self.tags)
        self.metadata["importance"] = self.importance_score

    def to_dict(self) -> dict[str, Any]:
        """Extend ``MemoryItem.to_dict`` with legacy typed fields."""

        data = super().to_dict()
        data["memory_type"] = self.memory_type
        data["context"] = dict(self.context)
        data["importance_score"] = self.importance_score
        data["tags"] = list(self.tags)
        return data


class MemoryStore(ContextualMemory):
    """Legacy bucket-per-type memory store layered on ``ContextualMemory``.

    Maintains a ``memories`` dict keyed by ``MemoryType`` for callers
    that iterate buckets directly, while keeping the underlying
    short-term/long-term tiers in sync. Persistence to SQLite is opt-in
    via ``enable_persistence`` + the ``WRITE_MEMORY=1`` env guard."""

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
        """Configure capacities, persistence, and embedding options.

        Args:
            max_short_term: Cap on the short-term bucket.
            max_working: Cap on the working bucket.
            max_long_term: Cap on the long-term bucket.
            enable_vector_search: Reserved for future vector search;
                stored for legacy compatibility.
            embedding_dimension: Stored embedding width hint.
            enable_persistence: When True (and ``WRITE_MEMORY=1`` is
                set), persist long-term items to SQLite at
                ``persistence_path``.
            persistence_path: SQLite file used when persistence is on.
            cache_size: Reserved legacy parameter.
            memory_type: Default bucket used when callers omit a type."""

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
        """Append a typed entry to the matching bucket and underlying tiers.

        Args:
            content: Text payload to remember.
            memory_type: Bucket the entry belongs to.
            agent_id: Producing agent identifier.
            context: Optional situational annotations.
            importance_score: Float in ``[0, 1]`` driving consolidation.
            tags: Free-form labels.
            **kwargs: Forwarded to the underlying ``MemoryEntry`` (e.g.
                ``user_id``, ``conversation_id``, ``timestamp``).

        Returns:
            The newly created ``MemoryEntry``."""

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
        """Return entries matching the supplied bucket/agent/tag filters.

        Results are sorted newest-first and truncated to ``limit``.

        Args:
            memory_types: Restrict to these buckets (defaults to all).
            agent_id: Optional producing-agent filter.
            tags: Require at least one tag overlap when set.
            limit: Maximum entries to return.
            min_importance: Drop entries below this importance score.
            query_embedding: Accepted for forward compatibility but
                currently unused.
            memory_type: Single-bucket shortcut when ``memory_types``
                is not supplied."""

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
        """Return entries created within the last ``minutes_ago`` minutes, newest-first."""

        cutoff = datetime.now() - timedelta(minutes=minutes_ago)
        recent = [entry for entries in self.memories.values() for entry in entries if entry.timestamp >= cutoff]
        recent.sort(key=lambda item: item.timestamp, reverse=True)
        return recent

    def clear_memories(
        self,
        memory_type: MemoryType | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Drop entries by bucket, agent, or both.

        Args:
            memory_type: Restrict deletion to this bucket; ``None``
                means every bucket.
            agent_id: Restrict deletion to entries produced by this
                agent; ``None`` means every agent."""

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
        """Promote important entries to long-term and produce a summary.

        Entries in short-term, working, and episodic buckets whose
        ``importance_score >= threshold`` move into long-term storage.
        Returns a human-readable summary block of important and recent
        memories for the given agent.

        Args:
            agent_id: Restrict promotion + summary to this agent.
            merge_similar: Accepted for API compatibility; currently
                unused.
            threshold: Importance cutoff for promotion."""

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
        """Augment base statistics with the total entry count across all buckets."""

        stats = super().get_statistics()
        stats["total_memories"] = sum(len(entries) for entries in self.memories.values())
        stats["cache_hit_rate"] = 0.0
        return stats

    def _enforce_limit(self, memory_type: MemoryType) -> None:
        """Trim the bucket to its configured cap, dropping the oldest overflow."""

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
        """Mirror a freshly added entry into the appropriate underlying tier."""

        if memory_type in {MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL}:
            self.long_term._items.append(entry)
            self.long_term._index[entry.memory_id] = entry
        else:
            self.short_term._items.append(entry)
            self.short_term._index[entry.memory_id] = entry
        self._items.append(entry)
        self._index[entry.memory_id] = entry

    def _rebuild_underlying_stores(self) -> None:
        """Replay every bucket into the short-term and long-term tiers from scratch."""

        self.short_term._items = []
        self.short_term._index = {}
        self.long_term._items = []
        self.long_term._index = {}
        self._items = []
        self._index = {}

        for memory_type, entries in self.memories.items():
            for entry in entries:
                self._sync_underlying_stores(entry, memory_type)
