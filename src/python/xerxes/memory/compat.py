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
"""Compat module for Xerxes.

Exports:
    - MemoryType
    - MemoryEntry
    - MemoryStore"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from .base import MemoryItem
from .contextual_memory import ContextualMemory
from .storage import SQLiteStorage


class MemoryType(StrEnum):
    """Memory type.

    Inherits from: StrEnum
    """

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry(MemoryItem):
    """Memory entry.

    Inherits from: MemoryItem

    Attributes:
        memory_type (str | MemoryType): memory type.
        context (dict[str, Any]): context.
        importance_score (float): importance score.
        tags (list[str]): tags."""

    memory_type: str | MemoryType = MemoryType.SHORT_TERM
    context: dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Dunder method for post init.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if isinstance(self.memory_type, MemoryType):
            self.memory_type = self.memory_type.value

        self.metadata = dict(self.metadata)
        if self.context:
            self.metadata["context"] = dict(self.context)
        if self.tags:
            self.metadata["tags"] = list(self.tags)
        self.metadata["importance"] = self.importance_score

    def to_dict(self) -> dict[str, Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        data = super().to_dict()
        data["memory_type"] = self.memory_type
        data["context"] = dict(self.context)
        data["importance_score"] = self.importance_score
        data["tags"] = list(self.tags)
        return data


class MemoryStore(ContextualMemory):
    """Memory store.

    Inherits from: ContextualMemory
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
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_short_term (int, optional): IN: max short term. Defaults to 100. OUT: Consumed during execution.
            max_working (int, optional): IN: max working. Defaults to 10. OUT: Consumed during execution.
            max_long_term (int, optional): IN: max long term. Defaults to 10000. OUT: Consumed during execution.
            enable_vector_search (bool, optional): IN: enable vector search. Defaults to False. OUT: Consumed during execution.
            embedding_dimension (int, optional): IN: embedding dimension. Defaults to 768. OUT: Consumed during execution.
            enable_persistence (bool, optional): IN: enable persistence. Defaults to False. OUT: Consumed during execution.
            persistence_path (str | None, optional): IN: persistence path. Defaults to None. OUT: Consumed during execution.
            cache_size (int, optional): IN: cache size. Defaults to 100. OUT: Consumed during execution.
            memory_type (MemoryType | None, optional): IN: memory type. Defaults to None. OUT: Consumed during execution."""

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
        """Add memory.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            memory_type (MemoryType): IN: memory type. OUT: Consumed during execution.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
            context (dict | None, optional): IN: context. Defaults to None. OUT: Consumed during execution.
            importance_score (float, optional): IN: importance score. Defaults to 0.5. OUT: Consumed during execution.
            tags (list | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            MemoryEntry: OUT: Result of the operation."""

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
        """Retrieve memories.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_types (list[MemoryType] | None, optional): IN: memory types. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            tags (list | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            min_importance (float, optional): IN: min importance. Defaults to 0.0. OUT: Consumed during execution.
            query_embedding (object, optional): IN: query embedding. Defaults to None. OUT: Consumed during execution.
            memory_type (MemoryType | None, optional): IN: memory type. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[MemoryEntry]: OUT: Result of the operation."""

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
        """Retrieve recent.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            minutes_ago (int, optional): IN: minutes ago. Defaults to 60. OUT: Consumed during execution.
        Returns:
            list[MemoryEntry]: OUT: Result of the operation."""

        cutoff = datetime.now() - timedelta(minutes=minutes_ago)
        recent = [entry for entries in self.memories.values() for entry in entries if entry.timestamp >= cutoff]
        recent.sort(key=lambda item: item.timestamp, reverse=True)
        return recent

    def clear_memories(
        self,
        memory_type: MemoryType | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Clear memories.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_type (MemoryType | None, optional): IN: memory type. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution."""

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
        """Consolidate memories.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            merge_similar (bool, optional): IN: merge similar. Defaults to True. OUT: Consumed during execution.
            threshold (float, optional): IN: threshold. Defaults to 0.7. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Retrieve the statistics.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict: OUT: Result of the operation."""

        stats = super().get_statistics()
        stats["total_memories"] = sum(len(entries) for entries in self.memories.values())
        stats["cache_hit_rate"] = 0.0
        return stats

    def _enforce_limit(self, memory_type: MemoryType) -> None:
        """Internal helper to enforce limit.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_type (MemoryType): IN: memory type. OUT: Consumed during execution."""

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
        """Internal helper to sync underlying stores.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            entry (MemoryEntry): IN: entry. OUT: Consumed during execution.
            memory_type (MemoryType): IN: memory type. OUT: Consumed during execution."""

        if memory_type in {MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL}:
            self.long_term._items.append(entry)
            self.long_term._index[entry.memory_id] = entry
        else:
            self.short_term._items.append(entry)
            self.short_term._index[entry.memory_id] = entry
        self._items.append(entry)
        self._index[entry.memory_id] = entry

    def _rebuild_underlying_stores(self) -> None:
        """Internal helper to rebuild underlying stores.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self.short_term._items = []
        self.short_term._index = {}
        self.long_term._items = []
        self.long_term._index = {}
        self._items = []
        self._index = {}

        for memory_type, entries in self.memories.items():
            for entry in entries:
                self._sync_underlying_stores(entry, memory_type)
