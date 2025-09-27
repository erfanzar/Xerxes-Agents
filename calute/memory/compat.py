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


"""Compatibility layer for old memory API"""

from enum import Enum

from .contextual_memory import ContextualMemory
from .storage import SQLiteStorage


class MemoryType(Enum):
    """Memory type enum for backward compatibility"""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MemoryStore(ContextualMemory):
    """
    Backward compatible MemoryStore that wraps ContextualMemory.
    Provides the old API while using the new memory system.
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
    ):
        """Initialize with backward compatible parameters"""
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

        self.max_working = max_working
        self.max_long_term = max_long_term
        self.enable_vector_search = enable_vector_search
        self.embedding_dimension = embedding_dimension
        self.cache_size = cache_size

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: str,
        context: dict | None = None,
        importance_score: float = 0.5,
        tags: list | None = None,
        **kwargs,
    ):
        """Add memory using old API"""
        metadata = context or {}
        if tags:
            metadata["tags"] = tags

        to_long_term = memory_type in [MemoryType.LONG_TERM, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]

        return self.save(
            content=content,
            metadata=metadata,
            importance=importance_score,
            to_long_term=to_long_term,
            agent_id=agent_id,
            **kwargs,
        )

    def retrieve_memories(
        self,
        memory_types: list | None = None,
        agent_id: str | None = None,
        tags: list | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        query_embedding=None,
    ):
        """Retrieve memories using old API"""
        filters = {}
        if agent_id:
            filters["agent_id"] = agent_id
        if tags:
            filters["tags"] = tags

        query = " ".join(tags) if tags else ""
        results = self.search(
            query=query,
            limit=limit,
            filters=filters,
            search_long_term=True,
        )

        filtered = [r for r in results if r.metadata.get("importance", 0.5) >= min_importance]
        return filtered[:limit]

    def consolidate_memories(self, agent_id: str) -> str:
        """Consolidate memories for an agent"""

        filters = {"agent_id": agent_id}
        memories = self.search(query="", limit=20, filters=filters)

        if not memories:
            return ""

        summary_parts = []

        important = [m for m in memories if m.metadata.get("importance", 0.5) >= 0.7]
        recent = memories[:5]

        if important:
            summary_parts.append("Important facts:")
            for mem in important[:5]:
                summary_parts.append(f"- {mem.content}")

        if recent:
            summary_parts.append("\nRecent context:")
            for mem in recent:
                if mem not in important:
                    summary_parts.append(f"- {mem.content}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> dict:
        """Get memory statistics"""
        stats = super().get_statistics()

        stats["total_memories"] = len(self.short_term) + len(self.long_term)
        stats["cache_hit_rate"] = 0.0
        return stats


MemoryEntry = None
