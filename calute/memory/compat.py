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


"""
Compatibility layer for old memory API.

Provides backward-compatible interfaces for the legacy memory system,
allowing existing code to work with the new contextual memory architecture.
"""

from enum import Enum

from .contextual_memory import ContextualMemory
from .storage import SQLiteStorage


class MemoryType(Enum):
    """
    Memory type enum for backward compatibility.

    Defines the different types of memories that can be stored,
    mapping to the appropriate storage tier in the new system.
    """

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MemoryStore(ContextualMemory):
    """
    Backward compatible MemoryStore that wraps ContextualMemory.

    Provides the old API while using the new memory system internally.
    This allows existing code to continue working without modifications
    while benefiting from the improved memory architecture.
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
        """
        Initialize with backward compatible parameters.

        Args:
            max_short_term: Maximum items in short-term memory
            max_working: Maximum items in working memory (legacy)
            max_long_term: Maximum items in long-term memory (legacy)
            enable_vector_search: Enable vector similarity search (legacy)
            embedding_dimension: Dimension of embeddings (legacy)
            enable_persistence: Enable persistent storage
            persistence_path: Path for SQLite database
            cache_size: Cache size for memory retrieval (legacy)
            memory_type: Default memory type (legacy)
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
        """
        Add memory using old API.

        Args:
            content: Memory content to store
            memory_type: Type of memory (determines storage tier)
            agent_id: ID of the agent creating the memory
            context: Optional context dictionary
            importance_score: Importance score from 0.0 to 1.0
            tags: Optional list of tags for categorization
            **kwargs: Additional fields passed to save

        Returns:
            Created MemoryItem
        """
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
        """
        Retrieve memories using old API.

        Args:
            memory_types: Filter by memory types (legacy, unused)
            agent_id: Filter by agent ID
            tags: Filter by tags (also used as search query)
            limit: Maximum number of results
            min_importance: Minimum importance score filter
            query_embedding: Query embedding vector (legacy, unused)

        Returns:
            List of MemoryItem objects matching criteria
        """
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

    def consolidate_memories(self, agent_id: str, merge_similar: bool = True) -> str:
        """
        Consolidate memories for an agent into a summary.

        Merges similar memories to reduce redundancy, then combines
        important facts and recent context into a formatted string
        suitable for including in agent prompts.

        Args:
            agent_id: ID of the agent to consolidate memories for
            merge_similar: Whether to merge similar long-term memories first

        Returns:
            Formatted summary string of important and recent memories
        """
        # First, run real consolidation on long-term memory to merge duplicates
        if merge_similar:
            self.long_term.consolidate(merge_similar=True)

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
                importance = mem.metadata.get("importance", 0.5)
                summary_parts.append(f"- [{importance:.1f}] {mem.content}")

        if recent:
            summary_parts.append("\nRecent context:")
            for mem in recent:
                if mem not in important:
                    summary_parts.append(f"- {mem.content}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> dict:
        """
        Get memory statistics.

        Returns:
            Dictionary containing memory statistics including
            total_memories and cache_hit_rate (legacy)
        """
        stats = super().get_statistics()

        stats["total_memories"] = len(self.short_term) + len(self.long_term)
        stats["cache_hit_rate"] = 0.0
        return stats


MemoryEntry = None
