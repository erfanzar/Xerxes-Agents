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

import hashlib
import heapq
import json
import pickle
import typing as tp
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type:ignore

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

    def cosine_similarity(X, Y):
        """Simple cosine similarity implementation."""
        X = np.array(X)
        Y = np.array(Y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return np.dot(X_norm, Y_norm.T)


class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""

    content: str
    timestamp: datetime
    memory_type: MemoryType
    agent_id: str
    context: dict = field(default_factory=dict)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    embedding: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type.value,
            "agent_id": self.agent_id,
            "context": self.context,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "tags": self.tags,
        }


class BasicMemoryStore:
    """[DEPRECATED] Basic memory store - use MemoryStore instead.
    Manages different types of memory with retrieval capabilities"""

    def __init__(self, max_short_term: int = 10, max_working: int = 5):
        self.memories: dict[MemoryType, list[MemoryEntry]] = {mem_type: [] for mem_type in MemoryType}
        self.max_short_term = max_short_term
        self.max_working = max_working
        self.memory_index: dict[str, MemoryEntry] = {}

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: str,
        context: dict | None = None,
        importance_score: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Add a new memory entry"""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            agent_id=agent_id,
            context=context or {},
            importance_score=importance_score,
            tags=tags or [],
        )

        self.memories[memory_type].append(entry)
        memory_id = f"{memory_type.value}_{len(self.memories[memory_type])}"
        self.memory_index[memory_id] = entry

        # Manage memory limits
        self._manage_memory_limits(memory_type)

        return entry

    def _manage_memory_limits(self, memory_type: MemoryType):
        """Enforce memory limits and promote/consolidate as needed"""
        if memory_type == MemoryType.SHORT_TERM:
            if len(self.memories[MemoryType.SHORT_TERM]) > self.max_short_term:
                # Promote important memories to long-term
                memories = sorted(
                    self.memories[MemoryType.SHORT_TERM],
                    key=lambda m: m.importance_score * (m.access_count + 1),
                    reverse=True,
                )

                # Move top memories to long-term
                for mem in memories[:3]:
                    mem.memory_type = MemoryType.LONG_TERM
                    self.memories[MemoryType.LONG_TERM].append(mem)

                # Keep only recent memories in short-term
                self.memories[MemoryType.SHORT_TERM] = memories[3 : self.max_short_term]

        elif memory_type == MemoryType.WORKING:
            if len(self.memories[MemoryType.WORKING]) > self.max_working:
                # Keep only most recent working memories
                self.memories[MemoryType.WORKING] = sorted(
                    self.memories[MemoryType.WORKING], key=lambda m: m.timestamp, reverse=True
                )[: self.max_working]

    def get_relevant_memories(self, agent_id: str, limit: int = 10) -> list[MemoryEntry]:
        """Get relevant memories for an agent"""
        relevant = []

        # Collect memories from all memory types for this agent
        for memory_type in MemoryType:
            agent_memories = [m for m in self.memories[memory_type] if m.agent_id == agent_id]
            relevant.extend(agent_memories)

        # Sort by importance score and timestamp
        relevant.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)

        # Update access count and last accessed
        for memory in relevant[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()

        return relevant[:limit]

    def retrieve_memories(
        self,
        memory_types: list[MemoryType] | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve memories based on criteria"""
        memory_types = memory_types or list(MemoryType)
        results = []

        for mem_type in memory_types:
            memories = self.memories[mem_type]

            if agent_id:
                memories = [m for m in memories if m.agent_id == agent_id]

            if tags:
                memories = [m for m in memories if any(tag in m.tags for tag in tags)]

            memories = [m for m in memories if m.importance_score >= min_importance]

            results.extend(memories)

        now = datetime.now()
        results.sort(
            key=lambda m: (
                m.importance_score * (1 / (1 + (now - m.timestamp).total_seconds() / 3600)) * (m.access_count + 1)
            ),
            reverse=True,
        )

        # Update access counts
        for mem in results[:limit]:
            mem.access_count += 1
            mem.last_accessed = now

        return results[:limit]

    def consolidate_memories(self, agent_id: str) -> str:
        """Consolidate memories into a summary"""
        short_term = self.retrieve_memories(memory_types=[MemoryType.SHORT_TERM], agent_id=agent_id, limit=5)

        working = self.retrieve_memories(memory_types=[MemoryType.WORKING], agent_id=agent_id, limit=3)

        summary_parts = []

        if working:
            summary_parts.append("Current Focus:")
            for mem in working:
                summary_parts.append(f"- {mem.content}")

        if short_term:
            summary_parts.append("\nRecent Context:")
            for mem in short_term:
                summary_parts.append(f"- {mem.content}")

        return "\n".join(summary_parts)

    def save_to_file(self, filepath: Path):
        """Save memory store to file"""
        data = {mem_type.value: [mem.to_dict() for mem in memories] for mem_type, memories in self.memories.items()}

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: Path):
        """Load memory store from file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        for mem_type_str, memories_data in data.items():
            mem_type = MemoryType(mem_type_str)
            for mem_data in memories_data:
                entry = MemoryEntry(
                    content=mem_data["content"],
                    timestamp=datetime.fromisoformat(mem_data["timestamp"]),
                    memory_type=mem_type,
                    agent_id=mem_data["agent_id"],
                    context=mem_data["context"],
                    importance_score=mem_data["importance_score"],
                    access_count=mem_data["access_count"],
                    last_accessed=datetime.fromisoformat(mem_data["last_accessed"])
                    if mem_data["last_accessed"]
                    else None,
                    tags=mem_data["tags"],
                )
                self.memories[mem_type].append(entry)


class MemoryIndex:
    """Index structure for fast memory retrieval."""

    def __init__(self):
        self.by_id: dict[str, MemoryEntry] = {}
        self.by_agent: dict[str, set[str]] = defaultdict(set)
        self.by_type: dict[MemoryType, set[str]] = defaultdict(set)
        self.by_tag: dict[str, set[str]] = defaultdict(set)
        self.by_timestamp: list[tuple[datetime, str]] = []
        self.by_importance: list[tuple[float, str]] = []

        # Vector index for semantic search
        self.embeddings: dict[str, np.ndarray] = {}
        self.embedding_index: Any | None = None  # For FAISS or similar


@dataclass
class MemoryEntry:
    """Enhanced memory entry with additional metadata and indexing."""

    id: str
    content: str
    timestamp: datetime
    memory_type: MemoryType
    agent_id: str
    context: dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    embedding: np.ndarray | None = None
    tags: list[str] = field(default_factory=list)

    # Additional metadata
    source: str | None = None
    confidence: float = 1.0
    decay_rate: float = 0.01
    relations: list[str] = field(default_factory=list)  # Related memory IDs
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make entry hashable for set operations."""
        return hash(self.id)

    def get_current_importance(self) -> float:
        """Calculate current importance with decay."""
        time_diff = (datetime.now() - self.timestamp).total_seconds() / 3600  # Hours
        decayed_importance = self.importance_score * np.exp(-self.decay_rate * time_diff)
        access_bonus = min(self.access_count * 0.01, 0.2)  # Cap at 0.2
        return min(decayed_importance + access_bonus, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type.value,
            "agent_id": self.agent_id,
            "context": self.context,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "relations": self.relations,
            "metadata": self.metadata,
        }


class MemoryStore:
    """Enhanced memory store with indexing, caching, and vector search."""

    def __init__(
        self,
        max_short_term: int = 100,
        max_working: int = 10,
        max_long_term: int = 10000,
        enable_vector_search: bool = False,
        embedding_dimension: int = 768,
        enable_persistence: bool = False,
        persistence_path: Path | str | None = None,
        cache_size: int = 100,
    ):
        self.max_short_term = max_short_term
        self.max_working = max_working
        self.max_long_term = max_long_term
        self.enable_vector_search = enable_vector_search
        self.embedding_dimension = embedding_dimension
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Memory storage
        self.memories: dict[MemoryType, deque[MemoryEntry]] = {
            MemoryType.SHORT_TERM: deque(maxlen=max_short_term),
            MemoryType.WORKING: deque(maxlen=max_working),
            MemoryType.LONG_TERM: deque(maxlen=max_long_term),
            MemoryType.EPISODIC: deque(maxlen=max_long_term),
            MemoryType.SEMANTIC: deque(maxlen=max_long_term),
            MemoryType.PROCEDURAL: deque(maxlen=max_long_term),
        }

        # Enhanced indexing
        self.index = MemoryIndex()

        # Caching
        self.cache: dict[str, tp.Any] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        # Load existing memories if persistence is enabled
        if self.enable_persistence and self.persistence_path and self.persistence_path.exists():
            self.load()

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_id: str,
        context: dict[str, Any] | None = None,
        importance_score: float = 0.5,
        tags: list[str] | None = None,
        embedding: np.ndarray | None = None,
        source: str | None = None,
        confidence: float = 1.0,
        relations: list[str] | None = None,
    ) -> MemoryEntry:
        """Add an enhanced memory entry with indexing."""
        # Generate unique ID
        memory_id = self._generate_id(content, agent_id)

        # Create enhanced entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            agent_id=agent_id,
            context=context or {},
            importance_score=importance_score,
            tags=tags or [],
            embedding=embedding,
            source=source,
            confidence=confidence,
            relations=relations or [],
        )

        # Add to storage
        self.memories[memory_type].append(entry)

        # Update indexes
        self._update_indexes(entry)

        # Clear cache
        self._invalidate_cache()

        # Auto-save if persistence is enabled
        if self.enable_persistence:
            self.save()

        return entry

    def retrieve_memories(
        self,
        memory_types: list[MemoryType] | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        query_embedding: np.ndarray | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve memories using enhanced indexing and search."""
        # Check cache
        cache_key = self._get_cache_key(memory_types, agent_id, tags, limit, min_importance)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1

        # Use indexes for fast retrieval
        candidate_ids = set()

        if agent_id:
            candidate_ids.update(self.index.by_agent.get(agent_id, set()))
        else:
            # All memory IDs
            candidate_ids = set(self.index.by_id.keys())

        if memory_types:
            type_ids = set()
            for mem_type in memory_types:
                type_ids.update(self.index.by_type.get(mem_type, set()))
            candidate_ids &= type_ids

        if tags:
            tag_ids = set()
            for tag in tags:
                tag_ids.update(self.index.by_tag.get(tag, set()))
            candidate_ids &= tag_ids

        # Get candidate memories
        candidates = [self.index.by_id[mid] for mid in candidate_ids if mid in self.index.by_id]

        # Filter by importance
        candidates = [m for m in candidates if m.get_current_importance() >= min_importance]

        # Sort by relevance
        candidates.sort(key=lambda m: m.get_current_importance(), reverse=True)

        # Limit results
        results = candidates[:limit]

        # Update access counts
        for mem in results:
            mem.access_count += 1
            mem.last_accessed = datetime.now()

        # Cache results
        self.cache[cache_key] = results
        if len(self.cache) > self.cache_size:
            # Remove oldest cache entry
            self.cache.pop(next(iter(self.cache)))

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get memory store statistics."""
        total = sum(len(mems) for mems in self.memories.values())
        by_type = {mem_type.value: len(mems) for mem_type, mems in self.memories.items()}

        cache_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        return {
            "total_memories": total,
            "by_type": by_type,
            "cache_hit_rate": cache_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "index_size": len(self.index.by_id),
        }

    def save(self):
        """Save memory store to disk."""
        if not self.persistence_path:
            return

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": {},
            "statistics": self.get_statistics(),
        }

        for mem_type, entries in self.memories.items():
            data["memories"][mem_type.value] = [entry.to_dict() for entry in entries]

        with open(self.persistence_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Load memory store from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        with open(self.persistence_path, "rb") as f:
            data = pickle.load(f)

        for mem_type_str, entries_data in data["memories"].items():
            mem_type = MemoryType(mem_type_str)
            for entry_data in entries_data:
                entry = self._entry_from_dict(entry_data)
                self.memories[mem_type].append(entry)
                self._update_indexes(entry)

    def _generate_id(self, content: str, agent_id: str) -> str:
        """Generate unique memory ID."""
        timestamp = datetime.now().isoformat()
        data = f"{content}{agent_id}{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _update_indexes(self, entry: MemoryEntry):
        """Update all indexes with new entry."""
        self.index.by_id[entry.id] = entry
        self.index.by_agent[entry.agent_id].add(entry.id)
        self.index.by_type[entry.memory_type].add(entry.id)

        for tag in entry.tags:
            self.index.by_tag[tag].add(entry.id)

        heapq.heappush(self.index.by_timestamp, (entry.timestamp, entry.id))
        heapq.heappush(self.index.by_importance, (-entry.importance_score, entry.id))

    def consolidate_memories(self, agent_id: str) -> str:
        """Consolidate memories into a summary for an agent"""
        memories = self.retrieve_memories(agent_id=agent_id, limit=20)

        if not memories:
            return ""

        # Group memories by type
        by_type = defaultdict(list)
        for mem in memories:
            by_type[mem.memory_type].append(mem)

        # Build consolidated summary
        summary_parts = []

        # Add long-term memories (most important)
        if MemoryType.LONG_TERM in by_type:
            long_term = by_type[MemoryType.LONG_TERM]
            summary_parts.append("Important facts:")
            for mem in long_term[:5]:
                summary_parts.append(f"- {mem.content}")

        # Add recent short-term memories
        if MemoryType.SHORT_TERM in by_type:
            short_term = by_type[MemoryType.SHORT_TERM]
            summary_parts.append("\nRecent context:")
            for mem in short_term[:3]:
                summary_parts.append(f"- {mem.content}")

        # Add working memory
        if MemoryType.WORKING in by_type:
            working = by_type[MemoryType.WORKING]
            summary_parts.append("\nCurrent tasks:")
            for mem in working[:2]:
                summary_parts.append(f"- {mem.content}")

        return "\n".join(summary_parts)

    def search_memories(self, query: str, agent_id: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        """Search memories by query string"""
        results = []
        query_lower = query.lower()

        # Simple text search
        for memory_list in self.memories.values():
            for entry in memory_list:
                if agent_id and entry.agent_id != agent_id:
                    continue

                # Check if query appears in content or tags
                if query_lower in entry.content.lower() or any(query_lower in tag.lower() for tag in entry.tags):
                    results.append(entry)

        # Sort by relevance (importance score)
        results.sort(key=lambda m: m.importance_score, reverse=True)

        return results[:limit]

    def analyze_memory_patterns(self, agent_id: str) -> dict[str, Any]:
        """Analyze patterns in agent's memories"""
        memories = self.retrieve_memories(agent_id=agent_id, limit=100)

        if not memories:
            return {}

        patterns = {
            "total_memories": len(memories),
            "memory_types": defaultdict(int),
            "common_tags": defaultdict(int),
            "avg_importance": 0,
            "access_patterns": {"most_accessed": None, "least_accessed": None},
        }

        total_importance = 0
        for mem in memories:
            patterns["memory_types"][mem.memory_type.value] += 1
            for tag in mem.tags:
                patterns["common_tags"][tag] += 1
            total_importance += mem.importance_score

        if memories:
            patterns["avg_importance"] = total_importance / len(memories)

            # Find most/least accessed
            sorted_by_access = sorted(memories, key=lambda m: m.access_count)
            if sorted_by_access:
                patterns["access_patterns"]["least_accessed"] = sorted_by_access[0].content[:50]
                patterns["access_patterns"]["most_accessed"] = sorted_by_access[-1].content[:50]

        return patterns

    def _invalidate_cache(self):
        """Clear the cache."""
        self.cache.clear()

    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        # Convert lists to tuples for hashing
        hashable_args = []
        for arg in args:
            if isinstance(arg, list):
                hashable_args.append(tuple(arg) if arg else None)
            elif isinstance(arg, dict):
                hashable_args.append(tuple(sorted(arg.items())) if arg else None)
            else:
                hashable_args.append(arg)
        return str(hash(tuple(hashable_args)))

    def _entry_from_dict(self, data: dict) -> MemoryEntry:
        """Create entry from dictionary."""
        return MemoryEntry(
            id=data["id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memory_type=MemoryType(data["memory_type"]),
            agent_id=data["agent_id"],
            context=data.get("context", {}),
            importance_score=data.get("importance_score", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            tags=data.get("tags", []),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            decay_rate=data.get("decay_rate", 0.01),
            relations=data.get("relations", []),
            metadata=data.get("metadata", {}),
        )
