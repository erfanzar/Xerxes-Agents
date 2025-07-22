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

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np


class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


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


class MemoryStore:
    """Manages different types of memory with retrieval capabilities"""

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
