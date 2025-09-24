
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

"""Base memory classes for Calute memory system"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class MemoryItem:
    """Individual memory item with comprehensive metadata"""

    content: str
    memory_type: str = "general"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_id: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    relevance_score: float = 1.0
    access_count: int = 0
    last_accessed: datetime | None = None
    embedding: list[float] | None = None
    memory_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert memory item to dictionary"""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """Create memory item from dictionary"""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class Memory(ABC):
    """Abstract base class for memory implementations"""

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int | None = None,
        enable_embeddings: bool = False,
    ):
        """
        Initialize memory.

        Args:
            storage: Storage backend for persistence
            max_items: Maximum number of items to store
            enable_embeddings: Whether to compute embeddings for semantic search
        """
        self.storage = storage
        self.max_items = max_items
        self.enable_embeddings = enable_embeddings
        self._items: list[MemoryItem] = []
        self._index: dict[str, MemoryItem] = {}

    @abstractmethod
    def save(self, content: str, metadata: dict[str, Any] | None = None, **kwargs) -> MemoryItem:
        """Save a memory item"""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        """Search for relevant memories"""
        pass

    @abstractmethod
    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Retrieve specific memory items"""
        pass

    @abstractmethod
    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory item"""
        pass

    @abstractmethod
    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete memory items"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories"""
        pass

    def get_context(self, limit: int = 10, format_type: str = "text") -> str:
        """
        Get formatted context from memories.

        Args:
            limit: Number of recent items to include
            format_type: Format type (text, json, markdown)

        Returns:
            Formatted context string
        """
        items = self._items[-limit:] if len(self._items) > limit else self._items

        if format_type == "json":
            import json

            return json.dumps([item.to_dict() for item in items], indent=2)
        elif format_type == "markdown":
            lines = []
            for item in items:
                timestamp = item.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                agent = f"**{item.agent_id}**" if item.agent_id else "**System**"
                lines.append(f"- [{timestamp}] {agent}: {item.content}")
            return "\n".join(lines)
        else:
            lines = []
            for item in items:
                if item.agent_id:
                    lines.append(f"[{item.agent_id}]: {item.content}")
                else:
                    lines.append(item.content)
            return "\n".join(lines)

    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_items": len(self._items),
            "max_items": self.max_items,
            "memory_types": {},
            "agents": set(),
            "users": set(),
            "conversations": set(),
        }

        for item in self._items:
            stats["memory_types"][item.memory_type] = stats["memory_types"].get(item.memory_type, 0) + 1

            if item.agent_id:
                stats["agents"].add(item.agent_id)
            if item.user_id:
                stats["users"].add(item.user_id)
            if item.conversation_id:
                stats["conversations"].add(item.conversation_id)

        stats["unique_agents"] = len(stats["agents"])
        stats["unique_users"] = len(stats["users"])
        stats["unique_conversations"] = len(stats["conversations"])
        del stats["agents"], stats["users"], stats["conversations"]

        return stats

    def __len__(self) -> int:
        """Get number of memory items"""
        return len(self._items)

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(items={len(self._items)}, max={self.max_items})"
