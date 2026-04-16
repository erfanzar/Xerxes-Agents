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


"""Base memory classes for Xerxes memory system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class MemoryItem:
    """Individual memory item with comprehensive metadata.

    Attributes:
        content: The text content stored in this memory item.
        memory_type: Category label for the memory (e.g., "general", "short_term").
        timestamp: When the memory was created.
        metadata: Arbitrary key-value data attached to this item.
        agent_id: Identifier of the agent that created this memory.
        task_id: Identifier of the task associated with this memory.
        conversation_id: Identifier of the conversation this memory belongs to.
        user_id: Identifier of the user associated with this memory.
        relevance_score: Search relevance score (0.0-1.0).
        access_count: Number of times this item has been retrieved.
        last_accessed: Timestamp of the most recent access.
        embedding: Optional dense vector embedding for semantic search.
        memory_id: Unique UUID string for this memory item.
    """

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
        """Convert this memory item to a JSON-serialisable dictionary.

        Returns:
            Dictionary representation with all fields, converting datetimes
            to ISO 8601 strings.
        """
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
        """Create a MemoryItem from a dictionary, converting ISO strings to datetimes.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.

        Returns:
            A new MemoryItem populated from the supplied data.
        """
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class Memory(ABC):
    """Abstract base class for all Xerxes memory implementations.

    Subclasses must implement :meth:`save`, :meth:`search`, :meth:`retrieve`,
    :meth:`update`, :meth:`delete`, and :meth:`clear`. The base class provides
    shared infrastructure: an ordered ``_items`` list, an ``_index`` mapping
    memory IDs to items, and the :meth:`get_context` / :meth:`get_statistics`
    convenience methods.
    """

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int | None = None,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialize memory.

        Args:
            storage: Storage backend for persistence.
            max_items: Maximum number of items to store before eviction.
            enable_embeddings: Whether to compute embeddings for semantic search.
        """
        self.storage = storage
        self.max_items = max_items
        self.enable_embeddings = enable_embeddings
        self._items: list[MemoryItem] = []
        self._index: dict[str, MemoryItem] = {}

    @abstractmethod
    def save(self, content: str, metadata: dict[str, Any] | None = None, **kwargs) -> MemoryItem:
        """Save a memory item.

        Args:
            content: Text content to store.
            metadata: Optional key-value metadata to attach.
            **kwargs: Additional implementation-specific fields.

        Returns:
            The newly created MemoryItem.
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        """Search for relevant memories matching a query.

        Args:
            query: Natural-language or keyword query string.
            limit: Maximum number of results to return.
            filters: Optional key-value criteria to narrow results.
            **kwargs: Additional implementation-specific parameters.

        Returns:
            List of matching MemoryItem instances, ordered by relevance.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Retrieve specific memory items by ID or filter criteria.

        Args:
            memory_id: Specific memory UUID to retrieve.
            filters: Optional key-value criteria for bulk retrieval.
            limit: Maximum number of items to return when using filters.

        Returns:
            Single MemoryItem if memory_id is provided and found,
            list of MemoryItem when using filters, or None if not found.
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update an existing memory item with new field values.

        Args:
            memory_id: UUID of the memory item to update.
            updates: Dictionary of attribute names and new values.

        Returns:
            True if the update succeeded, False if the item was not found.
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete memory items by ID or filter criteria.

        Args:
            memory_id: Specific memory UUID to delete.
            filters: Optional key-value criteria for bulk deletion.

        Returns:
            Number of items deleted.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all memory items from this store."""
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
        """Return aggregate statistics about stored memory items.

        Returns:
            Dictionary with keys: total_items, max_items, memory_types (counts
            per type), unique_agents, unique_users, and unique_conversations.
        """
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
        """Return the number of memory items currently stored."""
        return len(self._items)

    def __repr__(self) -> str:
        """Return a concise string representation of this memory store."""
        return f"{self.__class__.__name__}(items={len(self._items)}, max={self.max_items})"
