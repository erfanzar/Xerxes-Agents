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
"""Core memory primitives shared by every tier.

Defines ``MemoryItem`` — the unit of recall persisted across tiers —
and ``Memory``, the abstract base class implemented by short-term,
long-term, entity, contextual, and user-scoped memory stores."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class MemoryItem:
    """A single recallable record stored in any memory tier.

    Attributes:
        content: Free-form text payload (the recallable substance).
        memory_type: Tier or category label (e.g. ``"short_term"``,
            ``"long_term"``, ``"entity"``, ``"turn"``).
        timestamp: Wall-clock creation time.
        metadata: Arbitrary structured annotations (importance, tags,
            source, context, etc.).
        agent_id: Identifier of the agent that produced the memory.
        task_id: Optional task identifier tying the item to work-in-flight.
        conversation_id: Identifier of the conversation it belongs to.
        user_id: Identifier of the user the memory belongs to.
        relevance_score: Mutable score set by retrievers/searchers; not
            persisted semantics, just last-computed relevance.
        access_count: How many times the item has been retrieved; used
            for promotion and recency-aware eviction.
        last_accessed: Wall-clock time of the most recent retrieval.
        embedding: Optional cached dense vector for semantic search.
        memory_id: Stable UUID assigned on creation."""

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
        """Serialise the item to a JSON-safe dict for persistence.

        Timestamps are emitted as ISO 8601 strings. The embedding is
        intentionally omitted — embeddings live in vector storage."""

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
        """Reconstruct an item from its ``to_dict`` form.

        Parses ISO timestamp fields back into ``datetime`` objects.
        Unknown fields are passed through to the dataclass constructor."""

        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class Memory(ABC):
    """Abstract memory tier with a shared in-memory item buffer.

    Concrete subclasses (``ShortTermMemory``, ``LongTermMemory``,
    ``EntityMemory``, ``ContextualMemory``) implement ``save``,
    ``search``, ``retrieve``, ``update``, ``delete``, and ``clear``.
    They share ``_items`` (the ordered buffer) and ``_index`` (memory_id
    lookup table) so the base class can implement context formatting
    and statistics."""

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int | None = None,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialise common tier state.

        Args:
            storage: Optional backing store (``MemoryStorage`` subclass)
                used by subclasses for durable persistence.
            max_items: Soft cap on stored items; ``None`` means unbounded.
            enable_embeddings: Whether the tier should compute/store
                embeddings for semantic search."""

        self.storage = storage
        self.max_items = max_items
        self.enable_embeddings = enable_embeddings
        self._items: MutableSequence[MemoryItem] = []
        self._index: dict[str, MemoryItem] = {}

    @abstractmethod
    def save(self, content: str, metadata: dict[str, Any] | None = None, **kwargs) -> MemoryItem:
        """Persist a new memory item and return it.

        Args:
            content: The text payload to remember.
            metadata: Optional structured annotations to attach.
            **kwargs: Tier-specific keyword arguments (agent_id,
                user_id, importance, etc.)."""

        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        """Search the tier for items matching ``query``.

        Args:
            query: Free-form search string.
            limit: Maximum number of results.
            filters: Field-level filters applied before scoring.
            **kwargs: Tier-specific options (e.g. ``use_semantic``,
                ``min_relevance``, ``entity_filter``)."""

        pass

    @abstractmethod
    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Fetch one item by id or a filtered list.

        When ``memory_id`` is given, returns that item or ``None``.
        Otherwise returns up to ``limit`` items matching ``filters``."""

        pass

    @abstractmethod
    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Apply field-level updates to an existing item.

        Returns True when the item exists and was updated, False
        otherwise."""

        pass

    @abstractmethod
    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete a specific item or every item matching ``filters``.

        Returns the number of items removed."""

        pass

    @abstractmethod
    def clear(self) -> None:
        """Drop every item from this tier (including durable storage)."""

        pass

    def get_context(self, limit: int = 10, format_type: str = "text") -> str:
        """Render the most recent ``limit`` items as a context block.

        Args:
            limit: Number of trailing items to include.
            format_type: ``"text"`` (default, agent-tagged lines),
                ``"markdown"`` (timestamped bullet list), or ``"json"``
                (pretty-printed array of ``to_dict`` outputs)."""

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
        """Summarise tier contents — counts per type, unique agents/users/conversations."""

        stats: dict[str, Any] = {
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
        """Return the number of items currently held in the tier."""

        return len(self._items)

    def __repr__(self) -> str:
        """Return a debug repr including item count and capacity."""

        return f"{self.__class__.__name__}(items={len(self._items)}, max={self.max_items})"
