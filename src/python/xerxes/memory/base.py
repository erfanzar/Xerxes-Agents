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
"""Base module for Xerxes.

Exports:
    - MemoryItem
    - Memory"""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class MemoryItem:
    """Memory item.

    Attributes:
        content (str): content.
        memory_type (str): memory type.
        timestamp (datetime): timestamp.
        metadata (dict[str, Any]): metadata.
        agent_id (str | None): agent id.
        task_id (str | None): task id.
        conversation_id (str | None): conversation id.
        user_id (str | None): user id.
        relevance_score (float): relevance score.
        access_count (int): access count.
        last_accessed (datetime | None): last accessed.
        embedding (list[float] | None): embedding.
        memory_id (str): memory id."""

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
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, Any]): IN: data. OUT: Consumed during execution.
        Returns:
            'MemoryItem': OUT: Result of the operation."""

        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class Memory(ABC):
    """Memory.

    Inherits from: ABC
    """

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int | None = None,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            storage (Any | None, optional): IN: storage. Defaults to None. OUT: Consumed during execution.
            max_items (int | None, optional): IN: max items. Defaults to None. OUT: Consumed during execution.
            enable_embeddings (bool, optional): IN: enable embeddings. Defaults to False. OUT: Consumed during execution."""

        self.storage = storage
        self.max_items = max_items
        self.enable_embeddings = enable_embeddings
        self._items: MutableSequence[MemoryItem] = []
        self._index: dict[str, MemoryItem] = {}

    @abstractmethod
    def save(self, content: str, metadata: dict[str, Any] | None = None, **kwargs) -> MemoryItem:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            metadata (dict[str, Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            MemoryItem: OUT: Result of the operation."""

        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, **kwargs) -> list[MemoryItem]:
        """Search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[MemoryItem]: OUT: Result of the operation."""

        pass

    @abstractmethod
    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Retrieve.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str | None, optional): IN: memory id. Defaults to None. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
        Returns:
            MemoryItem | list[MemoryItem] | None: OUT: Result of the operation."""

        pass

    @abstractmethod
    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str): IN: memory id. OUT: Consumed during execution.
            updates (dict[str, Any]): IN: updates. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        pass

    @abstractmethod
    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str | None, optional): IN: memory id. Defaults to None. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        pass

    def get_context(self, limit: int = 10, format_type: str = "text") -> str:
        """Retrieve the context.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            format_type (str, optional): IN: format type. Defaults to 'text'. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Retrieve the statistics.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
        """Dunder method for len.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return len(self._items)

    def __repr__(self) -> str:
        """Dunder method for repr.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        return f"{self.__class__.__name__}(items={len(self._items)}, max={self.max_items})"
