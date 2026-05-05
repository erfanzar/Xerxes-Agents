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
"""Short term memory module for Xerxes.

Exports:
    - ShortTermMemory"""

from collections import deque
from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem


class ShortTermMemory(Memory):
    """Short term memory.

    Inherits from: Memory
    """

    def __init__(
        self,
        capacity: int = 20,
        storage: Any | None = None,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capacity (int, optional): IN: capacity. Defaults to 20. OUT: Consumed during execution.
            storage (Any | None, optional): IN: storage. Defaults to None. OUT: Consumed during execution.
            enable_embeddings (bool, optional): IN: enable embeddings. Defaults to False. OUT: Consumed during execution."""

        super().__init__(storage=storage, max_items=capacity, enable_embeddings=enable_embeddings)
        self._items = deque(maxlen=capacity)

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        **kwargs,
    ) -> MemoryItem:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            metadata (dict[str, Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            user_id (str | None, optional): IN: user id. Defaults to None. OUT: Consumed during execution.
            conversation_id (str | None, optional): IN: conversation id. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            MemoryItem: OUT: Result of the operation."""

        metadata = metadata or {}
        metadata.update(kwargs)

        item = MemoryItem(
            content=content,
            memory_type="short_term",
            metadata=metadata,
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        self._items.append(item)
        self._index[item.memory_id] = item

        if self.storage:
            self.storage.save(f"stm_{item.memory_id}", item.to_dict())

        return item

    def search(
        self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, min_relevance: float = 0.0, **kwargs
    ) -> list[MemoryItem]:
        """Search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
            min_relevance (float, optional): IN: min relevance. Defaults to 0.0. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[MemoryItem]: OUT: Result of the operation."""

        query_lower = query.lower()
        matches = []

        for item in reversed(self._items):
            if filters:
                if filters.get("agent_id") and item.agent_id != filters["agent_id"]:
                    continue
                if filters.get("user_id") and item.user_id != filters["user_id"]:
                    continue
                if filters.get("conversation_id") and item.conversation_id != filters["conversation_id"]:
                    continue

            content_lower = item.content.lower()
            relevance = 0.0

            if query_lower in content_lower:
                relevance = 1.0
            else:
                query_words = query_lower.split()
                if query_words:
                    matching = sum(1 for w in query_words if w in content_lower)
                    relevance = matching / len(query_words)

            if relevance >= min_relevance:
                item.relevance_score = relevance
                item.access_count += 1
                item.last_accessed = datetime.now()
                matches.append(item)

                if len(matches) >= limit:
                    break

        matches.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
        return matches

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

        if memory_id:
            item = self._index.get(memory_id)
            if item:
                item.access_count += 1
                item.last_accessed = datetime.now()
            return item

        results = []
        for item in reversed(self._items):
            if filters:
                match = True
                for key, value in filters.items():
                    if hasattr(item, key) and getattr(item, key) != value:
                        match = False
                        break
                if not match:
                    continue

            item.access_count += 1
            item.last_accessed = datetime.now()
            results.append(item)

            if len(results) >= limit:
                break

        return results

    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str): IN: memory id. OUT: Consumed during execution.
            updates (dict[str, Any]): IN: updates. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if memory_id not in self._index:
            return False

        item = self._index[memory_id]
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        if self.storage:
            self.storage.save(f"stm_{memory_id}", item.to_dict())

        return True

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str | None, optional): IN: memory id. Defaults to None. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

        count = 0

        if memory_id:
            if memory_id in self._index:
                item = self._index[memory_id]
                self._items.remove(item)
                del self._index[memory_id]
                if self.storage:
                    self.storage.delete(f"stm_{memory_id}")
                count = 1
        elif filters:
            to_remove = []
            for item in self._items:
                match = True
                for key, value in filters.items():
                    if hasattr(item, key) and getattr(item, key) != value:
                        match = False
                        break
                if match:
                    to_remove.append(item)

            for item in to_remove:
                self._items.remove(item)
                del self._index[item.memory_id]
                if self.storage:
                    self.storage.delete(f"stm_{item.memory_id}")
                count += 1

        return count

    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.storage:
            for item in self._items:
                self.storage.delete(f"stm_{item.memory_id}")

        self._items.clear()
        self._index.clear()

    def get_recent(self, n: int = 5) -> list[MemoryItem]:
        """Retrieve the recent.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            n (int, optional): IN: n. Defaults to 5. OUT: Consumed during execution.
        Returns:
            list[MemoryItem]: OUT: Result of the operation."""

        items = list(self._items)
        return items[-n:] if len(items) > n else items

    def summarize(self) -> str:
        """Summarize.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        if not self._items:
            return "No recent memories."

        summary = ["Recent activity:"]

        conversations: dict[str, list[MemoryItem]] = {}
        for item in self._items:
            conv_id = item.conversation_id or "default"
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(item)

        for conv_id, items in conversations.items():
            if conv_id != "default":
                summary.append(f"\nConversation {conv_id}:")
            for item in items[-3:]:
                agent = item.agent_id or "System"
                summary.append(f"  [{agent}]: {item.content[:100]}")

        return "\n".join(summary)
