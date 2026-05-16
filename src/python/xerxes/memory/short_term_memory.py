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
"""In-process short-term memory tier backed by a bounded deque.

``ShortTermMemory`` holds the most recent ``capacity`` items in FIFO
order, drops the oldest automatically once full, and performs lexical
search (substring or token-overlap) over the buffer. It is the hot
path for working context and is paired with ``LongTermMemory`` inside
``ContextualMemory``."""

from collections import deque
from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem


class ShortTermMemory(Memory):
    """Bounded recency-ordered memory tier."""

    def __init__(
        self,
        capacity: int = 20,
        storage: Any | None = None,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialise the deque with the given capacity.

        Args:
            capacity: Maximum items retained; oldest items are evicted
                automatically when the buffer is full.
            storage: Optional ``MemoryStorage`` used to mirror items.
            enable_embeddings: Whether to compute embeddings on save."""

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
        """Append a new short-term item and mirror it to storage if configured.

        Args:
            content: Text payload to remember.
            metadata: Annotations; ``**kwargs`` are folded in.
            agent_id: Producing agent identifier.
            user_id: Owning user identifier.
            conversation_id: Originating conversation identifier.
            **kwargs: Extra annotations merged into ``metadata``."""

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
        """Scan items newest-first, scoring by substring or token overlap.

        Each candidate is checked against simple ``agent_id``/``user_id``/
        ``conversation_id`` filters. Matches receive a ``relevance_score``
        of 1.0 for substring hits or the fraction of query words found
        for partial hits.

        Args:
            query: Free-form search string.
            limit: Maximum results.
            filters: Identity filters (agent/user/conversation).
            min_relevance: Drop items below this score.
            **kwargs: Accepted for protocol compatibility; unused."""

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
        """Fetch by id or by reverse-iterating filtered items up to ``limit``.

        Each returned item has its access counters bumped."""

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
        """Apply field updates and mirror the change to storage."""

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
        """Delete a single item by id or every item matching ``filters``."""

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
        """Drop every item and purge mirrored ``stm_*`` rows from storage."""

        if self.storage:
            for item in self._items:
                self.storage.delete(f"stm_{item.memory_id}")

        self._items.clear()
        self._index.clear()

    def get_recent(self, n: int = 5) -> list[MemoryItem]:
        """Return the ``n`` most recently added items in chronological order."""

        items = list(self._items)
        return items[-n:] if len(items) > n else items

    def summarize(self) -> str:
        """Render a per-conversation block of the trailing three items each."""

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
