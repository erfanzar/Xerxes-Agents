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
"""Contextual memory module for Xerxes.

Exports:
    - ContextualMemory"""

from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory


class ContextualMemory(Memory):
    """Contextual memory.

    Inherits from: Memory
    """

    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_storage: Any | None = None,
        promotion_threshold: int = 3,
        importance_threshold: float = 0.7,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            short_term_capacity (int, optional): IN: short term capacity. Defaults to 20. OUT: Consumed during execution.
            long_term_storage (Any | None, optional): IN: long term storage. Defaults to None. OUT: Consumed during execution.
            promotion_threshold (int, optional): IN: promotion threshold. Defaults to 3. OUT: Consumed during execution.
            importance_threshold (float, optional): IN: importance threshold. Defaults to 0.7. OUT: Consumed during execution."""

        super().__init__()
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage=long_term_storage)
        self.promotion_threshold = promotion_threshold
        self.importance_threshold = importance_threshold
        self.context_stack: list[dict[str, Any]] = []

    def push_context(self, context_type: str, context_data: dict[str, Any]) -> None:
        """Push context.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            context_type (str): IN: context type. OUT: Consumed during execution.
            context_data (dict[str, Any]): IN: context data. OUT: Consumed during execution."""

        self.context_stack.append(
            {
                "type": context_type,
                "data": context_data,
                "timestamp": datetime.now(),
            }
        )

    def pop_context(self) -> dict[str, Any] | None:
        """Pop context.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any] | None: OUT: Result of the operation."""

        return self.context_stack.pop() if self.context_stack else None

    def get_current_context(self) -> dict[str, Any] | None:
        """Retrieve the current context.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any] | None: OUT: Result of the operation."""

        return self.context_stack[-1] if self.context_stack else None

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        to_long_term: bool = False,
        **kwargs,
    ) -> MemoryItem:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            metadata (dict[str, Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
            importance (float, optional): IN: importance. Defaults to 0.5. OUT: Consumed during execution.
            to_long_term (bool, optional): IN: to long term. Defaults to False. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            MemoryItem: OUT: Result of the operation."""

        metadata = metadata or {}

        if self.context_stack:
            metadata["context"] = self.get_current_context()

        if to_long_term or importance >= self.importance_threshold:
            return self.long_term.save(content=content, metadata=metadata, importance=importance, **kwargs)

        item = self.short_term.save(content=content, metadata=metadata, **kwargs)
        item.metadata["importance"] = importance

        self._check_promotion(item)

        return item

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        search_long_term: bool = True,
        **kwargs,
    ) -> list[MemoryItem]:
        """Search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 10. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
            search_long_term (bool, optional): IN: search long term. Defaults to True. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            list[MemoryItem]: OUT: Result of the operation."""

        results = []

        st_results = self.short_term.search(query=query, limit=limit, filters=filters, **kwargs)
        for item in st_results:
            item.metadata["source"] = "short_term"
        results.extend(st_results)

        if search_long_term:
            lt_results = self.long_term.search(query=query, limit=limit, filters=filters, **kwargs)
            for item in lt_results:
                item.metadata["source"] = "long_term"
            results.extend(lt_results)

        if self.context_stack:
            results = self._rerank_by_context(results)

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

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
            item = self.short_term.retrieve(memory_id)
            if isinstance(item, MemoryItem):
                self._check_promotion(item)
                return item

            return self.long_term.retrieve(memory_id)

        results = []
        st_items = self.short_term.retrieve(filters=filters, limit=limit)
        if st_items:
            results.extend(st_items if isinstance(st_items, list) else [st_items])

        lt_items = self.long_term.retrieve(filters=filters, limit=limit - len(results))
        if lt_items:
            results.extend(lt_items if isinstance(lt_items, list) else [lt_items])

        return results[:limit]

    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str): IN: memory id. OUT: Consumed during execution.
            updates (dict[str, Any]): IN: updates. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if self.short_term.update(memory_id, updates):
            return True

        return self.long_term.update(memory_id, updates)

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            memory_id (str | None, optional): IN: memory id. Defaults to None. OUT: Consumed during execution.
            filters (dict[str, Any] | None, optional): IN: filters. Defaults to None. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

        count = 0
        count += self.short_term.delete(memory_id, filters)
        count += self.long_term.delete(memory_id, filters)
        return count

    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self.short_term.clear()
        self.long_term.clear()
        self.context_stack.clear()

    def get_context_summary(self) -> str:
        """Retrieve the context summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = []

        if self.context_stack:
            lines.append("Current context:")
            for ctx in self.context_stack[-3:]:
                lines.append(f"  - {ctx['type']}: {str(ctx['data'])[:100]}")

        recent = self.short_term.get_recent(5)
        if recent:
            lines.append("\nRecent activity:")
            for item in recent:
                lines.append(f"  - {item.content[:100]}")

        all_lt = self.long_term.search(query="", limit=20)
        important = [item for item in all_lt if item.metadata.get("importance", 0.5) >= 0.8][:3]
        if important:
            lines.append("\nImportant memories:")
            for item in important:
                lines.append(f"  - {item.content[:100]}")

        return "\n".join(lines) if lines else "No context available."

    def _check_promotion(self, item: MemoryItem) -> None:
        """Internal helper to check promotion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            item (MemoryItem): IN: item. OUT: Consumed during execution."""

        if item.access_count >= self.promotion_threshold:
            self.long_term.save(
                content=item.content,
                metadata=item.metadata,
                agent_id=item.agent_id,
                user_id=item.user_id,
                conversation_id=item.conversation_id,
                importance=item.metadata.get("importance", 0.6),
            )

            item.metadata["promoted"] = True

    def _rerank_by_context(self, results: list[MemoryItem]) -> list[MemoryItem]:
        """Internal helper to rerank by context.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            results (list[MemoryItem]): IN: results. OUT: Consumed during execution.
        Returns:
            list[MemoryItem]: OUT: Result of the operation."""

        current_context = self.get_current_context()
        if not current_context:
            return results

        for item in results:
            context_match = 0.0

            item_context = item.metadata.get("context", {})
            if item_context:
                if item_context.get("type") == current_context["type"]:
                    context_match += 0.5

                item_data = str(item_context.get("data", ""))
                current_data = str(current_context.get("data", ""))
                if item_data and current_data:
                    common_words = set(item_data.lower().split()) & set(current_data.lower().split())
                    if common_words:
                        context_match += 0.5 * (
                            len(common_words) / max(len(item_data.split()), len(current_data.split()))
                        )

            item.relevance_score = item.relevance_score * 0.7 + context_match * 0.3

        return results
