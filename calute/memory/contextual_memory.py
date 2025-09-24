
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

"""Contextual memory for maintaining conversation and task context"""

from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory


class ContextualMemory(Memory):
    """
    Hybrid memory combining short-term and long-term memories
    with context-aware retrieval and automatic promotion.
    """

    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_storage=None,
        promotion_threshold: int = 3,
        importance_threshold: float = 0.7,
    ):
        """
        Initialize contextual memory.

        Args:
            short_term_capacity: Capacity of short-term memory
            long_term_storage: Storage for long-term memories
            promotion_threshold: Access count to promote to long-term
            importance_threshold: Importance score to auto-promote
        """
        super().__init__()
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage=long_term_storage)
        self.promotion_threshold = promotion_threshold
        self.importance_threshold = importance_threshold
        self.context_stack: list[dict[str, Any]] = []

    def push_context(self, context_type: str, context_data: dict[str, Any]):
        """Push a new context onto the stack"""
        self.context_stack.append(
            {
                "type": context_type,
                "data": context_data,
                "timestamp": datetime.now(),
            }
        )

    def pop_context(self) -> dict[str, Any] | None:
        """Pop the most recent context from stack"""
        return self.context_stack.pop() if self.context_stack else None

    def get_current_context(self) -> dict[str, Any] | None:
        """Get current context without removing it"""
        return self.context_stack[-1] if self.context_stack else None

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        to_long_term: bool = False,
        **kwargs,
    ) -> MemoryItem:
        """
        Save memory with automatic context attachment.

        Args:
            content: Content to save
            metadata: Additional metadata
            importance: Importance score
            to_long_term: Force save to long-term
            **kwargs: Additional fields

        Returns:
            Created memory item
        """
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
        self, query: str, limit: int = 10, filters: dict[str, Any] | None = None, search_long_term: bool = True, **kwargs
    ) -> list[MemoryItem]:
        """
        Search both short-term and long-term memories.

        Args:
            query: Search query
            limit: Maximum results
            filters: Filter criteria
            search_long_term: Include long-term memory
            **kwargs: Additional arguments

        Returns:
            Combined and ranked results
        """
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
        """Retrieve from both memory stores"""
        if memory_id:
            item = self.short_term.retrieve(memory_id)
            if item:
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
        """Update in appropriate memory store"""

        if self.short_term.update(memory_id, updates):
            return True

        return self.long_term.update(memory_id, updates)

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete from both stores"""
        count = 0
        count += self.short_term.delete(memory_id, filters)
        count += self.long_term.delete(memory_id, filters)
        return count

    def clear(self) -> None:
        """Clear all memories and context"""
        self.short_term.clear()
        self.long_term.clear()
        self.context_stack.clear()

    def get_context_summary(self) -> str:
        """Get a summary of current context and recent memories"""
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

        important = self.long_term.search(query="", limit=3, filters={"importance": lambda x: x >= 0.8})
        if important:
            lines.append("\nImportant memories:")
            for item in important:
                lines.append(f"  - {item.content[:100]}")

        return "\n".join(lines) if lines else "No context available."

    def _check_promotion(self, item: MemoryItem):
        """Check if item should be promoted to long-term memory"""
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
        """Re-rank results based on current context"""
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
