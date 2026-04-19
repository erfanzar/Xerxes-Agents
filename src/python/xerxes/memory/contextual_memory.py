# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Contextual memory for maintaining conversation and task context.

Provides a hybrid memory system that combines short-term and long-term
storage with context-aware retrieval and automatic memory promotion
based on access patterns and importance scores.
"""

from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory


class ContextualMemory(Memory):
    """Hybrid memory combining short-term and long-term memories.

    Provides context-aware retrieval and automatic promotion of memories
    from short-term to long-term based on access patterns and importance
    scores. Maintains a context stack for tracking conversation state.

    The context stack allows callers to push and pop situational contexts
    (e.g. task, conversation) that influence how search results are ranked.
    Items whose metadata context matches the current stack context receive
    a relevance boost during search.

    Attributes:
        short_term: The underlying :class:`ShortTermMemory` instance used for
            recent, transient memories.
        long_term: The underlying :class:`LongTermMemory` instance used for
            persisted, important memories.
        promotion_threshold: Number of accesses required before a short-term
            memory item is promoted to long-term storage.
        importance_threshold: Minimum importance score (0.0--1.0) at which a
            new memory is saved directly to long-term storage.
        context_stack: Ordered list of context dictionaries representing the
            current situational context, used for re-ranking search results.

    Example:
        >>> from xerxes.memory import ContextualMemory
        >>> mem = ContextualMemory(short_term_capacity=50)
        >>> mem.push_context("task", {"name": "research"})
        >>> item = mem.save("Important finding", importance=0.9)
        >>> results = mem.search("finding")
    """

    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_storage: Any | None = None,
        promotion_threshold: int = 3,
        importance_threshold: float = 0.7,
    ) -> None:
        """Initialize contextual memory with short-term and long-term sub-stores.

        Args:
            short_term_capacity: Maximum number of items the short-term memory
                can hold before oldest items are evicted (FIFO).
            long_term_storage: Optional storage backend passed to the
                :class:`LongTermMemory` instance for persistence. When ``None``,
                ``LongTermMemory`` uses its default SQLite storage.
            promotion_threshold: Number of times a short-term item must be
                accessed before it is automatically promoted to long-term memory.
            importance_threshold: Minimum importance score (0.0--1.0). Items
                saved with an importance at or above this value bypass short-term
                memory and are stored directly in long-term memory.
        """
        super().__init__()
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage=long_term_storage)
        self.promotion_threshold = promotion_threshold
        self.importance_threshold = importance_threshold
        self.context_stack: list[dict[str, Any]] = []

    def push_context(self, context_type: str, context_data: dict[str, Any]) -> None:
        """Push a new context frame onto the context stack.

        The pushed context influences subsequent :meth:`search` calls by
        boosting the relevance of items whose stored context matches the
        current stack top.

        Args:
            context_type: Category label for the context frame (e.g.
                ``"task"``, ``"conversation"``, ``"tool_use"``).
            context_data: Arbitrary dictionary of context information that
                will be used for re-ranking during search.
        """
        self.context_stack.append(
            {
                "type": context_type,
                "data": context_data,
                "timestamp": datetime.now(),
            }
        )

    def pop_context(self) -> dict[str, Any] | None:
        """Pop and return the most recent context frame from the stack.

        Removes the topmost context so that subsequent search calls no
        longer consider it for re-ranking.

        Returns:
            The popped context dictionary containing ``"type"``, ``"data"``,
            and ``"timestamp"`` keys, or ``None`` if the stack is empty.
        """
        return self.context_stack.pop() if self.context_stack else None

    def get_current_context(self) -> dict[str, Any] | None:
        """Peek at the topmost context frame without removing it.

        Returns:
            The current context dictionary containing ``"type"``, ``"data"``,
            and ``"timestamp"`` keys, or ``None`` if the stack is empty.
        """
        return self.context_stack[-1] if self.context_stack else None

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        to_long_term: bool = False,
        **kwargs,
    ) -> MemoryItem:
        """Save a memory item with automatic context attachment.

        The current context stack (if non-empty) is attached to the item's
        metadata under the ``"context"`` key. Items with high importance or
        the ``to_long_term`` flag are stored directly in long-term memory;
        otherwise they go to short-term memory and are checked for promotion.

        Args:
            content: Text content to store.
            metadata: Optional key-value metadata to attach to the item.
            importance: Importance score (0.0--1.0). Items at or above
                :attr:`importance_threshold` are routed to long-term memory.
            to_long_term: When ``True``, forces storage in long-term memory
                regardless of the importance score.
            **kwargs: Additional keyword arguments forwarded to the underlying
                memory store's ``save`` method (e.g. ``agent_id``, ``user_id``).

        Returns:
            The newly created :class:`MemoryItem`.
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
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        search_long_term: bool = True,
        **kwargs,
    ) -> list[MemoryItem]:
        """Search both short-term and long-term memories.

        Searches short-term memory first, then (optionally) long-term memory.
        Each result is annotated with a ``"source"`` metadata key indicating
        which store it came from. If a context stack is active, results are
        re-ranked to boost items that match the current context before the
        final relevance sort.

        Args:
            query: Natural-language or keyword search query string.
            limit: Maximum number of results to return after merging and
                ranking results from both stores.
            filters: Optional key-value filter criteria forwarded to the
                underlying memory stores.
            search_long_term: When ``True`` (default), includes long-term
                memory in the search. Set to ``False`` to restrict results
                to short-term memory only.
            **kwargs: Additional keyword arguments forwarded to the
                underlying stores' ``search`` methods.

        Returns:
            List of :class:`MemoryItem` instances sorted by relevance score
            in descending order, with at most ``limit`` entries.
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
        """Retrieve memory items from both short-term and long-term stores.

        When a specific ``memory_id`` is provided, looks in short-term memory
        first, then falls back to long-term memory. Accessing an item in
        short-term memory may trigger automatic promotion to long-term memory
        if it has been accessed enough times.

        When ``memory_id`` is ``None``, performs a filter-based retrieval across
        both stores, filling up to ``limit`` items from short-term first and
        then from long-term.

        Args:
            memory_id: UUID of a specific memory item to retrieve. When
                provided, other parameters are ignored.
            filters: Optional key-value criteria for bulk retrieval. Applied
                to both short-term and long-term stores.
            limit: Maximum total number of items to return when using
                filter-based retrieval.

        Returns:
            A single :class:`MemoryItem` when ``memory_id`` is provided
            (or ``None`` if not found), or a list of :class:`MemoryItem`
            instances when performing filter-based retrieval.
        """
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
        """Update a memory item in the appropriate sub-store.

        Attempts the update in short-term memory first. If the item is not
        found there, falls back to long-term memory.

        Args:
            memory_id: UUID of the memory item to update.
            updates: Dictionary mapping attribute names to their new values.
                Supported keys include any :class:`MemoryItem` attribute
                (e.g. ``"content"``, ``"metadata"``).

        Returns:
            ``True`` if the item was found and updated in either store,
            ``False`` if the ``memory_id`` was not found in either store.
        """

        if self.short_term.update(memory_id, updates):
            return True

        return self.long_term.update(memory_id, updates)

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete memory items from both short-term and long-term stores.

        The deletion is applied to both stores regardless of where the item
        actually resides, and the counts are summed.

        Args:
            memory_id: UUID of a specific memory item to delete. When
                provided, the item is removed from whichever store holds it.
            filters: Optional key-value criteria for bulk deletion across
                both stores.

        Returns:
            Total number of items deleted across both stores.
        """
        count = 0
        count += self.short_term.delete(memory_id, filters)
        count += self.long_term.delete(memory_id, filters)
        return count

    def clear(self) -> None:
        """Clear all memories from both stores and reset the context stack."""
        self.short_term.clear()
        self.long_term.clear()
        self.context_stack.clear()

    def get_context_summary(self) -> str:
        """Build a human-readable summary of current context and recent memories.

        Combines up to three sections:

        1. **Current context** -- the last three entries on the context stack.
        2. **Recent activity** -- the five most recent short-term memory items.
        3. **Important memories** -- long-term items with importance >= 0.8.

        Returns:
            Multi-line formatted summary string. Returns
            ``"No context available."`` when all sections are empty.
        """
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
        """Check whether a short-term item qualifies for long-term promotion.

        An item is promoted when its :attr:`~MemoryItem.access_count` meets or
        exceeds :attr:`promotion_threshold`. On promotion, a copy of the item
        is saved to long-term memory and the original item's metadata is
        annotated with ``"promoted": True``.

        Args:
            item: The short-term :class:`MemoryItem` to evaluate.
        """
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
        """Re-rank search results by similarity to the current context.

        Adjusts each item's :attr:`~MemoryItem.relevance_score` using a
        weighted blend of the original score (70 %) and a context-match
        component (30 %). Context matching awards up to 0.5 for a matching
        context type and up to 0.5 for word overlap between the item's and
        the current context's data fields.

        Args:
            results: List of :class:`MemoryItem` instances to re-rank.
                Items are modified in place.

        Returns:
            The same list of items with updated relevance scores. If no
            context is active, the list is returned unchanged.
        """
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
