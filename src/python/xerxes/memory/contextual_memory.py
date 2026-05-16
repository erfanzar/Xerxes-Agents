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
"""Two-tier memory that combines short-term recall with long-term storage.

``ContextualMemory`` owns a ``ShortTermMemory`` deque and a
``LongTermMemory`` backed by persistent storage. Items can be routed
to either tier based on importance, and short-term items are
auto-promoted once their access count crosses ``promotion_threshold``.
A context stack allows callers to push/pop situational frames that are
mixed into search results."""

from datetime import datetime
from typing import Any

from .base import Memory, MemoryItem
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory


class ContextualMemory(Memory):
    """Composite short-term + long-term memory with a runtime context stack."""

    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_storage: Any | None = None,
        promotion_threshold: int = 3,
        importance_threshold: float = 0.7,
    ) -> None:
        """Wire up the two underlying tiers and tuning thresholds.

        Args:
            short_term_capacity: Maximum items held in the deque-backed
                short-term tier.
            long_term_storage: Optional ``MemoryStorage`` passed through
                to the long-term tier (defaults to its own SQLite).
            promotion_threshold: Number of accesses after which a
                short-term item is copied into long-term storage.
            importance_threshold: Importance score above which a saved
                item bypasses short-term and goes straight to long-term."""

        super().__init__()
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage=long_term_storage)
        self.promotion_threshold = promotion_threshold
        self.importance_threshold = importance_threshold
        self.context_stack: list[dict[str, Any]] = []

    def push_context(self, context_type: str, context_data: dict[str, Any]) -> None:
        """Push a labelled context frame onto the situational stack.

        Frames are timestamped and consulted during ``save`` (attached
        to metadata) and ``search`` (used for relevance reranking)."""

        self.context_stack.append(
            {
                "type": context_type,
                "data": context_data,
                "timestamp": datetime.now(),
            }
        )

    def pop_context(self) -> dict[str, Any] | None:
        """Pop and return the most recent context frame, or ``None`` if empty."""

        return self.context_stack.pop() if self.context_stack else None

    def get_current_context(self) -> dict[str, Any] | None:
        """Return the top context frame without removing it."""

        return self.context_stack[-1] if self.context_stack else None

    def save(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        to_long_term: bool = False,
        **kwargs,
    ) -> MemoryItem:
        """Save an item, routing to long-term when important or requested.

        Items go directly to long-term when ``to_long_term`` is True or
        ``importance >= importance_threshold``. Otherwise they enter
        short-term and may be auto-promoted on repeated access.

        Args:
            content: Text payload to remember.
            metadata: Annotations (importance + current context are
                merged in automatically).
            importance: Importance score in ``[0, 1]`` used for routing.
            to_long_term: Force long-term placement irrespective of
                importance.
            **kwargs: Forwarded to the underlying tier's ``save``."""

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
        """Search both tiers and rerank by current context.

        Each result is tagged with ``metadata["source"]`` (``short_term``
        or ``long_term``). When a context frame is active, scores are
        blended with a context-overlap factor before sorting.

        Args:
            query: Free-form search string.
            limit: Maximum number of merged results.
            filters: Field-level filters passed to each tier.
            search_long_term: When False, only short-term is queried.
            **kwargs: Passed through to each tier's ``search``."""

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
        """Retrieve by id or filtered list across both tiers.

        When ``memory_id`` is provided, short-term is checked first;
        a short-term hit triggers a promotion check before being
        returned. When ``memory_id`` is omitted, items from both tiers
        are concatenated up to ``limit``."""

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
        """Apply updates to the matching item in whichever tier owns it."""

        if self.short_term.update(memory_id, updates):
            return True

        return self.long_term.update(memory_id, updates)

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete from both tiers; returns the combined deletion count."""

        count = 0
        count += self.short_term.delete(memory_id, filters)
        count += self.long_term.delete(memory_id, filters)
        return count

    def clear(self) -> None:
        """Empty both tiers and the context stack."""

        self.short_term.clear()
        self.long_term.clear()
        self.context_stack.clear()

    def get_context_summary(self) -> str:
        """Render a human-readable status block of context + recent + important memories.

        Combines the trailing three context frames, the five most
        recent short-term items, and up to three high-importance
        long-term items."""

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
        """Copy the item into long-term storage once its access count crosses the threshold."""

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
        """Blend each item's relevance with a context-overlap factor."""

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
