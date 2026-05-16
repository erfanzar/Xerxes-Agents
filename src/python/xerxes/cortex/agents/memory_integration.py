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
"""Cortex memory integration for agent context and recall."""

from dataclasses import dataclass
from typing import Any

from ...memory import (
    ContextualMemory,
    EntityMemory,
    LongTermMemory,
    MemoryItem,
    ShortTermMemory,
    SQLiteStorage,
    UserMemory,
)


@dataclass
class CortexMemory:
    """Unified memory layer for Cortex agents.

    Combines short-term, long-term, entity, user, and contextual memory
    subsystems to provide rich context for agent tasks.

    Attributes:
        storage (SQLiteStorage | None): Persistent storage backend.
        short_term (ShortTermMemory | None): Recent interaction buffer.
        long_term (LongTermMemory | None): Durable memory store.
        entity_memory (EntityMemory | None): Entity tracking memory.
        user_memory (UserMemory | None): User-specific memory.
        contextual (ContextualMemory): Cross-memory search interface.
    """

    def __init__(
        self,
        enable_short_term: bool = True,
        enable_long_term: bool = True,
        enable_entity: bool = True,
        enable_user: bool = False,
        persistence_path: str | None = None,
        short_term_capacity: int = 50,
        long_term_capacity: int = 5000,
    ):
        """Build the four-tier memory stack for a Cortex run.

        Persistence is gated by the ``WRITE_MEMORY=1`` environment variable
        so tests and ephemeral runs never accidentally hit disk. When
        persistence is disabled the long-term, entity, and user memories
        still work — they just live in-memory.

        Args:
            enable_short_term: Allocate the recent-context buffer.
            enable_long_term: Allocate the durable knowledge store.
            enable_entity: Allocate the entity tracker.
            enable_user: Allocate per-user memory.
            persistence_path: SQLite database path used when persistence
                is enabled via ``WRITE_MEMORY``.
            short_term_capacity: Items kept in the short-term buffer.
            long_term_capacity: Hard cap on long-term store size.
        """

        import os

        write_memory = os.environ.get("WRITE_MEMORY", "0") == "1"

        self.storage = SQLiteStorage(persistence_path) if (persistence_path and write_memory) else None

        self.short_term = ShortTermMemory(capacity=short_term_capacity) if enable_short_term else None
        self.long_term = LongTermMemory(storage=self.storage, max_items=long_term_capacity) if enable_long_term else None
        self.entity_memory = EntityMemory(storage=self.storage) if enable_entity else None
        self.user_memory = UserMemory(storage=self.storage) if enable_user else None

        self.contextual = ContextualMemory(short_term_capacity=short_term_capacity, long_term_storage=self.storage)

    def build_context_for_task(
        self,
        task_description: str,
        agent_role: str | None = None,
        additional_context: str | None = None,
        max_items: int = 10,
    ) -> str:
        """Assemble a context string from every active memory layer.

        Pulls in (in order): supplied background, recent short-term items,
        agent-filtered long-term hits, extracted entities with their
        frequencies, and the top results from the contextual search.

        Args:
            task_description: Used as the search query and entity source.
            agent_role: Used to filter long-term retrieval by ``agent_id``.
            additional_context: Optional preamble shown verbatim.
            max_items: Cap on the contextual memory search.
        """

        context_parts = []

        if additional_context:
            context_parts.append(f"Background:\n{additional_context}")

        if self.short_term:
            recent = self.short_term.get_recent(n=5)
            if recent:
                context_parts.append("Recent context:")
                for item in recent:
                    context_parts.append(f"  • {item.content[:200]}")

        if self.long_term:
            relevant = self.long_term.search(
                query=task_description, limit=5, filters={"agent_id": agent_role} if agent_role else None
            )
            if relevant:
                context_parts.append("\nRelevant knowledge:")
                for item in relevant:
                    context_parts.append(f"  • {item.content[:200]}")

        if self.entity_memory:
            entities = self.entity_memory._extract_entities(task_description)
            if entities:
                context_parts.append("\nKnown entities:")
                for entity in entities[:5]:
                    info = self.entity_memory.get_entity_info(entity)
                    if info.get("frequency", 0) > 0:
                        context_parts.append(f"  • {entity}: {info.get('frequency')} mentions")

        comprehensive = self.contextual.search(query=task_description, limit=max_items, search_long_term=True)

        if comprehensive:
            context_parts.append("\nRelated memories:")
            for item in comprehensive[:3]:
                context_parts.append(f"  • {item.content[:150]}")

        return "\n".join(context_parts) if context_parts else ""

    def save_task_result(
        self,
        task_description: str,
        result: str,
        agent_role: str,
        importance: float = 0.5,
        task_metadata: dict[str, Any] | None = None,
    ):
        """Persist a completed task result across the memory tiers.

        Always writes to short-term, entity and contextual memory; long-term
        storage is only used when ``importance >= 0.7``.

        Args:
            task_description: Original task statement (stored truncated).
            result: Agent output to persist.
            agent_role: Tagged onto every memory entry for later filtering.
            importance: Score in ``[0, 1]`` gating long-term promotion.
            task_metadata: Extra metadata merged into the entry.
        """

        metadata = task_metadata or {}
        metadata["task"] = task_description[:100]
        metadata["agent_role"] = agent_role

        if self.short_term:
            self.short_term.save(
                content=f"Task completed: {task_description[:100]} - Result: {result[:200]}",
                metadata=metadata,
                agent_id=agent_role,
            )

        if self.long_term and importance >= 0.7:
            self.long_term.save(content=result, metadata=metadata, agent_id=agent_role, importance=importance)

        if self.entity_memory:
            self.entity_memory.save(content=f"{task_description} {result}", metadata=metadata)

        self.contextual.save(content=result, metadata=metadata, importance=importance, agent_id=agent_role)

    def save_agent_interaction(self, agent_role: str, action: str, content: str, importance: float = 0.3):
        """Log a one-line agent interaction (action + content).

        Always writes to short-term memory; promotes to long-term when
        ``importance >= 0.6``.
        """

        interaction = f"[{agent_role}] {action}: {content}"

        if self.short_term:
            self.short_term.save(content=interaction, metadata={"action": action}, agent_id=agent_role)

        if importance >= 0.6 and self.long_term:
            self.long_term.save(content=interaction, agent_id=agent_role, importance=importance)

    def save_cortex_decision(self, decision: str, context: str, outcome: str | None = None, importance: float = 0.7):
        """Record an orchestration-level decision under ``cortex_manager``.

        Writes to both long-term and contextual memory so the decision is
        searchable across future runs.
        """

        content = f"Decision: {decision}\nContext: {context}"
        if outcome:
            content += f"\nOutcome: {outcome}"

        metadata = {"type": "cortex_decision", "has_outcome": outcome is not None}

        if self.long_term:
            self.long_term.save(content=content, metadata=metadata, importance=importance, agent_id="cortex_manager")

        self.contextual.save(content=content, metadata=metadata, importance=importance)

    def get_agent_history(self, agent_role: str, limit: int = 20) -> list[str]:
        """Return up to ``limit`` content strings filtered to ``agent_role``.

        Short-term entries are returned first; the remainder of the budget
        is filled from long-term memory.
        """

        history = []

        if self.short_term:
            st_items = self.short_term.search(query="", limit=limit, filters={"agent_id": agent_role})
            history.extend([item.content for item in st_items])

        if self.long_term:
            lt_items = self.long_term.retrieve(filters={"agent_id": agent_role}, limit=max(0, limit - len(history)))
            if isinstance(lt_items, list):
                history.extend([item.content for item in lt_items])
            elif isinstance(lt_items, MemoryItem):
                history.append(lt_items.content)

        return history[:limit]

    def get_user_context(self, user_id: str) -> str:
        """Return ``UserMemory`` context for ``user_id`` or ``""`` if absent."""

        if self.user_memory:
            return self.user_memory.get_user_context(user_id)
        return ""

    def reset_short_term(self):
        """Clear the short-term memory buffer."""

        if self.short_term:
            self.short_term.clear()

    def reset_all(self):
        """Clear all memory subsystems."""

        if self.short_term:
            self.short_term.clear()
        if self.long_term:
            self.long_term.clear()
        if self.entity_memory:
            self.entity_memory.clear()
        if self.contextual:
            self.contextual.clear()

    def get_summary(self) -> str:
        """Return a human-readable summary across every active memory tier."""

        parts = []

        if self.short_term:
            parts.append(self.short_term.summarize())

        if self.long_term:
            parts.append(self.long_term.consolidate())

        if self.entity_memory:
            stats = self.entity_memory.get_statistics()
            if stats["total_items"] > 0:
                parts.append(f"Tracking {len(self.entity_memory.entities)} entities")

        return "\n\n".join(parts)
