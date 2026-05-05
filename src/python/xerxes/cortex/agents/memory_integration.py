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
        """Initialize the Cortex memory subsystem.

        Args:
            enable_short_term (bool): Whether to enable short-term memory.
                IN: Controls creation of the recent-memory buffer.
                OUT: Determines if ``self.short_term`` is instantiated.
            enable_long_term (bool): Whether to enable long-term memory.
                IN: Controls creation of the durable memory store.
                OUT: Determines if ``self.long_term`` is instantiated.
            enable_entity (bool): Whether to enable entity memory.
                IN: Controls creation of the entity tracker.
                OUT: Determines if ``self.entity_memory`` is instantiated.
            enable_user (bool): Whether to enable user memory.
                IN: Controls creation of the user-specific memory store.
                OUT: Determines if ``self.user_memory`` is instantiated.
            persistence_path (str | None): Path to an SQLite database for persistence.
                IN: Only used when ``WRITE_MEMORY`` env var is ``"1"``.
                OUT: Passed to ``SQLiteStorage`` when persistence is enabled.
            short_term_capacity (int): Maximum items in short-term memory.
                IN: Capacity limit for the recent-memory buffer.
                OUT: Passed to ``ShortTermMemory``.
            long_term_capacity (int): Maximum items in long-term memory.
                IN: Capacity limit for the durable memory store.
                OUT: Passed to ``LongTermMemory``.
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
        """Assemble a context string for a given task from all memory layers.

        Args:
            task_description (str): The current task to find relevant context for.
                IN: Used as a search query across memory subsystems.
                OUT: Drives retrieval of related memories and entities.
            agent_role (str | None): The role of the agent executing the task.
                IN: Filters long-term memory by ``agent_id``.
                OUT: Applied as a filter to improve relevance.
            additional_context (str | None): Extra background to prepend.
                IN: Arbitrary contextual information from the caller.
                OUT: Included verbatim at the top of the assembled context.
            max_items (int): Maximum number of contextual memory items.
                IN: Limits the number of comprehensive search results.
                OUT: Passed to ``self.contextual.search``.

        Returns:
            str: A formatted context string assembled from all memory sources.
                OUT: Empty string if no relevant context is found.
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
        """Persist the result of a completed task into memory.

        Args:
            task_description (str): The original task description.
                IN: Stored as metadata to link the result to its task.
                OUT: Truncated and saved in memory metadata.
            result (str): The output produced by the agent.
                IN: The content to store in memory.
                OUT: Saved to short-term, long-term (if important), entity, and
                contextual memory layers.
            agent_role (str): The role of the agent that completed the task.
                IN: Used to tag the memory entry.
                OUT: Stored in metadata for filtered retrieval.
            importance (float): Importance score from 0 to 1.
                IN: Determines whether the result is promoted to long-term memory.
                OUT: Compared against ``0.7`` to decide long-term storage.
            task_metadata (dict | None): Additional metadata to attach.
                IN: Optional caller-supplied metadata.
                OUT: Merged with internal metadata before storage.
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
        """Record an agent interaction in memory.

        Args:
            agent_role (str): The role of the agent performing the action.
                IN: Identifies which agent produced the interaction.
                OUT: Stored as metadata for retrieval.
            action (str): A short action label.
                IN: Describes what the agent did (e.g., ``"execute_task"``).
                OUT: Included in the stored content and metadata.
            content (str): The interaction content.
                IN: The text to persist in memory.
                OUT: Saved to short-term and optionally long-term memory.
            importance (float): Importance score from 0 to 1.
                IN: Determines long-term persistence; threshold is ``0.6``.
                OUT: Compared to decide whether to save to long-term memory.
        """

        interaction = f"[{agent_role}] {action}: {content}"

        if self.short_term:
            self.short_term.save(content=interaction, metadata={"action": action}, agent_id=agent_role)

        if importance >= 0.6 and self.long_term:
            self.long_term.save(content=interaction, agent_id=agent_role, importance=importance)

    def save_cortex_decision(self, decision: str, context: str, outcome: str | None = None, importance: float = 0.7):
        """Record a high-level Cortex orchestration decision.

        Args:
            decision (str): A description of the decision made.
                IN: Summarizes the orchestration choice.
                OUT: Stored as the primary content of the memory entry.
            context (str): The context in which the decision was made.
                IN: Background information surrounding the decision.
                OUT: Appended to the stored content.
            outcome (str | None): The result of the decision.
                IN: Optional outcome summary.
                OUT: Appended to the stored content when provided.
            importance (float): Importance score from 0 to 1.
                IN: Default is ``0.7`` to ensure durable storage.
                OUT: Passed to long-term and contextual memory.
        """

        content = f"Decision: {decision}\nContext: {context}"
        if outcome:
            content += f"\nOutcome: {outcome}"

        metadata = {"type": "cortex_decision", "has_outcome": outcome is not None}

        if self.long_term:
            self.long_term.save(content=content, metadata=metadata, importance=importance, agent_id="cortex_manager")

        self.contextual.save(content=content, metadata=metadata, importance=importance)

    def get_agent_history(self, agent_role: str, limit: int = 20) -> list[str]:
        """Retrieve the interaction history for a specific agent.

        Args:
            agent_role (str): The agent role to look up.
                IN: Filter value for memory queries.
                OUT: Applied to short-term and long-term retrieval.
            limit (int): Maximum number of history entries to return.
                IN: Caps the total number of results across memory layers.
                OUT: Split between short-term and long-term retrieval.

        Returns:
            list[str]: A list of content strings from the agent's history.
                OUT: Truncated to *limit* entries.
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
        """Fetch stored context for a specific user.

        Args:
            user_id (str): The identifier of the user.
                IN: Passed to ``UserMemory`` for lookup.
                OUT: Used to retrieve user-specific context.

        Returns:
            str: The user's context string, or an empty string if not found.
                OUT: Direct result from ``UserMemory.get_user_context``.
        """

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
        """Produce a human-readable summary of memory state.

        Returns:
            str: A summary string combining short-term, long-term, and entity
                statistics. OUT: Empty string if no memory layers are active.
        """

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
