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


"""Memory integration for Cortex framework.

This module provides the CortexMemory class, a unified memory system for
Cortex agents and tasks. It integrates multiple memory types (short-term,
long-term, entity, user, and contextual) to provide comprehensive context
management for AI agent operations.

The memory system supports:
- Short-term memory for recent interactions and temporary context
- Long-term memory for persistent knowledge storage
- Entity memory for tracking and relating entities mentioned in conversations
- User memory for personalized user-specific context
- Contextual memory for comprehensive search across all memory types

Memory persistence is optional and controlled via environment variables
and configuration parameters, allowing for both ephemeral and persistent
memory operations.

Typical usage example:
    memory = CortexMemory(
        enable_short_term=True,
        enable_long_term=True,
        persistence_path="./memory.db",
        short_term_capacity=50,
        long_term_capacity=5000
    )


    context = memory.build_context_for_task(
        task_description="Analyze sales data",
        agent_role="Data Analyst"
    )


    memory.save_task_result(
        task_description="Analyze sales data",
        result="Sales increased 15% this quarter",
        agent_role="Data Analyst",
        importance=0.8
    )
"""

from dataclasses import dataclass
from typing import Any

from ...memory import (
    ContextualMemory,
    EntityMemory,
    LongTermMemory,
    ShortTermMemory,
    SQLiteStorage,
    UserMemory,
)


@dataclass
class CortexMemory:
    """Unified memory system for Cortex agents and tasks.

    CortexMemory integrates multiple memory types to provide comprehensive
    context management for AI agents operating within the Cortex framework.
    It aggregates short-term, long-term, entity, user, and contextual memories
    to build rich context for task execution and maintain persistent knowledge.

    The memory system is designed to:
    - Provide relevant context for task execution
    - Store and retrieve task results and agent interactions
    - Track entities and their relationships across conversations
    - Maintain user-specific context for personalization
    - Support both ephemeral and persistent storage

    Attributes:
        storage: Optional SQLiteStorage instance for persistence.
            Only created if persistence_path is provided and WRITE_MEMORY
            environment variable is set to "1".
        short_term: Optional ShortTermMemory for recent interactions.
            Enabled by default with configurable capacity.
        long_term: Optional LongTermMemory for persistent knowledge.
            Enabled by default with configurable maximum items.
        entity_memory: Optional EntityMemory for tracking entities.
            Enabled by default for entity extraction and relationship tracking.
        user_memory: Optional UserMemory for user-specific context.
            Disabled by default, must be explicitly enabled.
        contextual: ContextualMemory for comprehensive cross-memory search.
            Always enabled, integrates with other memory types.

    Example:
        memory = CortexMemory(
            enable_short_term=True,
            enable_long_term=True,
            persistence_path="./cortex_memory.db"
        )

        context = memory.build_context_for_task(
            task_description="Write unit tests",
            agent_role="Test Engineer"
        )
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
        """Initialize the Cortex memory system.

        Creates a new CortexMemory instance with the specified memory types
        enabled. Each memory type can be individually enabled or disabled.
        Persistence is controlled by both the persistence_path parameter
        and the WRITE_MEMORY environment variable.

        Args:
            enable_short_term: Whether to enable short-term memory for
                recent interactions. Defaults to True.
            enable_long_term: Whether to enable long-term memory for
                persistent knowledge storage. Defaults to True.
            enable_entity: Whether to enable entity memory for tracking
                entities mentioned in conversations. Defaults to True.
            enable_user: Whether to enable user memory for user-specific
                context. Defaults to False.
            persistence_path: Optional path to SQLite database file for
                persistent storage. Requires WRITE_MEMORY=1 environment
                variable to be set. Defaults to None (no persistence).
            short_term_capacity: Maximum number of items to store in
                short-term memory. Defaults to 50.
            long_term_capacity: Maximum number of items to store in
                long-term memory. Defaults to 5000.

        Note:
            Persistence is only enabled when both persistence_path is
            provided AND the WRITE_MEMORY environment variable is set
            to "1". This provides an additional safety mechanism to
            prevent unintended writes.
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
        """Build contextual information for a task.

        Aggregates relevant memories from all enabled memory types to
        provide comprehensive context for task execution. The context
        includes recent interactions, relevant long-term knowledge,
        known entities, and related memories from contextual search.

        Args:
            task_description: Description of the task to build context for.
                Used as the search query for retrieving relevant memories.
            agent_role: Optional role identifier for filtering agent-specific
                memories. When provided, memories are filtered to those
                associated with this agent.
            additional_context: Optional background information to include
                at the beginning of the context. Useful for providing
                domain-specific information.
            max_items: Maximum number of items to retrieve from contextual
                memory search. Defaults to 10.

        Returns:
            Formatted string containing aggregated context from all memory
            types. Includes sections for background, recent context, relevant
            knowledge, known entities, and related memories. Returns empty
            string if no context is available.

        Note:
            Content from individual memory items is truncated to prevent
            context overflow (200 chars for short/long-term, 150 chars for
            contextual memories).
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
        """Save task execution result to memory.

        Stores the result of a completed task across multiple memory types
        based on its importance level. Short-term memory always receives the
        result, while long-term storage is reserved for high-importance items.

        Args:
            task_description: Description of the completed task. Truncated
                to 100 characters in stored metadata.
            result: The result or output of the task execution. Truncated
                to 200 characters for short-term storage.
            agent_role: Role identifier of the agent that executed the task.
                Used for filtering and retrieval.
            importance: Importance score from 0.0 to 1.0 indicating the
                significance of this result. Results with importance >= 0.7
                are stored in long-term memory. Defaults to 0.5.
            task_metadata: Optional dictionary of additional metadata to
                store with the result. Merged with auto-generated metadata.

        Side Effects:
            - Saves to short-term memory (if enabled)
            - Saves to long-term memory if importance >= 0.7 (if enabled)
            - Extracts and saves entities (if entity memory enabled)
            - Saves to contextual memory
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
        """Save agent interaction to memory.

        Records an agent's interaction or action for future reference.
        Interactions are formatted as "[agent_role] action: content" and
        stored in short-term memory. High-importance interactions are
        also persisted to long-term memory.

        Args:
            agent_role: Role identifier of the agent performing the action.
            action: Type of action being performed (e.g., "thinking",
                "delegating", "tool_call").
            content: Content or details of the interaction.
            importance: Importance score from 0.0 to 1.0. Interactions with
                importance >= 0.6 are stored in long-term memory.
                Defaults to 0.3.

        Side Effects:
            - Saves to short-term memory (if enabled)
            - Saves to long-term memory if importance >= 0.6 (if enabled)
        """
        interaction = f"[{agent_role}] {action}: {content}"

        if self.short_term:
            self.short_term.save(content=interaction, metadata={"action": action}, agent_id=agent_role)

        if importance >= 0.6 and self.long_term:
            self.long_term.save(content=interaction, agent_id=agent_role, importance=importance)

    def save_cortex_decision(self, decision: str, context: str, outcome: str | None = None, importance: float = 0.7):
        """Save cortex-level decisions to memory.

        Records strategic decisions made at the Cortex orchestration level.
        These are typically high-importance items that affect overall task
        execution flow and are persisted to long-term memory.

        Args:
            decision: Description of the decision that was made.
            context: Background context that led to this decision.
            outcome: Optional outcome or result of the decision. If provided,
                appended to the stored content.
            importance: Importance score from 0.0 to 1.0. Defaults to 0.7
                (high importance) as cortex decisions are typically significant.

        Side Effects:
            - Saves to long-term memory with agent_id "cortex_manager"
            - Saves to contextual memory for cross-memory search
        """
        content = f"Decision: {decision}\nContext: {context}"
        if outcome:
            content += f"\nOutcome: {outcome}"

        metadata = {"type": "cortex_decision", "has_outcome": outcome is not None}

        if self.long_term:
            self.long_term.save(content=content, metadata=metadata, importance=importance, agent_id="cortex_manager")

        self.contextual.save(content=content, metadata=metadata, importance=importance)

    def get_agent_history(self, agent_role: str, limit: int = 20) -> list[str]:
        """Get history for a specific agent.

        Retrieves the interaction and task history for a specific agent
        from both short-term and long-term memory. Short-term memory is
        searched first, then long-term memory fills remaining slots.

        Args:
            agent_role: Role identifier of the agent to get history for.
            limit: Maximum number of history items to return. Defaults to 20.

        Returns:
            List of content strings from the agent's history, up to the
            specified limit. Items from short-term memory appear first,
            followed by long-term memory items.
        """
        history = []

        if self.short_term:
            st_items = self.short_term.search(query="", limit=limit, filters={"agent_id": agent_role})
            history.extend([item.content for item in st_items])

        if self.long_term:
            lt_items = self.long_term.retrieve(filters={"agent_id": agent_role}, limit=max(0, limit - len(history)))
            if lt_items:
                history.extend([item.content for item in lt_items])

        return history[:limit]

    def get_user_context(self, user_id: str) -> str:
        """Get user-specific context.

        Retrieves personalized context for a specific user from user memory.
        This context can be used to customize agent responses based on
        user preferences, history, and characteristics.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            String containing user-specific context. Returns empty string
            if user memory is not enabled or no context exists for the user.
        """
        if self.user_memory:
            return self.user_memory.get_user_context(user_id)
        return ""

    def reset_short_term(self):
        """Clear short-term memory.

        Removes all items from short-term memory while preserving other
        memory types. Useful for starting a new conversation or session
        without losing long-term knowledge.

        Side Effects:
            - Clears all items from short-term memory (if enabled)
        """
        if self.short_term:
            self.short_term.clear()

    def reset_all(self):
        """Clear all memories.

        Removes all items from all enabled memory types, including short-term,
        long-term, entity, and contextual memory. Use with caution as this
        operation is irreversible and will delete all stored knowledge.

        Side Effects:
            - Clears short-term memory (if enabled)
            - Clears long-term memory (if enabled)
            - Clears entity memory (if enabled)
            - Clears contextual memory
        """
        if self.short_term:
            self.short_term.clear()
        if self.long_term:
            self.long_term.clear()
        if self.entity_memory:
            self.entity_memory.clear()
        if self.contextual:
            self.contextual.clear()

    def get_summary(self) -> str:
        """Get a summary of all memories.

        Generates a consolidated summary of the current memory state
        across all enabled memory types. Includes summaries from short-term
        memory, consolidated long-term knowledge, and entity statistics.

        Returns:
            Multi-line string containing summaries from each memory type,
            separated by blank lines. Includes:
            - Short-term memory summary (if enabled)
            - Long-term memory consolidation (if enabled)
            - Entity tracking statistics (if enabled and has items)
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
