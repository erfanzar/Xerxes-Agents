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


"""Memory integration for Cortex framework"""

from dataclasses import dataclass
from typing import Any

from ..memory import (
    ContextualMemory,
    EntityMemory,
    LongTermMemory,
    ShortTermMemory,
    SQLiteStorage,
    UserMemory,
)


@dataclass
class CortexMemory:
    """
    Unified memory system for Cortex cortexs.
    Manages different memory types and provides context for agents and tasks.
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
        """Initialize Cortex memory system"""
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
        """
        Build contextual information for a task.
        Aggregates relevant memories from all memory types.
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
        """Save task execution result to memory"""
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
        """Save agent interaction to memory"""
        interaction = f"[{agent_role}] {action}: {content}"

        if self.short_term:
            self.short_term.save(content=interaction, metadata={"action": action}, agent_id=agent_role)

        if importance >= 0.6 and self.long_term:
            self.long_term.save(content=interaction, agent_id=agent_role, importance=importance)

    def save_cortex_decision(self, decision: str, context: str, outcome: str | None = None, importance: float = 0.7):
        """Save cortex-level decisions"""
        content = f"Decision: {decision}\nContext: {context}"
        if outcome:
            content += f"\nOutcome: {outcome}"

        metadata = {"type": "cortex_decision", "has_outcome": outcome is not None}

        if self.long_term:
            self.long_term.save(content=content, metadata=metadata, importance=importance, agent_id="cortex_manager")

        self.contextual.save(content=content, metadata=metadata, importance=importance)

    def get_agent_history(self, agent_role: str, limit: int = 20) -> list[str]:
        """Get history for a specific agent"""
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
        """Get user-specific context"""
        if self.user_memory:
            return self.user_memory.get_user_context(user_id)
        return ""

    def reset_short_term(self):
        """Clear short-term memory"""
        if self.short_term:
            self.short_term.clear()

    def reset_all(self):
        """Clear all memories"""
        if self.short_term:
            self.short_term.clear()
        if self.long_term:
            self.long_term.clear()
        if self.entity_memory:
            self.entity_memory.clear()
        if self.contextual:
            self.contextual.clear()

    def get_summary(self) -> str:
        """Get a summary of all memories"""
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
