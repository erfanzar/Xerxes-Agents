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
"""Per-user memory facade combining contextual + entity memory and preferences.

``UserMemory`` lazily creates one ``ContextualMemory`` and one
``EntityMemory`` per user id and tracks a small preferences dict for
each. It is the entry point used by the daemon's user-aware code
paths; the contextual tiers underneath are independent across users."""

from typing import Any

from .contextual_memory import ContextualMemory
from .entity_memory import EntityMemory


class UserMemory:
    """Manage isolated memory + entity stores keyed by user identifier."""

    def __init__(self, storage: Any | None = None) -> None:
        """Wire up storage and hydrate cached user preferences.

        Args:
            storage: Optional ``MemoryStorage`` shared across every
                user's contextual and entity tiers."""

        self.storage = storage
        self.user_memories: dict[str, ContextualMemory] = {}
        self.user_entities: dict[str, EntityMemory] = {}
        self.user_preferences: dict[str, dict[str, Any]] = {}
        self._load_users()

    def _load_users(self) -> None:
        """Restore the persisted ``_user_preferences`` blob into memory."""

        if self.storage and self.storage.exists("_user_preferences"):
            self.user_preferences = self.storage.load("_user_preferences") or {}

    def get_or_create_user_memory(self, user_id: str) -> ContextualMemory:
        """Return the user's contextual memory, creating it (with entity store + default prefs) on first call."""

        if user_id not in self.user_memories:
            self.user_memories[user_id] = ContextualMemory(long_term_storage=self.storage)
            self.user_entities[user_id] = EntityMemory(storage=self.storage)
            self.user_preferences[user_id] = self._get_default_preferences()
            self._save_preferences()

        return self.user_memories[user_id]

    def save_memory(self, user_id: str, content: str, metadata: dict[str, Any] | None = None, **kwargs):
        """Save ``content`` into both the user's contextual and entity tiers.

        Args:
            user_id: Owning user identifier.
            content: Text payload.
            metadata: Annotations; ``user_id`` is set automatically.
            **kwargs: Forwarded to both underlying ``save`` calls."""

        memory = self.get_or_create_user_memory(user_id)
        metadata = metadata or {}
        metadata["user_id"] = user_id

        item = memory.save(content=content, metadata=metadata, user_id=user_id, **kwargs)

        entity_mem = self.user_entities.get(user_id)
        if entity_mem:
            entity_mem.save(content=content, metadata=metadata, **kwargs)

        return item

    def search_user_memory(self, user_id: str, query: str, limit: int = 10, **kwargs) -> list:
        """Search the user's contextual memory only (not entities)."""

        memory = self.get_or_create_user_memory(user_id)
        return memory.search(query=query, limit=limit, **kwargs)

    def get_user_context(self, user_id: str) -> str:
        """Render a multi-section context block (preferences + summary + known entities)."""

        memory = self.get_or_create_user_memory(user_id)
        entity_mem = self.user_entities.get(user_id)

        context_parts = []

        prefs = self.get_user_preferences(user_id)
        if prefs:
            context_parts.append(f"User preferences: {prefs}")

        context_parts.append(memory.get_context_summary())

        if entity_mem and entity_mem.entities:
            entities = list(entity_mem.entities.keys())[:10]
            context_parts.append(f"Known entities: {', '.join(entities)}")

        return "\n\n".join(context_parts)

    def update_user_preferences(self, user_id: str, preferences: dict[str, Any]) -> None:
        """Merge ``preferences`` into the user's stored preferences dict and persist."""

        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self._get_default_preferences()

        self.user_preferences[user_id].update(preferences)
        self._save_preferences()

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Return the user's preferences, falling back to the default template."""

        return self.user_preferences.get(user_id, self._get_default_preferences())

    def get_user_statistics(self, user_id: str) -> dict[str, Any]:
        """Return a statistics dict (memory counts, entity counts, preferences)."""

        stats = {
            "user_id": user_id,
            "total_memories": 0,
            "entities_known": 0,
            "preferences": self.get_user_preferences(user_id),
        }

        if user_id in self.user_memories:
            memory = self.user_memories[user_id]
            stats["total_memories"] = len(memory.short_term) + len(memory.long_term)
            stats["short_term_memories"] = len(memory.short_term)
            stats["long_term_memories"] = len(memory.long_term)

        if user_id in self.user_entities:
            entity_mem = self.user_entities[user_id]
            stats["entities_known"] = len(entity_mem.entities)
            stats["relationships"] = sum(len(rels) for rels in entity_mem.relationships.values())

        return stats

    def clear_user_memory(self, user_id: str) -> None:
        """Drop every tier and the preferences row for ``user_id``."""

        if user_id in self.user_memories:
            self.user_memories[user_id].clear()
            del self.user_memories[user_id]

        if user_id in self.user_entities:
            self.user_entities[user_id].clear()
            del self.user_entities[user_id]

        if user_id in self.user_preferences:
            del self.user_preferences[user_id]
            self._save_preferences()

    def _get_default_preferences(self) -> dict[str, Any]:
        """Return the canonical preference template for a freshly-seen user."""

        return {
            "response_style": "balanced",
            "verbosity": "normal",
            "technical_level": "intermediate",
            "language": "en",
            "timezone": "UTC",
            "memory_enabled": True,
            "max_context_items": 10,
        }

    def _save_preferences(self) -> None:
        """Persist the in-memory ``user_preferences`` map to storage."""

        if self.storage:
            self.storage.save("_user_preferences", self.user_preferences)
