# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""User-specific memory for personalization."""

from typing import Any

from .contextual_memory import ContextualMemory
from .entity_memory import EntityMemory


class UserMemory:
    """User-specific memory manager with per-user isolation.

    Maintains separate :class:`ContextualMemory` and :class:`EntityMemory`
    instances for each user, along with per-user preference dictionaries.
    This allows agents to personalise responses and retain context across
    sessions on a per-user basis.

    Attributes:
        storage: Optional :class:`~xerxes.memory.storage.MemoryStorage`
            backend used for persisting user preferences and passed to
            per-user memory instances.
        user_memories: Dictionary mapping user IDs to their
            :class:`ContextualMemory` instances.
        user_entities: Dictionary mapping user IDs to their
            :class:`EntityMemory` instances.
        user_preferences: Dictionary mapping user IDs to preference
            dictionaries (response style, verbosity, language, etc.).

    Example:
        >>> from xerxes.memory import UserMemory
        >>> um = UserMemory()
        >>> um.save_memory("user-1", "Prefers dark mode")
        >>> um.update_user_preferences("user-1", {"theme": "dark"})
        >>> ctx = um.get_user_context("user-1")
    """

    def __init__(self, storage: Any | None = None) -> None:
        """Initialize user memory manager with optional persistence.

        On initialisation, attempts to load previously persisted user
        preferences from the storage backend.

        Args:
            storage: Optional :class:`~xerxes.memory.storage.MemoryStorage`
                backend for persisting user preference data and for
                providing long-term storage to per-user memory instances.
        """
        self.storage = storage
        self.user_memories: dict[str, ContextualMemory] = {}
        self.user_entities: dict[str, EntityMemory] = {}
        self.user_preferences: dict[str, dict[str, Any]] = {}
        self._load_users()

    def _load_users(self) -> None:
        """Load persisted user preference data from storage on initialisation.

        Checks for a ``_user_preferences`` key in the storage backend and,
        if found, restores the :attr:`user_preferences` dictionary from it.
        """
        if self.storage and self.storage.exists("_user_preferences"):
            self.user_preferences = self.storage.load("_user_preferences") or {}

    def get_or_create_user_memory(self, user_id: str) -> ContextualMemory:
        """Get or lazily create the memory subsystem for a user.

        On first call for a given ``user_id``, creates:

        - A :class:`ContextualMemory` (using :attr:`storage` for long-term).
        - An :class:`EntityMemory` (using :attr:`storage` for persistence).
        - A default preferences dictionary (see :meth:`_get_default_preferences`).

        Subsequent calls return the existing :class:`ContextualMemory`.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            The :class:`ContextualMemory` instance associated with the user.
        """
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ContextualMemory(long_term_storage=self.storage)
            self.user_entities[user_id] = EntityMemory(storage=self.storage)
            self.user_preferences[user_id] = self._get_default_preferences()
            self._save_preferences()

        return self.user_memories[user_id]

    def save_memory(self, user_id: str, content: str, metadata: dict[str, Any] | None = None, **kwargs):
        """Save a memory item for a specific user.

        Stores the content in both the user's :class:`ContextualMemory`
        (for context-aware retrieval and promotion) and
        :class:`EntityMemory` (for entity tracking).

        Args:
            user_id: Unique identifier for the user. If the user does not
                yet exist, their memory subsystem is created automatically.
            content: Text content to store.
            metadata: Optional key-value metadata. A ``"user_id"`` key is
                added automatically.
            **kwargs: Additional keyword arguments forwarded to the
                underlying ``save`` methods (e.g. ``importance``,
                ``agent_id``).

        Returns:
            The :class:`~xerxes.memory.base.MemoryItem` created by the
            contextual memory store.
        """
        memory = self.get_or_create_user_memory(user_id)
        metadata = metadata or {}
        metadata["user_id"] = user_id

        item = memory.save(content=content, metadata=metadata, user_id=user_id, **kwargs)

        entity_mem = self.user_entities.get(user_id)
        if entity_mem:
            entity_mem.save(content=content, metadata=metadata, **kwargs)

        return item

    def search_user_memory(self, user_id: str, query: str, limit: int = 10, **kwargs) -> list:
        """Search memories for a specific user.

        Delegates to the user's :class:`ContextualMemory` search, which
        queries both short-term and long-term stores.

        Args:
            user_id: Unique identifier for the user. If the user does not
                yet exist, their memory subsystem is created automatically.
            query: Natural-language or keyword search query string.
            limit: Maximum number of results to return.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`ContextualMemory.search`.

        Returns:
            List of matching :class:`~xerxes.memory.base.MemoryItem`
            instances sorted by relevance.
        """
        memory = self.get_or_create_user_memory(user_id)
        return memory.search(query=query, limit=limit, **kwargs)

    def get_user_context(self, user_id: str) -> str:
        """Build a formatted context string for a user.

        Combines three sections into a single string separated by blank
        lines:

        1. **User preferences** -- the current preference dictionary.
        2. **Context summary** -- from the user's :class:`ContextualMemory`.
        3. **Known entities** -- up to 10 entity names from the user's
           :class:`EntityMemory`.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            Multi-line context string suitable for inclusion in an agent's
            system prompt or context window.
        """
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
        """Update user preferences by merging new values.

        Existing keys are overwritten, new keys are added, and the full
        preference dictionary is persisted to the storage backend.

        Args:
            user_id: Unique identifier for the user. If no preferences
                exist yet, defaults are initialised first.
            preferences: Dictionary of preference keys and their new
                values to merge into the existing preferences.
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self._get_default_preferences()

        self.user_preferences[user_id].update(preferences)
        self._save_preferences()

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Retrieve the preferences for a user.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            Dictionary of user preferences. Returns a fresh set of default
            preferences if the user has not been registered yet.
        """
        return self.user_preferences.get(user_id, self._get_default_preferences())

    def get_user_statistics(self, user_id: str) -> dict[str, Any]:
        """Compute aggregate statistics for a user's memory subsystem.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            Dictionary with keys:
                - ``"user_id"``: the queried user ID.
                - ``"total_memories"``: combined short-term and long-term count.
                - ``"short_term_memories"``: short-term item count (if available).
                - ``"long_term_memories"``: long-term item count (if available).
                - ``"entities_known"``: number of tracked entities.
                - ``"relationships"``: total relationship pair count.
                - ``"preferences"``: the user's current preference dictionary.
        """
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
        """Clear all data for a user and remove them from the manager.

        Clears and deletes the user's :class:`ContextualMemory`,
        :class:`EntityMemory`, and preference dictionary. After this call
        the user is treated as if they never existed; a subsequent call
        to :meth:`get_or_create_user_memory` will re-initialise them with
        fresh defaults.

        Args:
            user_id: Unique identifier for the user to clear.
        """
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
        """Build the default user preferences dictionary.

        Returns:
            Dictionary with default values:
                - ``"response_style"``: ``"balanced"``
                - ``"verbosity"``: ``"normal"``
                - ``"technical_level"``: ``"intermediate"``
                - ``"language"``: ``"en"``
                - ``"timezone"``: ``"UTC"``
                - ``"memory_enabled"``: ``True``
                - ``"max_context_items"``: ``10``
        """
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
        """Persist the full user preferences dictionary to storage.

        Writes the :attr:`user_preferences` mapping under the
        ``_user_preferences`` key. No-op if no storage backend is
        configured.
        """
        if self.storage:
            self.storage.save("_user_preferences", self.user_preferences)
