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


"""Entity memory for tracking information about specific entities."""

import re
from collections import defaultdict
from typing import Any

from .base import Memory, MemoryItem


class EntityMemory(Memory):
    """Memory system for tracking entities (people, organisations, concepts).

    Maintains a lightweight knowledge graph of entities and their
    relationships. Entities are automatically extracted from stored text
    using pattern-matching heuristics (capitalised phrases and quoted
    strings), and relationships are detected from common verb patterns
    (e.g. "works at", "knows", "created").

    Attributes:
        entities: Dictionary mapping entity names to tracking metadata
            including ``first_seen``, ``last_seen``, ``frequency``, and
            ``contexts`` (snippet list).
        relationships: Dictionary mapping relation types (e.g. ``"knows"``)
            to lists of ``(entity1, entity2)`` tuples.
        entity_mentions: Dictionary mapping entity names to lists of
            memory IDs in which that entity was mentioned.

    Example:
        >>> from calute.memory import EntityMemory
        >>> em = EntityMemory()
        >>> item = em.save("Alice works at Acme Corp")
        >>> em.get_entity_info("Alice")
        {'first_seen': ..., 'frequency': 1, ...}
    """

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int = 5000,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialize entity memory with optional persistence.

        Args:
            storage: Optional :class:`MemoryStorage` backend for persisting
                entity data and memory items. When ``None``, data is held
                in-memory only.
            max_items: Maximum number of memory items to store before the
                oldest items may be evicted.
            enable_embeddings: Whether to compute dense vector embeddings
                for semantic search over stored content.
        """
        super().__init__(storage=storage, max_items=max_items, enable_embeddings=enable_embeddings)
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.entity_mentions: dict[str, list[str]] = defaultdict(list)

    def save(
        self, content: str, metadata: dict[str, Any] | None = None, entities: list[str] | None = None, **kwargs
    ) -> MemoryItem:
        """Save a memory item and extract entities from its content.

        Entities are either provided explicitly or extracted automatically
        via :meth:`_extract_entities`. Relationships between co-occurring
        entities are also detected and recorded.

        Args:
            content: Text content to store. Entity extraction heuristics
                are applied to this text when ``entities`` is not provided.
            metadata: Optional key-value metadata to attach. An ``"entities"``
                key is added automatically with the resolved entity list.
            entities: Pre-identified entity names. When ``None``, entities
                are extracted from ``content`` using pattern matching.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            The newly created :class:`MemoryItem` with entity metadata.
        """
        metadata = metadata or {}

        if not entities:
            entities = self._extract_entities(content)

        metadata["entities"] = entities

        item = MemoryItem(
            content=content,
            memory_type="entity",
            metadata=metadata,
        )

        for entity in entities:
            self._update_entity(entity, item)

        relationships = self._extract_relationships(content, entities)
        for entity1, relation, entity2 in relationships:
            self.relationships[relation].append((entity1, entity2))

        self._items.append(item)
        self._index[item.memory_id] = item

        if self.storage:
            self.storage.save(f"entity_{item.memory_id}", item.to_dict())
            self._save_entity_data()

        return item

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        entity_filter: list[str] | None = None,
        **kwargs,
    ) -> list[MemoryItem]:
        """Search for memories related to specific entities.

        Entities are first extracted from the query (or taken from
        ``entity_filter``), and then all stored items are scored by the
        fraction of target entities they mention. Items with no overlapping
        entities are excluded unless no target entities were identified, in
        which case a simple substring match is used instead.

        Args:
            query: Natural-language query. Entity names are extracted from
                this text when ``entity_filter`` is not provided.
            limit: Maximum number of results to return.
            filters: Optional key-value criteria matched against item
                attributes (e.g. ``{"agent_id": "agent-1"}``).
            entity_filter: Explicit list of entity names to search for.
                Overrides automatic extraction from ``query``.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            List of :class:`MemoryItem` instances sorted by relevance
            (entity overlap ratio), with at most ``limit`` entries.
        """

        query_entities = self._extract_entities(query)
        target_entities = entity_filter or query_entities

        matches = []

        for item in self._items:
            item_entities = item.metadata.get("entities", [])

            if target_entities:
                overlap = set(item_entities) & set(target_entities)
                if not overlap:
                    continue

            if filters:
                skip = False
                for key, value in filters.items():
                    if hasattr(item, key) and getattr(item, key) != value:
                        skip = True
                        break
                if skip:
                    continue

            if target_entities:
                item.relevance_score = len(overlap) / len(target_entities)
            else:
                item.relevance_score = 1.0 if query.lower() in item.content.lower() else 0.5

            matches.append(item)

        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        return matches[:limit]

    def retrieve(
        self,
        memory_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> MemoryItem | list[MemoryItem] | None:
        """Retrieve memory items by ID or filter criteria.

        When ``memory_id`` is provided, performs an O(1) index lookup.
        Otherwise iterates through items and applies attribute-based filters.

        Args:
            memory_id: UUID of a specific memory item to retrieve.
            filters: Optional key-value criteria matched against item
                attributes (e.g. ``{"agent_id": "agent-1"}``). Items where
                any specified attribute does not match are excluded.
            limit: Maximum number of items to return when using filter-based
                retrieval.

        Returns:
            A single :class:`MemoryItem` when ``memory_id`` is provided
            (or ``None`` if not found), or a list of matching items.
        """
        if memory_id:
            return self._index.get(memory_id)

        results = []
        for item in self._items:
            if filters:
                skip = False
                for key, value in filters.items():
                    if hasattr(item, key) and getattr(item, key) != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append(item)
            if len(results) >= limit:
                break

        return results

    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory item and re-extract entities if content changes.

        When the ``"content"`` field is included in ``updates``, the old
        entity mentions are removed and new entities are extracted from the
        updated content. Changes are persisted to storage if configured.

        Args:
            memory_id: UUID of the memory item to update.
            updates: Dictionary mapping attribute names to their new values.
                When ``"content"`` is present, entity mention tracking is
                refreshed automatically.

        Returns:
            ``True`` if the item was found and updated, ``False`` if
            ``memory_id`` was not found in the index.
        """
        if memory_id not in self._index:
            return False

        item = self._index[memory_id]

        if "content" in updates:
            old_entities = item.metadata.get("entities", [])
            new_entities = self._extract_entities(updates["content"])
            updates.setdefault("metadata", {})["entities"] = new_entities

            for entity in old_entities:
                if entity in self.entity_mentions:
                    self.entity_mentions[entity].remove(memory_id)

            for entity in new_entities:
                self.entity_mentions[entity].append(memory_id)

        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        if self.storage:
            self.storage.save(f"entity_{memory_id}", item.to_dict())
            self._save_entity_data()

        return True

    def delete(self, memory_id: str | None = None, filters: dict[str, Any] | None = None) -> int:
        """Delete a memory item and update entity mention tracking.

        Removes the item from the in-memory index and list, cleans up
        entity mention references, and deletes from the storage backend
        if configured.

        Note:
            Filter-based bulk deletion is accepted by the signature for
            interface compatibility but is not currently implemented.

        Args:
            memory_id: UUID of the specific memory item to delete.
            filters: Reserved for future filter-based bulk deletion.

        Returns:
            Number of items deleted (0 or 1).
        """
        count = 0

        if memory_id and memory_id in self._index:
            item = self._index[memory_id]

            for entity in item.metadata.get("entities", []):
                if entity in self.entity_mentions:
                    self.entity_mentions[entity].remove(memory_id)

            self._items.remove(item)
            del self._index[memory_id]
            if self.storage:
                self.storage.delete(f"entity_{memory_id}")
            count = 1

        return count

    def clear(self) -> None:
        """Clear all memories, entities, relationships, and mention tracking."""
        self._items.clear()
        self._index.clear()
        self.entities.clear()
        self.relationships.clear()
        self.entity_mentions.clear()

        if self.storage:
            for key in self.storage.list_keys("entity_"):
                self.storage.delete(key)

    def get_entity_info(self, entity: str) -> dict[str, Any]:
        """Get comprehensive information about an entity.

        Collects the entity's tracking metadata, all memory IDs that
        mention it, and its direct relationships (both outgoing and
        inverse incoming).

        Args:
            entity: The entity name to look up (case-sensitive).

        Returns:
            Dictionary containing:
                - All stored tracking fields (``first_seen``, ``last_seen``,
                  ``frequency``, ``contexts``) if the entity exists.
                - ``"mentions"``: list of memory IDs referencing this entity.
                - ``"relationships"``: list of dicts with ``"relation"`` and
                  ``"target"`` keys describing outgoing and inverse relations.
        """
        info = self.entities.get(entity, {})
        info["mentions"] = self.entity_mentions.get(entity, [])
        info["relationships"] = []

        for relation, pairs in self.relationships.items():
            for e1, e2 in pairs:
                if e1 == entity:
                    info["relationships"].append({"relation": relation, "target": e2})
                elif e2 == entity:
                    info["relationships"].append({"relation": f"inverse_{relation}", "target": e1})

        return info

    def get_related_entities(self, entity: str, max_depth: int = 2) -> set[str]:
        """
        Get entities related to the given entity via relationship traversal.

        Uses breadth-first search to find connected entities up to max_depth
        hops away in the relationship graph.

        Args:
            entity: Starting entity name
            max_depth: Maximum relationship hops to traverse

        Returns:
            Set of related entity names
        """
        related = set()
        to_explore = [(entity, 0)]
        explored = set()

        while to_explore:
            current, depth = to_explore.pop(0)
            if current in explored or depth > max_depth:
                continue

            explored.add(current)

            for _relation, pairs in self.relationships.items():
                for e1, e2 in pairs:
                    if e1 == current:
                        related.add(e2)
                        if depth < max_depth:
                            to_explore.append((e2, depth + 1))
                    elif e2 == current:
                        related.add(e1)
                        if depth < max_depth:
                            to_explore.append((e1, depth + 1))

        return related

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract entities from text using pattern matching.

        Uses simple heuristics to identify entities:
        - Capitalized words and phrases (proper nouns)
        - Quoted strings

        Args:
            text: Text to extract entities from

        Returns:
            List of unique entity names found in text
        """

        entities = []

        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        matches = re.findall(pattern, text)
        entities.extend(matches)

        quoted = re.findall(r'"([^"]*)"', text)
        entities.extend(quoted)

        common_words = {"The", "This", "That", "These", "Those"}
        entities = list(set(e for e in entities if e not in common_words))

        return entities

    def _extract_relationships(self, text: str, entities: list[str]) -> list[tuple[str, str, str]]:
        """
        Extract relationships between entities from text.

        Identifies relationship patterns like "X works at Y" or "X knows Y"
        and returns structured relationship tuples.

        Args:
            text: Text to extract relationships from
            entities: List of known entities to match against

        Returns:
            List of (entity1, relation_type, entity2) tuples
        """
        relationships = []

        patterns = [
            (r"(\w+)\s+is\s+(?:a|an|the)?\s*(\w+)\s+of\s+(\w+)", "relation_of"),
            (r"(\w+)\s+works\s+(?:at|for|with)\s+(\w+)", "works_with"),
            (r"(\w+)\s+knows\s+(\w+)", "knows"),
            (r"(\w+)\s+created\s+(\w+)", "created"),
        ]

        for pattern, relation in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    e1, e2 = groups[0], groups[-1]
                    if e1 in entities and e2 in entities:
                        relationships.append((e1, relation, e2))

        return relationships

    def _update_entity(self, entity: str, memory_item: MemoryItem) -> None:
        """Update entity tracking data with a new memory mention.

        Creates the entity entry if it does not already exist, increments
        the frequency counter, records the latest timestamp, appends a
        content snippet, and logs the memory ID in :attr:`entity_mentions`.

        Args:
            entity: Entity name to update or create.
            memory_item: The :class:`MemoryItem` that mentions this entity.
        """
        if entity not in self.entities:
            self.entities[entity] = {"first_seen": memory_item.timestamp, "frequency": 0, "contexts": []}

        self.entities[entity]["frequency"] += 1
        self.entities[entity]["last_seen"] = memory_item.timestamp
        self.entities[entity]["contexts"].append(memory_item.content[:100])
        self.entity_mentions[entity].append(memory_item.memory_id)

    def _save_entity_data(self) -> None:
        """Persist entity tables (entities, relationships, mentions) to storage."""
        if self.storage:
            self.storage.save("_entity_entities", self.entities)
            self.storage.save("_entity_relationships", dict(self.relationships))
            self.storage.save("_entity_mentions", dict(self.entity_mentions))
