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


"""Entity memory for tracking information about specific entities"""

import re
from collections import defaultdict
from typing import Any

from .base import Memory, MemoryItem


class EntityMemory(Memory):
    """
    Memory system for tracking entities (people, organizations, concepts).
    Maintains a knowledge graph of entities and their relationships.
    """

    def __init__(
        self,
        storage=None,
        max_items: int = 5000,
        enable_embeddings: bool = False,
    ):
        """Initialize entity memory"""
        super().__init__(storage=storage, max_items=max_items, enable_embeddings=enable_embeddings)
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.entity_mentions: dict[str, list[str]] = defaultdict(list)

    def save(
        self, content: str, metadata: dict[str, Any] | None = None, entities: list[str] | None = None, **kwargs
    ) -> MemoryItem:
        """
        Save memory with entity extraction.

        Args:
            content: Content containing entities
            metadata: Additional metadata
            entities: Pre-identified entities (optional)
            **kwargs: Additional fields

        Returns:
            Created memory item
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
        """
        Search memories related to entities.

        Args:
            query: Search query (can be entity name)
            limit: Maximum results
            filters: Additional filters
            entity_filter: Filter by specific entities

        Returns:
            List of relevant memories
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
        """Retrieve memories"""
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
        """Update memory item"""
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
        """Delete memories"""
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
        """Clear all memories"""
        self._items.clear()
        self._index.clear()
        self.entities.clear()
        self.relationships.clear()
        self.entity_mentions.clear()

        if self.storage:
            for key in self.storage.list_keys("entity_"):
                self.storage.delete(key)

    def get_entity_info(self, entity: str) -> dict[str, Any]:
        """Get all information about an entity"""
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
        """Get entities related to the given entity"""
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
        """Extract entities from text (simple implementation)"""

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
        """Extract relationships between entities"""
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

    def _update_entity(self, entity: str, memory_item: MemoryItem):
        """Update entity information"""
        if entity not in self.entities:
            self.entities[entity] = {"first_seen": memory_item.timestamp, "frequency": 0, "contexts": []}

        self.entities[entity]["frequency"] += 1
        self.entities[entity]["last_seen"] = memory_item.timestamp
        self.entities[entity]["contexts"].append(memory_item.content[:100])
        self.entity_mentions[entity].append(memory_item.memory_id)

    def _save_entity_data(self):
        """Save entity data to storage"""
        if self.storage:
            self.storage.save("_entity_entities", self.entities)
            self.storage.save("_entity_relationships", dict(self.relationships))
            self.storage.save("_entity_mentions", dict(self.entity_mentions))
