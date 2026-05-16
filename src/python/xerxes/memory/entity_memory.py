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
"""Entity-centric memory tier with lightweight relationship tracking.

``EntityMemory`` extracts noun-phrase and quoted entities from saved
content, records first/last-seen timestamps and mention counts per
entity, and infers a handful of common relationships (works_with,
knows, created, ...) via regex patterns. It exposes graph-style
queries like ``get_entity_info`` and ``get_related_entities``."""

import re
from collections import defaultdict
from typing import Any

from .base import Memory, MemoryItem


class EntityMemory(Memory):
    """Memory tier that indexes content by extracted entities and relations."""

    def __init__(
        self,
        storage: Any | None = None,
        max_items: int = 5000,
        enable_embeddings: bool = False,
    ) -> None:
        """Initialise the entity tier and its auxiliary indexes.

        Args:
            storage: Optional ``MemoryStorage`` for durable persistence.
            max_items: Soft cap on retained items.
            enable_embeddings: Whether to compute embeddings on save."""

        super().__init__(storage=storage, max_items=max_items, enable_embeddings=enable_embeddings)
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.entity_mentions: dict[str, list[str]] = defaultdict(list)

    def save(
        self, content: str, metadata: dict[str, Any] | None = None, entities: list[str] | None = None, **kwargs
    ) -> MemoryItem:
        """Save ``content`` and index it under the supplied or extracted entities.

        Args:
            content: Text payload to remember.
            metadata: Annotations; receives an ``entities`` field.
            entities: Pre-extracted entity list; when omitted, entities
                are inferred from ``content`` via ``_extract_entities``.
            **kwargs: Accepted for protocol compatibility; unused."""

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
        """Return items whose indexed entities overlap with the query's.

        Each candidate item's ``relevance_score`` is set to the overlap
        ratio against the target entity set before sorting.

        Args:
            query: Text whose entities are extracted as the target set
                unless ``entity_filter`` is supplied.
            limit: Maximum results.
            filters: Attribute-level filters applied to items.
            entity_filter: Explicit entity allow-list overriding query
                extraction."""

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
        """Fetch by id, or scan items for the first ``limit`` matching ``filters``."""

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
        """Apply ``updates`` to the item, re-indexing entities when content changes."""

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
        """Remove the item with ``memory_id`` and unlink its entity mentions.

        Bulk filter-based deletion is not implemented; ``filters`` is
        accepted for protocol compatibility but ignored. Returns 1 on
        successful single delete, 0 otherwise."""

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
        """Drop every item, entity, relationship, and mention; purge storage entries."""

        self._items.clear()
        self._index.clear()
        self.entities.clear()
        self.relationships.clear()
        self.entity_mentions.clear()

        if self.storage:
            for key in self.storage.list_keys("entity_"):
                self.storage.delete(key)

    def get_entity_info(self, entity: str) -> dict[str, Any]:
        """Return a snapshot of ``entity``'s tracked metadata, mentions, and relations."""

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
        """Walk the relationship graph from ``entity`` up to ``max_depth`` hops.

        Returns the set of entities reachable through any relationship
        (in either direction), excluding ``entity`` itself."""

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
        """Heuristic entity extraction: capitalised phrases + quoted strings."""

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
        """Match a handful of regex relationship templates over ``text``.

        Returns ``(subject, relation, object)`` triples whose subject
        and object both appear in ``entities``."""

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
        """Increment the entity's frequency and append a context snippet."""

        if entity not in self.entities:
            self.entities[entity] = {"first_seen": memory_item.timestamp, "frequency": 0, "contexts": []}

        self.entities[entity]["frequency"] += 1
        self.entities[entity]["last_seen"] = memory_item.timestamp
        self.entities[entity]["contexts"].append(memory_item.content[:100])
        self.entity_mentions[entity].append(memory_item.memory_id)

    def _save_entity_data(self) -> None:
        """Persist the entity/relationship/mention tables to storage."""

        if self.storage:
            self.storage.save("_entity_entities", self.entities)
            self.storage.save("_entity_relationships", dict(self.relationships))
            self.storage.save("_entity_mentions", dict(self.entity_mentions))
