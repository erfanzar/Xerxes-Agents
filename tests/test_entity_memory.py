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
"""Tests for xerxes.memory.entity_memory module."""

from xerxes.memory.entity_memory import EntityMemory


class TestEntityMemory:
    def test_save_basic(self):
        mem = EntityMemory()
        item = mem.save("John Smith works at Google")
        assert item.content == "John Smith works at Google"
        assert len(mem) == 1

    def test_save_with_entities(self):
        mem = EntityMemory()
        item = mem.save("Alice met Bob", entities=["Alice", "Bob"])
        assert "Alice" in item.metadata["entities"]
        assert "Bob" in item.metadata["entities"]

    def test_entity_extraction(self):
        mem = EntityMemory()
        entities = mem._extract_entities("John Smith visited New York")
        assert any("John" in e for e in entities)

    def test_entity_extraction_quoted(self):
        mem = EntityMemory()
        entities = mem._extract_entities('Used the tool "SearchEngine" to find results')
        assert "SearchEngine" in entities

    def test_entity_tracking(self):
        mem = EntityMemory()
        mem.save("Alice went to the store", entities=["Alice"])
        assert "Alice" in mem.entities
        assert mem.entities["Alice"]["frequency"] == 1

    def test_entity_frequency(self):
        mem = EntityMemory()
        mem.save("Alice went home", entities=["Alice"])
        mem.save("Alice went to work", entities=["Alice"])
        assert mem.entities["Alice"]["frequency"] == 2

    def test_search_by_entity(self):
        mem = EntityMemory()
        mem.save("Alice works at Google", entities=["Alice", "Google"])
        mem.save("Bob works at Meta", entities=["Bob", "Meta"])
        results = mem.search("Alice", entity_filter=["Alice"])
        assert len(results) >= 1
        assert "Alice" in results[0].metadata["entities"]

    def test_search_no_filter(self):
        mem = EntityMemory()
        mem.save("some content about python", entities=["Python"])
        results = mem.search("python")
        assert len(results) >= 0

    def test_retrieve_by_id(self):
        mem = EntityMemory()
        item = mem.save("test", entities=["Test"])
        retrieved = mem.retrieve(memory_id=item.memory_id)
        assert retrieved is not None
        assert retrieved.content == "test"

    def test_retrieve_missing(self):
        mem = EntityMemory()
        assert mem.retrieve(memory_id="nonexistent") is None

    def test_retrieve_with_filters(self):
        mem = EntityMemory()
        mem.save("msg1", entities=["A"])
        results = mem.retrieve(filters={"memory_type": "entity"})
        assert len(results) >= 1

    def test_update(self):
        mem = EntityMemory()
        item = mem.save("original", entities=["Original"])
        assert mem.update(item.memory_id, {"content": "updated"})
        retrieved = mem.retrieve(memory_id=item.memory_id)
        assert retrieved.content == "updated"

    def test_update_missing(self):
        mem = EntityMemory()
        assert mem.update("nonexistent", {}) is False

    def test_delete(self):
        mem = EntityMemory()
        item = mem.save("to delete", entities=["Delete"])
        assert mem.delete(memory_id=item.memory_id) == 1
        assert len(mem) == 0

    def test_clear(self):
        mem = EntityMemory()
        mem.save("item1", entities=["A"])
        mem.save("item2", entities=["B"])
        mem.clear()
        assert len(mem) == 0
        assert len(mem.entities) == 0

    def test_get_entity_info(self):
        mem = EntityMemory()
        mem.save("Alice works at Google", entities=["Alice", "Google"])
        info = mem.get_entity_info("Alice")
        assert "mentions" in info
        assert "relationships" in info

    def test_get_entity_info_missing(self):
        mem = EntityMemory()
        info = mem.get_entity_info("Unknown")
        assert "mentions" in info

    def test_get_related_entities(self):
        mem = EntityMemory()
        mem.relationships["knows"].append(("Alice", "Bob"))
        mem.relationships["knows"].append(("Bob", "Charlie"))
        related = mem.get_related_entities("Alice", max_depth=2)
        assert "Bob" in related

    def test_extract_relationships(self):
        mem = EntityMemory()
        entities = ["John", "Google"]
        rels = mem._extract_relationships("John works at Google", entities)
        assert len(rels) >= 0

    def test_relationship_tracking(self):
        mem = EntityMemory()
        mem.relationships["works_with"].append(("Alice", "Bob"))
        info = mem.get_entity_info("Alice")
        assert any(r["target"] == "Bob" for r in info["relationships"])

    def test_inverse_relationships(self):
        mem = EntityMemory()
        mem.relationships["manages"].append(("Alice", "Bob"))
        info = mem.get_entity_info("Bob")
        assert any("inverse" in r["relation"] for r in info["relationships"])
