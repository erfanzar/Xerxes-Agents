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
"""Tests for xerxes.memory.base and xerxes.memory.short_term_memory modules."""

from xerxes.memory.base import MemoryItem
from xerxes.memory.short_term_memory import ShortTermMemory


class TestMemoryItem:
    def test_creation(self):
        item = MemoryItem(content="hello")
        assert item.content == "hello"
        assert item.memory_type == "general"
        assert item.memory_id is not None

    def test_to_dict(self):
        item = MemoryItem(content="test", agent_id="a1")
        d = item.to_dict()
        assert d["content"] == "test"
        assert d["agent_id"] == "a1"
        assert "memory_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        item = MemoryItem(content="original", agent_id="a1")
        d = item.to_dict()
        restored = MemoryItem.from_dict(d)
        assert restored.content == "original"
        assert restored.agent_id == "a1"

    def test_from_dict_with_timestamps(self):
        d = {
            "content": "test",
            "timestamp": "2025-01-01T00:00:00",
            "last_accessed": "2025-01-02T00:00:00",
            "memory_id": "abc",
        }
        item = MemoryItem.from_dict(d)
        assert item.content == "test"
        assert item.last_accessed is not None


class TestShortTermMemory:
    def test_save(self):
        mem = ShortTermMemory(capacity=10)
        item = mem.save("hello world")
        assert item.content == "hello world"
        assert len(mem) == 1

    def test_capacity(self):
        mem = ShortTermMemory(capacity=3)
        for i in range(5):
            mem.save(f"msg {i}")
        assert len(mem) == 3

    def test_search_exact(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("python programming")
        mem.save("java development")
        results = mem.search("python")
        assert len(results) >= 1
        assert results[0].content == "python programming"

    def test_search_partial(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("the quick brown fox")
        results = mem.search("quick fox")
        assert len(results) >= 1

    def test_search_with_filters(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("msg1", agent_id="a1")
        mem.save("msg2", agent_id="a2")
        results = mem.search("msg", filters={"agent_id": "a1"})
        assert all(r.agent_id == "a1" for r in results)

    def test_search_limit(self):
        mem = ShortTermMemory(capacity=20)
        for i in range(15):
            mem.save(f"message number {i}")
        results = mem.search("message", limit=5)
        assert len(results) <= 5

    def test_retrieve_by_id(self):
        mem = ShortTermMemory(capacity=10)
        item = mem.save("test content")
        retrieved = mem.retrieve(memory_id=item.memory_id)
        assert retrieved is not None
        assert retrieved.content == "test content"

    def test_retrieve_missing_id(self):
        mem = ShortTermMemory(capacity=10)
        assert mem.retrieve(memory_id="nonexistent") is None

    def test_retrieve_with_filters(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("msg1", agent_id="a1")
        mem.save("msg2", agent_id="a2")
        results = mem.retrieve(filters={"agent_id": "a1"})
        assert len(results) >= 1

    def test_update(self):
        mem = ShortTermMemory(capacity=10)
        item = mem.save("original")
        assert mem.update(item.memory_id, {"content": "updated"})
        retrieved = mem.retrieve(memory_id=item.memory_id)
        assert retrieved.content == "updated"

    def test_update_missing(self):
        mem = ShortTermMemory(capacity=10)
        assert mem.update("nonexistent", {"content": "x"}) is False

    def test_delete_by_id(self):
        mem = ShortTermMemory(capacity=10)
        item = mem.save("to delete")
        assert mem.delete(memory_id=item.memory_id) == 1
        assert len(mem) == 0

    def test_delete_with_filters(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("msg1", agent_id="a1")
        mem.save("msg2", agent_id="a1")
        mem.save("msg3", agent_id="a2")
        count = mem.delete(filters={"agent_id": "a1"})
        assert count == 2
        assert len(mem) == 1

    def test_clear(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("item1")
        mem.save("item2")
        mem.clear()
        assert len(mem) == 0

    def test_get_recent(self):
        mem = ShortTermMemory(capacity=10)
        for i in range(5):
            mem.save(f"msg {i}")
        recent = mem.get_recent(3)
        assert len(recent) == 3

    def test_summarize_empty(self):
        mem = ShortTermMemory(capacity=10)
        assert "No recent" in mem.summarize()

    def test_summarize_with_items(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("hello", agent_id="agent1")
        mem.save("world", agent_id="agent2", conversation_id="conv1")
        summary = mem.summarize()
        assert "hello" in summary or "agent1" in summary

    def test_get_context_text(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("item1", agent_id="a1")
        context = mem.get_context(format_type="text")
        assert "item1" in context

    def test_get_context_json(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("item1")
        context = mem.get_context(format_type="json")
        assert "item1" in context

    def test_get_context_markdown(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("item1", agent_id="a1")
        context = mem.get_context(format_type="markdown")
        assert "item1" in context

    def test_get_statistics(self):
        mem = ShortTermMemory(capacity=10)
        mem.save("msg1", agent_id="a1", user_id="u1")
        mem.save("msg2", agent_id="a2", conversation_id="c1")
        stats = mem.get_statistics()
        assert stats["total_items"] == 2
        assert stats["unique_agents"] == 2

    def test_repr(self):
        mem = ShortTermMemory(capacity=10)
        repr_str = repr(mem)
        assert "ShortTermMemory" in repr_str
