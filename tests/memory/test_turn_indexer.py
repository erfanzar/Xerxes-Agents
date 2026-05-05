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
"""Tests for the turn_indexer hook factory and memory provider."""

from __future__ import annotations

from xerxes.extensions.hooks import HookRunner
from xerxes.memory import LongTermMemory, make_memory_provider, make_turn_indexer_hook
from xerxes.memory.storage import SimpleStorage


class _FakeMemory:
    def __init__(self):
        self.saved = []

    def save(self, content, metadata=None, **kwargs):
        self.saved.append({"content": content, "metadata": metadata, **kwargs})

        class _Item:
            memory_id = f"item-{len(self.saved)}"

        return _Item()

    def search(self, query, limit=10, **kwargs):
        from xerxes.memory.base import MemoryItem

        return [MemoryItem(content=s["content"]) for s in self.saved[:limit]]


class TestTurnIndexerHook:
    def test_indexes_string_response(self):
        mem = _FakeMemory()
        hook = make_turn_indexer_hook(mem, min_chars=1)
        hook(agent_id="a1", response="A long enough response from the agent")
        assert len(mem.saved) == 1
        assert "long enough response" in mem.saved[0]["content"]
        assert mem.saved[0]["agent_id"] == "a1"

    def test_skips_short_responses(self):
        mem = _FakeMemory()
        hook = make_turn_indexer_hook(mem, min_chars=20)
        hook(agent_id="a1", response="ok")
        assert mem.saved == []

    def test_extracts_dict_content(self):
        mem = _FakeMemory()
        hook = make_turn_indexer_hook(mem, min_chars=1)
        hook(agent_id="a", response={"content": "hello from a dict response"})
        assert "hello from a dict response" in mem.saved[0]["content"]

    def test_extracts_object_content_attr(self):
        class Resp:
            content = "object content here long enough"

        mem = _FakeMemory()
        hook = make_turn_indexer_hook(mem, min_chars=1)
        hook(agent_id="a", response=Resp())
        assert "object content here" in mem.saved[0]["content"]

    def test_handles_none_response(self):
        mem = _FakeMemory()
        hook = make_turn_indexer_hook(mem, min_chars=1)
        hook(agent_id="a", response=None)
        assert mem.saved == []

    def test_swallows_save_errors(self):
        class BrokenMemory:
            def save(self, **kw):
                raise RuntimeError("disk full")

        hook = make_turn_indexer_hook(BrokenMemory(), min_chars=1)
        hook(agent_id="a", response="some content here")

    def test_integrates_with_hookrunner(self):
        mem = _FakeMemory()
        runner = HookRunner()
        runner.register("on_turn_end", make_turn_indexer_hook(mem, min_chars=1))
        runner.run("on_turn_end", agent_id="x", response="hello world here")
        assert len(mem.saved) == 1


class TestMemoryProvider:
    def test_returns_strings_from_memory_search(self):
        mem = _FakeMemory()
        mem.save(content="alpha")
        mem.save(content="beta")
        provider = make_memory_provider(mem, use_semantic=False)
        out = provider("agent", 5)
        assert "alpha" in out and "beta" in out

    def test_swallows_search_errors(self):
        class BrokenMemory:
            def search(self, *a, **kw):
                raise RuntimeError("boom")

        provider = make_memory_provider(BrokenMemory())
        assert provider("a", 5) == []

    def test_end_to_end_with_long_term_memory(self):
        ltm = LongTermMemory(storage=SimpleStorage(), enable_embeddings=False)
        ltm.save("the project deadline is march 15", importance=0.9)
        ltm.save("user prefers terse responses", importance=0.7)
        provider = make_memory_provider(ltm, use_semantic=False)
        snippets = provider("default", 5)
        assert any("project deadline" in s for s in snippets)
