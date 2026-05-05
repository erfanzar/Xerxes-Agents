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
"""Tests for the MemoryStorage.semantic_search protocol method."""

from __future__ import annotations

from xerxes.memory.embedders import HashEmbedder
from xerxes.memory.storage import FileStorage, RAGStorage, SimpleStorage, SQLiteStorage


class TestSemanticSearchProtocol:
    def test_simple_storage_returns_empty(self):
        s = SimpleStorage()
        assert s.semantic_search("anything") == []
        assert s.supports_semantic_search() is False

    def test_sqlite_storage_returns_empty(self, tmp_path):
        s = SQLiteStorage(str(tmp_path / "m.db"))
        assert s.semantic_search("anything") == []
        assert s.supports_semantic_search() is False

    def test_file_storage_returns_empty(self, tmp_path):
        s = FileStorage(str(tmp_path / "fs"))
        assert s.semantic_search("anything") == []
        assert s.supports_semantic_search() is False

    def test_rag_storage_supports_semantic_search(self):
        rag = RAGStorage(SimpleStorage(), embedder=HashEmbedder())
        assert rag.supports_semantic_search() is True
        rag.save("k1", "the quick brown fox")
        rag.save("k2", "completely unrelated text")
        results = rag.semantic_search("the quick", limit=1)
        assert len(results) == 1
        assert results[0][0] == "k1"

    def test_semantic_search_threshold_filter(self):
        rag = RAGStorage(SimpleStorage(), embedder=HashEmbedder())
        rag.save("k1", "alpha beta gamma")
        results = rag.semantic_search("zzz qqq xxx", limit=10, threshold=0.99)
        assert results == [] or all(r[1] >= 0.99 for r in results)

    def test_semantic_search_alias_matches_search_similar(self):
        rag = RAGStorage(SimpleStorage(), embedder=HashEmbedder())
        rag.save("a", "apple banana")
        rag.save("b", "carrot daikon")
        a = rag.semantic_search("apple", limit=2)
        b = rag.search_similar("apple", limit=2)
        assert [(k, s) for k, s, _ in a] == [(k, s) for k, s, _ in b]
