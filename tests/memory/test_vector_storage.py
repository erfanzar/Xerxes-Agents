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
"""Tests for SQLiteVectorStorage."""

from __future__ import annotations

import pytest
from xerxes.memory import HashEmbedder, SQLiteVectorStorage


@pytest.fixture
def store(tmp_path):
    return SQLiteVectorStorage(str(tmp_path / "v.db"), embedder=HashEmbedder())


class TestSQLiteVectorStorage:
    def test_save_and_load_string(self, store):
        store.save("doc1", "the quick brown fox")
        assert store.load("doc1") == "the quick brown fox"

    def test_save_and_load_dict(self, store):
        d = {"name": "alice", "age": 30}
        store.save("u1", d)
        assert store.load("u1") == d

    def test_load_missing_returns_none(self, store):
        assert store.load("nope") is None

    def test_exists(self, store):
        assert not store.exists("k")
        store.save("k", "v")
        assert store.exists("k")

    def test_delete(self, store):
        store.save("k", "v")
        assert store.delete("k") is True
        assert store.delete("k") is False

    def test_list_keys(self, store):
        store.save("a", "alpha")
        store.save("b", "beta")
        keys = store.list_keys()
        assert set(keys) == {"a", "b"}

    def test_list_keys_pattern(self, store):
        store.save("user:1", "alice")
        store.save("user:2", "bob")
        store.save("post:1", "hello")
        users = store.list_keys("user:")
        assert set(users) == {"user:1", "user:2"}

    def test_clear_returns_count(self, store):
        store.save("a", "x")
        store.save("b", "y")
        assert store.clear() == 2
        assert store.list_keys() == []

    def test_supports_semantic_search(self, store):
        assert store.supports_semantic_search() is True

    def test_semantic_search_finds_relevant(self, store):
        store.save("fruit", "apple banana cherry mango")
        store.save("car", "honda toyota ford bmw")
        store.save("color", "red blue green yellow")
        results = store.semantic_search("apple banana", limit=1)
        assert len(results) == 1
        assert results[0][0] == "fruit"

    def test_semantic_search_threshold(self, store):
        store.save("k1", "alpha beta")
        results = store.semantic_search("xxx yyy zzz", limit=10, threshold=0.99)
        assert all(r[1] >= 0.99 for r in results)

    def test_semantic_search_empty_query(self, store):
        store.save("k1", "anything")
        assert store.semantic_search("") == []

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "p.db")
        s1 = SQLiteVectorStorage(path, embedder=HashEmbedder())
        s1.save("doc", "hello world")
        s2 = SQLiteVectorStorage(path, embedder=HashEmbedder())
        assert s2.load("doc") == "hello world"
        results = s2.semantic_search("hello", limit=1)
        assert len(results) == 1
        assert results[0][0] == "doc"

    def test_replace_on_duplicate_key(self, store):
        store.save("k", "first")
        store.save("k", "second")
        assert store.load("k") == "second"
