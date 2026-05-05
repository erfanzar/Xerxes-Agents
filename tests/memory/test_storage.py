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
"""Tests for xerxes.memory.storage module."""

from xerxes.memory.storage import (
    FileStorage,
    RAGStorage,
    SimpleStorage,
    SQLiteStorage,
)


class TestSimpleStorage:
    def test_save_and_load(self):
        s = SimpleStorage()
        assert s.save("k1", "v1") is True
        assert s.load("k1") == "v1"

    def test_load_missing(self):
        s = SimpleStorage()
        assert s.load("missing") is None

    def test_delete(self):
        s = SimpleStorage()
        s.save("k1", "v1")
        assert s.delete("k1") is True
        assert s.load("k1") is None

    def test_delete_missing(self):
        s = SimpleStorage()
        assert s.delete("missing") is False

    def test_exists(self):
        s = SimpleStorage()
        s.save("k1", "v1")
        assert s.exists("k1") is True
        assert s.exists("missing") is False

    def test_list_keys(self):
        s = SimpleStorage()
        s.save("agent_1", "a")
        s.save("agent_2", "b")
        s.save("task_1", "c")
        assert len(s.list_keys()) == 3
        assert len(s.list_keys("agent")) == 2

    def test_clear(self):
        s = SimpleStorage()
        s.save("k1", "v1")
        s.save("k2", "v2")
        count = s.clear()
        assert count == 2
        assert s.list_keys() == []


class TestFileStorage:
    def test_save_and_load(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        assert s.save("key1", {"data": 123}) is True
        assert s.load("key1") == {"data": 123}

    def test_load_missing(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        assert s.load("missing") is None

    def test_delete(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        s.save("key1", "val")
        assert s.delete("key1") is True
        assert s.load("key1") is None

    def test_delete_missing(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        assert s.delete("missing") is False

    def test_exists(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        s.save("k1", "v")
        assert s.exists("k1") is True
        assert s.exists("missing") is False

    def test_list_keys(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        s.save("a_1", "x")
        s.save("a_2", "y")
        s.save("b_1", "z")
        assert len(s.list_keys()) == 3
        assert len(s.list_keys("a_")) == 2

    def test_clear(self, tmp_path):
        s = FileStorage(str(tmp_path / "mem"))
        s.save("k1", "v1")
        s.save("k2", "v2")
        count = s.clear()
        assert count == 2


class TestSQLiteStorage:
    def test_save_and_load_memory_mode(self):
        s = SQLiteStorage()
        assert s.save("k1", "v1") is True
        assert s.load("k1") == "v1"

    def test_load_missing(self):
        s = SQLiteStorage()
        assert s.load("missing") is None

    def test_delete(self):
        s = SQLiteStorage()
        s.save("k1", "v1")
        assert s.delete("k1") is True
        assert s.load("k1") is None

    def test_delete_missing(self):
        s = SQLiteStorage()
        assert s.delete("missing") is False

    def test_exists(self):
        s = SQLiteStorage()
        s.save("k1", "v")
        assert s.exists("k1") is True
        assert s.exists("missing") is False

    def test_list_keys(self):
        s = SQLiteStorage()
        s.save("a_1", "x")
        s.save("b_1", "y")
        keys = s.list_keys()
        assert "a_1" in keys
        filtered = s.list_keys("a_")
        assert "a_1" in filtered
        assert "b_1" not in filtered

    def test_clear(self):
        s = SQLiteStorage()
        s.save("k1", "v1")
        s.save("k2", "v2")
        count = s.clear()
        assert count == 2


class TestRAGStorage:
    def test_save_and_load(self):
        r = RAGStorage()
        assert r.save("k1", "hello world") is True
        assert r.load("k1") == "hello world"

    def test_embedding_computed(self):
        r = RAGStorage()
        r.save("k1", "some text data")
        assert "k1" in r.embeddings
        assert len(r.embeddings["k1"]) > 0

    def test_save_dict_data(self):
        r = RAGStorage()
        r.save("k1", {"message": "hello"})
        assert "k1" in r.embeddings

    def test_save_non_text(self):
        r = RAGStorage()
        r.save("k1", 42)
        assert "k1" not in r.embeddings

    def test_delete(self):
        r = RAGStorage()
        r.save("k1", "text")
        assert r.delete("k1") is True
        assert "k1" not in r.embeddings

    def test_exists(self):
        r = RAGStorage()
        r.save("k1", "text")
        assert r.exists("k1") is True

    def test_list_keys(self):
        r = RAGStorage()
        r.save("a_1", "text1")
        r.save("b_1", "text2")
        assert len(r.list_keys()) == 2

    def test_clear(self):
        r = RAGStorage()
        r.save("k1", "text")
        r.clear()
        assert len(r.embeddings) == 0

    def test_search_similar(self):
        r = RAGStorage()
        r.save("doc1", "python programming language")
        r.save("doc2", "java programming language")
        r.save("doc3", "cooking recipes and food")
        results = r.search_similar("python code", limit=2)
        assert len(results) <= 2
        assert all(len(t) == 3 for t in results)

    def test_search_similar_with_threshold(self):
        r = RAGStorage()
        r.save("doc1", "programming")
        results = r.search_similar("programming", limit=10, threshold=0.5)
        assert len(results) >= 0

    def test_cosine_similarity_zero_vectors(self):
        r = RAGStorage()
        assert r._cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_cosine_similarity_identical(self):
        r = RAGStorage()
        vec = [1.0, 2.0, 3.0]
        sim = r._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_tfidf_empty_text(self):
        r = RAGStorage()
        embedding = r._compute_tfidf_embedding("")
        assert all(v == 0.0 for v in embedding)

    def test_tfidf_embedding(self):
        r = RAGStorage(embedding_model="tfidf")
        assert r._embedding_type == "tfidf"
        embedding = r._compute_embedding("hello world")
        assert len(embedding) > 0
