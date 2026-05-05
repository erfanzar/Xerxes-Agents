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
"""Tests for RAGStorage embedding persistence and Embedder integration."""

from __future__ import annotations

from xerxes.memory.embedders import HashEmbedder
from xerxes.memory.storage import FileStorage, RAGStorage, SimpleStorage


class TestRAGStoragePersistence:
    def test_embedding_persisted_alongside_data(self):
        backend = SimpleStorage()
        rag = RAGStorage(backend, embedder=HashEmbedder())
        rag.save("doc1", "the quick brown fox")
        emb_key = rag.EMBEDDING_KEY_PREFIX + "doc1"
        stored = backend.load(emb_key)
        assert stored is not None
        assert isinstance(stored, list)
        assert len(stored) == 256

    def test_embeddings_restored_on_new_instance(self, tmp_path):
        backend = FileStorage(str(tmp_path / "store"))
        rag1 = RAGStorage(backend, embedder=HashEmbedder())
        rag1.save("doc1", "the quick brown fox")
        rag1.save("doc2", "lazy dog jumps over moon")

        backend2 = FileStorage(str(tmp_path / "store"))
        rag2 = RAGStorage(backend2, embedder=HashEmbedder())
        assert "doc1" in rag2.embeddings
        assert "doc2" in rag2.embeddings
        results = rag2.search_similar("the quick", limit=2)
        keys = [r[0] for r in results]
        assert "doc1" in keys

    def test_explicit_embedder_takes_precedence(self):
        called = {"count": 0}

        class CountingEmbedder(HashEmbedder):
            def embed(self, text):
                called["count"] += 1
                return super().embed(text)

        rag = RAGStorage(SimpleStorage(), embedder=CountingEmbedder())
        rag.save("k1", "hello world")
        assert called["count"] == 1

    def test_delete_removes_embedding(self):
        backend = SimpleStorage()
        rag = RAGStorage(backend, embedder=HashEmbedder())
        rag.save("doc1", "hello world")
        emb_key = rag.EMBEDDING_KEY_PREFIX + "doc1"
        assert backend.load(emb_key) is not None
        rag.delete("doc1")
        assert backend.load(emb_key) is None
        assert "doc1" not in rag.embeddings

    def test_clear_removes_persisted_embeddings(self):
        backend = SimpleStorage()
        rag = RAGStorage(backend, embedder=HashEmbedder())
        rag.save("d1", "alpha")
        rag.save("d2", "beta")
        rag.clear()
        assert rag.embeddings == {}
        assert backend.list_keys(rag.EMBEDDING_KEY_PREFIX) == []

    def test_list_keys_hides_embedding_entries(self):
        backend = SimpleStorage()
        rag = RAGStorage(backend, embedder=HashEmbedder())
        rag.save("d1", "alpha")
        rag.save("d2", "beta")
        keys = rag.list_keys()
        assert set(keys) == {"d1", "d2"}

    def test_save_under_emb_prefix_does_not_recurse(self):
        backend = SimpleStorage()
        rag = RAGStorage(backend, embedder=HashEmbedder())
        rag.save(rag.EMBEDDING_KEY_PREFIX + "raw", [0.1, 0.2])
        assert backend.load(rag.EMBEDDING_KEY_PREFIX + "raw") == [0.1, 0.2]

    def test_search_similar_works_after_restore(self, tmp_path):
        backend1 = FileStorage(str(tmp_path / "store"))
        rag1 = RAGStorage(backend1, embedder=HashEmbedder())
        rag1.save("apple", "apple banana cherry")
        rag1.save("car", "car bus train")
        backend2 = FileStorage(str(tmp_path / "store"))
        rag2 = RAGStorage(backend2, embedder=HashEmbedder())
        results = rag2.search_similar("car bus", limit=1)
        assert len(results) >= 1
        assert results[0][0] == "car"
