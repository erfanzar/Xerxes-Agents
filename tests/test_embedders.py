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
"""Tests for memory embedder protocol and HashEmbedder fallback."""

from __future__ import annotations

import pytest
from xerxes.memory.embedders import (
    HashEmbedder,
    cosine_similarity,
    get_default_embedder,
    reset_default_embedder,
)


class TestHashEmbedder:
    def test_default_dim_is_256(self):
        e = HashEmbedder()
        v = e.embed("hello world")
        assert len(v) == 256

    def test_custom_dim(self):
        e = HashEmbedder(dim=64)
        v = e.embed("hello")
        assert len(v) == 64

    def test_empty_text_yields_zero_vector(self):
        e = HashEmbedder()
        v = e.embed("")
        assert v == [0.0] * 256

    def test_l2_normalised(self):
        e = HashEmbedder()
        v = e.embed("the quick brown fox jumps over the lazy dog")
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_deterministic(self):
        e = HashEmbedder()
        a = e.embed("hello world")
        b = e.embed("hello world")
        assert a == b

    def test_different_texts_produce_different_vectors(self):
        e = HashEmbedder()
        a = e.embed("the cat sat on the mat")
        b = e.embed("a completely unrelated sentence")
        assert a != b

    def test_batch_matches_single_calls(self):
        e = HashEmbedder()
        texts = ["hello", "world", "foo bar"]
        single = [e.embed(t) for t in texts]
        batch = e.embed_batch(texts)
        assert single == batch

    def test_batch_empty(self):
        e = HashEmbedder()
        assert e.embed_batch([]) == []


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_dims_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


class TestGetDefaultEmbedder:
    def test_forced_hash(self, monkeypatch):
        reset_default_embedder()
        monkeypatch.setenv("XERXES_EMBEDDER", "hash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        e = get_default_embedder()
        assert isinstance(e, HashEmbedder)
        reset_default_embedder()

    def test_falls_back_to_hash_without_st_or_openai(self, monkeypatch):
        reset_default_embedder()
        monkeypatch.delenv("XERXES_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import sys

        st_modules = [m for m in list(sys.modules) if m.startswith("sentence_transformers")]
        for m in st_modules:
            sys.modules[m] = None  # type: ignore[assignment]
        try:
            e = get_default_embedder()
            assert isinstance(e, HashEmbedder)
        finally:
            for m in st_modules:
                sys.modules.pop(m, None)
            reset_default_embedder()

    def test_caches_result(self, monkeypatch):
        reset_default_embedder()
        monkeypatch.setenv("XERXES_EMBEDDER", "hash")
        a = get_default_embedder()
        b = get_default_embedder()
        assert a is b
        reset_default_embedder()
