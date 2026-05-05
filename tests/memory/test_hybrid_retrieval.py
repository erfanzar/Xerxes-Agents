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
"""Tests for hybrid (semantic + BM25 + recency) retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta

from xerxes.memory import HashEmbedder, HybridRetriever, MemoryItem, RetrievalWeights


def _item(content: str, days_old: int = 0) -> MemoryItem:
    return MemoryItem(
        content=content,
        timestamp=datetime.now() - timedelta(days=days_old),
    )


class TestRetrievalWeights:
    def test_normalised_sums_to_one(self):
        w = RetrievalWeights(0.6, 0.3, 0.1).normalised()
        assert abs(w.semantic + w.bm25 + w.recency - 1.0) < 1e-9

    def test_zero_weights_use_default(self):
        w = RetrievalWeights(0, 0, 0).normalised()
        assert abs(w.semantic + w.bm25 + w.recency - 1.0) < 1e-9


class TestHybridRetriever:
    def test_empty_items_returns_empty(self):
        r = HybridRetriever(embedder=HashEmbedder())
        assert r.rank("query", []) == []

    def test_returns_top_k(self):
        r = HybridRetriever(embedder=HashEmbedder())
        items = [_item(f"doc number {i}") for i in range(20)]
        results = r.rank("doc number 5", items, k=5)
        assert len(results) == 5

    def test_exact_match_ranks_high(self):
        r = HybridRetriever(embedder=HashEmbedder())
        items = [
            _item("the project deadline is march 15"),
            _item("birthday party planning"),
            _item("grocery list potatoes onions"),
        ]
        results = r.rank("project deadline", items, k=3)
        assert results[0].item.content == "the project deadline is march 15"

    def test_recency_boosts_recent_items(self):
        r = HybridRetriever(
            embedder=HashEmbedder(),
            weights=RetrievalWeights(semantic=0.0, bm25=0.0, recency=1.0),
        )
        items = [
            _item("anything goes here", days_old=100),
            _item("anything goes here", days_old=1),
        ]
        results = r.rank("query", items, k=2)
        assert results[0].item.timestamp > results[1].item.timestamp

    def test_bm25_only_finds_lexical_match(self):
        r = HybridRetriever(
            embedder=HashEmbedder(),
            weights=RetrievalWeights(semantic=0.0, bm25=1.0, recency=0.0),
        )
        items = [
            _item("alpha beta gamma"),
            _item("delta epsilon zeta"),
            _item("alpha alpha alpha"),
        ]
        results = r.rank("alpha", items, k=3)
        assert results[0].item.content.startswith("alpha")

    def test_score_in_unit_range(self):
        r = HybridRetriever(embedder=HashEmbedder())
        items = [_item("hello world")]
        results = r.rank("hello world", items, k=1)
        assert 0.0 <= results[0].score <= 1.0
        assert 0.0 <= results[0].semantic_score <= 1.0
        assert 0.0 <= results[0].bm25_score <= 1.0
        assert 0.0 <= results[0].recency_score <= 1.0

    def test_uses_existing_embedding_when_present(self):
        r = HybridRetriever(embedder=HashEmbedder())
        item = _item("the quick brown fox")
        item.embedding = HashEmbedder().embed(item.content)
        results = r.rank("brown fox", [item], k=1)
        assert results[0].semantic_score > 0
