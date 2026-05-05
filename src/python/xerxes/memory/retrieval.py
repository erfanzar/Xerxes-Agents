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
"""Retrieval module for Xerxes.

Exports:
    - logger
    - RetrievalWeights
    - RetrievalResult
    - HybridRetriever"""

from __future__ import annotations

import logging
import math
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from .base import MemoryItem
from .embedders import Embedder, cosine_similarity, get_default_embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalWeights:
    """Retrieval weights.

    Attributes:
        semantic (float): semantic.
        bm25 (float): bm25.
        recency (float): recency."""

    semantic: float = 0.55
    bm25: float = 0.30
    recency: float = 0.15

    def normalised(self) -> RetrievalWeights:
        """Normalised.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            RetrievalWeights: OUT: Result of the operation."""

        total = self.semantic + self.bm25 + self.recency
        if total == 0.0:
            return RetrievalWeights(0.55, 0.30, 0.15)
        return RetrievalWeights(self.semantic / total, self.bm25 / total, self.recency / total)


@dataclass
class RetrievalResult:
    """Retrieval result.

    Attributes:
        item (MemoryItem): item.
        score (float): score.
        semantic_score (float): semantic score.
        bm25_score (float): bm25 score.
        recency_score (float): recency score."""

    item: MemoryItem
    score: float
    semantic_score: float
    bm25_score: float
    recency_score: float


class HybridRetriever:
    """Hybrid retriever."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        weights: RetrievalWeights | None = None,
        recency_half_life_days: float = 14.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            embedder (Embedder | None, optional): IN: embedder. Defaults to None. OUT: Consumed during execution.
            weights (RetrievalWeights | None, optional): IN: weights. Defaults to None. OUT: Consumed during execution.
            recency_half_life_days (float, optional): IN: recency half life days. Defaults to 14.0. OUT: Consumed during execution.
            bm25_k1 (float, optional): IN: bm25 k1. Defaults to 1.5. OUT: Consumed during execution.
            bm25_b (float, optional): IN: bm25 b. Defaults to 0.75. OUT: Consumed during execution."""

        self.embedder = embedder or get_default_embedder()
        self.weights = (weights or RetrievalWeights()).normalised()
        self.recency_half_life_days = recency_half_life_days
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def rank(
        self,
        query: str,
        items: tp.Sequence[MemoryItem],
        k: int = 10,
        now: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Rank.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            items (tp.Sequence[MemoryItem]): IN: items. OUT: Consumed during execution.
            k (int, optional): IN: k. Defaults to 10. OUT: Consumed during execution.
            now (datetime | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[RetrievalResult]: OUT: Result of the operation."""

        if not items:
            return []
        now = now or datetime.now()
        try:
            qvec = self.embedder.embed(query)
        except Exception:
            logger.warning("Embedder failed for query; semantic component disabled", exc_info=True)
            qvec = None
        bm25_scores = self._bm25_lite(query, items)
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        results: list[RetrievalResult] = []
        for item, bm25 in zip(items, bm25_scores, strict=False):
            sem = 0.0
            if qvec is not None and item.embedding:
                sem = max(0.0, cosine_similarity(qvec, item.embedding))
            elif qvec is not None and not item.embedding:
                try:
                    item_vec = self.embedder.embed(item.content)
                    sem = max(0.0, cosine_similarity(qvec, item_vec))
                except Exception:
                    sem = 0.0
            bm = (bm25 / max_bm25) if max_bm25 > 0 else 0.0
            rec = self._recency(item.timestamp, now)
            score = self.weights.semantic * sem + self.weights.bm25 * bm + self.weights.recency * rec
            results.append(
                RetrievalResult(
                    item=item,
                    score=score,
                    semantic_score=sem,
                    bm25_score=bm,
                    recency_score=rec,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _recency(self, timestamp: datetime, now: datetime) -> float:
        """Internal helper to recency.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            timestamp (datetime): IN: timestamp. OUT: Consumed during execution.
            now (datetime): IN: now. OUT: Consumed during execution.
        Returns:
            float: OUT: Result of the operation."""

        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return float(2.0 ** (-age_days / max(self.recency_half_life_days, 0.001)))

    def _bm25_lite(self, query: str, items: tp.Sequence[MemoryItem]) -> list[float]:
        """Internal helper to bm25 lite.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            items (tp.Sequence[MemoryItem]): IN: items. OUT: Consumed during execution.
        Returns:
            list[float]: OUT: Result of the operation."""

        q_terms = self._tokenize(query)
        if not q_terms:
            return [0.0] * len(items)
        docs = [self._tokenize(it.content) for it in items]
        n_docs = len(docs)
        doc_lens = [len(d) for d in docs]
        avgdl = (sum(doc_lens) / n_docs) if n_docs else 1.0
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        scores = []
        for d, dl in zip(docs, doc_lens, strict=False):
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for q in set(q_terms):
                if q not in tf:
                    continue
                idf = math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5))
                num = tf[q] * (self.bm25_k1 + 1.0)
                denom = tf[q] + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * dl / max(avgdl, 1.0))
                score += idf * num / max(denom, 1e-9)
            scores.append(score)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Internal helper to tokenize.

        Args:
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        out: list[str] = []
        for raw in text.lower().split():
            tok = "".join(c for c in raw if c.isalnum())
            if tok:
                out.append(tok)
        return out
