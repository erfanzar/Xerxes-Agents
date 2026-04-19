# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Hybrid memory retrieval — fuses semantic, lexical, and recency signals.

Hermes-style "what's relevant now" recall blends three signals:

1. **Semantic similarity** (cosine over embeddings) — captures meaning.
2. **BM25-lite lexical match** — catches exact phrases the embedder smooths over.
3. **Recency decay** — newer memories matter more.

Each :class:`MemoryItem` is scored::

    score = w_sem * cosine + w_bm25 * bm25 + w_recency * decay

Weights default to ``(0.55, 0.30, 0.15)`` and can be tuned per call.
"""

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
    """Weighting for hybrid retrieval components.

    Attributes:
        semantic: Weight for cosine similarity over embeddings.
        bm25: Weight for the BM25-lite lexical score.
        recency: Weight for time-decay (newer = higher).
    """

    semantic: float = 0.55
    bm25: float = 0.30
    recency: float = 0.15

    def normalised(self) -> RetrievalWeights:
        """Return a copy whose weights sum to 1.0."""
        total = self.semantic + self.bm25 + self.recency
        if total == 0.0:
            return RetrievalWeights(0.55, 0.30, 0.15)
        return RetrievalWeights(self.semantic / total, self.bm25 / total, self.recency / total)


@dataclass
class RetrievalResult:
    """A scored item returned by :class:`HybridRetriever`.

    Attributes:
        item: The original :class:`MemoryItem`.
        score: Composite score in ``[0, 1]``.
        semantic_score: Raw cosine similarity (or 0 if no embedder).
        bm25_score: Raw BM25-lite score (normalised to ``[0, 1]``).
        recency_score: Raw recency decay (``[0, 1]``).
    """

    item: MemoryItem
    score: float
    semantic_score: float
    bm25_score: float
    recency_score: float


class HybridRetriever:
    """Blends semantic, BM25-lite, and recency signals into one score.

    Stateless apart from the embedder reference; callers pass items at
    query time. For very large corpora, prefer wrapping a vector store
    that returns top-K candidates and feed those into :meth:`rank`.

    Example:
        >>> from xerxes.memory import HybridRetriever, HashEmbedder
        >>> r = HybridRetriever(embedder=HashEmbedder())
        >>> ranked = r.rank("what is the project deadline", items=memories, k=5)
        >>> for hit in ranked:
        ...     print(hit.score, hit.item.content)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        weights: RetrievalWeights | None = None,
        recency_half_life_days: float = 14.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedder: Embedder for the semantic component. Defaults to
                :func:`get_default_embedder`.
            weights: Component weights. Defaults to ``(0.55, 0.30, 0.15)``.
            recency_half_life_days: Days after which the recency score
                decays to 0.5.
            bm25_k1: BM25 ``k1`` parameter (term frequency saturation).
            bm25_b: BM25 ``b`` parameter (length normalisation).
        """
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
        """Score and rank items against the query.

        Args:
            query: Free-text query.
            items: Candidate :class:`MemoryItem` collection.
            k: Maximum number of results.
            now: Reference timestamp for recency. Defaults to ``datetime.now()``.

        Returns:
            Top-K :class:`RetrievalResult` instances by descending score.
        """
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
        """Half-life decay: ``2 ** (-age_days / half_life)``."""
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return float(2.0 ** (-age_days / max(self.recency_half_life_days, 0.001)))

    def _bm25_lite(self, query: str, items: tp.Sequence[MemoryItem]) -> list[float]:
        """Compute a simplified BM25 score for each item.

        Uses standard Okapi-BM25 over a single field (``item.content``)
        with ``k1`` and ``b`` configured at construction time. IDF is
        computed in-batch over the candidate set — fine for top-K
        candidates returned from a vector pre-filter.

        Args:
            query: Query text.
            items: Candidate items.

        Returns:
            One BM25 score per item, in the same order as ``items``.
        """
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
        """Lowercase whitespace tokenisation, alpha-numeric filtered."""
        out: list[str] = []
        for raw in text.lower().split():
            tok = "".join(c for c in raw if c.isalnum())
            if tok:
                out.append(tok)
        return out
