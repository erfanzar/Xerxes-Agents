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
"""Semantic skill matching using text embeddings.

``SkillMatcher`` embeds a user query and each registered skill, then returns
the best matches sorted by cosine similarity.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

from ...memory.embedders import Embedder, cosine_similarity, get_default_embedder

if tp.TYPE_CHECKING:
    from ..skills import Skill, SkillRegistry
logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """A skill paired with its cosine-similarity score against a query.

    Attributes:
        skill: The matched skill.
        score: Cosine similarity in [-1, 1].
    """

    skill: Skill
    score: float


class SkillMatcher:
    """Embedder-driven skill recommender."""

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        embedder: Embedder | None = None,
        min_score: float = 0.15,
    ) -> None:
        """Initialize the matcher.

        Args:
            skill_registry: Registry whose skills are searched in ``match``.
            embedder: Embedding backend; defaults to the project default.
            min_score: Cosine similarity threshold; lower-scoring hits are dropped.
        """
        self.registry = skill_registry
        self.embedder = embedder or get_default_embedder()
        self.min_score = min_score
        self._cache: dict[tuple[str, str], list[float]] = {}

    def match(
        self,
        query: str,
        k: int = 5,
        skills: tp.Sequence[Skill] | None = None,
    ) -> list[SkillMatch]:
        """Return the top-``k`` skills most similar to ``query``.

        Args:
            query: Free-text query embedded for comparison.
            k: Maximum number of matches to return.
            skills: Candidate list; defaults to all registered skills.

        Returns:
            Matches sorted by cosine similarity, descending.
        """

        if not query:
            return []
        candidates = list(skills) if skills is not None else self._all_skills()
        if not candidates:
            return []
        try:
            qvec = self.embedder.embed(query)
        except Exception:
            logger.warning("Embedder failed for query; matcher returning []", exc_info=True)
            return []
        out: list[SkillMatch] = []
        for skill in candidates:
            svec = self._embed_skill(skill)
            if svec is None:
                continue
            score = cosine_similarity(qvec, svec)
            if score >= self.min_score:
                out.append(SkillMatch(skill=skill, score=score))
        out.sort(key=lambda m: m.score, reverse=True)
        return out[:k]

    def best(self, query: str) -> SkillMatch | None:
        """Return the single highest-scoring match for ``query``, or ``None``."""

        hits = self.match(query, k=1)
        return hits[0] if hits else None

    def invalidate(self) -> None:
        """Drop every cached embedding."""

        self._cache.clear()

    def _all_skills(self) -> list[Skill]:
        """Return every skill from the configured registry."""

        if self.registry is None:
            return []
        try:
            return list(self.registry.get_all())
        except Exception:
            return []

    def _embed_skill(self, skill: Skill) -> list[float] | None:
        """Return the cached or freshly computed embedding vector for ``skill``."""

        meta = skill.metadata
        key = (meta.name, getattr(meta, "version", ""))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        text = self._skill_text(skill)
        if not text:
            return None
        try:
            vec = self.embedder.embed(text)
        except Exception:
            return None
        self._cache[key] = vec
        return vec

    @staticmethod
    def _skill_text(skill: Skill) -> str:
        """Build the text payload used to embed ``skill`` (name, tags, body excerpt)."""

        meta = skill.metadata
        parts = [meta.name, getattr(meta, "description", ""), " ".join(getattr(meta, "tags", []))]
        body = getattr(skill, "instructions", "") or ""
        if body:
            parts.append(body[:1000])
        return " ".join(p for p in parts if p)
