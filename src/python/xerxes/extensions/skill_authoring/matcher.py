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
    """Result of matching a skill against a query.

    Attributes:
        skill (Skill): IN: Matched skill instance. OUT: Stored.
        score (float): IN: Cosine similarity score. OUT: Stored.
    """

    skill: Skill
    score: float


class SkillMatcher:
    """Embedder-based skill recommender.

    Args:
        skill_registry (SkillRegistry | None): IN: Registry to enumerate.
            OUT: Used by ``_all_skills``.
        embedder (Embedder | None): IN: Embedding backend. OUT: Defaults to
            the project default embedder.
        min_score (float): IN: Minimum cosine similarity threshold. OUT:
            Filters results in ``match``.
    """

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        embedder: Embedder | None = None,
        min_score: float = 0.15,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            embedder (Embedder | None, optional): IN: embedder. Defaults to None. OUT: Consumed during execution.
            min_score (float, optional): IN: min score. Defaults to 0.15. OUT: Consumed during execution."""
        self.registry = skill_registry
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            embedder (Embedder | None, optional): IN: embedder. Defaults to None. OUT: Consumed during execution.
            min_score (float, optional): IN: min score. Defaults to 0.15. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            embedder (Embedder | None, optional): IN: embedder. Defaults to None. OUT: Consumed during execution.
            min_score (float, optional): IN: min score. Defaults to 0.15. OUT: Consumed during execution."""
        self.embedder = embedder or get_default_embedder()
        self.min_score = min_score
        self._cache: dict[tuple[str, str], list[float]] = {}

    def match(
        self,
        query: str,
        k: int = 5,
        skills: tp.Sequence[Skill] | None = None,
    ) -> list[SkillMatch]:
        """Find the top-``k`` skills most similar to ``query``.

        Args:
            query (str): IN: Free-text query. OUT: Embedded and compared.
            k (int): IN: Maximum results. OUT: Limits the returned list.
            skills (tp.Sequence[Skill] | None): IN: Optional candidate list.
                OUT: Defaults to all registered skills.

        Returns:
            list[SkillMatch]: OUT: Descending by similarity score.
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
        """Return the single best match for ``query``.

        Args:
            query (str): IN: Free-text query. OUT: Passed to ``match``.

        Returns:
            SkillMatch | None: OUT: Top result or ``None``.
        """

        hits = self.match(query, k=1)
        return hits[0] if hits else None

    def invalidate(self) -> None:
        """Clear the embedding cache.

        Returns:
            None: OUT: All cached embeddings are dropped.
        """

        self._cache.clear()

    def _all_skills(self) -> list[Skill]:
        """Return all skills from the registry.

        Returns:
            list[Skill]: OUT: Empty list if no registry or on error.
        """

        if self.registry is None:
            return []
        try:
            return list(self.registry.get_all())
        except Exception:
            return []

    def _embed_skill(self, skill: Skill) -> list[float] | None:
        """Return the embedding vector for a skill, using the cache.

        Args:
            skill (Skill): IN: Skill to embed. OUT: Text is extracted and
                passed to the embedder.

        Returns:
            list[float] | None: OUT: Embedding vector or ``None`` on failure.
        """

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
        """Concatenate skill metadata into a single text string for embedding.

        Args:
            skill (Skill): IN: Skill to summarise. OUT: Fields are extracted.

        Returns:
            str: OUT: Combined text.
        """

        meta = skill.metadata
        parts = [meta.name, getattr(meta, "description", ""), " ".join(getattr(meta, "tags", []))]
        body = getattr(skill, "instructions", "") or ""
        if body:
            parts.append(body[:1000])
        return " ".join(p for p in parts if p)
