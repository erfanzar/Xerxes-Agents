# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Semantic skill matcher.

Given the user's current task description, ranks skills in the registry
by how well they match. Uses the embedding subsystem so that natural
language overlap counts more than keyword matching. Lazily computes and
caches one embedding per skill (description + tag bag).
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
    """A scored skill match.

    Attributes:
        skill: The matched :class:`Skill` instance.
        score: Cosine similarity in ``[-1, 1]``.
    """

    skill: Skill
    score: float


class SkillMatcher:
    """Embedding-based skill matcher.

    Builds a per-skill embedding lazily on first match and caches it
    keyed by ``(skill.name, skill.metadata.version)``. The cache survives
    repeated matches but is invalidated when a new skill version is
    drafted (the version number changes).

    Example:
        >>> m = SkillMatcher(registry)
        >>> for hit in m.match("set up CI for the project", k=3):
        ...     print(hit.score, hit.skill.name)
    """

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        embedder: Embedder | None = None,
        min_score: float = 0.15,
    ) -> None:
        """Initialise the matcher.

        Args:
            skill_registry: Optional registry to query. If ``None``, only
                explicit ``match(... skills=...)`` calls work.
            embedder: Embedder used for skill+query encoding. Defaults
                to :func:`get_default_embedder`.
            min_score: Skills with cosine similarity below this are
                filtered from results.
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
        """Return the top-K skills relevant to *query*.

        Args:
            query: Free-text description of the task.
            k: Maximum number of matches to return.
            skills: Optional explicit skill list. When ``None``, the
                bound ``skill_registry`` is queried.

        Returns:
            Top-K :class:`SkillMatch` instances sorted by descending
            score, filtered by :attr:`min_score`.
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
        """Return the single best match, or ``None`` if none clear ``min_score``."""
        hits = self.match(query, k=1)
        return hits[0] if hits else None

    def invalidate(self) -> None:
        """Clear the embedding cache (e.g. after registry mutation)."""
        self._cache.clear()

    def _all_skills(self) -> list[Skill]:
        """Return every skill in the registry, or ``[]`` on failure.

        Returns:
            A materialised list of registry skills; empty if no registry
            is attached or ``get_all`` raises.
        """
        if self.registry is None:
            return []
        try:
            return list(self.registry.get_all())
        except Exception:
            return []

    def _embed_skill(self, skill: Skill) -> list[float] | None:
        """Embed *skill* once per ``(name, version)`` and cache the result.

        Args:
            skill: The :class:`Skill` whose searchable text should be embedded.

        Returns:
            The embedding vector, or ``None`` if the skill has no text or
            the embedder fails.
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
        """Concatenate the fields of *skill* that should participate in matching.

        Joins the name, description, tags, and the first 1000 characters
        of the instruction body into a single whitespace-separated string.

        Args:
            skill: The :class:`Skill` to flatten into search text.

        Returns:
            A single string suitable for embedding.
        """
        meta = skill.metadata
        parts = [meta.name, getattr(meta, "description", ""), " ".join(getattr(meta, "tags", []))]
        body = getattr(skill, "instructions", "") or ""
        if body:
            parts.append(body[:1000])
        return " ".join(p for p in parts if p)
