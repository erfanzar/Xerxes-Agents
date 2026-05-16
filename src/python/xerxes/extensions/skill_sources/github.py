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
"""GitHub-backed skill source.

Tests avoid hitting GitHub by injecting a ``fetch_callable`` that returns the
raw markdown body for a given ``identifier``.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import SkillBundle, SkillSearchHit, SkillSource

FetchCallable = Callable[[str], str]
SearchCallable = Callable[[str, int], list[dict]]


class GitHubSkillSource(SkillSource):
    """Skill source that loads ``SKILL.md`` files from a GitHub repository."""

    name = "github"

    def __init__(
        self,
        *,
        repo: str = "agentskills/community",
        fetch_callable: FetchCallable | None = None,
        search_callable: SearchCallable | None = None,
    ) -> None:
        """Bind the source to ``repo`` with optional fetch/search callables.

        Args:
            repo: ``owner/name`` of the GitHub repository.
            fetch_callable: Function returning markdown body for an identifier.
            search_callable: Function returning raw search rows for a query.
        """
        self._repo = repo
        self._fetch_fn = fetch_callable
        self._search_fn = search_callable

    def search(self, query: str, *, limit: int = 20) -> list[SkillSearchHit]:
        """Invoke the configured search callable and adapt rows to hits."""
        if self._search_fn is None:
            return []
        raw = self._search_fn(query, limit)
        out: list[SkillSearchHit] = []
        for row in raw:
            out.append(
                SkillSearchHit(
                    name=row["name"],
                    description=row.get("description", ""),
                    source_name=self.name,
                    version=row.get("version", ""),
                    tags=list(row.get("tags", [])),
                )
            )
        return out

    def fetch(self, identifier: str) -> SkillBundle:
        """Build a ``SkillBundle`` from the configured fetch callable.

        Raises:
            RuntimeError: No fetch callable was provided at construction time.
        """
        if self._fetch_fn is None:
            raise RuntimeError("GitHubSkillSource was not configured with a fetch callable")
        body = self._fetch_fn(identifier)
        return SkillBundle(
            name=identifier,
            version="github",
            body_markdown=body,
            metadata={"repo": self._repo},
            source_name=self.name,
        )


__all__ = ["FetchCallable", "GitHubSkillSource", "SearchCallable"]
