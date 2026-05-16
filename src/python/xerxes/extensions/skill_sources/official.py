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
"""Xerxes-curated official skill registry (HTTP-fetched)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import SkillBundle, SkillSearchHit, SkillSource


class OfficialSkillSource(SkillSource):
    """HTTP-backed skill source for the Xerxes-curated registry."""

    name = "official"

    def __init__(self, *, base_url: str = "https://skills.xerxes-agent.dev", client: httpx.Client | None = None) -> None:
        """Bind the source to ``base_url`` and optionally an existing client.

        Args:
            base_url: Base URL of the registry.
            client: Optional pre-configured ``httpx.Client`` for testing.
        """
        self._base_url = base_url
        self._client = client

    def _http(self) -> httpx.Client:
        """Return the injected client or build a fresh ``httpx.Client``."""
        return self._client or httpx.Client(base_url=self._base_url, timeout=20.0)

    def search(self, query: str, *, limit: int = 20) -> list[SkillSearchHit]:
        """Query the ``/search`` endpoint and return at most ``limit`` hits."""
        with self._http() as c:
            try:
                resp = c.get("/search", params={"q": query, "limit": limit})
                resp.raise_for_status()
            except httpx.HTTPError:
                return []
            try:
                rows: list[dict[str, Any]] = resp.json()
            except json.JSONDecodeError:
                return []
        return [
            SkillSearchHit(
                name=r["name"],
                description=r.get("description", ""),
                source_name=self.name,
                version=r.get("version", ""),
                tags=list(r.get("tags", [])),
            )
            for r in rows
        ]

    def fetch(self, identifier: str) -> SkillBundle:
        """Download the ``SKILL.md`` for ``identifier`` from the registry."""
        with self._http() as c:
            resp = c.get(f"/skills/{identifier}/SKILL.md")
            resp.raise_for_status()
            body = resp.text
        return SkillBundle(name=identifier, version="official", body_markdown=body, source_name=self.name)


__all__ = ["OfficialSkillSource"]
