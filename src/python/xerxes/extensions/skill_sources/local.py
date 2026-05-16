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
"""``LocalSkillSource`` — read skills from a directory tree of SKILL.md files."""

from __future__ import annotations

from pathlib import Path

from .base import SkillBundle, SkillSearchHit, SkillSource


class LocalSkillSource(SkillSource):
    """Skill source that reads from a local directory tree."""

    name = "local"

    def __init__(self, root: str | Path) -> None:
        """Initialize with the root directory containing ``SKILL.md`` files."""
        self._root = Path(root)

    def search(self, query: str, *, limit: int = 20) -> list[SkillSearchHit]:
        """Return case-insensitive matches of ``query`` against body or folder name."""
        q = query.lower()
        hits: list[SkillSearchHit] = []
        if not self._root.is_dir():
            return hits
        for skill_path in self._root.rglob("SKILL.md"):
            try:
                body = skill_path.read_text(encoding="utf-8")
            except OSError:
                continue
            haystack = body.lower()
            if q in haystack or q in skill_path.parent.name.lower():
                hits.append(
                    SkillSearchHit(
                        name=skill_path.parent.name,
                        description=_first_line(body),
                        source_name=self.name,
                    )
                )
                if len(hits) >= limit:
                    break
        return hits

    def fetch(self, identifier: str) -> SkillBundle:
        """Return the bundle for the skill folder named ``identifier``.

        Raises:
            KeyError: ``identifier`` does not match any local skill.
        """
        candidates = list(self._root.rglob(f"{identifier}/SKILL.md"))
        if not candidates:
            raise KeyError(f"skill not found in {self._root}: {identifier}")
        skill_path = candidates[0]
        return SkillBundle(
            name=identifier,
            version=_extract_version(skill_path.read_text(encoding="utf-8")),
            body_markdown=skill_path.read_text(encoding="utf-8"),
            source_name=self.name,
        )


def _first_line(body: str) -> str:
    """Return the first non-frontmatter, non-heading line of ``body`` (max 200 chars)."""
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("---") and not line.startswith("#"):
            return line[:200]
    return ""


def _extract_version(body: str) -> str:
    """Read ``version:`` from the first 20 lines of frontmatter, default ``"0.0.1"``."""
    for line in body.splitlines()[:20]:
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return "0.0.1"


__all__ = ["LocalSkillSource"]
