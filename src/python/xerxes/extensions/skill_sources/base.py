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
"""Abstract base class and shared dataclasses for skill source backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SkillBundle:
    """A single skill payload ready to install on disk.

    Attributes:
        name: Skill identifier.
        version: Version reported by the source backend.
        body_markdown: Raw ``SKILL.md`` content.
        metadata: Backend-specific metadata.
        source_name: Name of the originating ``SkillSource``.
    """

    name: str
    version: str
    body_markdown: str
    metadata: dict = field(default_factory=dict)
    source_name: str = ""


@dataclass
class SkillSearchHit:
    """One hit from a ``SkillSource.search`` query.

    Attributes:
        name: Skill identifier.
        description: Human-readable summary.
        source_name: Name of the originating ``SkillSource``.
        version: Version string when reported by the backend.
        tags: Categorisation labels.
    """

    name: str
    description: str
    source_name: str
    version: str = ""
    tags: list[str] = field(default_factory=list)


class SkillSource(ABC):
    """Backend interface for searching and fetching skills."""

    name: str = ""

    @abstractmethod
    def search(self, query: str, *, limit: int = 20) -> list[SkillSearchHit]:
        """Return at most ``limit`` hits matching ``query``."""

    @abstractmethod
    def fetch(self, identifier: str) -> SkillBundle:
        """Return the ``SkillBundle`` identified by ``identifier``."""
