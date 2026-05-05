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
"""Tests for SkillMatcher."""

from __future__ import annotations

import pytest
from xerxes.extensions.skill_authoring import SkillMatcher
from xerxes.extensions.skills import Skill, SkillMetadata, SkillRegistry
from xerxes.memory import HashEmbedder


def _skill(tmp_path, name, description, tags, instructions="body"):
    return Skill(
        metadata=SkillMetadata(name=name, description=description, tags=tags),
        instructions=instructions,
        source_path=tmp_path / f"{name}.md",
    )


@pytest.fixture
def registry(tmp_path):
    r = SkillRegistry()
    r._skills["ci"] = _skill(tmp_path, "ci", "set up continuous integration with github actions", ["ci", "github"])
    r._skills["docs"] = _skill(tmp_path, "docs", "generate documentation from python docstrings", ["docs", "sphinx"])
    r._skills["release"] = _skill(
        tmp_path, "release", "tag a new semver release and publish to pypi", ["release", "pypi"]
    )
    return r


class TestSkillMatcher:
    def test_matches_relevant_skill(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder(), min_score=0.0)
        hits = m.match("set up continuous integration", k=1)
        assert hits[0].skill.name == "ci"

    def test_returns_top_k(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder(), min_score=0.0)
        hits = m.match("set up", k=2)
        assert len(hits) <= 2

    def test_min_score_filters(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder(), min_score=0.999)
        assert m.match("xyz unrelated nonsense") == []

    def test_best_returns_single(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder(), min_score=0.0)
        best = m.best("python documentation generation")
        assert best is not None
        assert best.skill.name == "docs"

    def test_empty_query_returns_empty(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder())
        assert m.match("") == []

    def test_explicit_skills_override_registry(self, registry, tmp_path):
        only = [_skill(tmp_path, "only", "the only skill", ["only"])]
        m = SkillMatcher(registry, embedder=HashEmbedder(), min_score=0.0)
        hits = m.match("anything", k=5, skills=only)
        assert {h.skill.name for h in hits} <= {"only"}

    def test_caches_embeddings(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder())
        m.match("setup ci", k=3)
        cached = len(m._cache)
        m.match("set up CI again", k=3)
        assert len(m._cache) == cached  # no additional cache entries

    def test_invalidate_clears_cache(self, registry):
        m = SkillMatcher(registry, embedder=HashEmbedder())
        m.match("setup ci", k=3)
        assert m._cache
        m.invalidate()
        assert m._cache == {}

    def test_no_registry_returns_empty(self):
        m = SkillMatcher(skill_registry=None, embedder=HashEmbedder())
        assert m.match("anything") == []
