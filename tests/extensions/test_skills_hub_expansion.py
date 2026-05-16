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
"""Tests for the Plan 23 skills-hub expansion."""

from __future__ import annotations

import httpx
import pytest
from xerxes.extensions.skill_sources import (
    AgentskillsIOSource,
    GitHubSkillSource,
    LocalSkillSource,
    OfficialSkillSource,
)
from xerxes.extensions.skill_sources.base import SkillBundle
from xerxes.extensions.skills_sync import ManifestEntry, install_bundle, sync_manifest

# ---------------------------- LocalSkillSource -----------------------------


class TestLocalSource:
    def _seed(self, tmp_path):
        (tmp_path / "alpha" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "alpha" / "SKILL.md").write_text("---\nversion: 1.2.3\n---\n\nThe alpha skill helps.")
        (tmp_path / "beta" / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "beta" / "SKILL.md").write_text("---\n---\n\nBeta does things.")
        return tmp_path

    def test_search_matches_filename(self, tmp_path):
        s = LocalSkillSource(self._seed(tmp_path))
        out = s.search("alpha")
        assert any(h.name == "alpha" for h in out)

    def test_search_matches_body(self, tmp_path):
        s = LocalSkillSource(self._seed(tmp_path))
        out = s.search("does things")
        assert any(h.name == "beta" for h in out)

    def test_fetch_returns_bundle(self, tmp_path):
        s = LocalSkillSource(self._seed(tmp_path))
        bundle = s.fetch("alpha")
        assert bundle.name == "alpha"
        assert bundle.version == "1.2.3"
        assert "alpha skill" in bundle.body_markdown

    def test_fetch_missing_raises(self, tmp_path):
        s = LocalSkillSource(self._seed(tmp_path))
        with pytest.raises(KeyError):
            s.fetch("ghost")


# ---------------------------- GitHubSkillSource ----------------------------


class TestGitHubSource:
    def test_fetch_invokes_callable(self):
        captured = {}

        def fake_fetch(identifier):
            captured["id"] = identifier
            return "# body"

        s = GitHubSkillSource(fetch_callable=fake_fetch)
        bundle = s.fetch("python-pep8")
        assert bundle.name == "python-pep8"
        assert bundle.body_markdown == "# body"
        assert captured["id"] == "python-pep8"

    def test_search_invokes_callable(self):
        def fake_search(query, limit):
            return [{"name": "x", "description": "desc"}]

        s = GitHubSkillSource(search_callable=fake_search)
        hits = s.search("anything")
        assert len(hits) == 1
        assert hits[0].name == "x"

    def test_fetch_without_callable_raises(self):
        s = GitHubSkillSource()
        with pytest.raises(RuntimeError):
            s.fetch("x")

    def test_search_without_callable_empty(self):
        s = GitHubSkillSource()
        assert s.search("x") == []


# ---------------------------- OfficialSkillSource --------------------------


class TestOfficialSource:
    def test_search_returns_results(self):
        def handler(req):
            return httpx.Response(200, json=[{"name": "a", "description": "first"}])

        c = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test")
        s = OfficialSkillSource(client=c)
        hits = s.search("anything")
        assert hits[0].name == "a"

    def test_fetch_returns_body(self):
        def handler(req):
            return httpx.Response(200, text="# the body")

        c = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test")
        s = OfficialSkillSource(client=c)
        bundle = s.fetch("identifier")
        assert bundle.body_markdown == "# the body"

    def test_search_network_error_returns_empty(self):
        def handler(req):
            raise httpx.RequestError("boom")

        c = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test")
        assert OfficialSkillSource(client=c).search("x") == []


class TestAgentskillsIO:
    def test_name(self):
        assert AgentskillsIOSource().name == "agentskills.io"


# ---------------------------- sync -----------------------------------------


class TestSync:
    def test_install_bundle_writes_skill_md(self, tmp_path):
        out = install_bundle(SkillBundle(name="x", version="1", body_markdown="b"), tmp_path)
        assert out.read_text() == "b"
        assert (tmp_path / "x" / "SKILL.md").exists()

    def test_sync_installs_new(self, tmp_path):
        captured = {}

        class FakeSource:
            name = "fake"

            def search(self, q, *, limit=20):
                return []

            def fetch(self, identifier):
                captured["id"] = identifier
                return SkillBundle(name=identifier, version="1", body_markdown="body")

        result = sync_manifest(
            [ManifestEntry(source="fake", identifier="alpha")],
            sources={"fake": FakeSource()},
            target_dir=tmp_path,
        )
        assert result.installed == ["alpha"]
        assert captured["id"] == "alpha"

    def test_sync_skips_existing(self, tmp_path):
        # Pre-install one.
        install_bundle(SkillBundle(name="alpha", version="1", body_markdown="x"), tmp_path)

        class _Source:
            name = "fake"

            def search(self, *a, **kw):
                return []

            def fetch(self, identifier):
                raise AssertionError("should not be called")

        result = sync_manifest(
            [ManifestEntry(source="fake", identifier="alpha")],
            sources={"fake": _Source()},
            target_dir=tmp_path,
        )
        assert result.skipped == ["alpha"]

    def test_sync_unknown_source_records_failure(self, tmp_path):
        result = sync_manifest(
            [ManifestEntry(source="ghost", identifier="alpha")],
            sources={},
            target_dir=tmp_path,
        )
        assert result.failed and result.failed[0][0] == "alpha"

    def test_sync_prune_removes_strays(self, tmp_path):
        install_bundle(SkillBundle(name="stray", version="1", body_markdown="x"), tmp_path)
        result = sync_manifest([], sources={}, target_dir=tmp_path, prune=True)
        assert result.removed == ["stray"]
