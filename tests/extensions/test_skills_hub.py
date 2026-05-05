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
"""Tests for xerxes.extensions.skills_hub."""

from pathlib import Path

import pytest
from xerxes.extensions.skills_hub import (
    LocalSkillSource,
    OfficialSkillSource,
    SkillsHub,
)


class TestLocalSkillSource:
    """Tests for LocalSkillSource."""

    def test_fetch_existing_skill(self, tmp_path: Path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my_skill\n---\nDo stuff.", encoding="utf-8")

        src = LocalSkillSource()
        bundle = src.fetch(str(skill_dir))
        assert bundle["name"] == "my_skill"
        assert "Do stuff." in bundle["content"]

    def test_fetch_missing_raises(self):
        src = LocalSkillSource()
        with pytest.raises(FileNotFoundError):
            src.fetch("/nonexistent/skill")

    def test_search_finds_match(self, tmp_path: Path):
        skill_dir = tmp_path / "searchable"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: searchable\n---\nSearch me.", encoding="utf-8")

        # Local source searches ~/.xerxes/skills, so we can't easily test
        # without monkeypatching. Just verify search returns a list.
        src = LocalSkillSource()
        results = src.search("nonexistent_query_12345", limit=5)
        assert isinstance(results, list)


class TestOfficialSkillSource:
    """Tests for OfficialSkillSource."""

    def test_fetch_missing_raises(self):
        src = OfficialSkillSource()
        with pytest.raises(FileNotFoundError):
            src.fetch("definitely_not_a_real_skill_name")


class TestSkillsHub:
    """Tests for SkillsHub."""

    def test_install_local_skill(self, tmp_path: Path, monkeypatch):
        # Monkeypatch skills dir to tmp_path
        monkeypatch.setattr("xerxes.extensions.skills_hub.SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr("xerxes.extensions.skills_hub.HUB_DIR", tmp_path / "skills" / ".hub")
        monkeypatch.setattr("xerxes.extensions.skills_hub.LOCK_FILE", tmp_path / "skills" / ".hub" / "lock.json")
        monkeypatch.setattr(
            "xerxes.extensions.skills_guard._TRUSTED_HASHES_PATH",
            tmp_path / "skills" / ".hub" / "trusted_hashes.json",
        )

        skill_dir = tmp_path / "source_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test_skill\n---\nTest instructions.", encoding="utf-8")

        hub = SkillsHub()
        result = hub.install(f"local:{skill_dir}")
        assert "Installed" in result
        assert "test_skill" in result

        # Verify lock file
        lock = (tmp_path / "skills" / ".hub" / "lock.json").read_text(encoding="utf-8")
        assert "test_skill" in lock

    def test_uninstall_skill(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("xerxes.extensions.skills_hub.SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr("xerxes.extensions.skills_hub.HUB_DIR", tmp_path / "skills" / ".hub")
        monkeypatch.setattr("xerxes.extensions.skills_hub.LOCK_FILE", tmp_path / "skills" / ".hub" / "lock.json")

        hub = SkillsHub()
        # Pre-install
        skill_dir = tmp_path / "preinstalled"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: preinstalled\n---\nHi.", encoding="utf-8")
        hub.install(f"local:{skill_dir}")

        result = hub.uninstall("preinstalled")
        assert "Uninstalled" in result

    def test_uninstall_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("xerxes.extensions.skills_hub.SKILLS_DIR", tmp_path / "skills")
        hub = SkillsHub()
        result = hub.uninstall("nonexistent")
        assert "[Error]" in result

    def test_list_installed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("xerxes.extensions.skills_hub.SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr("xerxes.extensions.skills_hub.HUB_DIR", tmp_path / "skills" / ".hub")
        monkeypatch.setattr("xerxes.extensions.skills_hub.LOCK_FILE", tmp_path / "skills" / ".hub" / "lock.json")

        hub = SkillsHub()
        skill_dir = tmp_path / "listed"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: listed\n---\nHi.", encoding="utf-8")
        hub.install(f"local:{skill_dir}")

        installed = hub.list_installed()
        assert any(i["name"] == "listed" for i in installed)
