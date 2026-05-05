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
"""Tests for enhanced skill metadata and config injection."""

import sys
from pathlib import Path

from xerxes.extensions.skills import (
    Skill,
    SkillMetadata,
    SkillRegistry,
    inject_skill_config,
    parse_skill_md,
    resolve_skill_config,
    skill_matches_platform,
)


class TestEnhancedSkillMetadata:
    """Tests for new SkillMetadata fields."""

    def test_platforms_parsed(self):
        content = "---\nname: test\nplatforms: [macos, linux]\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.platforms == ["macos", "linux"]

    def test_config_vars_parsed(self):
        content = "---\nname: test\nconfig_vars: [api_key, region]\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.config_vars == ["api_key", "region"]

    def test_trust_level_parsed(self):
        content = "---\nname: test\ntrust_level: trusted\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.trust_level == "trusted"

    def test_source_parsed(self):
        content = "---\nname: test\nsource: github\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.source == "github"

    def test_setup_command_parsed(self):
        content = "---\nname: test\nsetup_command: pip install foo\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.setup_command == "pip install foo"

    def test_defaults_for_new_fields(self):
        content = "---\nname: test\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.platforms == []
        assert skill.metadata.config_vars == []
        assert skill.metadata.trust_level == "community"
        assert skill.metadata.source == "local"
        assert skill.metadata.setup_command == ""

    def test_string_tag_coerced_to_list(self):
        content = "---\nname: test\ntags: research\n---\nDo something."
        skill = parse_skill_md(content, Path("/skills/test/SKILL.md"))
        assert skill.metadata.tags == ["research"]


class TestSkillPlatformMatching:
    """Tests for skill_matches_platform."""

    def test_no_platforms_means_universal(self):
        skill = Skill(
            metadata=SkillMetadata(name="universal"),
            instructions="",
            source_path=Path("/skills/u/SKILL.md"),
        )
        assert skill_matches_platform(skill, "darwin") is True
        assert skill_matches_platform(skill, "linux") is True

    def test_matching_platform(self):
        skill = Skill(
            metadata=SkillMetadata(name="mac", platforms=["macos"]),
            instructions="",
            source_path=Path("/skills/m/SKILL.md"),
        )
        assert skill_matches_platform(skill, "darwin") is True
        assert skill_matches_platform(skill, "linux") is False

    def test_multiple_platforms(self):
        skill = Skill(
            metadata=SkillMetadata(name="nix", platforms=["macos", "linux"]),
            instructions="",
            source_path=Path("/skills/n/SKILL.md"),
        )
        assert skill_matches_platform(skill, "darwin") is True
        assert skill_matches_platform(skill, "linux2") is True
        assert skill_matches_platform(skill, "win32") is False

    def test_defaults_to_sys_platform(self):
        skill = Skill(
            metadata=SkillMetadata(name="any", platforms=[sys.platform]),
            instructions="",
            source_path=Path("/skills/a/SKILL.md"),
        )
        assert skill_matches_platform(skill) is True


class TestSkillConfigInjection:
    """Tests for skill config resolution and injection."""

    def test_resolve_skill_config(self):
        skill = Skill(
            metadata=SkillMetadata(name="news", config_vars=["api_key", "region"]),
            instructions="Read news.",
            source_path=Path("/skills/news/SKILL.md"),
        )
        user_cfg = {"news": {"api_key": "secret123", "region": "us"}}
        resolved = resolve_skill_config(skill, user_cfg)
        assert resolved == {"api_key": "secret123", "region": "us"}

    def test_resolve_missing_vars_omitted(self):
        skill = Skill(
            metadata=SkillMetadata(name="news", config_vars=["api_key", "region"]),
            instructions="Read news.",
            source_path=Path("/skills/news/SKILL.md"),
        )
        user_cfg = {"news": {"api_key": "secret123"}}
        resolved = resolve_skill_config(skill, user_cfg)
        assert resolved == {"api_key": "secret123"}
        assert "region" not in resolved

    def test_resolve_no_config_vars(self):
        skill = Skill(
            metadata=SkillMetadata(name="simple"),
            instructions="Simple.",
            source_path=Path("/skills/simple/SKILL.md"),
        )
        resolved = resolve_skill_config(skill, {})
        assert resolved == {}

    def test_inject_skill_config(self):
        skill = Skill(
            metadata=SkillMetadata(name="news", config_vars=["api_key"]),
            instructions="Read news.",
            source_path=Path("/skills/news/SKILL.md"),
        )
        user_cfg = {"news": {"api_key": "secret123"}}
        block = inject_skill_config(skill, user_cfg)
        assert "[Skill config" in block
        assert "api_key = secret123" in block

    def test_inject_skill_config_empty(self):
        skill = Skill(
            metadata=SkillMetadata(name="news", config_vars=["api_key"]),
            instructions="Read news.",
            source_path=Path("/skills/news/SKILL.md"),
        )
        block = inject_skill_config(skill, {})
        assert block == ""


class TestSkillRegistryPlatformFiltering:
    """Tests for platform-aware skill discovery."""

    def test_skills_filtered_by_platform(self, tmp_path: Path, monkeypatch):
        # Create two skills: one universal, one linux-only
        linux_dir = tmp_path / "linux_skill"
        linux_dir.mkdir()
        (linux_dir / "SKILL.md").write_text(
            "---\nname: linux_only\nplatforms: [linux]\n---\nLinux stuff.",
            encoding="utf-8",
        )

        universal_dir = tmp_path / "universal_skill"
        universal_dir.mkdir()
        (universal_dir / "SKILL.md").write_text(
            "---\nname: universal\n---\nWorks everywhere.",
            encoding="utf-8",
        )

        registry = SkillRegistry()
        discovered = registry.discover(tmp_path)
        assert "linux_only" in discovered
        assert "universal" in discovered

        # On non-linux platforms, linux_only should not match
        monkeypatch.setattr(sys, "platform", "darwin")
        linux_skill = registry.get("linux_only")
        assert linux_skill is not None
        assert skill_matches_platform(linux_skill) is False
        assert skill_matches_platform(registry.get("universal")) is True
