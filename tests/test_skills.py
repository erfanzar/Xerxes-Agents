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
"""Tests for xerxes.skills — skill discovery and indexing."""

from pathlib import Path

from xerxes.extensions.skills import Skill, SkillMetadata, SkillRegistry, parse_skill_md

SAMPLE_SKILL_MD = """---
name: web_research
description: Search the web and synthesize findings
version: "1.0"
tags: [research, web]
resources:
  - templates/query.md
---

# Web Research Skill

When asked to research a topic:
1. Break the query into sub-questions
2. Search for each sub-question
3. Synthesize the findings
"""

SIMPLE_SKILL_MD = """---
name: simple_skill
description: A simple skill
---

Do the thing.
"""

NO_FRONTMATTER_SKILL = """# Just Instructions

Some instructions without frontmatter.
"""


class TestParseSkillMd:
    def test_full_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text(SAMPLE_SKILL_MD)
        skill = parse_skill_md(SAMPLE_SKILL_MD, path)
        assert skill.name == "web_research"
        assert skill.metadata.description == "Search the web and synthesize findings"
        assert skill.metadata.version == "1.0"
        assert "research" in skill.metadata.tags
        assert "web" in skill.metadata.tags
        assert "templates/query.md" in skill.metadata.resources
        assert "Break the query" in skill.instructions

    def test_simple_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        skill = parse_skill_md(SIMPLE_SKILL_MD, path)
        assert skill.name == "simple_skill"
        assert skill.instructions == "Do the thing."

    def test_no_frontmatter_uses_dirname(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        path = skill_dir / "SKILL.md"
        skill = parse_skill_md(NO_FRONTMATTER_SKILL, path)
        assert skill.name == "my_skill"
        assert "Just Instructions" in skill.instructions

    def test_to_prompt_section(self, tmp_path):
        path = tmp_path / "SKILL.md"
        skill = parse_skill_md(SAMPLE_SKILL_MD, path)
        section = skill.to_prompt_section()
        assert "## Skill: web_research" in section
        assert "Search the web" in section
        assert "Break the query" in section


class TestSkillRegistry:
    def _create_skill_dir(self, tmp_path, name, content):
        skill_dir = tmp_path / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)
        return skill_dir

    def test_discover_finds_skills(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        self._create_skill_dir(tmp_path, "simple_skill", SIMPLE_SKILL_MD)

        registry = SkillRegistry()
        discovered = registry.discover(tmp_path)
        assert "web_research" in discovered
        assert "simple_skill" in discovered
        assert len(registry.skill_names) == 2

    def test_get_skill(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        registry = SkillRegistry()
        registry.discover(tmp_path)
        skill = registry.get("web_research")
        assert skill is not None
        assert skill.name == "web_research"

    def test_get_nonexistent_returns_none(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_no_duplicate_registration(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        registry = SkillRegistry()
        d1 = registry.discover(tmp_path)
        d2 = registry.discover(tmp_path)
        assert len(d1) == 1
        assert len(d2) == 0  # Already registered

    def test_manual_register(self):
        registry = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(name="test_skill", description="test"),
            instructions="Do stuff",
            source_path=Path("/fake"),
        )
        registry.register(skill)
        assert registry.get("test_skill") is skill

    def test_search_by_query(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        self._create_skill_dir(tmp_path, "simple_skill", SIMPLE_SKILL_MD)
        registry = SkillRegistry()
        registry.discover(tmp_path)
        results = registry.search("web")
        assert any(s.name == "web_research" for s in results)

    def test_search_by_tags(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        registry = SkillRegistry()
        registry.discover(tmp_path)
        results = registry.search(tags=["research"])
        assert len(results) == 1
        assert results[0].name == "web_research"

    def test_build_skills_index(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        registry = SkillRegistry()
        registry.discover(tmp_path)
        index = registry.build_skills_index()
        assert "web_research" in index
        assert "research, web" in index

    def test_empty_index(self):
        registry = SkillRegistry()
        assert registry.build_skills_index() == ""

    def test_nonexistent_directory(self):
        registry = SkillRegistry()
        discovered = registry.discover("/nonexistent/path")
        assert discovered == []


SKILL_WITH_DEPS_MD = """---
name: advanced_research
description: Advanced research with dependencies
version: "2.0"
tags: [research]
dependencies:
  - web_research
required_tools:
  - web_search
  - summarize
---

# Advanced Research

Use web_research skill and web_search tool.
"""


class TestSkillDependencies:
    def _create_skill_dir(self, tmp_path, name, content):
        skill_dir = tmp_path / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content)
        return skill_dir

    def test_parse_dependencies_from_frontmatter(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text(SKILL_WITH_DEPS_MD)
        skill = parse_skill_md(SKILL_WITH_DEPS_MD, path)
        assert skill.metadata.dependencies == ["web_research"]
        assert skill.metadata.required_tools == ["web_search", "summarize"]

    def test_parse_no_deps_defaults_empty(self, tmp_path):
        path = tmp_path / "SKILL.md"
        skill = parse_skill_md(SIMPLE_SKILL_MD, path)
        assert skill.metadata.dependencies == []
        assert skill.metadata.required_tools == []

    def test_validate_dependencies_satisfied(self, tmp_path):
        self._create_skill_dir(tmp_path, "web_research", SAMPLE_SKILL_MD)
        self._create_skill_dir(tmp_path, "advanced_research", SKILL_WITH_DEPS_MD)
        registry = SkillRegistry()
        registry.discover(tmp_path)
        # Without plugin registry, only skill-to-skill deps are checked
        errors = registry.validate_dependencies()
        assert errors == []

    def test_validate_missing_skill_dependency(self):
        registry = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="orphan",
                description="needs missing skill",
                dependencies=["nonexistent_skill"],
            ),
            instructions="Do stuff",
            source_path=Path("/fake"),
        )
        registry.register(skill)
        errors = registry.validate_dependencies()
        assert len(errors) == 1
        assert "nonexistent_skill" in errors[0]

    def test_validate_missing_tool_dependency(self):
        from xerxes.extensions.plugins import PluginMeta, PluginRegistry

        plugin_reg = PluginRegistry()
        plugin_reg.register_plugin(PluginMeta(name="base", version="1.0"))
        # Register only one tool
        plugin_reg.register_tool("web_search", lambda q: q, plugin_name="base")

        skill_reg = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="needs_tools",
                description="needs tools",
                required_tools=["web_search", "summarize"],
            ),
            instructions="Do stuff",
            source_path=Path("/fake"),
        )
        skill_reg.register(skill)
        errors = skill_reg.validate_dependencies(plugin_registry=plugin_reg)
        assert len(errors) == 1
        assert "summarize" in errors[0]

    def test_validate_all_tools_satisfied(self):
        from xerxes.extensions.plugins import PluginMeta, PluginRegistry

        plugin_reg = PluginRegistry()
        plugin_reg.register_plugin(PluginMeta(name="base", version="1.0"))
        plugin_reg.register_tool("web_search", lambda q: q, plugin_name="base")
        plugin_reg.register_tool("summarize", lambda t: t, plugin_name="base")

        skill_reg = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="needs_tools",
                description="needs tools",
                required_tools=["web_search", "summarize"],
            ),
            instructions="Do stuff",
            source_path=Path("/fake"),
        )
        skill_reg.register(skill)
        errors = skill_reg.validate_dependencies(plugin_registry=plugin_reg)
        assert errors == []

    def test_validate_without_plugin_registry_skips_tools(self):
        """When no plugin registry is provided, tool deps are not checked."""
        skill_reg = SkillRegistry()
        skill = Skill(
            metadata=SkillMetadata(
                name="needs_tools",
                required_tools=["nonexistent"],
            ),
            instructions="Do stuff",
            source_path=Path("/fake"),
        )
        skill_reg.register(skill)
        errors = skill_reg.validate_dependencies()
        assert errors == []
