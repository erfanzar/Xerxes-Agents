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
"""Tests for xerxes.prompt_profiles — prompt profile configs and truncation."""

from pathlib import Path

from xerxes.extensions.hooks import HookRunner
from xerxes.extensions.skills import Skill, SkillMetadata, SkillRegistry
from xerxes.runtime.context import PromptContextBuilder
from xerxes.runtime.profiles import PromptProfile, PromptProfileConfig, get_profile_config
from xerxes.security.sandbox import SandboxConfig, SandboxMode


class TestPromptProfileEnum:
    def test_full_value(self):
        assert PromptProfile.FULL.value == "full"

    def test_compact_value(self):
        assert PromptProfile.COMPACT.value == "compact"

    def test_minimal_value(self):
        assert PromptProfile.MINIMAL.value == "minimal"

    def test_none_value(self):
        assert PromptProfile.NONE.value == "none"

    def test_all_members(self):
        assert set(PromptProfile) == {
            PromptProfile.FULL,
            PromptProfile.COMPACT,
            PromptProfile.MINIMAL,
            PromptProfile.NONE,
        }


class TestGetProfileConfig:
    def test_full_profile_all_enabled(self):
        cfg = get_profile_config(PromptProfile.FULL)
        assert cfg.profile == PromptProfile.FULL
        assert cfg.include_runtime_info is True
        assert cfg.include_workspace_info is True
        assert cfg.include_sandbox_info is True
        assert cfg.include_skills_index is True
        assert cfg.include_enabled_skills is True
        assert cfg.include_tools_list is True
        assert cfg.include_guardrails is True
        assert cfg.include_bootstrap is True
        assert cfg.max_skill_instructions_length is None
        assert cfg.max_tools_listed is None

    def test_compact_profile(self):
        cfg = get_profile_config(PromptProfile.COMPACT)
        assert cfg.profile == PromptProfile.COMPACT
        assert cfg.include_runtime_info is True
        assert cfg.include_workspace_info is False
        assert cfg.include_sandbox_info is True
        assert cfg.include_skills_index is True
        assert cfg.include_enabled_skills is True
        assert cfg.include_tools_list is True
        assert cfg.include_guardrails is True
        assert cfg.include_bootstrap is False
        assert cfg.max_skill_instructions_length == 500
        assert cfg.max_tools_listed == 20

    def test_minimal_profile(self):
        cfg = get_profile_config(PromptProfile.MINIMAL)
        assert cfg.profile == PromptProfile.MINIMAL
        assert cfg.include_runtime_info is False
        assert cfg.include_workspace_info is False
        assert cfg.include_sandbox_info is True
        assert cfg.include_skills_index is False
        assert cfg.include_enabled_skills is False
        assert cfg.include_tools_list is True
        assert cfg.include_guardrails is True
        assert cfg.include_bootstrap is False
        assert cfg.max_tools_listed == 10

    def test_none_profile(self):
        cfg = get_profile_config(PromptProfile.NONE)
        assert cfg.profile == PromptProfile.NONE
        assert cfg.include_runtime_info is False
        assert cfg.include_workspace_info is False
        assert cfg.include_sandbox_info is False
        assert cfg.include_skills_index is False
        assert cfg.include_enabled_skills is False
        assert cfg.include_tools_list is False
        assert cfg.include_guardrails is False
        assert cfg.include_bootstrap is False
        assert cfg.max_skill_instructions_length is None
        assert cfg.max_tools_listed is None

    def test_string_profile_alias_is_accepted(self):
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(profile="none")
        assert prefix == ("You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.")


class TestPromptProfileConfigDefaults:
    def test_defaults_match_full(self):
        cfg = PromptProfileConfig()
        full = get_profile_config(PromptProfile.FULL)
        assert cfg.include_runtime_info == full.include_runtime_info
        assert cfg.include_workspace_info == full.include_workspace_info
        assert cfg.include_bootstrap == full.include_bootstrap
        assert cfg.max_skill_instructions_length is None
        assert cfg.max_tools_listed is None

    def test_custom_config(self):
        cfg = PromptProfileConfig(
            profile=PromptProfile.COMPACT,
            include_runtime_info=False,
            include_guardrails=False,
            max_tools_listed=5,
        )
        assert cfg.include_runtime_info is False
        assert cfg.include_guardrails is False
        assert cfg.max_tools_listed == 5


def _make_builder_with_everything() -> PromptContextBuilder:
    """Helper: builder with skills, sandbox, guardrails, hooks."""
    registry = SkillRegistry()
    registry.register(
        Skill(
            metadata=SkillMetadata(name="web_research", description="Search the web", tags=["search"]),
            instructions="Use DuckDuckGo to search. Summarise results concisely.",
            source_path=Path("/fake/web_research"),
        )
    )
    runner = HookRunner()
    runner.register("bootstrap_files", lambda agent_id: "Bootstrap context from hook")

    return PromptContextBuilder(
        skill_registry=registry,
        hook_runner=runner,
        sandbox_config=SandboxConfig(mode=SandboxMode.STRICT, sandboxed_tools={"exec"}),
        guardrails=["Be safe", "Respect limits"],
    )


TOOL_NAMES = ["search", "read_file", "execute_shell"]


class TestFullProfileBackwardCompat:
    """FULL profile must produce identical output to no-profile (backward compat)."""

    def test_full_same_as_default(self):
        builder = _make_builder_with_everything()
        default = builder.assemble_system_prompt_prefix(agent_id="a1", tool_names=TOOL_NAMES)
        full = builder.assemble_system_prompt_prefix(agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.FULL)

        import re

        def _strip_ts(s):
            return re.sub(r"(Local time: ).+\n", r"\1<TS>\n", s)

        assert _strip_ts(default) == _strip_ts(full)

    def test_full_profile_includes_all_sections(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.FULL)
        assert "[Identity]" in prefix
        assert "[Tooling]" in prefix
        assert "[Safety]" in prefix
        assert "[Runtime]" in prefix
        assert "[Runtime Context]" in prefix
        assert "[Workspace]" in prefix
        assert "[Sandbox]" in prefix
        assert "[Guardrails]" in prefix
        assert "[Skills]" in prefix
        assert "[Available Tools]" in prefix
        assert "Bootstrap context from hook" in prefix


class TestCompactProfile:
    def test_includes_runtime_info(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.COMPACT
        )
        assert "[Runtime Context]" in prefix

    def test_excludes_workspace(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.COMPACT
        )
        assert "[Workspace]" not in prefix

    def test_includes_sandbox(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.COMPACT
        )
        assert "[Sandbox]" in prefix

    def test_includes_guardrails(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.COMPACT
        )
        assert "[Safety]" in prefix
        assert "Be safe" in prefix

    def test_excludes_bootstrap(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.COMPACT
        )
        assert "Bootstrap context from hook" not in prefix

    def test_truncates_skill_instructions(self):
        """Skill instructions exceeding max_skill_instructions_length are truncated."""
        registry = SkillRegistry()
        long_instructions = "A" * 2000
        registry.register(
            Skill(
                metadata=SkillMetadata(name="verbose_skill", description="Wordy", tags=[]),
                instructions=long_instructions,
                source_path=Path("/fake"),
            )
        )
        builder = PromptContextBuilder(skill_registry=registry)
        skill = registry.get("verbose_skill")
        assert skill is not None

        prefix = builder.assemble_system_prompt_prefix(
            tool_names=["t"],
            enabled_skills=[skill],
            profile=PromptProfile.COMPACT,
        )

        assert "..." in prefix

        assert "A" * 2000 not in prefix

    def test_limits_tools_list(self):
        """Tool list is capped at max_tools_listed with a '... and N more' trailer."""
        tools = [f"tool_{i}" for i in range(30)]
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(tool_names=tools, profile=PromptProfile.COMPACT)
        assert "tool_0" in prefix
        assert "tool_19" in prefix
        assert "tool_20" not in prefix
        assert "... and 10 more" in prefix


class TestMinimalProfile:
    def test_excludes_runtime(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "[Runtime Context]" not in prefix

    def test_excludes_workspace(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "[Workspace]" not in prefix

    def test_excludes_skills_index(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "[Skills]" not in prefix

    def test_excludes_enabled_skills(self):
        builder = _make_builder_with_everything()
        skill = builder.skill_registry.get("web_research")  # type: ignore[union-attr]
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1",
            tool_names=TOOL_NAMES,
            enabled_skills=[skill],
            profile=PromptProfile.MINIMAL,
        )
        assert "[Enabled Skill Instructions]" not in prefix

    def test_includes_sandbox(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "[Sandbox]" in prefix

    def test_includes_guardrails(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "[Safety]" in prefix

    def test_limits_tools_to_10(self):
        tools = [f"tool_{i}" for i in range(25)]
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(tool_names=tools, profile=PromptProfile.MINIMAL)
        assert "tool_9" in prefix
        assert "tool_10" not in prefix
        assert "... and 15 more" in prefix

    def test_excludes_bootstrap(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1", tool_names=TOOL_NAMES, profile=PromptProfile.MINIMAL
        )
        assert "Bootstrap context from hook" not in prefix


class TestNoneProfile:
    def test_emits_only_identity_line(self):
        builder = _make_builder_with_everything()
        prefix = builder.assemble_system_prompt_prefix(
            agent_id="a1",
            tool_names=TOOL_NAMES,
            profile=PromptProfile.NONE,
        )
        assert prefix == ("You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.")
        assert "[Identity]" not in prefix
        assert "[Tooling]" not in prefix
        assert "[Safety]" not in prefix
        assert "[Runtime]" not in prefix


class TestTruncation:
    def test_skill_instruction_exact_limit(self):
        """Skill section at exactly the limit is not truncated."""
        registry = SkillRegistry()

        skill = Skill(
            metadata=SkillMetadata(name="s", description="d", tags=[]),
            instructions="X" * 10,
            source_path=Path("/f"),
        )
        registry.register(skill)
        section_text = skill.to_prompt_section()

        cfg = PromptProfileConfig(max_skill_instructions_length=len(section_text))
        builder = PromptContextBuilder(skill_registry=registry)
        prefix = builder.assemble_system_prompt_prefix(enabled_skills=[skill], tool_names=["t"], profile=cfg)
        assert "..." not in prefix
        assert "X" * 10 in prefix

    def test_skill_instruction_just_over_limit(self):
        skill = Skill(
            metadata=SkillMetadata(name="s", description="d", tags=[]),
            instructions="Y" * 600,
            source_path=Path("/f"),
        )
        cfg = PromptProfileConfig(max_skill_instructions_length=50)
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(enabled_skills=[skill], tool_names=["t"], profile=cfg)
        assert "..." in prefix
        assert "Y" * 600 not in prefix

    def test_tools_not_truncated_under_limit(self):
        tools = [f"tool_{i}" for i in range(5)]
        cfg = PromptProfileConfig(max_tools_listed=10)
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(tool_names=tools, profile=cfg)
        assert "more" not in prefix
        for t in tools:
            assert t in prefix

    def test_tools_truncated_at_exact_limit(self):
        tools = [f"tool_{i}" for i in range(10)]
        cfg = PromptProfileConfig(max_tools_listed=10)
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(tool_names=tools, profile=cfg)

        assert "more" not in prefix
        for t in tools:
            assert t in prefix
