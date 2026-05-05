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
"""Tests for xerxes.runtime_context — enriched prompt assembly."""

from pathlib import Path

from xerxes.extensions.hooks import HookRunner
from xerxes.extensions.skills import Skill, SkillMetadata, SkillRegistry
from xerxes.runtime.context import PromptContextBuilder, RuntimeInfo
from xerxes.runtime.profiles import PromptProfile, PromptProfileConfig
from xerxes.security.sandbox import SandboxConfig, SandboxMode


class TestRuntimeInfo:
    def test_capture(self):
        info = RuntimeInfo.capture()
        assert info.timestamp != ""
        assert info.platform != ""
        assert info.python_version != ""
        assert info.xerxes_version != ""
        assert info.working_directory != ""

    def test_capture_uses_explicit_workspace_root(self, tmp_path):
        info = RuntimeInfo.capture(str(tmp_path))
        assert info.working_directory == str(tmp_path)
        assert info.workspace_name == tmp_path.name


class TestPromptContextBuilder:
    def test_build_runtime_section(self):
        builder = PromptContextBuilder()
        ctx = builder.build()
        assert "[Runtime Context]" in ctx.runtime_section
        assert "Xerxes:" in ctx.runtime_section

    def test_build_workspace_section(self):
        builder = PromptContextBuilder()
        ctx = builder.build()
        assert "[Workspace]" in ctx.workspace_section

    def test_build_sandbox_off(self):
        builder = PromptContextBuilder(sandbox_config=SandboxConfig(mode=SandboxMode.OFF))
        ctx = builder.build()
        assert "off" in ctx.sandbox_section.lower()

    def test_build_sandbox_strict(self):
        config = SandboxConfig(
            mode=SandboxMode.STRICT,
            sandboxed_tools={"execute_shell"},
            elevated_tools={"read_file"},
        )
        builder = PromptContextBuilder(sandbox_config=config)
        ctx = builder.build()
        assert "strict" in ctx.sandbox_section.lower()
        assert "execute_shell" in ctx.sandbox_section

    def test_build_skills_section(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="test_skill", description="A test", tags=["test"]),
                instructions="Do testing",
                source_path=Path("/fake"),
            )
        )
        builder = PromptContextBuilder(skill_registry=registry)
        ctx = builder.build()
        assert "test_skill" in ctx.skills_section

    def test_build_tools_section(self):
        builder = PromptContextBuilder()
        ctx = builder.build(tool_names=["search", "read_file", "execute_shell"])
        assert "search" in ctx.tools_section
        assert "read_file" in ctx.tools_section

    def test_build_guardrails(self):
        builder = PromptContextBuilder(guardrails=["Do not access private data", "Respect rate limits"])
        ctx = builder.build()
        assert "Do not access private data" in ctx.guardrails_section

    def test_build_bootstrap_with_hooks(self):
        runner = HookRunner()
        runner.register("bootstrap_files", lambda agent_id: "Extra context from bootstrap hook")
        builder = PromptContextBuilder(hook_runner=runner)
        ctx = builder.build(agent_id="test")
        assert "Extra context" in ctx.bootstrap_section

    def test_assemble_system_prompt_prefix(self):
        runner = HookRunner()
        runner.register("bootstrap_files", lambda agent_id: "Bootstrap content")
        builder = PromptContextBuilder(
            hook_runner=runner,
            guardrails=["Be safe"],
            sandbox_config=SandboxConfig(mode=SandboxMode.OFF),
        )
        prefix = builder.assemble_system_prompt_prefix(agent_id="a1", tool_names=["search"])
        assert "[Identity]" in prefix
        assert "[Tooling]" in prefix
        assert "[Safety]" in prefix
        assert "[Runtime]" in prefix
        assert "[Runtime Context]" in prefix
        assert "search" in prefix
        assert "Be safe" in prefix
        assert "Bootstrap content" in prefix

    def test_full_prefix_discourages_fake_tool_markup_and_reasoning_leaks(self):
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(tool_names=["search"], profile=PromptProfile.FULL)

        assert "Do not use or simulate tools for greetings, simple arithmetic" in prefix
        assert "Do not simulate tool calls or wrap normal answers in tool/XML markup." in prefix
        assert "keep internal reasoning private" in prefix
        assert "Keep internal reasoning out of the visible answer" in prefix
        assert "not in a scratchpad or reasoning field" in prefix

    def test_full_prefix_includes_tool_selection_guidance(self):
        builder = PromptContextBuilder()
        prefix = builder.assemble_system_prompt_prefix(
            tool_names=["web.search_query", "web.open", "ReadFile"],
            profile=PromptProfile.FULL,
        )

        assert "Tool selection guidance:" in prefix
        assert "user explicitly asks to search/look up/browse the web" in prefix
        assert "Generic web-search follow-ups" in prefix
        assert "Use this after search when you need the contents of a specific result page" in prefix
        assert "File-reading tools: Use them for project-specific facts" in prefix
        assert "do not claim that you cannot browse or access current information" in prefix
        assert "Search snippets and result titles are leads, not verification" in prefix

    def test_empty_sections_omitted(self):
        builder = PromptContextBuilder()
        ctx = builder.build()
        assert ctx.skills_section == ""
        assert ctx.bootstrap_section == ""


# ── Profile-aware building ────────────────────────────────────────────


class TestPromptContextBuilderProfiles:
    """Tests for profile-aware prompt building (backward compatibility + new profiles)."""

    def test_default_no_profile_unchanged(self):
        """Calling build() without a profile produces the same result as before."""
        builder = PromptContextBuilder(
            guardrails=["Be safe"],
            sandbox_config=SandboxConfig(mode=SandboxMode.OFF),
        )
        ctx_default = builder.build(tool_names=["search"])
        ctx_full = builder.build(tool_names=["search"], profile=PromptProfile.FULL)
        # Both should produce identical contexts (skip runtime_section due to timestamps)
        assert "[Runtime Context]" in ctx_default.runtime_section
        assert "[Runtime Context]" in ctx_full.runtime_section
        assert ctx_default.sandbox_section == ctx_full.sandbox_section
        assert ctx_default.guardrails_section == ctx_full.guardrails_section
        assert ctx_default.tools_section == ctx_full.tools_section
        # Workspace sections have timestamps too
        assert "[Workspace]" in ctx_default.workspace_section
        assert "[Workspace]" in ctx_full.workspace_section

    def test_compact_profile_excludes_workspace_and_bootstrap(self):
        runner = HookRunner()
        runner.register("bootstrap_files", lambda agent_id: "Bootstrap data")
        builder = PromptContextBuilder(hook_runner=runner)
        ctx = builder.build(agent_id="a1", profile=PromptProfile.COMPACT)
        assert ctx.workspace_section == ""
        assert ctx.bootstrap_section == ""
        assert "[Runtime Context]" in ctx.runtime_section

    def test_minimal_profile_excludes_runtime_workspace_skills(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="sk", description="d", tags=[]),
                instructions="Do stuff",
                source_path=Path("/f"),
            )
        )
        builder = PromptContextBuilder(
            skill_registry=registry,
            guardrails=["Stay safe"],
            sandbox_config=SandboxConfig(mode=SandboxMode.OFF),
        )
        ctx = builder.build(tool_names=["t1"], profile=PromptProfile.MINIMAL)
        assert ctx.runtime_section == ""
        assert ctx.workspace_section == ""
        assert ctx.skills_section == ""
        assert ctx.enabled_skills_section == ""
        assert ctx.bootstrap_section == ""
        assert "[Guardrails]" in ctx.guardrails_section
        assert "off" in ctx.sandbox_section.lower()
        assert "t1" in ctx.tools_section

    def test_build_compact_prefix(self):
        builder = PromptContextBuilder(guardrails=["G1"])
        prefix = builder.build_compact_prefix(tool_names=["a", "b"])
        assert "[Identity]" in prefix
        assert "[Runtime Context]" in prefix
        assert "[Workspace]" not in prefix
        assert "[Safety]" in prefix

    def test_build_minimal_prefix(self):
        builder = PromptContextBuilder(guardrails=["G1"])
        prefix = builder.build_minimal_prefix(tool_names=["a", "b"])
        assert "[Runtime Context]" not in prefix
        assert "[Workspace]" not in prefix
        assert "[Safety]" in prefix

    def test_build_none_prefix(self):
        builder = PromptContextBuilder(guardrails=["G1"])
        prefix = builder.build_none_prefix()
        assert prefix == ("You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.")

    def test_custom_profile_config(self):
        cfg = PromptProfileConfig(
            include_runtime_info=False,
            include_workspace_info=False,
            include_guardrails=False,
            include_bootstrap=False,
        )
        builder = PromptContextBuilder(guardrails=["G1"])
        prefix = builder.assemble_system_prompt_prefix(tool_names=["t"], profile=cfg)
        assert "[Runtime Context]" not in prefix
        assert "[Guardrails]" not in prefix

    def test_builder_default_profile(self):
        """Builder-level default profile is respected when per-call profile is None."""
        builder = PromptContextBuilder(
            guardrails=["G1"],
            profile=PromptProfile.MINIMAL,
        )
        prefix = builder.assemble_system_prompt_prefix(tool_names=["t"])
        assert "[Runtime Context]" not in prefix
        assert "[Safety]" in prefix

    def test_none_profile_excludes_all_sections(self):
        builder = PromptContextBuilder(
            guardrails=["G1"],
            sandbox_config=SandboxConfig(mode=SandboxMode.OFF),
        )
        prefix = builder.assemble_system_prompt_prefix(tool_names=["t"], profile=PromptProfile.NONE)
        assert prefix == ("You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.")
        assert "[Tooling]" not in prefix
        assert "[Safety]" not in prefix
        assert "[Sandbox]" not in prefix
        assert "[Runtime]" not in prefix
