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
"""Composable system-prompt prefix builder.

Captures the runtime, workspace, sandbox, skill, tool, guardrail, memory, and
user-profile context used by every Xerxes session and renders it as a single
deterministic system-prompt prefix. :class:`PromptContextBuilder` is the
public entry point; :class:`PromptContext` is the bag of rendered sections;
:class:`RuntimeInfo` is the per-session environment snapshot. Each section is
gated by the active :class:`PromptProfileConfig` so the same builder serves
``FULL``, ``COMPACT``, ``MINIMAL``, and ``NONE`` prompt profiles.
"""

from __future__ import annotations

import os
import platform
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from .profiles import PromptProfile, PromptProfileConfig, get_profile_config

if tp.TYPE_CHECKING:
    from ..extensions.hooks import HookRunner
    from ..extensions.plugins import PluginRegistry
    from ..extensions.skills import Skill, SkillRegistry
    from ..security.sandbox import SandboxConfig


@dataclass
class RuntimeInfo:
    """Snapshot of host metadata embedded in the prompt prefix.

    Attributes:
        timestamp: ISO-8601 local time at capture.
        timezone: Local timezone name (or ``"local"`` when unknown).
        platform: ``"<system> <release>"`` (e.g. ``"Darwin 25.4.0"``).
        python_version: Interpreter version string.
        xerxes_version: Installed ``xerxes-agent`` version.
        working_directory: Absolute path of the workspace root.
        workspace_name: Basename of ``working_directory`` used as project label.
    """

    timestamp: str = ""
    timezone: str = ""
    platform: str = ""
    python_version: str = ""
    xerxes_version: str = ""
    working_directory: str = ""
    workspace_name: str = ""

    @classmethod
    def capture(cls, workspace_root: str | None = None) -> RuntimeInfo:
        """Snapshot the running environment.

        Args:
            workspace_root: Path to treat as the workspace root; falls back to
                :func:`os.getcwd` when ``None``.
        """

        from xerxes import __version__

        now = datetime.now().astimezone()
        cwd = os.path.abspath(workspace_root or os.getcwd())
        return cls(
            timestamp=now.isoformat(timespec="seconds"),
            timezone=now.tzname() or "local",
            platform=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            xerxes_version=__version__,
            working_directory=cwd,
            workspace_name=os.path.basename(cwd),
        )


@dataclass
class PromptContext:
    """Container of rendered prompt-prefix sections.

    Each attribute is a self-contained Markdown-ish string the builder
    emitted for the active profile; sections suppressed by the profile or
    missing inputs come through as ``""``.

    Attributes:
        runtime_section: Platform/python/xerxes version banner.
        workspace_section: Workspace root and project name.
        datetime_section: Local time and timezone.
        reasoning_section: Profile-aware response-style guidance.
        sandbox_section: Active sandbox mode and per-tool overrides.
        skills_section: Index of available skills.
        enabled_skills_section: Full instructions for enabled skills.
        tools_section: Bullet list of tool names available this run.
        guardrails_section: Active guardrail identifiers.
        bootstrap_section: Project/bootstrap notes from hooks.
        memory_section: Relevant memory snippets returned by the provider.
        user_profile_section: User profile blurb from the provider.
    """

    runtime_section: str = ""
    workspace_section: str = ""
    datetime_section: str = ""
    reasoning_section: str = ""
    sandbox_section: str = ""
    skills_section: str = ""
    enabled_skills_section: str = ""
    tools_section: str = ""
    guardrails_section: str = ""
    bootstrap_section: str = ""
    memory_section: str = ""
    user_profile_section: str = ""


def _resolve_profile_config(
    profile: PromptProfile | PromptProfileConfig | str | None,
) -> PromptProfileConfig:
    """Normalise ``profile`` (enum/string/config/``None``) to a :class:`PromptProfileConfig`."""

    if profile is None:
        return get_profile_config(PromptProfile.FULL)
    if isinstance(profile, PromptProfile):
        return get_profile_config(profile)
    if isinstance(profile, str):
        return get_profile_config(PromptProfile(profile.strip().lower()))
    return profile


class PromptContextBuilder:
    """Renders the system-prompt prefix from registries, hooks, and providers."""

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
        hook_runner: HookRunner | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        profile: PromptProfile | PromptProfileConfig | None = None,
        workspace_root: str | None = None,
        memory_provider: tp.Callable[[str | None, int], list[str]] | None = None,
        user_profile_provider: tp.Callable[[str | None], str] | None = None,
    ):
        """Wire the builder against the live runtime dependencies.

        Args:
            skill_registry: Source for the skills index and skill instructions.
            plugin_registry: Reserved for future plugin-provided context.
            hook_runner: Used to invoke the ``bootstrap_files`` hook.
            sandbox_config: Default sandbox config when no per-agent override
                is supplied at build time.
            guardrails: Default guardrail list when no override is provided.
            profile: Default prompt profile for :meth:`build` calls.
            workspace_root: Path treated as the project root.
            memory_provider: ``(agent_id, k) -> list[str]`` returning the
                top-``k`` memory snippets to inject.
            user_profile_provider: ``(agent_id) -> str`` returning a user
                profile blurb for the prompt.
        """

        self.skill_registry = skill_registry
        self.plugin_registry = plugin_registry
        self.hook_runner = hook_runner
        self.sandbox_config = sandbox_config
        self.guardrails = guardrails or []
        self.default_profile_config = _resolve_profile_config(profile)
        self.workspace_root = workspace_root
        self.memory_provider = memory_provider
        self.user_profile_provider = user_profile_provider

    def build(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        profile: PromptProfile | PromptProfileConfig | str | None = None,
    ) -> PromptContext:
        """Build a :class:`PromptContext` using only the builder's defaults.

        Thin wrapper around :meth:`build_with_overrides` that omits every
        per-agent override.
        """
        return self.build_with_overrides(agent_id=agent_id, tool_names=tool_names, profile=profile)

    def build_with_overrides(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        enabled_skills: list[Skill] | None = None,
        profile: PromptProfile | PromptProfileConfig | str | None = None,
    ) -> PromptContext:
        """Build a :class:`PromptContext` with explicit per-agent overrides.

        Each ``override`` argument falls back to the builder's default when
        ``None``. Section inclusion is governed by the resolved
        :class:`PromptProfileConfig` — disabled sections come back as ``""``.
        """

        pcfg = _resolve_profile_config(profile) if profile is not None else self.default_profile_config
        runtime_info = RuntimeInfo.capture(self.workspace_root)

        ctx = PromptContext()
        ctx.runtime_section = self._build_runtime(runtime_info) if pcfg.include_runtime_info else ""
        ctx.workspace_section = self._build_workspace(runtime_info) if pcfg.include_workspace_info else ""
        ctx.datetime_section = self._build_datetime(runtime_info) if pcfg.include_runtime_info else ""
        ctx.reasoning_section = self._build_reasoning(pcfg) if pcfg.include_runtime_info else ""
        ctx.sandbox_section = self._build_sandbox(sandbox_config=sandbox_config) if pcfg.include_sandbox_info else ""
        ctx.skills_section = self._build_skills(pcfg) if pcfg.include_skills_index else ""
        ctx.enabled_skills_section = (
            self._build_enabled_skills(enabled_skills=enabled_skills, profile_config=pcfg)
            if pcfg.include_enabled_skills
            else ""
        )
        ctx.tools_section = self._build_tools(tool_names, pcfg) if pcfg.include_tools_list else ""
        ctx.guardrails_section = self._build_guardrails(guardrails=guardrails) if pcfg.include_guardrails else ""
        ctx.bootstrap_section = self._build_bootstrap(agent_id) if pcfg.include_bootstrap else ""
        ctx.memory_section = self._build_memory_section(agent_id, pcfg) if pcfg.include_relevant_memories else ""
        ctx.user_profile_section = self._build_user_profile_section(agent_id) if pcfg.include_user_profile else ""
        return ctx

    def build_compact_prefix(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        enabled_skills: list[Skill] | None = None,
    ) -> str:
        """Render the prompt prefix under the :attr:`PromptProfile.COMPACT` profile."""
        return self.assemble_system_prompt_prefix(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=sandbox_config,
            guardrails=guardrails,
            enabled_skills=enabled_skills,
            profile=PromptProfile.COMPACT,
        )

    def build_minimal_prefix(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        enabled_skills: list[Skill] | None = None,
    ) -> str:
        """Render the prompt prefix under the :attr:`PromptProfile.MINIMAL` profile."""
        return self.assemble_system_prompt_prefix(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=sandbox_config,
            guardrails=guardrails,
            enabled_skills=enabled_skills,
            profile=PromptProfile.MINIMAL,
        )

    def build_none_prefix(self) -> str:
        """Render the bare identity-only prefix used by the ``NONE`` profile."""
        return self.assemble_system_prompt_prefix(profile=PromptProfile.NONE)

    def _build_runtime(self, info: RuntimeInfo) -> str:
        """Render the ``[Runtime Context]`` section."""
        return (
            f"[Runtime Context]\n"
            f"  Platform: {info.platform}\n"
            f"  Python: {info.python_version}\n"
            f"  Xerxes: v{info.xerxes_version}\n"
        )

    def _build_workspace(self, info: RuntimeInfo) -> str:
        """Render the ``[Workspace]`` section with the project root and name."""
        return f"[Workspace]\n  Directory: {info.working_directory}\n  Project: {info.workspace_name}\n"

    def _build_datetime(self, info: RuntimeInfo) -> str:
        """Render the ``[Current Date & Time]`` section."""
        return f"[Current Date & Time]\n  Local time: {info.timestamp}\n  Time zone: {info.timezone}\n"

    def _build_reasoning(self, pcfg: PromptProfileConfig | None = None) -> str:
        """Render the ``[Response Guidance]`` section reflecting the active profile."""

        profile_name = pcfg.profile.value if pcfg is not None else PromptProfile.FULL.value
        return (
            f"[Response Guidance]\n"
            f"  Profile: {profile_name}\n"
            f"  Guidance: answer from actual tool and workspace results; avoid speculative claims; keep internal reasoning private; and put the final answer in the normal assistant response content, not in a scratchpad or reasoning field.\n"
        )

    def _build_sandbox(self, sandbox_config: SandboxConfig | None = None) -> str:
        """Render the ``[Sandbox]`` section from the active sandbox config."""

        config = sandbox_config or self.sandbox_config
        if not config:
            return ""
        from ..security.sandbox import SandboxMode

        if config.mode == SandboxMode.OFF:
            return "[Sandbox] Mode: off (all execution on host)\n"
        return (
            f"[Sandbox]\n"
            f"  Mode: {config.mode.value}\n"
            f"  Sandboxed tools: {', '.join(sorted(config.sandboxed_tools)) or 'none'}\n"
            f"  Elevated tools: {', '.join(sorted(config.elevated_tools)) or 'none'}\n"
        )

    def _build_skills(self, pcfg: PromptProfileConfig | None = None) -> str:
        """Render the ``[Skills]`` index from :attr:`skill_registry`."""

        if not self.skill_registry:
            return ""
        index = self.skill_registry.build_skills_index()
        return f"[Skills]\n{index}\n" if index else ""

    def _build_enabled_skills(
        self,
        enabled_skills: list[Skill] | None = None,
        profile_config: PromptProfileConfig | None = None,
    ) -> str:
        """Render full instructions for ``enabled_skills``, truncating per profile."""

        if not enabled_skills:
            return ""

        max_len = profile_config.max_skill_instructions_length if profile_config else None

        sections: list[str] = []
        for skill in enabled_skills:
            section = skill.to_prompt_section()
            if max_len is not None and len(section) > max_len:
                section = section[:max_len] + "..."
            sections.append(section)

        rendered = "\n\n".join(sections)
        return f"[Enabled Skill Instructions]\n{rendered}\n"

    def _build_tools(
        self,
        tool_names: list[str] | None = None,
        pcfg: PromptProfileConfig | None = None,
    ) -> str:
        """Render the ``[Available Tools]`` bullet list, capped by profile."""

        if not tool_names:
            return ""

        max_tools = pcfg.max_tools_listed if pcfg else None

        if max_tools is not None and len(tool_names) > max_tools:
            shown = tool_names[:max_tools]
            remaining = len(tool_names) - max_tools
            lines = [f"  - {name}" for name in shown]
            lines.append(f"  ... and {remaining} more")
        else:
            lines = [f"  - {name}" for name in tool_names]

        return "[Available Tools]\n" + "\n".join(lines) + "\n"

    def _build_guardrails(self, guardrails: list[str] | None = None) -> str:
        """Render the ``[Guardrails]`` section from the active guardrail list."""

        active_guardrails = self.guardrails if guardrails is None else guardrails
        if not active_guardrails:
            return ""
        lines = ["[Guardrails]"]
        for g in active_guardrails:
            lines.append(f"  - {g}")
        return "\n".join(lines) + "\n"

    def _build_memory_section(
        self,
        agent_id: str | None,
        pcfg: PromptProfileConfig,
    ) -> str:
        """Render the ``[Relevant Memories]`` section using :attr:`memory_provider`."""

        if self.memory_provider is None:
            return ""
        k = max(0, int(getattr(pcfg, "max_memories_injected", 5)))
        if k == 0:
            return ""
        try:
            snippets = self.memory_provider(agent_id, k)
        except Exception:
            return ""
        if not snippets:
            return ""
        lines = ["[Relevant Memories]"]
        for s in snippets[:k]:
            text = str(s).strip().replace("\n", " ")
            if text:
                lines.append(f"  - {text}")
        return "\n".join(lines) + "\n" if len(lines) > 1 else ""

    def _build_user_profile_section(self, agent_id: str | None) -> str:
        """Render the ``[User Profile]`` section using :attr:`user_profile_provider`."""

        if self.user_profile_provider is None:
            return ""
        try:
            text = self.user_profile_provider(agent_id)
        except Exception:
            return ""
        if not text or not text.strip():
            return ""
        return f"[User Profile]\n{text.strip()}\n"

    def _build_bootstrap(self, agent_id: str | None = None) -> str:
        """Collect bootstrap text from the ``bootstrap_files`` hook chain."""

        if not self.hook_runner or not self.hook_runner.has_hooks("bootstrap_files"):
            return ""
        results = self.hook_runner.run("bootstrap_files", agent_id=agent_id)
        if not results:
            return ""
        sections = []
        for content in results:
            if isinstance(content, list):
                sections.extend(content)
            elif isinstance(content, str):
                sections.append(content)
        return "\n".join(sections) if sections else ""

    def assemble_system_prompt_prefix(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        enabled_skills: list[Skill] | None = None,
        profile: PromptProfile | PromptProfileConfig | str | None = None,
    ) -> str:
        """Render the full system-prompt prefix string for the active profile.

        Builds a :class:`PromptContext` then concatenates identity, tooling,
        safety, skill, workspace, sandbox, runtime, execution-policy, and
        output-style blocks. Returns a single short identity line when the
        ``NONE`` profile is selected.
        """

        resolved_profile = _resolve_profile_config(profile) if profile is not None else self.default_profile_config
        if resolved_profile.profile == PromptProfile.NONE:
            return "You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment."

        ctx = self.build_with_overrides(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=sandbox_config,
            guardrails=guardrails,
            enabled_skills=enabled_skills,
            profile=resolved_profile,
        )
        parts = [
            self._build_identity_block(resolved_profile),
            self._build_tooling_block(ctx),
            self._build_safety_block(ctx),
            self._build_skill_block(ctx),
            ctx.user_profile_section.rstrip() if ctx.user_profile_section else "",
            ctx.memory_section.rstrip() if ctx.memory_section else "",
            self._build_workspace_block(ctx),
            self._build_sandbox_block(ctx),
            self._build_runtime_block(ctx),
            self._build_execution_policy_block(resolved_profile),
            self._build_output_style_block(resolved_profile),
        ]
        return "\n\n".join(part for part in parts if part)

    def _build_identity_block(self, profile: PromptProfileConfig) -> str:
        """Render the ``[Identity]`` block; sub-agents get a delegated variant."""

        if profile.profile == PromptProfile.FULL:
            lines = [
                "[Identity]",
                "- You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.",
                "- Complete the user's task accurately, efficiently, and safely using the available tools, skills, and workspace context.",
                "- Follow runtime policy, sandbox limits, and tool restrictions.",
            ]
        else:
            lines = [
                "[Identity]",
                "- You are Xerxes, a delegated sub-agent running inside a controlled runtime.",
                "- Stay within the assigned subtask and return integration-friendly output.",
                "- Follow runtime policy, sandbox limits, and tool restrictions.",
            ]
        return "\n".join(lines)

    def _build_tooling_block(self, ctx: PromptContext) -> str:
        """Render the ``[Tooling]`` block, including per-tool selection guidance."""

        tools_section = ctx.tools_section.rstrip() if ctx.tools_section else "[Available Tools]\n  - none"
        tools_section_lower = tools_section.lower()
        lines = [
            "[Tooling]",
            "Available tools in this run:",
            tools_section,
            "Tool rules:",
            "- Use tools only when you need live external state, workspace contents, shell execution, or another real action you cannot complete from the current conversation alone.",
            "- Do not use or simulate tools for greetings, simple arithmetic, direct explanations, summaries, or code-writing requests that can be answered directly.",
            "- Do not repeat the same tool call with the same arguments if it did not make progress.",
            "- If a tool result already answers the task, use it directly.",
            "- For a simple tool-backed request, prefer one necessary tool call followed by the final answer instead of extended planning or multiple retries.",
            "- If a tool fails, adjust strategy instead of blindly retrying.",
        ]
        guidance_lines: list[str] = []
        if "web.search_query" in tools_section_lower:
            guidance_lines.append(
                '- `web.search_query`: Use this when the user explicitly asks to search/look up/browse the web, or when the answer depends on live recent information, news, or source discovery. Pass a clean search phrase in `q`; prefer `search_type="news"` for latest/news queries and `"text"` for general research.'
            )
            guidance_lines.append(
                "- Generic web-search follow-ups: If the user says something like `search the web`, `look it up`, or `find it` right after discussing a topic, infer the topic from the latest relevant user request instead of asking the same clarification again, then call `web.search_query` with that inferred query."
            )
        if "web.open" in tools_section_lower:
            guidance_lines.append(
                "- `web.open`: Use this after search when you need the contents of a specific result page, direct quotes, or details that are not in the search snippets. Do this before presenting a claim as confirmed, official, or verified."
            )
        if "web.find" in tools_section_lower:
            guidance_lines.append(
                "- `web.find`: Use this on an already opened page to jump to a specific term, section, or citation instead of guessing where the information is."
            )
        if "readfile" in tools_section_lower or "read_file" in tools_section_lower:
            guidance_lines.append(
                "- File-reading tools: Use them for project-specific facts, exact code behavior, config values, or anything the workspace can answer more reliably than memory."
            )
        if "listdir" in tools_section_lower or "list_dir" in tools_section_lower:
            guidance_lines.append(
                "- Directory-listing tools: Use them to discover repo structure or confirm what files exist before claiming paths from memory."
            )
        if (
            "exec_command" in tools_section_lower
            or "executeshell" in tools_section_lower
            or "execute_shell" in tools_section_lower
        ):
            guidance_lines.append(
                "- Shell tools: Use them for real command execution, environment inspection, tests, and filesystem queries that require current machine state."
            )
        if guidance_lines:
            lines.extend(["Tool selection guidance:", *guidance_lines])
        lines.extend(
            [
                "Search grounding rules:",
                "- If a web/search tool is available or has already been used in the conversation, do not claim that you cannot browse or access current information.",
                "- Search snippets and result titles are leads, not verification. Phrase them as 'search results indicate' or 'the top result says' unless you opened a source and confirmed it.",
            ]
        )
        return "\n".join(lines)

    def _build_safety_block(self, ctx: PromptContext) -> str:
        """Render the ``[Safety]`` block, merging guardrails and core safety rules."""

        lines = ["[Safety]", "Safety guidance:"]
        if ctx.guardrails_section:
            lines.append(ctx.guardrails_section.rstrip())
        else:
            lines.append("- No additional runtime guardrails are configured for this run.")
        lines.extend(
            [
                "Safety rules:",
                "- Do not try to bypass oversight, sandboxing, or tool restrictions.",
                "- Do not invent tool results, file contents, or execution outcomes.",
                "- If blocked by runtime policy or sandbox limits, say so plainly.",
            ]
        )
        return "\n".join(lines)

    def _build_skill_block(self, ctx: PromptContext) -> str:
        """Render the combined ``[Skills & Instructions]`` block."""

        if not ctx.skills_section and not ctx.enabled_skills_section:
            return ""
        lines = ["[Skills & Instructions]"]
        if ctx.skills_section:
            lines.extend(["Available skills:", ctx.skills_section.rstrip()])
        if ctx.enabled_skills_section:
            lines.extend(["Enabled skill instructions:", ctx.enabled_skills_section.rstrip()])
        lines.extend(
            [
                "Skill rules:",
                "- Use enabled skills as task-specific operating instructions.",
                "- If a skill is listed but not fully injected, load or apply it only when relevant.",
                "- Do not assume a skill exists unless it is present in runtime context.",
            ]
        )
        return "\n".join(lines)

    def _build_workspace_block(self, ctx: PromptContext) -> str:
        """Render the ``[Workspace Context]`` block including bootstrap notes."""

        if not ctx.workspace_section and not ctx.bootstrap_section:
            return ""
        lines = ["[Workspace Context]"]
        if ctx.workspace_section:
            lines.append(ctx.workspace_section.rstrip())
        if ctx.bootstrap_section:
            lines.extend(["Project/bootstrap context:", ctx.bootstrap_section.rstrip()])
        lines.extend(
            [
                "Workspace rules:",
                "- Treat workspace files as the source of truth for project-specific behavior.",
                "- Prefer minimal, targeted changes over broad rewrites.",
            ]
        )
        return "\n".join(lines)

    def _build_sandbox_block(self, ctx: PromptContext) -> str:
        """Render the ``[Sandbox Runtime]`` block with the active sandbox rules."""

        if not ctx.sandbox_section:
            return ""
        lines = [
            "[Sandbox Runtime]",
            ctx.sandbox_section.rstrip(),
            "Sandbox rules:",
            "- Treat sandboxed tools as sandboxed.",
            "- Elevated execution is exceptional and must be explicit.",
            "- Never describe host execution as sandboxed if it was not.",
        ]
        return "\n".join(lines)

    def _build_runtime_block(self, ctx: PromptContext) -> str:
        """Render the ``[Runtime]`` block (runtime + datetime + reasoning sections)."""

        if not ctx.runtime_section and not ctx.datetime_section and not ctx.reasoning_section:
            return ""
        lines = ["[Runtime]"]
        if ctx.runtime_section:
            lines.append(ctx.runtime_section.rstrip())
        if ctx.datetime_section:
            lines.append(ctx.datetime_section.rstrip())
        if ctx.reasoning_section:
            lines.append(ctx.reasoning_section.rstrip())
        return "\n".join(lines)

    def _build_execution_policy_block(self, profile: PromptProfileConfig) -> str:
        """Render the ``[Execution Policy]`` block, profile-aware."""

        if profile.profile == PromptProfile.FULL:
            lines = [
                "[Execution Policy]",
                "1. Understand the request and use the workspace context first.",
                "2. Choose the smallest correct action that moves the task forward.",
                "3. Use tools only when you need missing live information, file contents, execution, or verification that cannot be done from the conversation alone.",
                "4. Do not simulate tool calls or wrap normal answers in tool/XML markup.",
                "5. If one successful tool result is enough, stop and give the final answer immediately.",
                "6. After tool use, answer from the actual result.",
                "7. Surface blockers, assumptions, and risks clearly.",
                "8. Do not loop.",
            ]
        else:
            lines = [
                "[Execution Policy]",
                "- Stay within the assigned subtask.",
                "- Use tools only when needed for missing live information or real actions.",
                "- Do not simulate tool calls or emit tool/XML wrappers in normal answers.",
                "- If one tool result is enough, answer immediately instead of continuing to plan.",
                "- Answer from actual tool and workspace results.",
                "- Keep output compact and integration-friendly.",
            ]
        return "\n".join(lines)

    def _build_output_style_block(self, profile: PromptProfileConfig) -> str:
        """Render the ``[Output Style]`` block (only emitted for the FULL profile)."""

        if profile.profile != PromptProfile.FULL:
            return ""
        return "\n".join(
            [
                "[Output Style]",
                "- Be precise, technical, and pragmatic.",
                "- Prefer concrete outcomes over general advice.",
                "- Keep internal reasoning out of the visible answer unless the user explicitly asks for it.",
                "- Put the user-facing answer in the normal assistant response content, not in a scratchpad or reasoning field.",
                "- If code or files were changed, mention the real result.",
                "- If tests were run, report the actual scope and outcome.",
            ]
        )
