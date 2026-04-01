# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Runtime context assembly for Calute system prompts.

Builds rich system prompt sections that include:
- Runtime/environment context (date, platform, version)
- Workspace context (working directory, project info)
- Sandbox status
- Skills index
- Tooling summary
- Safety/guardrail reminders
- Bootstrap file injections from hooks

Supports :class:`~calute.runtime.profiles.PromptProfile` to control
verbosity (full, compact, minimal, none) for sub-agent delegation.
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
    """Snapshot of runtime environment information.

    Captures a point-in-time view of the host environment including
    timestamps, platform details, Python and Calute versions, and the
    current workspace directory. Used by :class:`PromptContextBuilder`
    to populate runtime and workspace prompt sections.

    Attributes:
        timestamp: ISO 8601 formatted local timestamp with timezone offset.
        timezone: Name of the local timezone (e.g. ``"UTC"``, ``"PST"``).
        platform: Operating system name and release (e.g. ``"Darwin 23.4.0"``).
        python_version: Python interpreter version string (e.g. ``"3.12.3"``).
        calute_version: Installed Calute package version.
        working_directory: Absolute path to the resolved workspace directory.
        workspace_name: Base name of the workspace directory.
    """

    timestamp: str = ""
    timezone: str = ""
    platform: str = ""
    python_version: str = ""
    calute_version: str = ""
    working_directory: str = ""
    workspace_name: str = ""

    @classmethod
    def capture(cls, workspace_root: str | None = None) -> RuntimeInfo:
        """Capture a snapshot of the current runtime environment.

        Reads the local clock, platform details, Python version, and the
        installed Calute version to produce a frozen :class:`RuntimeInfo`
        instance.

        Args:
            workspace_root: Optional explicit workspace directory. When
                ``None``, falls back to the current working directory.

        Returns:
            A new :class:`RuntimeInfo` populated with current environment
            data.
        """
        from calute import __version__

        now = datetime.now().astimezone()
        cwd = os.path.abspath(workspace_root or os.getcwd())
        return cls(
            timestamp=now.isoformat(timespec="seconds"),
            timezone=now.tzname() or "local",
            platform=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            calute_version=__version__,
            working_directory=cwd,
            workspace_name=os.path.basename(cwd),
        )


@dataclass
class PromptContext:
    """Assembled context sections for system prompt enrichment.

    Each field holds a pre-rendered string for one logical section of the
    system prompt. Fields that are empty (``""``) indicate that the
    corresponding section was disabled by the active
    :class:`~calute.runtime.profiles.PromptProfileConfig`.

    Attributes:
        runtime_section: Platform, Python version, and Calute version block.
        workspace_section: Working directory and project name block.
        datetime_section: Current local date/time and timezone block.
        reasoning_section: Profile name and reasoning guidance block.
        sandbox_section: Sandbox mode and tool routing block.
        skills_section: Index of all discovered skills from the registry.
        enabled_skills_section: Full instruction text for skills that are
            currently enabled for the agent.
        tools_section: List of available tool names for this run.
        guardrails_section: Active guardrail rules for safety enforcement.
        bootstrap_section: Injected project/bootstrap content from hook
            runners.
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


def _resolve_profile_config(
    profile: PromptProfile | PromptProfileConfig | str | None,
) -> PromptProfileConfig:
    """Normalise a profile argument into a concrete :class:`PromptProfileConfig`.

    Accepts several input forms for ergonomic use across the codebase:

    * ``None`` -- returns the default :data:`PromptProfile.FULL` config.
    * A :class:`PromptProfile` enum member -- looked up via
      :func:`get_profile_config`.
    * A lowercase string (``"compact"``, ``"minimal"``, etc.) -- converted
      to :class:`PromptProfile` then looked up.
    * An existing :class:`PromptProfileConfig` -- returned as-is.

    Args:
        profile: The profile specification to resolve. May be ``None``,
            a :class:`PromptProfile` enum value, a string name, or an
            already-resolved :class:`PromptProfileConfig`.

    Returns:
        A fully resolved :class:`PromptProfileConfig` instance.

    Raises:
        ValueError: If *profile* is a string that does not match any
            :class:`PromptProfile` member.
    """
    if profile is None:
        return get_profile_config(PromptProfile.FULL)
    if isinstance(profile, PromptProfile):
        return get_profile_config(profile)
    if isinstance(profile, str):
        return get_profile_config(PromptProfile(profile.strip().lower()))
    return profile


class PromptContextBuilder:
    """Builds enriched prompt context from runtime state.

    Orchestrates the construction of all system-prompt sections that
    describe the runtime environment, available tools, skills, sandbox
    configuration, guardrails, and workspace context. The builder
    supports multiple verbosity levels via
    :class:`~calute.runtime.profiles.PromptProfile` and can produce
    agent-specific overrides.

    Attributes:
        skill_registry: Registry of discovered skills.
        plugin_registry: Registry of discovered plugins.
        hook_runner: Hook runner for bootstrap file injection.
        sandbox_config: Default sandbox configuration.
        guardrails: Default list of guardrail rule strings.
        default_profile_config: Resolved default prompt profile config.
        workspace_root: Optional explicit workspace directory path.

    Example:
        >>> builder = PromptContextBuilder(skill_registry=my_registry)
        >>> ctx = builder.build(agent_id="coder")
        >>> print(ctx.runtime_section)
    """

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
        hook_runner: HookRunner | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        profile: PromptProfile | PromptProfileConfig | None = None,
        workspace_root: str | None = None,
    ):
        """Initialise the prompt context builder.

        Args:
            skill_registry: Optional skill registry for building the
                skills index section.
            plugin_registry: Optional plugin registry (reserved for
                future plugin-contributed prompt sections).
            hook_runner: Optional hook runner used to invoke
                ``bootstrap_files`` hooks for project context injection.
            sandbox_config: Optional default sandbox configuration
                applied when no agent-specific override is provided.
            guardrails: Optional default list of guardrail rule strings.
            profile: Default prompt profile controlling section
                verbosity. Defaults to :data:`PromptProfile.FULL`.
            workspace_root: Explicit workspace root directory. When
                ``None``, the current working directory is used.
        """
        self.skill_registry = skill_registry
        self.plugin_registry = plugin_registry
        self.hook_runner = hook_runner
        self.sandbox_config = sandbox_config
        self.guardrails = guardrails or []
        self.default_profile_config = _resolve_profile_config(profile)
        self.workspace_root = workspace_root

    def build(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        profile: PromptProfile | PromptProfileConfig | str | None = None,
    ) -> PromptContext:
        """Build all prompt context sections using default overrides.

        Convenience wrapper around :meth:`build_with_overrides` that
        forwards the most common parameters.

        Args:
            agent_id: Optional agent identifier for bootstrap hook
                dispatch and per-agent customisation.
            tool_names: Optional list of tool names available to the
                agent in this run.
            profile: Optional prompt profile override. When ``None``,
                the builder's default profile is used.

        Returns:
            A :class:`PromptContext` with all applicable sections
            populated.
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
        """Build all prompt context sections with agent-specific overrides.

        This is the full-control entry point. Each parameter can override
        the builder's defaults for a single invocation, enabling
        per-agent customisation without mutating shared state.

        Args:
            agent_id: Optional agent identifier forwarded to bootstrap
                hooks.
            tool_names: Optional list of tool names available to the
                agent.
            sandbox_config: Optional per-agent sandbox configuration
                that overrides the builder's default.
            guardrails: Optional per-agent guardrail list that overrides
                the builder's default.
            enabled_skills: Optional list of resolved :class:`Skill`
                objects whose instruction text should be injected.
            profile: Optional prompt profile override. Accepts a
                :class:`PromptProfile` enum, a string name, or a
                :class:`PromptProfileConfig`. When ``None``, the
                builder's default profile is used.

        Returns:
            A :class:`PromptContext` with each section populated (or
            left empty) according to the resolved profile configuration.
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
        return ctx

    def build_compact_prefix(
        self,
        agent_id: str | None = None,
        tool_names: list[str] | None = None,
        sandbox_config: SandboxConfig | None = None,
        guardrails: list[str] | None = None,
        enabled_skills: list[Skill] | None = None,
    ) -> str:
        """Build a system prompt prefix using the COMPACT profile.

        The compact profile drops workspace/bootstrap sections and caps
        skill instructions and tool lists, yielding a shorter prefix
        suitable for sub-agent delegation.

        Args:
            agent_id: Optional agent identifier for hook dispatch.
            tool_names: Optional list of available tool names.
            sandbox_config: Optional sandbox configuration override.
            guardrails: Optional guardrail list override.
            enabled_skills: Optional list of enabled :class:`Skill`
                objects.

        Returns:
            A compact system prompt prefix string.
        """
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
        """Build a system prompt prefix using the MINIMAL profile.

        The minimal profile includes only sandbox info, guardrails, and
        a short tool list (capped at 10 entries). All other sections are
        omitted, producing the smallest useful prefix for internal
        delegation.

        Args:
            agent_id: Optional agent identifier for hook dispatch.
            tool_names: Optional list of available tool names.
            sandbox_config: Optional sandbox configuration override.
            guardrails: Optional guardrail list override.
            enabled_skills: Optional list of enabled :class:`Skill`
                objects.

        Returns:
            A minimal system prompt prefix string.
        """
        return self.assemble_system_prompt_prefix(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=sandbox_config,
            guardrails=guardrails,
            enabled_skills=enabled_skills,
            profile=PromptProfile.MINIMAL,
        )

    def build_none_prefix(self) -> str:
        """Build an OpenClaw-style identity-only system prompt prefix.

        Returns only the bare identity line with no runtime sections,
        useful when the caller supplies all context externally.

        Returns:
            A single-line identity string for the system prompt.
        """
        return self.assemble_system_prompt_prefix(profile=PromptProfile.NONE)

    def _build_runtime(self, info: RuntimeInfo) -> str:
        """Build the runtime environment section string.

        Args:
            info: Captured runtime information snapshot.

        Returns:
            A multi-line string describing the platform, Python version,
            and Calute version.
        """
        return (
            f"[Runtime Context]\n"
            f"  Platform: {info.platform}\n"
            f"  Python: {info.python_version}\n"
            f"  Calute: v{info.calute_version}\n"
        )

    def _build_workspace(self, info: RuntimeInfo) -> str:
        """Build the workspace directory section string.

        Args:
            info: Captured runtime information snapshot.

        Returns:
            A string containing the working directory path and project
            name.
        """
        return f"[Workspace]\n  Directory: {info.working_directory}\n  Project: {info.workspace_name}\n"

    def _build_datetime(self, info: RuntimeInfo) -> str:
        """Build the current date/time section string.

        Args:
            info: Captured runtime information snapshot.

        Returns:
            A string with the local timestamp and timezone name.
        """
        return f"[Current Date & Time]\n  Local time: {info.timestamp}\n  Time zone: {info.timezone}\n"

    def _build_reasoning(self, pcfg: PromptProfileConfig | None = None) -> str:
        """Build the reasoning/profile guidance section string.

        Emits the active profile name and a general guidance reminder
        that answers should be grounded in actual tool results.

        Args:
            pcfg: Optional profile config whose profile name is
                displayed. Defaults to :data:`PromptProfile.FULL`.

        Returns:
            A multi-line reasoning guidance block string.
        """
        profile_name = pcfg.profile.value if pcfg is not None else PromptProfile.FULL.value
        return (
            f"[Response Guidance]\n"
            f"  Profile: {profile_name}\n"
            f"  Guidance: answer from actual tool and workspace results; avoid speculative claims; keep internal reasoning private; and put the final answer in the normal assistant response content, not in a scratchpad or reasoning field.\n"
        )

    def _build_sandbox(self, sandbox_config: SandboxConfig | None = None) -> str:
        """Build the sandbox status section string.

        Describes the sandbox mode and lists which tools are sandboxed
        versus elevated. Returns an empty string when no sandbox
        configuration is available.

        Args:
            sandbox_config: Optional override sandbox configuration. When
                ``None``, falls back to the builder's default
                ``sandbox_config``.

        Returns:
            A sandbox description block, or an empty string if sandbox
            is not configured.
        """
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
        """Build the skills index section string from the skill registry.

        Delegates to the skill registry's ``build_skills_index`` method
        to produce a human-readable summary of all discovered skills.

        Args:
            pcfg: Optional profile config (reserved for future
                profile-aware skill index formatting).

        Returns:
            A formatted skills index block, or an empty string when no
            skill registry is configured or no skills are discovered.
        """
        if not self.skill_registry:
            return ""
        index = self.skill_registry.build_skills_index()
        return f"[Skills]\n{index}\n" if index else ""

    def _build_enabled_skills(
        self,
        enabled_skills: list[Skill] | None = None,
        profile_config: PromptProfileConfig | None = None,
    ) -> str:
        """Build the enabled-skill instruction sections.

        Renders each enabled skill's prompt section and concatenates
        them. When ``max_skill_instructions_length`` is set in the
        profile config, individual skill sections are truncated to that
        length.

        Args:
            enabled_skills: List of resolved :class:`Skill` objects to
                render. Returns an empty string when ``None`` or empty.
            profile_config: Optional profile config providing the
                ``max_skill_instructions_length`` truncation cap.

        Returns:
            A formatted block of enabled skill instructions, or an
            empty string if no skills are provided.
        """
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
        """Build the available tools list section string.

        When ``max_tools_listed`` is set in the profile config and the
        tool list exceeds that cap, the output is truncated with a
        summary count of remaining tools.

        Args:
            tool_names: List of tool name strings. Returns an empty
                string when ``None`` or empty.
            pcfg: Optional profile config providing the
                ``max_tools_listed`` truncation cap.

        Returns:
            A formatted available-tools block, or an empty string if no
            tool names are provided.
        """
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
        """Build the guardrails section string.

        Args:
            guardrails: Optional override guardrail list. When ``None``,
                falls back to the builder's default ``guardrails``.

        Returns:
            A formatted guardrails block, or an empty string if no
            guardrails are active.
        """
        active_guardrails = self.guardrails if guardrails is None else guardrails
        if not active_guardrails:
            return ""
        lines = ["[Guardrails]"]
        for g in active_guardrails:
            lines.append(f"  - {g}")
        return "\n".join(lines) + "\n"

    def _build_bootstrap(self, agent_id: str | None = None) -> str:
        """Build the bootstrap/project context section by running hooks.

        Invokes the ``bootstrap_files`` hook point and concatenates
        all returned content. Hook results may be individual strings or
        lists of strings.

        Args:
            agent_id: Optional agent identifier passed to the hook
                runner for agent-specific bootstrap content.

        Returns:
            Concatenated bootstrap content, or an empty string if no
            hook runner is configured or no results are returned.
        """
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
        """Build the full enriched prefix for a system prompt.

        Assembles identity, tooling, safety, skills, workspace, sandbox,
        runtime, execution policy, and output style blocks into a single
        string that should be prepended to the agent's instructions in
        the system message.

        For the :data:`PromptProfile.NONE` profile, returns only a
        single identity line with no runtime context sections.

        Args:
            agent_id: Optional agent identifier for hook dispatch and
                per-agent customisation.
            tool_names: Optional list of tool names available to the
                agent.
            sandbox_config: Optional sandbox configuration override.
            guardrails: Optional guardrail list override.
            enabled_skills: Optional list of enabled :class:`Skill`
                objects.
            profile: Optional prompt profile override. When ``None``,
                the builder's default profile is used (which itself
                defaults to :data:`PromptProfile.FULL`).

        Returns:
            The assembled system prompt prefix string with all
            applicable blocks joined by double newlines.
        """
        resolved_profile = _resolve_profile_config(profile) if profile is not None else self.default_profile_config
        if resolved_profile.profile == PromptProfile.NONE:
            return "You are Calute, a runtime-managed AI agent operating inside a controlled tool environment."

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
            self._build_workspace_block(ctx),
            self._build_sandbox_block(ctx),
            self._build_runtime_block(ctx),
            self._build_execution_policy_block(resolved_profile),
            self._build_output_style_block(resolved_profile),
        ]
        return "\n\n".join(part for part in parts if part)

    def _build_identity_block(self, profile: PromptProfileConfig) -> str:
        """Build the identity block lines for the assembled system prompt.

        For the FULL profile, describes the agent as a primary Calute
        agent. For all other profiles, describes it as a delegated
        sub-agent with narrower responsibilities.

        Args:
            profile: The resolved profile configuration determining
                which identity variant to emit.

        Returns:
            A multi-line identity block string.
        """
        if profile.profile == PromptProfile.FULL:
            lines = [
                "[Identity]",
                "- You are Calute, a runtime-managed AI agent operating inside a controlled tool environment.",
                "- Complete the user's task accurately, efficiently, and safely using the available tools, skills, and workspace context.",
                "- Follow runtime policy, sandbox limits, and tool restrictions.",
            ]
        else:
            lines = [
                "[Identity]",
                "- You are Calute, a delegated sub-agent running inside a controlled runtime.",
                "- Stay within the assigned subtask and return integration-friendly output.",
                "- Follow runtime policy, sandbox limits, and tool restrictions.",
            ]
        return "\n".join(lines)

    def _build_tooling_block(self, ctx: PromptContext) -> str:
        """Build the tooling block including tool list and tool-use rules.

        Combines the rendered tools section from *ctx* with a fixed set
        of tool-use guidance rules.

        Args:
            ctx: The assembled prompt context containing the tools
                section.

        Returns:
            A multi-line tooling block string.
        """
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
            "- Never emit fake tool syntax such as <tool_call>, <response>, <function=name>, or <parameter=name>. Either call a real tool through the tool interface or answer normally.",
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
        """Build the safety/guardrails block for the assembled system prompt.

        Combines any active guardrails from *ctx* with fixed safety
        rules that prevent sandbox bypass and fabricated outputs.

        Args:
            ctx: The assembled prompt context containing the guardrails
                section.

        Returns:
            A multi-line safety block string.
        """
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
        """Build the skills and enabled-skill instructions block.

        Combines the skills index and enabled-skill instructions from
        *ctx* with rules governing skill usage.

        Args:
            ctx: The assembled prompt context containing skills and
                enabled-skills sections.

        Returns:
            A multi-line skills block string, or an empty string if
            neither the skills index nor enabled-skill instructions
            are present.
        """
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
        """Build the workspace context block including bootstrap content.

        Combines the workspace directory section and bootstrap-injected
        project context with rules for treating workspace files as the
        source of truth.

        Args:
            ctx: The assembled prompt context containing workspace and
                bootstrap sections.

        Returns:
            A multi-line workspace block string, or an empty string if
            neither section is populated.
        """
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
        """Build the sandbox runtime block if sandbox info is present.

        Renders the sandbox mode details with rules enforcing correct
        sandboxed vs. elevated tool handling.

        Args:
            ctx: The assembled prompt context containing the sandbox
                section.

        Returns:
            A multi-line sandbox block string, or an empty string if
            no sandbox information is available.
        """
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
        """Build the runtime info block combining runtime, datetime, and reasoning sections.

        Concatenates the runtime environment, date/time, and reasoning
        guidance sections under a single ``[Runtime]`` header.

        Args:
            ctx: The assembled prompt context containing runtime,
                datetime, and reasoning sections.

        Returns:
            A multi-line runtime block string, or an empty string if
            all three sub-sections are empty.
        """
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
        """Build the execution policy block appropriate for the given profile.

        The FULL profile emits a detailed numbered policy, while all
        other profiles emit a condensed bullet-list variant.

        Args:
            profile: The resolved profile configuration controlling
                which policy variant to emit.

        Returns:
            A multi-line execution policy block string.
        """
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
        """Build the output style block; only emitted for the FULL profile.

        Provides stylistic guidance encouraging precise, technical, and
        results-oriented output. Omitted for non-FULL profiles to save
        tokens.

        Args:
            profile: The resolved profile configuration. Only the FULL
                profile triggers output.

        Returns:
            A multi-line output style block string for the FULL profile,
            or an empty string for all other profiles.
        """
        if profile.profile != PromptProfile.FULL:
            return ""
        return "\n".join(
            [
                "[Output Style]",
                "- Be precise, technical, and pragmatic.",
                "- Prefer concrete outcomes over general advice.",
                "- Give the final answer directly, without <response>, <tool_call>, or XML-style wrappers.",
                "- Keep internal reasoning out of the visible answer unless the user explicitly asks for it.",
                "- Put the user-facing answer in the normal assistant response content, not in a scratchpad or reasoning field.",
                "- If code or files were changed, mention the real result.",
                "- If tests were run, report the actual scope and outcome.",
            ]
        )
