# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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

"""Runtime feature integration for Xerxes.

This module owns the opt-in OpenClaw-style runtime capability layer:
- plugin and skill discovery
- hook registration
- tool policy configuration
- loop detection configuration
- sandbox routing configuration
- prompt enrichment helpers
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..audit import AuditCollector, AuditEmitter
from ..core.utils import get_callable_public_name
from ..extensions.hooks import HOOK_POINTS, HookRunner
from ..extensions.plugins import PluginRegistry
from ..extensions.skills import Skill, SkillRegistry
from ..operators import HIGH_POWER_OPERATOR_TOOLS, OperatorRuntimeConfig, OperatorState
from ..security.policy import PolicyEngine, ToolPolicy
from ..security.sandbox import SandboxBackend, SandboxConfig, SandboxRouter
from ..session import SessionManager, SessionStore
from ..types import Agent
from .context import PromptContextBuilder
from .loop_detection import LoopDetectionConfig, LoopDetector
from .profiles import PromptProfile, PromptProfileConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeOverrides:
    """Per-agent runtime feature overrides.

    Allows individual agents to deviate from global runtime settings.
    A field set to ``None`` means "inherit the global runtime setting".
    Empty lists explicitly clear list-valued globals (e.g. setting
    ``guardrails=[]`` removes all guardrails for that agent even when
    the global config has guardrails defined).

    Attributes:
        policy: Agent-specific tool policy. ``None`` inherits the global
            policy.
        loop_detection: Agent-specific loop detection configuration.
            ``None`` inherits the global loop detection config.
        sandbox: Agent-specific sandbox configuration. ``None`` inherits
            the global sandbox config.
        enabled_skills: List of skill names enabled for the agent.
            ``None`` inherits the global enabled skills.
        guardrails: List of guardrail rule strings for the agent.
            ``None`` inherits the global guardrails.
        prompt_profile: Prompt profile name or enum for system prompt
            verbosity. ``None`` inherits the global default.
    """

    policy: ToolPolicy | None = None
    loop_detection: LoopDetectionConfig | None = None
    sandbox: SandboxConfig | None = None
    enabled_skills: list[str] | None = None
    guardrails: list[str] | None = None
    prompt_profile: PromptProfile | str | None = None


@dataclass
class RuntimeFeaturesConfig:
    """Public configuration for Xerxes runtime features.

    This is the top-level configuration object passed by the user to
    enable and customise the runtime feature layer. All fields have safe
    defaults, so the minimal configuration is simply
    ``RuntimeFeaturesConfig(enabled=True)``.

    Attributes:
        enabled: Master switch. When ``False``, the entire runtime
            feature layer is a no-op.
        workspace_root: Explicit workspace root directory. When ``None``,
            the current working directory is used.
        plugin_dirs: List of directory paths to scan for plugins.
        skill_dirs: List of directory paths to scan for skills.
        discover_conventional_extensions: When ``True``, also look for
            ``plugins/`` and ``skills/`` directories under the workspace
            root using conventional layout conventions.
        guardrails: Global list of guardrail rule strings injected into
            every agent's system prompt.
        policy: Global tool policy applied to all agents unless
            overridden per-agent.
        loop_detection: Global loop detection configuration. ``None``
            disables loop detection.
        sandbox: Global sandbox configuration. ``None`` disables
            sandboxing.
        enabled_skills: Global list of skill names to enable for all
            agents.
        default_prompt_profile: Default prompt profile name or enum
            controlling system prompt verbosity.
        audit_collector: Optional audit collector for emitting audit
            events during tool execution.
        session_store: Optional session store backend for persisting
            agent session state.
        operator: Optional runtime operator configuration for power
            tools.
        agent_overrides: Mapping from agent ID to per-agent runtime
            overrides.
    """

    enabled: bool = False
    workspace_root: str | None = None
    plugin_dirs: list[str] = field(default_factory=list)
    skill_dirs: list[str] = field(default_factory=list)
    discover_conventional_extensions: bool = True
    guardrails: list[str] = field(default_factory=list)
    policy: ToolPolicy | None = None
    loop_detection: LoopDetectionConfig | None = None
    sandbox: SandboxConfig | None = None
    enabled_skills: list[str] = field(default_factory=list)
    default_prompt_profile: PromptProfile | str | None = None
    audit_collector: AuditCollector | None = None
    session_store: SessionStore | None = None
    operator: OperatorRuntimeConfig | None = None
    agent_overrides: dict[str, AgentRuntimeOverrides] = field(default_factory=dict)


@dataclass
class RuntimeFeaturesState:
    """Internal state holder for runtime feature integration.

    Created from a :class:`RuntimeFeaturesConfig` and manages all
    runtime sub-systems: plugin/skill discovery, hook registration,
    policy engine, sandbox routing, audit emission, session management,
    and prompt context building. Intended to be instantiated once per
    Xerxes runtime lifecycle.

    Attributes:
        config: The user-provided runtime features configuration.
        plugin_registry: Registry of discovered plugins.
        skill_registry: Registry of discovered skills.
        hook_runner: Hook runner for plugin-contributed callbacks.
        sandbox_backend: Optional instantiated sandbox execution
            backend.
        operator_state: Optional runtime operator state for power
            tools.
    """

    config: RuntimeFeaturesConfig
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    skill_registry: SkillRegistry = field(default_factory=SkillRegistry)
    hook_runner: HookRunner = field(default_factory=HookRunner)
    sandbox_backend: SandboxBackend | None = None
    operator_state: OperatorState | None = None

    def __post_init__(self) -> None:
        """Initialise derived state from the provided configuration.

        Sets up the policy engine, prompt context builder, sandbox
        backend, audit emitter, session manager, and operator state.
        Also discovers extensions and registers plugin hooks.

        Raises:
            ValueError: If plugin or skill dependency validation fails.
        """
        global_policy = self.config.policy or ToolPolicy()
        if self.config.operator is not None and self.config.operator.enabled:
            if self.config.operator.power_tools_enabled:
                global_policy.optional_tools.difference_update(HIGH_POWER_OPERATOR_TOOLS)
            else:
                global_policy.optional_tools.update(HIGH_POWER_OPERATOR_TOOLS)

        agent_policies = {
            agent_id: overrides.policy
            for agent_id, overrides in self.config.agent_overrides.items()
            if overrides.policy is not None
        }
        self.policy_engine = PolicyEngine(
            global_policy=global_policy,
            agent_policies=agent_policies,
        )
        self.prompt_context_builder = PromptContextBuilder(
            skill_registry=self.skill_registry,
            plugin_registry=self.plugin_registry,
            hook_runner=self.hook_runner,
            workspace_root=self.config.workspace_root,
        )
        self._sandbox_routers: dict[str, SandboxRouter] = {}
        if self.sandbox_backend is None and self.config.sandbox is not None and self.config.sandbox.backend_type:
            try:
                from ..security.sandbox_backends import get_backend

                self.sandbox_backend = get_backend(self.config.sandbox.backend_type, self.config.sandbox)
                logger.info(
                    "Sandbox backend '%s' instantiated (available=%s)",
                    self.config.sandbox.backend_type,
                    self.sandbox_backend.is_available(),
                )
            except Exception:
                logger.warning(
                    "Failed to instantiate sandbox backend '%s'",
                    self.config.sandbox.backend_type,
                    exc_info=True,
                )
        self.audit_emitter: AuditEmitter | None = (
            AuditEmitter(collector=self.config.audit_collector) if self.config.audit_collector is not None else None
        )
        self.session_manager: SessionManager | None = (
            SessionManager(store=self.config.session_store) if self.config.session_store is not None else None
        )
        if self.config.operator is not None and self.config.operator.enabled:
            self.operator_state = OperatorState(self.config.operator)
        self.discover_extensions()
        self._register_plugin_hooks()

    def discover_extensions(self) -> None:
        """Discover configured and conventional plugins/skills, then validate dependencies.

        Scans all configured plugin and skill directories (including
        conventional ``plugins/`` and ``skills/`` directories under the
        workspace root when ``discover_conventional_extensions`` is
        enabled). After discovery, validates that all plugin and skill
        dependency requirements are satisfied.

        Raises:
            ValueError: If any plugin or skill has unmet dependencies.
        """
        plugin_dirs = self._resolve_dirs(self.config.plugin_dirs, "plugins")
        skill_dirs = self._resolve_dirs(self.config.skill_dirs, "skills")

        for plugin_dir in plugin_dirs:
            self.plugin_registry.discover(plugin_dir)

        if skill_dirs:
            self.skill_registry.discover(*skill_dirs)

        plugin_dep_errors = self.plugin_registry.validate_dependencies()
        skill_dep_errors = self.skill_registry.validate_dependencies(plugin_registry=self.plugin_registry)
        errors = [f"Plugin dependency issue: {err}" for err in plugin_dep_errors]
        errors.extend(f"Skill dependency issue: {err}" for err in skill_dep_errors)
        if errors:
            raise ValueError("Runtime extension dependency validation failed:\n" + "\n".join(errors))

    def _resolve_dirs(self, configured_dirs: list[str], conventional_name: str) -> list[Path]:
        """Resolve configured directories plus optional conventional local paths.

        Expands user home (``~``) and resolves each configured directory
        to an absolute :class:`Path`. When
        ``discover_conventional_extensions`` is enabled, also appends
        the conventional ``<workspace_root>/<conventional_name>/``
        directory if it exists.

        Args:
            configured_dirs: List of raw directory path strings from
                the user configuration.
            conventional_name: Subdirectory name to look for under the
                workspace root (e.g. ``"plugins"`` or ``"skills"``).

        Returns:
            Ordered, deduplicated list of resolved :class:`Path`
            objects.
        """
        ordered: list[Path] = []
        seen: set[Path] = set()

        for raw_dir in configured_dirs:
            path = Path(raw_dir).expanduser().resolve()
            if path not in seen:
                ordered.append(path)
                seen.add(path)

        if self.config.discover_conventional_extensions:
            base_root = Path(self.config.workspace_root or os.getcwd()).resolve()
            conventional = (base_root / conventional_name).resolve()
            if conventional.is_dir() and conventional not in seen:
                ordered.append(conventional)
                seen.add(conventional)

        return ordered

    def _register_plugin_hooks(self) -> None:
        """Register all plugin-contributed hook callbacks with the hook runner.

        Iterates over every defined hook point and queries each
        discovered plugin for matching callbacks, registering them
        with the central :class:`HookRunner`.
        """
        for hook_name in HOOK_POINTS:
            for callback in self.plugin_registry.get_hooks(hook_name):
                self.hook_runner.register(hook_name, callback)

    def merge_plugin_tools(self, agent: Agent) -> None:
        """Attach plugin-contributed tools to an agent's function list.

        Iterates over all tools registered by discovered plugins and
        appends them to the agent's ``functions`` list. Raises an error
        if any plugin tool name collides with an existing function.

        Args:
            agent: The agent whose ``functions`` list will be extended
                with plugin tools.

        Raises:
            ValueError: If a plugin tool name conflicts with an
                existing function already registered on the agent.
        """
        if agent.functions is None:
            agent.functions = []

        existing_names = {get_callable_public_name(func) for func in agent.functions}
        for tool_name, func in self.plugin_registry.get_all_tools().items():
            if tool_name in existing_names:
                raise ValueError(
                    f"Plugin tool '{tool_name}' conflicts with an existing function on agent '{agent.id or agent.name or 'default'}'"
                )
            agent.functions.append(func)
            existing_names.add(tool_name)

    def merge_operator_tools(self, agent: Agent) -> None:
        """Attach runtime operator tools to an agent if enabled.

        If the operator state is active, builds the operator tool set
        and appends each allowed tool to the agent's ``functions`` list.
        Silently skips tools that are already present or not in the
        operator's allowed tool set.

        Args:
            agent: The agent whose ``functions`` list will be extended
                with operator tools.
        """
        if self.operator_state is None:
            return
        if agent.functions is None:
            agent.functions = []

        existing_names = {get_callable_public_name(func) for func in agent.functions}
        for func in self.operator_state.build_tools():
            tool_name = get_callable_public_name(func)
            if tool_name in existing_names:
                continue
            if tool_name not in self.operator_state.config.allowed_tool_names:
                continue
            agent.functions.append(func)
            existing_names.add(tool_name)

    def get_agent_overrides(self, agent_id: str | None) -> AgentRuntimeOverrides:
        """Return per-agent runtime overrides, or an empty default if not configured.

        Args:
            agent_id: The agent identifier to look up. When ``None`` or
                not found in the overrides map, returns a default
                :class:`AgentRuntimeOverrides` with all fields as
                ``None``.

        Returns:
            The :class:`AgentRuntimeOverrides` for the given agent, or
            a default instance.
        """
        if not agent_id:
            return AgentRuntimeOverrides()
        return self.config.agent_overrides.get(agent_id, AgentRuntimeOverrides())

    def get_guardrails(self, agent_id: str | None) -> list[str]:
        """Return the effective guardrail list for an agent.

        Checks for per-agent overrides first; falls back to the global
        guardrail list from the configuration.

        Args:
            agent_id: The agent identifier to look up overrides for.

        Returns:
            The list of guardrail rule strings applicable to this agent.
        """
        overrides = self.get_agent_overrides(agent_id)
        if overrides.guardrails is not None:
            return overrides.guardrails
        return self.config.guardrails

    def get_enabled_skill_names(self, agent_id: str | None) -> list[str]:
        """Return the list of enabled skill names for an agent.

        Checks for per-agent overrides first; falls back to the global
        enabled skills list from the configuration.

        Args:
            agent_id: The agent identifier to look up overrides for.

        Returns:
            The list of skill name strings enabled for this agent.
        """
        overrides = self.get_agent_overrides(agent_id)
        if overrides.enabled_skills is not None:
            return overrides.enabled_skills
        return self.config.enabled_skills

    def get_enabled_skills(self, agent_id: str | None) -> list[Skill]:
        """Return resolved Skill objects for all enabled skill names for an agent.

        Looks up each enabled skill name in the skill registry. Skills
        that were configured but not discovered are logged as warnings
        and omitted from the result.

        Args:
            agent_id: The agent identifier used to determine which
                skills are enabled.

        Returns:
            List of resolved :class:`Skill` objects that are both
            enabled and discovered.
        """
        skills: list[Skill] = []
        for skill_name in self.get_enabled_skill_names(agent_id):
            skill = self.skill_registry.get(skill_name)
            if skill is None:
                logger.warning("Configured runtime skill '%s' was not discovered", skill_name)
                continue
            skills.append(skill)
        return skills

    def get_loop_detection_config(self, agent_id: str | None) -> LoopDetectionConfig | None:
        """Return the effective loop detection config for an agent.

        Checks for per-agent overrides first; falls back to the global
        loop detection configuration.

        Args:
            agent_id: The agent identifier to look up overrides for.

        Returns:
            The :class:`LoopDetectionConfig` for this agent, or
            ``None`` if loop detection is not configured.
        """
        overrides = self.get_agent_overrides(agent_id)
        if overrides.loop_detection is not None:
            return overrides.loop_detection
        return self.config.loop_detection

    def create_loop_detector(self, agent_id: str | None) -> LoopDetector | None:
        """Create a LoopDetector for an agent.

        Resolves the effective loop detection config for the agent and
        instantiates a fresh :class:`LoopDetector`. Returns ``None``
        when loop detection is not configured.

        Args:
            agent_id: The agent identifier used to resolve the
                effective loop detection configuration.

        Returns:
            A new :class:`LoopDetector` instance, or ``None`` if no
            loop detection config is available.
        """
        config = self.get_loop_detection_config(agent_id)
        if config is None:
            return None
        return LoopDetector(config)

    def get_sandbox_config(self, agent_id: str | None) -> SandboxConfig | None:
        """Return the effective sandbox config for an agent.

        Checks for per-agent overrides first; falls back to the global
        sandbox configuration.

        Args:
            agent_id: The agent identifier to look up overrides for.

        Returns:
            The :class:`SandboxConfig` for this agent, or ``None`` if
            sandboxing is not configured.
        """
        overrides = self.get_agent_overrides(agent_id)
        if overrides.sandbox is not None:
            return overrides.sandbox
        return self.config.sandbox

    def get_sandbox_router(self, agent_id: str | None) -> SandboxRouter | None:
        """Return a cached-or-new SandboxRouter for an agent.

        Maintains an internal cache of :class:`SandboxRouter` instances
        keyed by agent ID. A cached router is reused when the config
        and backend match; otherwise a new router is created and cached.

        Args:
            agent_id: The agent identifier used to resolve sandbox
                config and cache the router.

        Returns:
            A :class:`SandboxRouter` instance for the agent, or
            ``None`` if sandbox is not configured.
        """
        config = self.get_sandbox_config(agent_id)
        if config is None:
            return None

        key = agent_id or "__default__"
        cached = self._sandbox_routers.get(key)
        if cached is not None and cached.config == config and cached.backend is self.sandbox_backend:
            return cached

        router = SandboxRouter(config=config, backend=self.sandbox_backend)
        self._sandbox_routers[key] = router
        return router

    def get_prompt_profile(self, agent_id: str | None) -> PromptProfile | PromptProfileConfig | str | None:
        """Resolve the prompt profile for an agent.

        Checks for a per-agent prompt profile override first; falls
        back to the global default prompt profile from the
        configuration.

        Args:
            agent_id: The agent identifier to look up overrides for.

        Returns:
            The prompt profile specification for this agent, or
            ``None`` if no profile is configured globally or per-agent.
        """
        overrides = self.get_agent_overrides(agent_id)
        if overrides.prompt_profile is not None:
            return overrides.prompt_profile
        return self.config.default_prompt_profile

    def build_prompt_prefix(
        self,
        agent_id: str | None,
        tool_names: list[str] | None = None,
        profile: PromptProfile | PromptProfileConfig | str | None = None,
    ) -> str:
        """Build the enriched system prompt prefix for a specific agent.

        Resolves all agent-specific overrides (sandbox, guardrails,
        enabled skills, prompt profile) and delegates to the
        :class:`PromptContextBuilder` to assemble the full prefix.

        Args:
            agent_id: The agent identifier used to resolve per-agent
                overrides.
            tool_names: Optional list of tool names available to the
                agent in this run.
            profile: Optional prompt profile override. When ``None``,
                the agent's resolved prompt profile is used.

        Returns:
            The assembled system prompt prefix string.
        """
        resolved_profile = profile or self.get_prompt_profile(agent_id)
        return self.prompt_context_builder.assemble_system_prompt_prefix(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=self.get_sandbox_config(agent_id),
            guardrails=self.get_guardrails(agent_id),
            enabled_skills=self.get_enabled_skills(agent_id),
            profile=resolved_profile,
        )
