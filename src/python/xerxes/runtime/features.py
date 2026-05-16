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
"""Runtime feature wiring for plugins, skills, sandboxing, audit, and policy.

This module owns the heavyweight :class:`RuntimeFeaturesState` that the
streaming loop reaches into during a turn. It assembles plugins, skills, the
hook runner, sandbox routers, audit emitter, session manager, operator
state, and skill-authoring pipeline, then exposes per-agent overrides
through :class:`AgentRuntimeOverrides`. Construction is intentionally
heavy-handed (one-time on session start) so per-turn lookups stay cheap.
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
from ..extensions.skill_authoring.pipeline import SkillAuthoringPipeline
from ..extensions.skill_authoring.telemetry import SkillTelemetry
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
    """Per-agent overrides applied on top of :class:`RuntimeFeaturesConfig`.

    Attributes:
        policy: Custom :class:`ToolPolicy` for this agent; ``None`` inherits.
        loop_detection: Custom :class:`LoopDetectionConfig`, or ``None``.
        sandbox: Custom :class:`SandboxConfig`, or ``None``.
        enabled_skills: Explicit list of enabled skill names; ``None``
            inherits the runtime-wide list.
        guardrails: Explicit list of guardrail identifiers; ``None``
            inherits the runtime-wide list.
        prompt_profile: Override prompt profile (instance, config, or name).
    """

    policy: ToolPolicy | None = None
    loop_detection: LoopDetectionConfig | None = None
    sandbox: SandboxConfig | None = None
    enabled_skills: list[str] | None = None
    guardrails: list[str] | None = None
    prompt_profile: PromptProfile | str | None = None


@dataclass
class RuntimeFeaturesConfig:
    """Declarative configuration of the runtime feature surface.

    Attributes:
        enabled: Master switch consulted by the streaming loop.
        workspace_root: Project root used for conventional extension discovery.
        plugin_dirs: Explicit plugin directories to discover.
        skill_dirs: Explicit skill directories to discover.
        discover_conventional_extensions: When ``True`` also probe
            ``<workspace_root>/plugins`` and ``<workspace_root>/skills``.
        guardrails: Default list of guardrail identifiers (per-agent
            overridable).
        policy: Default :class:`ToolPolicy`; merged with operator power-tool
            policy when applicable.
        loop_detection: Default loop-detection config (per-agent overridable).
        sandbox: Default :class:`SandboxConfig` (per-agent overridable).
        enabled_skills: Default list of enabled skill names.
        default_prompt_profile: Default :class:`PromptProfile` or its name.
        audit_collector: Optional :class:`AuditCollector` for audit events.
        session_store: Optional :class:`SessionStore` for session persistence.
        operator: Optional :class:`OperatorRuntimeConfig` enabling PTY/browser
            operators.
        agent_overrides: Per-agent :class:`AgentRuntimeOverrides`.
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
    """Fully constructed feature-state object consumed by the streaming loop.

    Attributes:
        config: The originating :class:`RuntimeFeaturesConfig`.
        plugin_registry: Registry of discovered plugins.
        skill_registry: Registry of discovered skills.
        hook_runner: Hook dispatcher used at lifecycle events.
        sandbox_backend: Instantiated sandbox backend, or ``None``.
        operator_state: Live :class:`OperatorState` when operators are
            enabled.
        authoring_pipeline: Skill-authoring pipeline, when configured.
        authoring_telemetry: Telemetry sink paired with the authoring
            pipeline.
    """

    config: RuntimeFeaturesConfig
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    skill_registry: SkillRegistry = field(default_factory=SkillRegistry)
    hook_runner: HookRunner = field(default_factory=HookRunner)
    sandbox_backend: SandboxBackend | None = None
    operator_state: OperatorState | None = None
    authoring_pipeline: SkillAuthoringPipeline | None = None
    authoring_telemetry: SkillTelemetry | None = None

    def __post_init__(self) -> None:
        """Materialise policies, prompt builder, sandbox, audit, sessions, and operators.

        Side-effects: discovers plugins/skills, registers their hooks, and
        kicks off skill-authoring telemetry when ``skill_dirs`` is set.
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
            AuditEmitter(collector=self.config.audit_collector, hook_runner=self.hook_runner)
            if self.config.audit_collector is not None
            else None
        )
        self.session_manager: SessionManager | None = (
            SessionManager(store=self.config.session_store) if self.config.session_store is not None else None
        )
        if self.config.operator is not None and self.config.operator.enabled:
            self.operator_state = OperatorState(self.config.operator)
        self.discover_extensions()
        self._register_plugin_hooks()
        self._register_builtin_hooks()

        if self.config.skill_dirs:
            from ..extensions.skill_authoring import SkillAuthoringPipeline, SkillTelemetry

            self.authoring_telemetry = SkillTelemetry()
            skills_dir = self.config.skill_dirs[0] if self.config.skill_dirs else str(Path.home() / ".xerxes" / "skills")
            self.authoring_pipeline = SkillAuthoringPipeline(
                skills_dir=skills_dir,
                skill_registry=self.skill_registry,
                telemetry=self.authoring_telemetry,
                audit_emitter=self.audit_emitter,
            )

            self.hook_runner.register("on_turn_start", self._on_turn_start_hook)
            self.hook_runner.register("before_tool_call", self._before_tool_call_hook)
            self.hook_runner.register("on_turn_end", self._on_turn_end_hook)

    def discover_extensions(self) -> None:
        """Walk configured plus conventional plugin/skill directories.

        Discovered modules are registered into ``plugin_registry`` and
        ``skill_registry``; dependency validation runs against the union and
        raises :class:`ValueError` on any unresolved declaration.
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
        """Expand ``configured_dirs`` and optionally append a conventional dir.

        Returns absolute :class:`Path` objects in stable order with duplicates
        removed. When ``discover_conventional_extensions`` is set, appends
        ``<workspace_root>/<conventional_name>`` if it exists.
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
        """Register every plugin-declared hook against the hook runner."""

        for hook_name in HOOK_POINTS:
            for callback in self.plugin_registry.get_hooks(hook_name):
                self.hook_runner.register(hook_name, callback)

    def _register_builtin_hooks(self) -> None:
        """Wire built-in hooks (currently: audit loop-warning emission)."""

        if self.audit_emitter is not None:
            self.hook_runner.register("on_loop_warning", self._on_loop_warning_hook)

    def _on_turn_start_hook(self, agent_id=None, messages=None) -> None:
        """Hand the turn's opening user prompt to the skill-authoring pipeline."""

        if self.authoring_pipeline is not None:
            prompt = ""
            if messages and hasattr(messages[-1], "content"):
                prompt = messages[-1].content
            self.authoring_pipeline.begin_turn(agent_id=agent_id, user_prompt=prompt)

    def _before_tool_call_hook(self, tool_name=None, arguments=None, agent_id=None) -> dict | None:
        """Forward each tool invocation to the authoring pipeline for telemetry."""

        if self.authoring_pipeline is not None:
            self.authoring_pipeline.record_call(tool_name, dict(arguments or {}))
        return None

    def _on_turn_end_hook(self, agent_id=None, response=None) -> None:
        """Notify the authoring pipeline that a turn finished and pass the response."""

        if self.authoring_pipeline is not None:
            self.authoring_pipeline.on_turn_end(final_response=str(response or ""))

    def _on_loop_warning_hook(
        self,
        tool_name: str = "",
        pattern: str = "",
        severity: str = "",
        count: int = 0,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a tool-loop-warning audit event when loop detection trips."""

        if self.audit_emitter is not None:
            self.audit_emitter.emit_tool_loop_warning(
                tool_name=tool_name,
                pattern=pattern,
                severity=severity,
                count=count,
                agent_id=agent_id,
                turn_id=turn_id,
            )

    def merge_plugin_tools(self, agent: Agent) -> None:
        """Append every discovered plugin tool onto ``agent.functions``.

        Raises:
            ValueError: A plugin tool name collides with one already declared
                on the agent.
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
        """Append allowed operator tools to ``agent.functions``.

        No-op when operators are disabled or :attr:`operator_state` is unset.
        Tools already on the agent are skipped silently to make repeated
        merges idempotent.
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
        """Return the override set for ``agent_id`` (empty default when unset)."""
        if not agent_id:
            return AgentRuntimeOverrides()
        return self.config.agent_overrides.get(agent_id, AgentRuntimeOverrides())

    def get_guardrails(self, agent_id: str | None) -> list[str]:
        """Return effective guardrails for ``agent_id``, preferring overrides."""
        overrides = self.get_agent_overrides(agent_id)
        if overrides.guardrails is not None:
            return overrides.guardrails
        return self.config.guardrails

    def get_enabled_skill_names(self, agent_id: str | None) -> list[str]:
        """Return the enabled skill name list for ``agent_id``."""
        overrides = self.get_agent_overrides(agent_id)
        if overrides.enabled_skills is not None:
            return overrides.enabled_skills
        return self.config.enabled_skills

    def get_enabled_skills(self, agent_id: str | None) -> list[Skill]:
        """Resolve :meth:`get_enabled_skill_names` to live :class:`Skill` objects.

        Skill names that fail to resolve in the registry are skipped with a
        WARNING log rather than raising.
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
        """Return the loop-detection config for ``agent_id`` or its inherited default."""
        overrides = self.get_agent_overrides(agent_id)
        if overrides.loop_detection is not None:
            return overrides.loop_detection
        return self.config.loop_detection

    def create_loop_detector(self, agent_id: str | None) -> LoopDetector | None:
        """Instantiate a fresh :class:`LoopDetector` for ``agent_id``, or ``None``."""
        config = self.get_loop_detection_config(agent_id)
        if config is None:
            return None
        return LoopDetector(config)

    def get_sandbox_config(self, agent_id: str | None) -> SandboxConfig | None:
        """Return the sandbox config for ``agent_id`` or its inherited default."""
        overrides = self.get_agent_overrides(agent_id)
        if overrides.sandbox is not None:
            return overrides.sandbox
        return self.config.sandbox

    def get_sandbox_router(self, agent_id: str | None) -> SandboxRouter | None:
        """Return a (cached) :class:`SandboxRouter` for ``agent_id``.

        Routers are memoised by agent id; the cache is invalidated when the
        underlying config or backend changes between calls.
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
        """Return the prompt profile for ``agent_id`` or the configured default."""
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
        """Render the runtime-aware system-prompt prefix for ``agent_id``.

        Combines guardrails, enabled skills, sandbox metadata, and the agent's
        prompt profile through :class:`PromptContextBuilder`. ``profile``
        overrides the per-agent profile when supplied.
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
