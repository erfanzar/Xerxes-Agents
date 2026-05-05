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
"""Features module for Xerxes.

Exports:
    - logger
    - AgentRuntimeOverrides
    - RuntimeFeaturesConfig
    - RuntimeFeaturesState"""

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
    """Agent runtime overrides.

    Attributes:
        policy (ToolPolicy | None): policy.
        loop_detection (LoopDetectionConfig | None): loop detection.
        sandbox (SandboxConfig | None): sandbox.
        enabled_skills (list[str] | None): enabled skills.
        guardrails (list[str] | None): guardrails.
        prompt_profile (PromptProfile | str | None): prompt profile."""

    policy: ToolPolicy | None = None
    loop_detection: LoopDetectionConfig | None = None
    sandbox: SandboxConfig | None = None
    enabled_skills: list[str] | None = None
    guardrails: list[str] | None = None
    prompt_profile: PromptProfile | str | None = None


@dataclass
class RuntimeFeaturesConfig:
    """Runtime features config.

    Attributes:
        enabled (bool): enabled.
        workspace_root (str | None): workspace root.
        plugin_dirs (list[str]): plugin dirs.
        skill_dirs (list[str]): skill dirs.
        discover_conventional_extensions (bool): discover conventional extensions.
        guardrails (list[str]): guardrails.
        policy (ToolPolicy | None): policy.
        loop_detection (LoopDetectionConfig | None): loop detection.
        sandbox (SandboxConfig | None): sandbox.
        enabled_skills (list[str]): enabled skills.
        default_prompt_profile (PromptProfile | str | None): default prompt profile.
        audit_collector (AuditCollector | None): audit collector.
        session_store (SessionStore | None): session store.
        operator (OperatorRuntimeConfig | None): operator.
        agent_overrides (dict[str, AgentRuntimeOverrides]): agent overrides."""

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
    """Runtime features state.

    Attributes:
        config (RuntimeFeaturesConfig): config.
        plugin_registry (PluginRegistry): plugin registry.
        skill_registry (SkillRegistry): skill registry.
        hook_runner (HookRunner): hook runner.
        sandbox_backend (SandboxBackend | None): sandbox backend.
        operator_state (OperatorState | None): operator state.
        authoring_pipeline (SkillAuthoringPipeline | None): authoring pipeline.
        authoring_telemetry (SkillTelemetry | None): authoring telemetry."""

    config: RuntimeFeaturesConfig
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    skill_registry: SkillRegistry = field(default_factory=SkillRegistry)
    hook_runner: HookRunner = field(default_factory=HookRunner)
    sandbox_backend: SandboxBackend | None = None
    operator_state: OperatorState | None = None
    authoring_pipeline: SkillAuthoringPipeline | None = None
    authoring_telemetry: SkillTelemetry | None = None

    def __post_init__(self) -> None:
        """Dunder method for post init.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Discover extensions.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Internal helper to resolve dirs.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            configured_dirs (list[str]): IN: configured dirs. OUT: Consumed during execution.
            conventional_name (str): IN: conventional name. OUT: Consumed during execution.
        Returns:
            list[Path]: OUT: Result of the operation."""

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
        """Internal helper to register plugin hooks.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        for hook_name in HOOK_POINTS:
            for callback in self.plugin_registry.get_hooks(hook_name):
                self.hook_runner.register(hook_name, callback)

    def _register_builtin_hooks(self) -> None:
        """Internal helper to register builtin hooks.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.audit_emitter is not None:
            self.hook_runner.register("on_loop_warning", self._on_loop_warning_hook)

    def _on_turn_start_hook(self, agent_id=None, messages=None) -> None:
        """Internal helper to on turn start hook.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (Any, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            messages (Any, optional): IN: messages. Defaults to None. OUT: Consumed during execution."""

        if self.authoring_pipeline is not None:
            prompt = ""
            if messages and hasattr(messages[-1], "content"):
                prompt = messages[-1].content
            self.authoring_pipeline.begin_turn(agent_id=agent_id, user_prompt=prompt)

    def _before_tool_call_hook(self, tool_name=None, arguments=None, agent_id=None) -> dict | None:
        """Internal helper to before tool call hook.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (Any, optional): IN: tool name. Defaults to None. OUT: Consumed during execution.
            arguments (Any, optional): IN: arguments. Defaults to None. OUT: Consumed during execution.
            agent_id (Any, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict | None: OUT: Result of the operation."""

        if self.authoring_pipeline is not None:
            self.authoring_pipeline.record_call(tool_name, dict(arguments or {}))
        return None

    def _on_turn_end_hook(self, agent_id=None, response=None) -> None:
        """Internal helper to on turn end hook.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (Any, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            response (Any, optional): IN: response. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to on loop warning hook.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str, optional): IN: tool name. Defaults to ''. OUT: Consumed during execution.
            pattern (str, optional): IN: pattern. Defaults to ''. OUT: Consumed during execution.
            severity (str, optional): IN: severity. Defaults to ''. OUT: Consumed during execution.
            count (int, optional): IN: count. Defaults to 0. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            turn_id (str | None, optional): IN: turn id. Defaults to None. OUT: Consumed during execution."""

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
        """Merge plugin tools.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent (Agent): IN: agent. OUT: Consumed during execution."""

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
        """Merge operator tools.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent (Agent): IN: agent. OUT: Consumed during execution."""

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
        """Retrieve the agent overrides.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            AgentRuntimeOverrides: OUT: Result of the operation."""

        if not agent_id:
            return AgentRuntimeOverrides()
        return self.config.agent_overrides.get(agent_id, AgentRuntimeOverrides())

    def get_guardrails(self, agent_id: str | None) -> list[str]:
        """Retrieve the guardrails.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        overrides = self.get_agent_overrides(agent_id)
        if overrides.guardrails is not None:
            return overrides.guardrails
        return self.config.guardrails

    def get_enabled_skill_names(self, agent_id: str | None) -> list[str]:
        """Retrieve the enabled skill names.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        overrides = self.get_agent_overrides(agent_id)
        if overrides.enabled_skills is not None:
            return overrides.enabled_skills
        return self.config.enabled_skills

    def get_enabled_skills(self, agent_id: str | None) -> list[Skill]:
        """Retrieve the enabled skills.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            list[Skill]: OUT: Result of the operation."""

        skills: list[Skill] = []
        for skill_name in self.get_enabled_skill_names(agent_id):
            skill = self.skill_registry.get(skill_name)
            if skill is None:
                logger.warning("Configured runtime skill '%s' was not discovered", skill_name)
                continue
            skills.append(skill)
        return skills

    def get_loop_detection_config(self, agent_id: str | None) -> LoopDetectionConfig | None:
        """Retrieve the loop detection config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            LoopDetectionConfig | None: OUT: Result of the operation."""

        overrides = self.get_agent_overrides(agent_id)
        if overrides.loop_detection is not None:
            return overrides.loop_detection
        return self.config.loop_detection

    def create_loop_detector(self, agent_id: str | None) -> LoopDetector | None:
        """Create loop detector.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            LoopDetector | None: OUT: Result of the operation."""

        config = self.get_loop_detection_config(agent_id)
        if config is None:
            return None
        return LoopDetector(config)

    def get_sandbox_config(self, agent_id: str | None) -> SandboxConfig | None:
        """Retrieve the sandbox config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            SandboxConfig | None: OUT: Result of the operation."""

        overrides = self.get_agent_overrides(agent_id)
        if overrides.sandbox is not None:
            return overrides.sandbox
        return self.config.sandbox

    def get_sandbox_router(self, agent_id: str | None) -> SandboxRouter | None:
        """Retrieve the sandbox router.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            SandboxRouter | None: OUT: Result of the operation."""

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
        """Retrieve the prompt profile.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
        Returns:
            PromptProfile | PromptProfileConfig | str | None: OUT: Result of the operation."""

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
        """Build prompt prefix.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            tool_names (list[str] | None, optional): IN: tool names. Defaults to None. OUT: Consumed during execution.
            profile (PromptProfile | PromptProfileConfig | str | None, optional): IN: profile. Defaults to None. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        resolved_profile = profile or self.get_prompt_profile(agent_id)
        return self.prompt_context_builder.assemble_system_prompt_prefix(
            agent_id=agent_id,
            tool_names=tool_names,
            sandbox_config=self.get_sandbox_config(agent_id),
            guardrails=self.get_guardrails(agent_id),
            enabled_skills=self.get_enabled_skills(agent_id),
            profile=resolved_profile,
        )
