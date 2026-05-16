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
"""Tool allow/deny policy enforcement.

A ``PolicyEngine`` holds one global ``ToolPolicy`` plus optional
per-agent overrides. Each tool invocation is evaluated against the
appropriate policy and either allowed or denied; denials raise
``ToolPolicyViolation`` and are observable via listener callbacks
(used by the audit trail and TUI status line). This is the coarse
"is this tool reachable at all" gate that sits in front of the
permission/sandbox stack."""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Outcome of a policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ToolPolicy:
    """Per-scope tool admission rules.

    The lists interact as follows. When ``allow`` is non-empty it acts as
    an exclusive allow-list (anything not present is denied). Otherwise
    ``deny`` and ``optional_tools`` veto specific names; ``optional_tools``
    are tools that exist but are off by default and must be explicitly
    enabled by the operator.

    Attributes:
        allow: explicit allow-list; empty means "no allow-list active".
        deny: names that are always denied.
        optional_tools: tools that exist but are gated off by default.
    """

    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    optional_tools: set[str] = field(default_factory=set)

    def evaluate(self, tool_name: str) -> PolicyAction:
        """Return ``ALLOW`` or ``DENY`` for ``tool_name`` under this policy."""

        if self.allow:
            return PolicyAction.ALLOW if tool_name in self.allow else PolicyAction.DENY
        if tool_name in self.deny:
            return PolicyAction.DENY
        if tool_name in self.optional_tools:
            return PolicyAction.DENY
        return PolicyAction.ALLOW


class PolicyEngine:
    """Evaluate tool calls against a layered policy.

    Holds one ``global_policy`` and optional per-agent overrides keyed by
    ``agent_id``. When an ``agent_id`` is supplied to :meth:`check` and a
    matching agent policy exists, it shadows the global one entirely
    (there is no merge); otherwise the global policy applies. Listener
    callbacks are notified on every decision and are typically used for
    audit logging."""

    def __init__(
        self,
        global_policy: ToolPolicy | None = None,
        agent_policies: dict[str, ToolPolicy] | None = None,
    ) -> None:
        """Initialise the engine with a global policy and optional agent overrides."""

        self.global_policy = global_policy or ToolPolicy()
        self.agent_policies: dict[str, ToolPolicy] = agent_policies or {}
        self._listeners: list[tp.Callable[[str, str | None, PolicyAction], None]] = []

    def set_global_policy(self, policy: ToolPolicy) -> None:
        """Replace the global policy applied when no agent-specific one exists."""

        self.global_policy = policy

    def set_agent_policy(self, agent_id: str, policy: ToolPolicy) -> None:
        """Install or replace the policy override for ``agent_id``."""

        self.agent_policies[agent_id] = policy

    def remove_agent_policy(self, agent_id: str) -> None:
        """Drop the policy override for ``agent_id`` (no-op if none)."""

        self.agent_policies.pop(agent_id, None)

    def add_listener(self, callback: tp.Callable[[str, str | None, PolicyAction], None]) -> None:
        """Register a ``(tool_name, agent_id, action)`` listener for decisions."""

        self._listeners.append(callback)

    def check(self, tool_name: str, agent_id: str | None = None) -> PolicyAction:
        """Return ``ALLOW``/``DENY`` for ``tool_name`` and notify listeners.

        Picks the agent-specific policy if one exists for ``agent_id``,
        otherwise falls back to the global policy. Listener exceptions are
        swallowed and logged; they never block enforcement."""

        policy = self.agent_policies.get(agent_id) if agent_id else None
        if policy is None:
            policy = self.global_policy

        action = policy.evaluate(tool_name)

        for listener in self._listeners:
            try:
                listener(tool_name, agent_id, action)
            except Exception:
                logger.warning("Policy listener error", exc_info=True)

        if action == PolicyAction.DENY:
            logger.info("Policy DENIED tool=%s agent=%s", tool_name, agent_id)
        return action

    def enforce(self, tool_name: str, agent_id: str | None = None) -> None:
        """Run :meth:`check`; raise :class:`ToolPolicyViolation` on DENY."""

        action = self.check(tool_name, agent_id)
        if action == PolicyAction.DENY:
            raise ToolPolicyViolation(tool_name, agent_id)


class ToolPolicyViolation(Exception):
    """Raised when a tool call is rejected by the configured policy."""

    def __init__(self, tool_name: str, agent_id: str | None = None) -> None:
        """Capture which tool was denied and (optionally) for which agent."""

        self.tool_name = tool_name
        self.agent_id = agent_id
        agent_part = f" for agent '{agent_id}'" if agent_id else ""
        super().__init__(f"Tool '{tool_name}' is denied by policy{agent_part}")
