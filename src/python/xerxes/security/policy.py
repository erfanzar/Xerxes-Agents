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
"""Policy module for Xerxes.

Exports:
    - logger
    - PolicyAction
    - ToolPolicy
    - PolicyEngine
    - ToolPolicyViolation"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Policy action.

    Inherits from: Enum
    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ToolPolicy:
    """Tool policy.

    Attributes:
        allow (set[str]): allow.
        deny (set[str]): deny.
        optional_tools (set[str]): optional tools."""

    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    optional_tools: set[str] = field(default_factory=set)

    def evaluate(self, tool_name: str) -> PolicyAction:
        """Evaluate.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
        Returns:
            PolicyAction: OUT: Result of the operation."""

        if self.allow:
            return PolicyAction.ALLOW if tool_name in self.allow else PolicyAction.DENY
        if tool_name in self.deny:
            return PolicyAction.DENY
        if tool_name in self.optional_tools:
            return PolicyAction.DENY
        return PolicyAction.ALLOW


class PolicyEngine:
    """Policy engine."""

    def __init__(
        self,
        global_policy: ToolPolicy | None = None,
        agent_policies: dict[str, ToolPolicy] | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            global_policy (ToolPolicy | None, optional): IN: global policy. Defaults to None. OUT: Consumed during execution.
            agent_policies (dict[str, ToolPolicy] | None, optional): IN: agent policies. Defaults to None. OUT: Consumed during execution."""

        self.global_policy = global_policy or ToolPolicy()
        self.agent_policies: dict[str, ToolPolicy] = agent_policies or {}
        self._listeners: list[tp.Callable[[str, str | None, PolicyAction], None]] = []

    def set_global_policy(self, policy: ToolPolicy) -> None:
        """Set the global policy.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            policy (ToolPolicy): IN: policy. OUT: Consumed during execution."""

        self.global_policy = policy

    def set_agent_policy(self, agent_id: str, policy: ToolPolicy) -> None:
        """Set the agent policy.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
            policy (ToolPolicy): IN: policy. OUT: Consumed during execution."""

        self.agent_policies[agent_id] = policy

    def remove_agent_policy(self, agent_id: str) -> None:
        """Remove agent policy.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str): IN: agent id. OUT: Consumed during execution."""

        self.agent_policies.pop(agent_id, None)

    def add_listener(self, callback: tp.Callable[[str, str | None, PolicyAction], None]) -> None:
        """Add listener.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            callback (tp.Callable[[str, str | None, PolicyAction], None]): IN: callback. OUT: Consumed during execution."""

        self._listeners.append(callback)

    def check(self, tool_name: str, agent_id: str | None = None) -> PolicyAction:
        """Check.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        Returns:
            PolicyAction: OUT: Result of the operation."""

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
        """Enforce.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution."""

        action = self.check(tool_name, agent_id)
        if action == PolicyAction.DENY:
            raise ToolPolicyViolation(tool_name, agent_id)


class ToolPolicyViolation(Exception):
    """Tool policy violation.

    Inherits from: Exception
    """

    def __init__(self, tool_name: str, agent_id: str | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution."""

        self.tool_name = tool_name
        self.agent_id = agent_id
        agent_part = f" for agent '{agent_id}'" if agent_id else ""
        super().__init__(f"Tool '{tool_name}' is denied by policy{agent_part}")
