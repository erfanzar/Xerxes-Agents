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


"""Tool policy enforcement layer for Xerxes.

Provides configurable allow/deny policies for tool execution at both
global and per-agent levels. Policies are evaluated before any tool
call is dispatched, blocking unauthorized calls with a clear error.

Design:
    - Global policy applies to all agents unless overridden.
    - Per-agent policy takes precedence over global for that agent.
    - An explicit allow-list means only those tools are permitted.
    - An explicit deny-list means all tools *except* those are permitted.
    - If both allow and deny are set, allow takes precedence (intersection).
    - Optional tools require explicit opt-in via the allow list.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Result of a policy evaluation.

    Represents the two possible outcomes when a tool invocation is checked
    against a :class:`ToolPolicy`.

    Attributes:
        ALLOW: The tool invocation is permitted by the policy.
        DENY: The tool invocation is blocked by the policy.

    Example:
        >>> action = PolicyAction.ALLOW
        >>> action.value
        'allow'
    """

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ToolPolicy:
    """A single allow/deny policy for tool invocation.

    Attributes:
        allow: Explicit set of tool names that are permitted.
            If non-empty, only these tools can be called.
        deny: Explicit set of tool names that are blocked.
            If non-empty, these tools cannot be called.
        optional_tools: Tools that exist but require explicit opt-in.
            They are denied unless they appear in ``allow``.
    """

    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    optional_tools: set[str] = field(default_factory=set)

    def evaluate(self, tool_name: str) -> PolicyAction:
        """Evaluate whether a given tool name is permitted by this policy.

        The evaluation follows a strict precedence order:

        1. If the ``allow`` set is non-empty, the tool must be present in it;
           otherwise it is denied.
        2. If the ``deny`` set is non-empty and the tool is in it, the tool
           is denied.
        3. If the tool is in ``optional_tools`` but not explicitly in
           ``allow``, the tool is denied.
        4. Otherwise, the tool is allowed.

        Args:
            tool_name: The name of the tool to evaluate against this policy.

        Returns:
            PolicyAction.ALLOW if the tool is permitted, PolicyAction.DENY
            if the tool is blocked.

        Example:
            >>> policy = ToolPolicy(deny={"execute_shell"})
            >>> policy.evaluate("execute_shell")
            <PolicyAction.DENY: 'deny'>
            >>> policy.evaluate("read_file")
            <PolicyAction.ALLOW: 'allow'>
        """
        if self.allow:
            return PolicyAction.ALLOW if tool_name in self.allow else PolicyAction.DENY
        if tool_name in self.deny:
            return PolicyAction.DENY
        if tool_name in self.optional_tools:
            return PolicyAction.DENY
        return PolicyAction.ALLOW


class PolicyEngine:
    """Evaluates tool policies at global and per-agent level.

    The engine holds a *global_policy* that applies to every agent and an
    optional dict of *agent_policies* keyed by agent ID.  Per-agent policies
    fully override the global policy for that agent (no merging).

    Listeners can be registered to observe every policy check, which is
    useful for audit logging or metrics collection.

    Attributes:
        global_policy: The default :class:`ToolPolicy` applied when no
            per-agent policy matches.
        agent_policies: Mapping of agent ID to :class:`ToolPolicy`. When an
            agent ID matches a key in this dict, that policy is used instead
            of the global policy.

    Example:
        >>> engine = PolicyEngine(
        ...     global_policy=ToolPolicy(deny={"execute_shell"}),
        ... )
        >>> engine.check("execute_shell", agent_id="coder")
        PolicyAction.DENY
        >>> engine.set_agent_policy("coder", ToolPolicy(allow={"execute_shell"}))
        >>> engine.check("execute_shell", agent_id="coder")
        PolicyAction.ALLOW
    """

    def __init__(
        self,
        global_policy: ToolPolicy | None = None,
        agent_policies: dict[str, ToolPolicy] | None = None,
    ) -> None:
        """Initialise the policy engine.

        Args:
            global_policy: The default policy applied to all agents that do
                not have a per-agent override. Defaults to an empty
                :class:`ToolPolicy` (allow all).
            agent_policies: Optional mapping of agent IDs to their specific
                policies. Per-agent policies fully replace (not merge with)
                the global policy for that agent.
        """
        self.global_policy = global_policy or ToolPolicy()
        self.agent_policies: dict[str, ToolPolicy] = agent_policies or {}
        self._listeners: list[tp.Callable[[str, str | None, PolicyAction], None]] = []

    def set_global_policy(self, policy: ToolPolicy) -> None:
        """Replace the global policy applied to all agents without a per-agent override.

        Args:
            policy: The new :class:`ToolPolicy` to use as the global default.
        """
        self.global_policy = policy

    def set_agent_policy(self, agent_id: str, policy: ToolPolicy) -> None:
        """Set or replace the per-agent policy for a specific agent.

        When set, this policy fully overrides the global policy for the
        given agent (no merging occurs).

        Args:
            agent_id: The unique identifier of the agent.
            policy: The :class:`ToolPolicy` to assign to this agent.
        """
        self.agent_policies[agent_id] = policy

    def remove_agent_policy(self, agent_id: str) -> None:
        """Remove a per-agent policy so the agent falls back to the global policy.

        If no per-agent policy exists for the given agent, this is a no-op.

        Args:
            agent_id: The unique identifier of the agent whose policy
                should be removed.
        """
        self.agent_policies.pop(agent_id, None)

    def add_listener(self, callback: tp.Callable[[str, str | None, PolicyAction], None]) -> None:
        """Register a listener that is notified on every policy check.

        Listeners are called synchronously after each policy evaluation.
        If a listener raises an exception, the error is logged as a warning
        and the remaining listeners are still invoked.

        Args:
            callback: A callable that receives ``(tool_name, agent_id, action)``
                where *tool_name* is the tool being checked, *agent_id* is
                the optional agent identifier, and *action* is the resulting
                :class:`PolicyAction`.
        """
        self._listeners.append(callback)

    def check(self, tool_name: str, agent_id: str | None = None) -> PolicyAction:
        """Check whether a tool invocation is allowed for a given agent.

        Resolves the applicable policy (per-agent if available, otherwise
        global), evaluates it, notifies all registered listeners, and logs
        denied actions at INFO level.

        Args:
            tool_name: The name of the tool to check.
            agent_id: Optional identifier of the agent requesting the tool.
                When ``None``, only the global policy is consulted.

        Returns:
            :attr:`PolicyAction.ALLOW` if the tool is permitted, or
            :attr:`PolicyAction.DENY` if it is blocked.
        """
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
        """Check a tool invocation and raise on denial.

        This is a convenience wrapper around :meth:`check` that raises a
        :class:`ToolPolicyViolation` when the policy decision is DENY,
        making it suitable for use in enforcement points where a blocked
        tool should halt execution.

        Args:
            tool_name: The name of the tool to check.
            agent_id: Optional identifier of the agent requesting the tool.

        Raises:
            ToolPolicyViolation: If the tool is denied by the applicable
                policy.
        """
        action = self.check(tool_name, agent_id)
        if action == PolicyAction.DENY:
            raise ToolPolicyViolation(tool_name, agent_id)


class ToolPolicyViolation(Exception):
    """Raised when a tool call is blocked by policy.

    Attributes:
        tool_name: The name of the tool that was denied.
        agent_id: The agent identifier that attempted the call, or ``None``
            if no agent context was provided.

    Example:
        >>> raise ToolPolicyViolation("execute_shell", agent_id="coder")
        Traceback (most recent call last):
            ...
        ToolPolicyViolation: Tool 'execute_shell' is denied by policy for agent 'coder'
    """

    def __init__(self, tool_name: str, agent_id: str | None = None) -> None:
        """Initialise the violation with context about the blocked call.

        Args:
            tool_name: The name of the tool that was denied.
            agent_id: Optional identifier of the agent that attempted the
                call.
        """
        self.tool_name = tool_name
        self.agent_id = agent_id
        agent_part = f" for agent '{agent_id}'" if agent_id else ""
        super().__init__(f"Tool '{tool_name}' is denied by policy{agent_part}")
