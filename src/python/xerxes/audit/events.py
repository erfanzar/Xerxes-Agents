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
"""Audit event dataclasses.

This module defines the structured event types used throughout the audit
subsystem, each inheriting from :class:`AuditEvent`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string.

    Returns:
        str: OUT: Current UTC time in ISO format.
    """
    return datetime.now(UTC).isoformat()


@dataclass
class AuditEvent:
    """Base class for all audit events.

    Attributes:
        event_type (str): Event type discriminator.
        timestamp (str): ISO-8601 timestamp.
        agent_id (str | None): Associated agent identifier.
        turn_id (str | None): Associated turn identifier.
        session_id (str | None): Session identifier (injected by emitter).
        severity (str): Severity level (e.g., ``"info"``, ``"error"``).
        metadata (dict[str, Any]): Arbitrary additional metadata.
    """

    event_type: str = "base"
    timestamp: str = field(default_factory=_now_iso)
    agent_id: str | None = None
    turn_id: str | None = None
    session_id: str | None = None
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a dictionary.

        Returns:
            dict[str, Any]: OUT: Flat dictionary of all fields.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the event to a JSON string.

        Returns:
            str: OUT: JSON representation of the event.
        """
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TurnStartEvent(AuditEvent):
    """Emitted when an agent turn begins.

    Attributes:
        prompt_preview (str): Truncated preview of the user prompt.
    """

    event_type: str = field(default="turn_start", init=False)
    prompt_preview: str = ""


@dataclass
class TurnEndEvent(AuditEvent):
    """Emitted when an agent turn ends.

    Attributes:
        content_preview (str): Truncated preview of the assistant response.
        function_calls_count (int): Number of function calls in the turn.
    """

    event_type: str = field(default="turn_end", init=False)
    content_preview: str = ""
    function_calls_count: int = 0


@dataclass
class ToolCallAttemptEvent(AuditEvent):
    """Emitted when a tool call is attempted.

    Attributes:
        tool_name (str): Name of the tool.
        arguments_preview (str): Truncated preview of the arguments.
    """

    event_type: str = field(default="tool_call_attempt", init=False)
    tool_name: str = ""
    arguments_preview: str = ""


@dataclass
class ToolCallCompleteEvent(AuditEvent):
    """Emitted when a tool call completes successfully.

    Attributes:
        tool_name (str): Name of the tool.
        status (str): Completion status.
        duration_ms (float): Execution duration in milliseconds.
        result_preview (str): Truncated preview of the result.
    """

    event_type: str = field(default="tool_call_complete", init=False)
    tool_name: str = ""
    status: str = "success"
    duration_ms: float = 0.0
    result_preview: str = ""


@dataclass
class ToolCallFailureEvent(AuditEvent):
    """Emitted when a tool call fails.

    Attributes:
        severity (str): Always ``"error"``.
        tool_name (str): Name of the tool.
        error_type (str): Error classification.
        error_message (str): Error message.
    """

    event_type: str = field(default="tool_call_failure", init=False)
    severity: str = "error"
    tool_name: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class ToolPolicyDecisionEvent(AuditEvent):
    """Emitted when a policy decision is made about a tool.

    Attributes:
        tool_name (str): Name of the tool.
        action (str): Decision action.
        policy_source (str): Source of the policy.
    """

    event_type: str = field(default="tool_policy_decision", init=False)
    tool_name: str = ""
    action: str = ""
    policy_source: str = ""


@dataclass
class SandboxDecisionEvent(AuditEvent):
    """Emitted when a sandbox execution decision is made.

    Attributes:
        tool_name (str): Name of the tool.
        context (str): Execution context.
        reason (str): Decision rationale.
        backend_type (str): Sandbox backend type.
    """

    event_type: str = field(default="sandbox_decision", init=False)
    tool_name: str = ""
    context: str = ""
    reason: str = ""
    backend_type: str = ""


@dataclass
class ToolLoopWarningEvent(AuditEvent):
    """Emitted when a potential tool loop is detected.

    Attributes:
        severity (str): Always ``"warning"``.
        tool_name (str): Name of the tool.
        pattern (str): Detected loop pattern.
        severity_level (str): Severity level string.
        call_count (int): Number of repeated calls.
    """

    event_type: str = field(default="tool_loop_warning", init=False)
    severity: str = "warning"
    tool_name: str = ""
    pattern: str = ""
    severity_level: str = ""
    call_count: int = 0


@dataclass
class ToolLoopBlockEvent(AuditEvent):
    """Emitted when a tool loop is blocked.

    Attributes:
        severity (str): Always ``"error"``.
        tool_name (str): Name of the tool.
        pattern (str): Detected loop pattern.
        call_count (int): Number of repeated calls.
    """

    event_type: str = field(default="tool_loop_block", init=False)
    severity: str = "error"
    tool_name: str = ""
    pattern: str = ""
    call_count: int = 0


@dataclass
class HookMutationEvent(AuditEvent):
    """Emitted when a hook mutates a field.

    Attributes:
        hook_name (str): Name of the hook.
        tool_name (str): Related tool name.
        mutated_field (str): Name of the mutated field.
    """

    event_type: str = field(default="hook_mutation", init=False)
    hook_name: str = ""
    tool_name: str = ""
    mutated_field: str = ""


@dataclass
class ErrorEvent(AuditEvent):
    """Emitted for generic errors.

    Attributes:
        severity (str): Always ``"error"``.
        error_type (str): Error classification.
        error_message (str): Error message.
        error_context (str): Additional error context.
    """

    event_type: str = field(default="error", init=False)
    severity: str = "error"
    error_type: str = ""
    error_message: str = ""
    error_context: str = ""


@dataclass
class SkillUsedEvent(AuditEvent):
    """Emitted when a skill is invoked.

    Attributes:
        skill_name (str): Name of the skill.
        version (str): Skill version.
        outcome (str): Execution outcome.
        duration_ms (float): Execution duration.
        triggered_automatically (bool): Whether triggered automatically.
    """

    event_type: str = field(default="skill_used", init=False)
    skill_name: str = ""
    version: str = ""
    outcome: str = "unknown"
    duration_ms: float = 0.0
    triggered_automatically: bool = True


@dataclass
class SkillAuthoredEvent(AuditEvent):
    """Emitted when a skill is authored or updated.

    Attributes:
        skill_name (str): Name of the skill.
        version (str): Skill version.
        source_path (str): Path to the skill source.
        tool_count (int): Number of tools in the skill.
        unique_tools (list[str]): Unique tool names.
        confirmed_by_user (bool): Whether user confirmed the skill.
    """

    event_type: str = field(default="skill_authored", init=False)
    skill_name: str = ""
    version: str = ""
    source_path: str = ""
    tool_count: int = 0
    unique_tools: list[str] = field(default_factory=list)
    confirmed_by_user: bool = False


@dataclass
class SkillFeedbackEvent(AuditEvent):
    """Emitted when feedback is given on a skill.

    Attributes:
        skill_name (str): Name of the skill.
        rating (str): Feedback rating.
        reason (str): Feedback reason.
        source (str): Feedback source.
    """

    event_type: str = field(default="skill_feedback", init=False)
    skill_name: str = ""
    rating: str = "neutral"
    reason: str = ""
    source: str = "user"


@dataclass
class AgentSwitchEvent(AuditEvent):
    """Emitted when the active agent switches during execution.

    Attributes:
        from_agent (str): Previous agent name.
        to_agent (str): New agent name.
        reason (str): Switch reason.
    """

    event_type: str = field(default="agent_switch", init=False)
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""
