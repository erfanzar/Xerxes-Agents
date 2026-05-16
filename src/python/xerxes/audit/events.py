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
"""Structured event dataclasses for the audit subsystem.

Every event inherits from :class:`AuditEvent`, which carries the
common fields (event type discriminator, UTC timestamp, agent/turn/
session id, severity, free-form metadata). Each subclass adds the
fields specific to that event type and pins ``event_type`` to a stable
string used by downstream consumers (JSONL sinks, OTel spans).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass
class AuditEvent:
    """Base class shared by every audit event.

    Attributes:
        event_type: discriminator overridden by each subclass.
        timestamp: UTC ISO-8601 instant the event was constructed.
        agent_id: the agent that produced the event, if known.
        turn_id: the turn the event belongs to, if any.
        session_id: injected by :class:`AuditEmitter`.
        severity: ``"info"``, ``"warning"``, ``"error"``, ...
        metadata: free-form annotations.
    """

    event_type: str = "base"
    timestamp: str = field(default_factory=_now_iso)
    agent_id: str | None = None
    turn_id: str | None = None
    session_id: str | None = None
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the event as a JSON string (``str`` fallback for unknowns)."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TurnStartEvent(AuditEvent):
    """Emitted at the start of an agent turn.

    Attributes:
        prompt_preview: leading slice of the user prompt (up to 200 chars).
    """

    event_type: str = field(default="turn_start", init=False)
    prompt_preview: str = ""


@dataclass
class TurnEndEvent(AuditEvent):
    """Emitted when a turn finishes (success or model stop).

    Attributes:
        content_preview: leading slice of the assistant response.
        function_calls_count: tools the model invoked during the turn.
    """

    event_type: str = field(default="turn_end", init=False)
    content_preview: str = ""
    function_calls_count: int = 0


@dataclass
class ToolCallAttemptEvent(AuditEvent):
    """Emitted before a tool call runs (post-policy, pre-execution).

    Attributes:
        tool_name: registered tool name.
        arguments_preview: leading slice of the JSON-encoded arguments.
    """

    event_type: str = field(default="tool_call_attempt", init=False)
    tool_name: str = ""
    arguments_preview: str = ""


@dataclass
class ToolCallCompleteEvent(AuditEvent):
    """Emitted when a tool call returns without raising.

    Attributes:
        tool_name: registered tool name.
        status: outcome string (``"success"`` by default).
        duration_ms: wall-clock execution time in milliseconds.
        result_preview: leading slice of the result text.
    """

    event_type: str = field(default="tool_call_complete", init=False)
    tool_name: str = ""
    status: str = "success"
    duration_ms: float = 0.0
    result_preview: str = ""


@dataclass
class ToolCallFailureEvent(AuditEvent):
    """Emitted when a tool call raises or returns a failure status.

    Attributes:
        severity: always ``"error"``.
        tool_name: registered tool name.
        error_type: short classification (e.g. ``"TimeoutError"``).
        error_message: human-readable message from the raised exception.
    """

    event_type: str = field(default="tool_call_failure", init=False)
    severity: str = "error"
    tool_name: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class ToolPolicyDecisionEvent(AuditEvent):
    """Emitted when permission/policy resolves a tool call.

    Attributes:
        tool_name: registered tool name.
        action: ``"allow"``, ``"deny"``, ``"ask"``, ...
        policy_source: which policy fired (``"agent"``, ``"user"``, ``"plan"``).
    """

    event_type: str = field(default="tool_policy_decision", init=False)
    tool_name: str = ""
    action: str = ""
    policy_source: str = ""


@dataclass
class SandboxDecisionEvent(AuditEvent):
    """Emitted when the sandbox layer decides where a tool runs.

    Attributes:
        tool_name: registered tool name.
        context: requesting context (e.g. ``"shell"``, ``"file_write"``).
        reason: short justification for the chosen backend.
        backend_type: ``"docker"``, ``"firejail"``, ``"native"``, ...
    """

    event_type: str = field(default="sandbox_decision", init=False)
    tool_name: str = ""
    context: str = ""
    reason: str = ""
    backend_type: str = ""


@dataclass
class ToolLoopWarningEvent(AuditEvent):
    """Emitted when the loop-detector flags suspicious repetition.

    Attributes:
        severity: always ``"warning"``.
        tool_name: tool being repeated.
        pattern: short label describing the pattern type.
        severity_level: nuance flag (e.g. ``"low"``, ``"high"``).
        call_count: how many consecutive calls triggered the heuristic.
    """

    event_type: str = field(default="tool_loop_warning", init=False)
    severity: str = "warning"
    tool_name: str = ""
    pattern: str = ""
    severity_level: str = ""
    call_count: int = 0


@dataclass
class ToolLoopBlockEvent(AuditEvent):
    """Emitted when the loop-detector aborts further tool calls.

    Attributes:
        severity: always ``"error"``.
        tool_name: tool that was blocked.
        pattern: short label describing the loop type.
        call_count: how many consecutive calls were observed.
    """

    event_type: str = field(default="tool_loop_block", init=False)
    severity: str = "error"
    tool_name: str = ""
    pattern: str = ""
    call_count: int = 0


@dataclass
class HookMutationEvent(AuditEvent):
    """Emitted when a hook rewrites tool arguments or output.

    Attributes:
        hook_name: registered hook identifier.
        tool_name: tool the hook acted upon.
        mutated_field: which field the hook changed.
    """

    event_type: str = field(default="hook_mutation", init=False)
    hook_name: str = ""
    tool_name: str = ""
    mutated_field: str = ""


@dataclass
class ErrorEvent(AuditEvent):
    """Emitted for errors not tied to a specific tool call.

    Attributes:
        severity: always ``"error"``.
        error_type: short classification (e.g. ``"ValueError"``).
        error_message: human-readable description.
        error_context: where the error happened (function, subsystem).
    """

    event_type: str = field(default="error", init=False)
    severity: str = "error"
    error_type: str = ""
    error_message: str = ""
    error_context: str = ""


@dataclass
class SkillUsedEvent(AuditEvent):
    """Emitted when a skill bundle is invoked.

    Attributes:
        skill_name: registered skill identifier.
        version: skill's declared version string.
        outcome: ``"success"`` / ``"failure"`` / ``"partial"`` / ...
        duration_ms: end-to-end skill execution time.
        triggered_automatically: ``True`` when matched by description,
            ``False`` when the user explicitly invoked it.
    """

    event_type: str = field(default="skill_used", init=False)
    skill_name: str = ""
    version: str = ""
    outcome: str = "unknown"
    duration_ms: float = 0.0
    triggered_automatically: bool = True


@dataclass
class SkillAuthoredEvent(AuditEvent):
    """Emitted when a skill bundle is created or updated.

    Attributes:
        skill_name: registered skill identifier.
        version: declared version string.
        source_path: filesystem path to the skill's root.
        tool_count: total tools defined in the skill manifest.
        unique_tools: sorted list of distinct tool names.
        confirmed_by_user: ``True`` if the user explicitly approved it.
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
    """Emitted when a user or agent rates a skill execution.

    Attributes:
        skill_name: registered skill identifier.
        rating: ``"positive"``, ``"neutral"``, ``"negative"``.
        reason: free-form justification.
        source: ``"user"`` or ``"agent"``.
    """

    event_type: str = field(default="skill_feedback", init=False)
    skill_name: str = ""
    rating: str = "neutral"
    reason: str = ""
    source: str = "user"


@dataclass
class AgentSwitchEvent(AuditEvent):
    """Emitted when control transfers between agents mid-turn.

    Attributes:
        from_agent: agent that yielded control.
        to_agent: agent that received control.
        reason: short justification (often the trigger label).
    """

    event_type: str = field(default="agent_switch", init=False)
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""
