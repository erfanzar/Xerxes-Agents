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

"""Typed audit event definitions for structured event export.

This module defines a hierarchy of dataclass-based audit events that capture
every significant decision and transition in a Calute agent run. All events
inherit from :class:`AuditEvent` and are fully JSON-serializable via their
:meth:`AuditEvent.to_dict` and :meth:`AuditEvent.to_json` methods.

The event taxonomy covers the following domains:

    * **Turn lifecycle** -- :class:`TurnStartEvent`, :class:`TurnEndEvent`
    * **Tool execution** -- :class:`ToolCallAttemptEvent`,
      :class:`ToolCallCompleteEvent`, :class:`ToolCallFailureEvent`
    * **Policy & security** -- :class:`ToolPolicyDecisionEvent`,
      :class:`SandboxDecisionEvent`
    * **Loop detection** -- :class:`ToolLoopWarningEvent`,
      :class:`ToolLoopBlockEvent`
    * **Hook introspection** -- :class:`HookMutationEvent`
    * **General errors** -- :class:`ErrorEvent`

Example:
    Creating and serializing a tool-call event::

        event = ToolCallAttemptEvent(
            tool_name="web_search",
            arguments_preview='{"query": "calute docs"}',
            agent_id="agent-1",
            turn_id="abc123",
        )
        print(event.to_json())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    This is used as the ``default_factory`` for the ``timestamp`` field on
    every :class:`AuditEvent` instance, ensuring that each event is stamped
    at construction time.

    Returns:
        str: A UTC timestamp in ISO-8601 format
            (e.g. ``"2025-06-15T14:30:00.123456+00:00"``).
    """
    return datetime.now(UTC).isoformat()


@dataclass
class AuditEvent:
    """Base audit event carrying the common metadata envelope.

    Every audit event in the Calute audit system inherits from this
    dataclass. It provides the shared envelope fields (timestamps, IDs,
    severity) and the serialization helpers that collectors rely on.

    Subclasses should **not** override ``event_type`` via ``__init__``;
    instead they declare it as a ``field(default=..., init=False)`` so
    that the type tag is automatically set and immutable.

    Attributes:
        event_type: A short string tag identifying the event kind
            (e.g. ``"turn_start"``, ``"tool_call_attempt"``).
        timestamp: ISO-8601 UTC timestamp captured at construction time.
        agent_id: Optional identifier of the agent that produced the event.
        turn_id: Optional identifier of the conversational turn.
        session_id: Optional session-level identifier, typically stamped
            by the :class:`~calute.audit.emitter.AuditEmitter`.
        severity: Log-style severity level. One of ``"info"``,
            ``"warning"``, or ``"error"``.
        metadata: Free-form dictionary for attaching additional
            context that does not warrant its own field.

    Example:
        Constructing a bare base event::

            event = AuditEvent(agent_id="agent-1", severity="warning")
            payload = event.to_dict()
            assert payload["event_type"] == "base"
    """

    event_type: str = "base"
    timestamp: str = field(default_factory=_now_iso)
    agent_id: str | None = None
    turn_id: str | None = None
    session_id: str | None = None
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a plain dictionary.

        Converts the entire dataclass tree (including nested dataclasses)
        into a dictionary of primitive types suitable for JSON encoding.

        Returns:
            dict[str, Any]: A recursively-unwrapped dictionary of all
                fields and their values.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the event to a compact JSON string.

        Non-serializable values (e.g. ``datetime`` objects stored in
        ``metadata``) are coerced to strings via the ``default=str``
        fallback.

        Returns:
            str: A single-line JSON representation of the event.
        """
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TurnStartEvent(AuditEvent):
    """Emitted when a new agent turn begins.

    Captures the opening of a conversational turn, including a truncated
    preview of the user prompt that initiated it.

    Attributes:
        event_type: Fixed to ``"turn_start"`` (not settable via init).
        prompt_preview: The first portion (up to 200 characters) of the
            user prompt that triggered this turn. May be empty if no
            prompt text was available.

    Example:
        ::

            event = TurnStartEvent(
                agent_id="agent-1",
                turn_id="t-001",
                prompt_preview="Summarize the latest report",
            )
    """

    event_type: str = field(default="turn_start", init=False)
    prompt_preview: str = ""


@dataclass
class TurnEndEvent(AuditEvent):
    """Emitted when an agent turn finishes.

    Records the outcome of a completed turn, including a truncated preview
    of the assistant's response and the total number of function calls
    that were executed during the turn.

    Attributes:
        event_type: Fixed to ``"turn_end"`` (not settable via init).
        content_preview: The first portion (up to 200 characters) of the
            assistant's response content. May be empty when no textual
            content was produced.
        function_calls_count: The total number of tool / function calls
            that were dispatched during this turn.

    Example:
        ::

            event = TurnEndEvent(
                agent_id="agent-1",
                turn_id="t-001",
                content_preview="The report shows ...",
                function_calls_count=3,
            )
    """

    event_type: str = field(default="turn_end", init=False)
    content_preview: str = ""
    function_calls_count: int = 0


@dataclass
class ToolCallAttemptEvent(AuditEvent):
    """Emitted just before a tool call is dispatched to its executor.

    This event is created *before* the tool actually runs, making it
    useful for pre-execution auditing and for correlating with the
    subsequent :class:`ToolCallCompleteEvent` or
    :class:`ToolCallFailureEvent`.

    Attributes:
        event_type: Fixed to ``"tool_call_attempt"`` (not settable via init).
        tool_name: The registered name of the tool being invoked.
        arguments_preview: A truncated (up to 200 characters) string
            representation of the arguments passed to the tool.

    Example:
        ::

            event = ToolCallAttemptEvent(
                tool_name="web_search",
                arguments_preview='{"query": "Python dataclasses"}',
                agent_id="agent-1",
            )
    """

    event_type: str = field(default="tool_call_attempt", init=False)
    tool_name: str = ""
    arguments_preview: str = ""


@dataclass
class ToolCallCompleteEvent(AuditEvent):
    """Emitted when a tool call completes successfully.

    Captures the outcome of a successful tool invocation, including its
    wall-clock duration and a truncated preview of the result payload.

    Attributes:
        event_type: Fixed to ``"tool_call_complete"`` (not settable via init).
        tool_name: The registered name of the tool that was invoked.
        status: Completion status string, typically ``"success"``.
        duration_ms: Wall-clock execution time in milliseconds.
        result_preview: A truncated (up to 200 characters) string
            representation of the tool's return value.

    Example:
        ::

            event = ToolCallCompleteEvent(
                tool_name="web_search",
                status="success",
                duration_ms=142.5,
                result_preview="Found 10 results for ...",
            )
    """

    event_type: str = field(default="tool_call_complete", init=False)
    tool_name: str = ""
    status: str = "success"
    duration_ms: float = 0.0
    result_preview: str = ""


@dataclass
class ToolCallFailureEvent(AuditEvent):
    """Emitted when a tool call raises an exception or otherwise fails.

    The event automatically sets ``severity`` to ``"error"`` to ensure
    failures are prominently surfaced in any downstream log consumer.

    Attributes:
        event_type: Fixed to ``"tool_call_failure"`` (not settable via init).
        severity: Defaults to ``"error"`` (overrides the base ``"info"``).
        tool_name: The registered name of the tool that failed.
        error_type: A short classifier for the error, typically the
            exception class name (e.g. ``"ValueError"``).
        error_message: The human-readable error description or the
            stringified exception message.

    Example:
        ::

            event = ToolCallFailureEvent(
                tool_name="file_read",
                error_type="FileNotFoundError",
                error_message="/tmp/missing.txt not found",
                agent_id="agent-1",
            )
    """

    event_type: str = field(default="tool_call_failure", init=False)
    severity: str = "error"
    tool_name: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class ToolPolicyDecisionEvent(AuditEvent):
    """Emitted when the policy engine makes an allow/deny decision.

    Records the outcome of the tool-policy evaluation step that occurs
    before a tool call is dispatched. This is critical for auditing
    which policy rules allowed or blocked a given tool invocation.

    Attributes:
        event_type: Fixed to ``"tool_policy_decision"`` (not settable via init).
        tool_name: The registered name of the tool under evaluation.
        action: The policy verdict, typically ``"allow"`` or ``"deny"``.
        policy_source: An identifier for the policy rule or configuration
            file that produced the decision (e.g. ``"default_policy"``
            or ``"operator_override"``).

    Example:
        ::

            event = ToolPolicyDecisionEvent(
                tool_name="shell_exec",
                action="deny",
                policy_source="operator_override",
                agent_id="agent-1",
            )
    """

    event_type: str = field(default="tool_policy_decision", init=False)
    tool_name: str = ""
    action: str = ""
    policy_source: str = ""


@dataclass
class SandboxDecisionEvent(AuditEvent):
    """Emitted when the sandbox router selects an execution backend.

    The sandbox router inspects the tool call and its context to decide
    which isolation backend (e.g. local, Docker, subprocess jail) should
    execute the call. This event captures that routing decision.

    Attributes:
        event_type: Fixed to ``"sandbox_decision"`` (not settable via init).
        tool_name: The registered name of the tool being routed.
        context: A short description of the execution context that
            influenced the routing decision.
        reason: A human-readable explanation of *why* the particular
            backend was selected.
        backend_type: The identifier of the chosen sandbox backend
            (e.g. ``"local"``, ``"docker"``, ``"subprocess"``).

    Example:
        ::

            event = SandboxDecisionEvent(
                tool_name="shell_exec",
                context="untrusted_input",
                reason="Input contains user-supplied shell commands",
                backend_type="docker",
            )
    """

    event_type: str = field(default="sandbox_decision", init=False)
    tool_name: str = ""
    context: str = ""
    reason: str = ""
    backend_type: str = ""


@dataclass
class ToolLoopWarningEvent(AuditEvent):
    """Emitted when the loop detector identifies a potential tool-call loop.

    This is a *soft* warning -- the tool call is still allowed to proceed,
    but downstream consumers (dashboards, logs) are alerted that the
    agent may be stuck in a repetitive pattern.

    Attributes:
        event_type: Fixed to ``"tool_loop_warning"`` (not settable via init).
        severity: Defaults to ``"warning"`` (overrides the base ``"info"``).
        tool_name: The registered name of the tool involved in the
            suspected loop.
        pattern: A short description of the repetitive pattern that was
            detected (e.g. ``"same_args_3x"``).
        severity_level: An additional severity qualifier supplied by the
            loop detector (e.g. ``"warning"``, ``"critical"``).
        call_count: The number of consecutive or recent calls that
            matched the loop pattern.

    Example:
        ::

            event = ToolLoopWarningEvent(
                tool_name="web_search",
                pattern="same_args_3x",
                severity_level="warning",
                call_count=3,
            )
    """

    event_type: str = field(default="tool_loop_warning", init=False)
    severity: str = "warning"
    tool_name: str = ""
    pattern: str = ""
    severity_level: str = ""
    call_count: int = 0


@dataclass
class ToolLoopBlockEvent(AuditEvent):
    """Emitted when a tool-call loop triggers a hard block.

    Unlike :class:`ToolLoopWarningEvent`, this event indicates that the
    loop detector has *prevented* the tool call from executing. The
    ``severity`` is automatically set to ``"error"``.

    Attributes:
        event_type: Fixed to ``"tool_loop_block"`` (not settable via init).
        severity: Defaults to ``"error"`` (overrides the base ``"info"``).
        tool_name: The registered name of the tool that was blocked.
        pattern: A short description of the repetitive pattern that
            triggered the block (e.g. ``"same_args_5x"``).
        call_count: The number of consecutive or recent calls that
            matched the loop pattern before the block was imposed.

    Example:
        ::

            event = ToolLoopBlockEvent(
                tool_name="web_search",
                pattern="same_args_5x",
                call_count=5,
            )
    """

    event_type: str = field(default="tool_loop_block", init=False)
    severity: str = "error"
    tool_name: str = ""
    pattern: str = ""
    call_count: int = 0


@dataclass
class HookMutationEvent(AuditEvent):
    """Emitted when a hook mutates a tool call or its result.

    Hooks can intercept tool calls before execution (pre-hooks) or
    modify results after execution (post-hooks). This event records
    which hook altered which field, providing an audit trail for any
    data transformations applied outside the tool's own logic.

    Attributes:
        event_type: Fixed to ``"hook_mutation"`` (not settable via init).
        hook_name: The identifier of the hook that performed the mutation
            (e.g. ``"sanitize_output"``, ``"inject_context"``).
        tool_name: The registered name of the tool whose call or result
            was mutated.
        mutated_field: The specific field that was changed
            (e.g. ``"arguments"``, ``"result"``).

    Example:
        ::

            event = HookMutationEvent(
                hook_name="sanitize_output",
                tool_name="shell_exec",
                mutated_field="result",
                agent_id="agent-1",
            )
    """

    event_type: str = field(default="hook_mutation", init=False)
    hook_name: str = ""
    tool_name: str = ""
    mutated_field: str = ""


@dataclass
class ErrorEvent(AuditEvent):
    """Emitted for generic errors not tied to a specific tool call.

    Use this event type for infrastructure-level or agent-level errors
    that cannot be attributed to a single tool invocation (e.g. LLM
    API failures, serialization errors, or unexpected state transitions).
    For tool-specific failures, prefer :class:`ToolCallFailureEvent`.

    Attributes:
        event_type: Fixed to ``"error"`` (not settable via init).
        severity: Defaults to ``"error"`` (overrides the base ``"info"``).
        error_type: A short classifier for the error, typically the
            exception class name (e.g. ``"RuntimeError"``).
        error_message: The human-readable error description or the
            stringified exception message.
        error_context: Additional context about where or why the error
            occurred (e.g. ``"during response parsing"``).

    Example:
        ::

            event = ErrorEvent(
                error_type="TimeoutError",
                error_message="LLM request timed out after 30s",
                error_context="during turn execution",
                agent_id="agent-1",
            )
    """

    event_type: str = field(default="error", init=False)
    severity: str = "error"
    error_type: str = ""
    error_message: str = ""
    error_context: str = ""
