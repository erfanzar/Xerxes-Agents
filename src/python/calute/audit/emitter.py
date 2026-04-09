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

"""High-level audit emitter with convenience methods.

:class:`AuditEmitter` wraps an :class:`~calute.audit.collector.AuditCollector`
and exposes one dedicated ``emit_*`` method for every event type defined in
:mod:`calute.audit.events`. This keeps call-sites in the executor and
response loop clean -- callers never need to construct event dataclasses
directly.

The emitter is fully thread-safe: an internal :class:`threading.Lock`
serializes all writes to the underlying collector, and an optional
``session_id`` is automatically stamped onto every event.

Example:
    ::

        from calute.audit import AuditEmitter, InMemoryCollector

        collector = InMemoryCollector()
        emitter = AuditEmitter(collector=collector, session_id="sess-01")
        tid = emitter.emit_turn_start(agent_id="agent-1", prompt="Hello")
        emitter.emit_tool_call_attempt("web_search", args='{"q":"hi"}', turn_id=tid)
        emitter.emit_turn_end(agent_id="agent-1", turn_id=tid, fc_count=1)
        emitter.flush()
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from .collector import AuditCollector, InMemoryCollector
from .events import (
    AuditEvent,
    ErrorEvent,
    HookMutationEvent,
    SandboxDecisionEvent,
    ToolCallAttemptEvent,
    ToolCallCompleteEvent,
    ToolCallFailureEvent,
    ToolLoopBlockEvent,
    ToolLoopWarningEvent,
    ToolPolicyDecisionEvent,
    TurnEndEvent,
    TurnStartEvent,
)


def _generate_turn_id() -> str:
    """Generate a compact random turn identifier.

    Produces a 12-character hexadecimal string derived from a UUID4,
    providing sufficient uniqueness for correlating events within a
    single session while remaining compact enough for log readability.

    Returns:
        str: A 12-character lowercase hexadecimal string
            (e.g. ``"a1b2c3d4e5f6"``).
    """
    return uuid.uuid4().hex[:12]


class AuditEmitter:
    """Thread-safe emitter that converts method calls into typed audit events.

    The emitter provides a dedicated ``emit_*`` convenience method for each
    event type so that call-sites do not need to import or construct event
    dataclasses directly. Every event is automatically stamped with the
    emitter's ``session_id`` before being forwarded to the underlying
    collector.

    Attributes:
        _collector: The audit collector that receives forwarded events.
        _session_id: Optional session-level identifier stamped onto
            every event.
        _lock: Internal threading lock serializing writes.

    Args:
        collector: The audit collector that receives all emitted events.
            Defaults to a new :class:`InMemoryCollector` when ``None``.
        session_id: An optional session identifier that is stamped onto
            every event emitted through this instance.

    Example:
        ::

            from calute.audit import AuditEmitter, InMemoryCollector

            collector = InMemoryCollector()
            emitter = AuditEmitter(collector=collector, session_id="s1")
            tid = emitter.emit_turn_start(agent_id="a1", prompt="Hi")
            emitter.emit_turn_end(agent_id="a1", turn_id=tid)
            assert len(collector) == 2
    """

    def __init__(
        self,
        collector: AuditCollector | InMemoryCollector | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the audit emitter.

        Args:
            collector: The audit collector that receives all emitted
                events. When ``None``, an :class:`InMemoryCollector` is
                created automatically.
            session_id: An optional session identifier stamped onto
                every emitted event.
        """
        self._collector: Any = collector if collector is not None else InMemoryCollector()
        self._session_id = session_id
        self._lock = threading.Lock()

    @property
    def collector(self) -> Any:
        """Return the underlying audit collector instance.

        Returns:
            The collector that this emitter forwards events to.
        """
        return self._collector

    @property
    def session_id(self) -> str | None:
        """Return the session identifier stamped onto every event.

        Returns:
            str | None: The session ID, or ``None`` if not set.
        """
        return self._session_id

    def _emit(self, event: AuditEvent) -> None:
        """Stamp the session ID and forward the event to the collector.

        This is the internal dispatch point used by all public
        ``emit_*`` methods. It sets the event's ``session_id`` field
        and writes to the collector under the internal lock.

        Args:
            event: The fully-constructed audit event to emit.
        """
        event.session_id = self._session_id
        with self._lock:
            self._collector.emit(event)

    def emit_turn_start(
        self,
        agent_id: str | None = None,
        turn_id: str | None = None,
        prompt: str = "",
    ) -> str:
        """Emit a :class:`TurnStartEvent` and return the turn identifier.

        If no ``turn_id`` is provided, a new random 12-hex-character ID
        is generated via :func:`_generate_turn_id`. The prompt is
        truncated to 200 characters for the ``prompt_preview`` field.

        Args:
            agent_id: Optional identifier of the agent starting the turn.
            turn_id: Optional pre-assigned turn identifier. When ``None``,
                a new one is generated automatically.
            prompt: The full user prompt text. Only the first 200
                characters are stored in the event.

        Returns:
            str: The turn identifier (either the provided ``turn_id``
                or the auto-generated one).
        """
        tid = turn_id or _generate_turn_id()
        self._emit(
            TurnStartEvent(
                agent_id=agent_id,
                turn_id=tid,
                prompt_preview=prompt[:200] if prompt else "",
            )
        )
        return tid

    def emit_turn_end(
        self,
        agent_id: str | None = None,
        turn_id: str | None = None,
        content: str = "",
        fc_count: int = 0,
    ) -> None:
        """Emit a :class:`TurnEndEvent` recording turn completion details.

        The assistant's response content is truncated to 200 characters
        for the ``content_preview`` field.

        Args:
            agent_id: Optional identifier of the agent that completed
                the turn.
            turn_id: Optional turn identifier correlating this end event
                with its corresponding :class:`TurnStartEvent`.
            content: The full assistant response text. Only the first
                200 characters are stored in the event.
            fc_count: The total number of function / tool calls that
                were executed during this turn.
        """
        self._emit(
            TurnEndEvent(
                agent_id=agent_id,
                turn_id=turn_id,
                content_preview=content[:200] if content else "",
                function_calls_count=fc_count,
            )
        )

    def emit_tool_call_attempt(
        self,
        tool_name: str,
        args: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolCallAttemptEvent` before a tool is dispatched.

        Should be called immediately before the tool executor runs the
        tool, so that the audit trail records the intent even if the
        tool subsequently fails.

        Args:
            tool_name: The registered name of the tool about to be
                invoked.
            args: A string representation of the tool arguments. Only
                the first 200 characters are stored in the event.
            agent_id: Optional identifier of the agent making the call.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolCallAttemptEvent(
                tool_name=tool_name,
                arguments_preview=args[:200] if args else "",
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_tool_call_complete(
        self,
        tool_name: str,
        status: str = "success",
        duration_ms: float = 0.0,
        result: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolCallCompleteEvent` after a tool returns.

        Should be called immediately after a successful tool execution.
        The result text is truncated to 200 characters for the
        ``result_preview`` field.

        Args:
            tool_name: The registered name of the tool that completed.
            status: Completion status string, typically ``"success"``.
            duration_ms: Wall-clock execution time of the tool in
                milliseconds.
            result: The full string representation of the tool result.
                Only the first 200 characters are stored.
            agent_id: Optional identifier of the agent that invoked the
                tool.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolCallCompleteEvent(
                tool_name=tool_name,
                status=status,
                duration_ms=duration_ms,
                result_preview=result[:200] if result else "",
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_tool_call_failure(
        self,
        tool_name: str,
        error_type: str = "",
        error_msg: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolCallFailureEvent` when a tool raises an error.

        This should be called in the exception handler of the tool
        executor, capturing the exception class name and message for
        the audit trail.

        Args:
            tool_name: The registered name of the tool that failed.
            error_type: A short error classifier, typically the
                exception class name (e.g. ``"ValueError"``).
            error_msg: The human-readable error description or
                stringified exception message.
            agent_id: Optional identifier of the agent that invoked the
                tool.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolCallFailureEvent(
                tool_name=tool_name,
                error_type=error_type,
                error_message=error_msg,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_tool_policy_decision(
        self,
        tool_name: str,
        agent_id: str | None = None,
        action: str = "",
        source: str = "",
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolPolicyDecisionEvent` for a policy verdict.

        Records whether the tool-policy engine allowed or denied a
        given tool invocation, along with the policy rule that produced
        the decision.

        Args:
            tool_name: The registered name of the tool under evaluation.
            agent_id: Optional identifier of the agent requesting the
                tool call.
            action: The policy verdict, typically ``"allow"`` or
                ``"deny"``.
            source: An identifier for the policy rule or configuration
                that produced the decision.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolPolicyDecisionEvent(
                tool_name=tool_name,
                agent_id=agent_id,
                action=action,
                policy_source=source,
                turn_id=turn_id,
            )
        )

    def emit_tool_loop_warning(
        self,
        tool_name: str,
        pattern: str = "",
        severity: str = "warning",
        count: int = 0,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolLoopWarningEvent` for a suspected call loop.

        This is a soft warning -- the tool call is still dispatched, but
        the event alerts downstream consumers that the agent may be
        stuck in a repetitive invocation pattern.

        Args:
            tool_name: The registered name of the tool involved in the
                suspected loop.
            pattern: A short descriptor of the detected repetitive
                pattern (e.g. ``"same_args_3x"``).
            severity: The severity qualifier assigned by the loop
                detector (e.g. ``"warning"``, ``"critical"``).
            count: The number of consecutive or recent calls that
                matched the loop pattern.
            agent_id: Optional identifier of the agent exhibiting the
                loop behavior.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolLoopWarningEvent(
                tool_name=tool_name,
                pattern=pattern,
                severity_level=severity,
                call_count=count,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_tool_loop_block(
        self,
        tool_name: str,
        pattern: str = "",
        count: int = 0,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`ToolLoopBlockEvent` when a loop causes a hard block.

        Unlike :meth:`emit_tool_loop_warning`, this indicates that the
        loop detector has **prevented** the tool call from executing.

        Args:
            tool_name: The registered name of the tool that was blocked.
            pattern: A short descriptor of the repetitive pattern that
                triggered the block (e.g. ``"same_args_5x"``).
            count: The number of consecutive or recent calls that
                matched the loop pattern before the block was imposed.
            agent_id: Optional identifier of the agent exhibiting the
                loop behavior.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ToolLoopBlockEvent(
                tool_name=tool_name,
                pattern=pattern,
                call_count=count,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_sandbox_decision(
        self,
        tool_name: str,
        context: str = "",
        reason: str = "",
        backend_type: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`SandboxDecisionEvent` for a sandbox routing choice.

        Records which execution backend the sandbox router selected for
        a given tool call, along with the context and reasoning behind
        the decision.

        Args:
            tool_name: The registered name of the tool being routed.
            context: A short description of the execution context that
                influenced the routing decision.
            reason: A human-readable explanation of why the particular
                backend was chosen.
            backend_type: The identifier of the selected sandbox backend
                (e.g. ``"local"``, ``"docker"``, ``"subprocess"``).
            agent_id: Optional identifier of the agent whose tool call
                is being routed.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            SandboxDecisionEvent(
                tool_name=tool_name,
                context=context,
                reason=reason,
                backend_type=backend_type,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_hook_mutation(
        self,
        hook_name: str,
        tool_name: str = "",
        agent_id: str | None = None,
        field: str = "",
        turn_id: str | None = None,
    ) -> None:
        """Emit a :class:`HookMutationEvent` when a hook alters data.

        Records which hook mutated which field of a tool call or its
        result, providing a full audit trail for data transformations
        applied outside the tool's own logic.

        Args:
            hook_name: The identifier of the hook that performed the
                mutation (e.g. ``"sanitize_output"``).
            tool_name: The registered name of the tool whose call or
                result was mutated.
            agent_id: Optional identifier of the agent whose pipeline
                includes the hook.
            field: The specific field that was changed
                (e.g. ``"arguments"``, ``"result"``).
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            HookMutationEvent(
                hook_name=hook_name,
                tool_name=tool_name,
                agent_id=agent_id,
                mutated_field=field,
                turn_id=turn_id,
            )
        )

    def emit_error(
        self,
        error_type: str = "",
        error_msg: str = "",
        context: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a generic :class:`ErrorEvent` for non-tool-specific errors.

        Use this for infrastructure-level or agent-level errors that
        cannot be attributed to a single tool invocation (e.g. LLM API
        failures, serialization errors). For tool-specific failures,
        prefer :meth:`emit_tool_call_failure`.

        Args:
            error_type: A short error classifier, typically the
                exception class name (e.g. ``"RuntimeError"``).
            error_msg: The human-readable error description or
                stringified exception message.
            context: Additional context about where or why the error
                occurred (e.g. ``"during response parsing"``).
            agent_id: Optional identifier of the agent that encountered
                the error.
            turn_id: Optional turn identifier for event correlation.
        """
        self._emit(
            ErrorEvent(
                error_type=error_type,
                error_message=error_msg,
                error_context=context,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def flush(self) -> None:
        """Flush the underlying collector's buffered state.

        Acquires the internal lock and delegates to the collector's
        :meth:`~calute.audit.collector.AuditCollector.flush` method,
        ensuring that all previously emitted events have been fully
        persisted or transmitted.
        """
        with self._lock:
            self._collector.flush()
