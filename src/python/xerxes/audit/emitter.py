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
"""High-level audit event emitter.

This module provides :class:`AuditEmitter`, a convenience facade that constructs
specific audit event dataclasses and forwards them to a collector.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from .collector import AuditCollector, InMemoryCollector
from .events import (
    AgentSwitchEvent,
    AuditEvent,
    ErrorEvent,
    HookMutationEvent,
    SandboxDecisionEvent,
    SkillAuthoredEvent,
    SkillFeedbackEvent,
    SkillUsedEvent,
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
    """Generate a short unique turn identifier.

    Returns:
        str: OUT: A 12-character hexadecimal UUID fragment.
    """
    return uuid.uuid4().hex[:12]


class AuditEmitter:
    """Facade for emitting structured audit events to a collector.

    Provides typed convenience methods for each event subclass and handles
    session ID injection.
    """

    def __init__(
        self,
        collector: AuditCollector | InMemoryCollector | None = None,
        session_id: str | None = None,
        hook_runner: Any = None,
    ) -> None:
        """Initialize the emitter.

        Args:
            collector (AuditCollector | InMemoryCollector | None): IN: Target
                collector. OUT: Defaults to a new :class:`InMemoryCollector`.
            session_id (str | None): IN: Session identifier. OUT: Injected into
                all emitted events.
            hook_runner (Any): IN: Optional hook runner for loop warnings. OUT:
                Used by :meth:`emit_loop_warning`.
        """
        self._collector: Any = collector if collector is not None else InMemoryCollector()
        self._session_id = session_id
        self._lock = threading.Lock()
        self._hook_runner = hook_runner

    @property
    def collector(self) -> Any:
        """Return the underlying collector.

        Returns:
            Any: OUT: The collector instance.
        """
        return self._collector

    @property
    def session_id(self) -> str | None:
        """Return the session identifier.

        Returns:
            str | None: OUT: The session ID, if set.
        """
        return self._session_id

    def _emit(self, event: AuditEvent) -> None:
        """Inject the session ID and forward the event to the collector.

        Args:
            event (AuditEvent): IN: Event to emit. OUT: ``session_id`` is set
                before forwarding.
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
        """Emit a turn start event.

        Args:
            agent_id (str | None): IN: Agent identifier. OUT: Stored in the event.
            turn_id (str | None): IN: Optional turn ID. OUT: Auto-generated if not provided.
            prompt (str): IN: Prompt preview text. OUT: Truncated and stored.

        Returns:
            str: OUT: The turn ID (generated or provided).
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
        """Emit a turn end event.

        Args:
            agent_id (str | None): IN: Agent identifier. OUT: Stored in the event.
            turn_id (str | None): IN: Turn identifier. OUT: Stored in the event.
            content (str): IN: Response content preview. OUT: Truncated and stored.
            fc_count (int): IN: Number of function calls in the turn. OUT: Stored.
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
        """Emit a tool call attempt event.

        Args:
            tool_name (str): IN: Name of the tool being called. OUT: Stored.
            args (str): IN: Arguments preview. OUT: Truncated and stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a tool call completion event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            status (str): IN: Completion status. OUT: Stored.
            duration_ms (float): IN: Execution duration in milliseconds. OUT: Stored.
            result (str): IN: Result preview. OUT: Truncated and stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a tool call failure event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            error_type (str): IN: Error classification. OUT: Stored.
            error_msg (str): IN: Error message. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a tool policy decision event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            action (str): IN: Policy action taken. OUT: Stored.
            source (str): IN: Policy source. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a tool loop warning event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            pattern (str): IN: Detected loop pattern. OUT: Stored.
            severity (str): IN: Severity level. OUT: Stored.
            count (int): IN: Call count. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a tool loop block event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            pattern (str): IN: Detected loop pattern. OUT: Stored.
            count (int): IN: Call count. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a sandbox decision event.

        Args:
            tool_name (str): IN: Tool name. OUT: Stored.
            context (str): IN: Execution context. OUT: Stored.
            reason (str): IN: Decision rationale. OUT: Stored.
            backend_type (str): IN: Sandbox backend type. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit a hook mutation event.

        Args:
            hook_name (str): IN: Name of the mutated hook. OUT: Stored.
            tool_name (str): IN: Related tool name. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            field (str): IN: Mutated field name. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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
        """Emit an error event.

        Args:
            error_type (str): IN: Error classification. OUT: Stored.
            error_msg (str): IN: Error message. OUT: Stored.
            context (str): IN: Error context. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
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

    def emit_skill_used(
        self,
        skill_name: str,
        version: str = "",
        outcome: str = "unknown",
        duration_ms: float = 0.0,
        triggered_automatically: bool = True,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a skill used event.

        Args:
            skill_name (str): IN: Skill name. OUT: Stored.
            version (str): IN: Skill version. OUT: Stored.
            outcome (str): IN: Execution outcome. OUT: Stored.
            duration_ms (float): IN: Execution duration. OUT: Stored.
            triggered_automatically (bool): IN: Whether the skill was auto-triggered.
                OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
        """
        self._emit(
            SkillUsedEvent(
                skill_name=skill_name,
                version=version,
                outcome=outcome,
                duration_ms=duration_ms,
                triggered_automatically=triggered_automatically,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_skill_authored(
        self,
        skill_name: str,
        version: str = "",
        source_path: str = "",
        tool_count: int = 0,
        unique_tools: list[str] | None = None,
        confirmed_by_user: bool = False,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a skill authored event.

        Args:
            skill_name (str): IN: Skill name. OUT: Stored.
            version (str): IN: Skill version. OUT: Stored.
            source_path (str): IN: Filesystem path to the skill source. OUT: Stored.
            tool_count (int): IN: Number of tools in the skill. OUT: Stored.
            unique_tools (list[str] | None): IN: List of unique tool names. OUT: Stored.
            confirmed_by_user (bool): IN: Whether the user confirmed the skill. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
        """
        self._emit(
            SkillAuthoredEvent(
                skill_name=skill_name,
                version=version,
                source_path=source_path,
                tool_count=tool_count,
                unique_tools=list(unique_tools or []),
                confirmed_by_user=confirmed_by_user,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_skill_feedback(
        self,
        skill_name: str,
        rating: str = "neutral",
        reason: str = "",
        source: str = "user",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a skill feedback event.

        Args:
            skill_name (str): IN: Skill name. OUT: Stored.
            rating (str): IN: Feedback rating. OUT: Stored.
            reason (str): IN: Feedback reason. OUT: Stored.
            source (str): IN: Feedback source. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
        """
        self._emit(
            SkillFeedbackEvent(
                skill_name=skill_name,
                rating=rating,
                reason=reason,
                source=source,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def flush(self) -> None:
        """Flush the underlying collector."""
        with self._lock:
            self._collector.flush()

    def emit_agent_switch(
        self,
        from_agent: str,
        to_agent: str,
        reason: str = "",
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit an agent switch event.

        Args:
            from_agent (str): IN: Previous agent name. OUT: Stored.
            to_agent (str): IN: New agent name. OUT: Stored.
            reason (str): IN: Switch reason. OUT: Stored.
            agent_id (str | None): IN: Agent identifier. OUT: Stored.
            turn_id (str | None): IN: Turn identifier. OUT: Stored.
        """
        self._emit(
            AgentSwitchEvent(
                from_agent=from_agent,
                to_agent=to_agent,
                reason=reason,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        )

    def emit_loop_warning(
        self,
        tool_name: str,
        pattern: str = "",
        severity: str = "warning",
        count: int = 0,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        """Emit a loop warning, preferring hooks if available.

        If a hook runner is configured and has ``on_loop_warning`` hooks, they
        are executed instead of emitting a standard event.

        Args:
            tool_name (str): IN: Tool name. OUT: Passed to hooks or event.
            pattern (str): IN: Detected loop pattern. OUT: Passed to hooks or event.
            severity (str): IN: Severity level. OUT: Passed to hooks or event.
            count (int): IN: Call count. OUT: Passed to hooks or event.
            agent_id (str | None): IN: Agent identifier. OUT: Passed to hooks or event.
            turn_id (str | None): IN: Turn identifier. OUT: Passed to hooks or event.
        """
        if hasattr(self, "_hook_runner") and self._hook_runner.has_hooks("on_loop_warning"):
            self._hook_runner.run(
                "on_loop_warning",
                tool_name=tool_name,
                pattern=pattern,
                severity=severity,
                count=count,
                agent_id=agent_id,
                turn_id=turn_id,
            )
        else:
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
