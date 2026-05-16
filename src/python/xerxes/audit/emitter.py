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
"""Convenience facade for constructing and dispatching audit events.

:class:`AuditEmitter` exposes one ``emit_*`` method per event subtype.
Each constructs the right dataclass, injects ``session_id``, takes the
emitter lock, and forwards to the configured collector. The emitter
also wires the optional ``hook_runner`` used to route loop-warning
events through user hooks.
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
    """Return a 12-char hexadecimal turn id derived from a uuid4."""
    return uuid.uuid4().hex[:12]


class AuditEmitter:
    """Construct typed audit events and forward them to a collector.

    Holds an :class:`InMemoryCollector` by default and injects the
    bound ``session_id`` onto every emitted event. Loop-warning
    emission is special-cased to delegate to user hooks when a
    compatible ``hook_runner`` is wired.
    """

    def __init__(
        self,
        collector: AuditCollector | InMemoryCollector | None = None,
        session_id: str | None = None,
        hook_runner: Any = None,
    ) -> None:
        """Bind the collector, session id, and optional hook runner.

        Args:
            collector: target sink; defaults to a new :class:`InMemoryCollector`.
            session_id: stamped on every event.
            hook_runner: object with ``has_hooks`` / ``run`` used by
                :meth:`emit_loop_warning`.
        """
        self._collector: Any = collector if collector is not None else InMemoryCollector()
        self._session_id = session_id
        self._lock = threading.Lock()
        self._hook_runner = hook_runner

    @property
    def collector(self) -> Any:
        """Return the bound collector instance."""
        return self._collector

    @property
    def session_id(self) -> str | None:
        """Return the session id stamped onto outgoing events."""
        return self._session_id

    def _emit(self, event: AuditEvent) -> None:
        """Inject ``session_id`` and forward ``event`` to the collector."""
        event.session_id = self._session_id
        with self._lock:
            self._collector.emit(event)

    def emit_turn_start(
        self,
        agent_id: str | None = None,
        turn_id: str | None = None,
        prompt: str = "",
    ) -> str:
        """Emit :class:`TurnStartEvent` and return the resolved turn id.

        ``prompt`` is truncated to 200 chars for the preview. When
        ``turn_id`` is omitted a fresh id is generated.
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
        """Emit :class:`TurnEndEvent` for the turn just finished.

        ``content`` is truncated to 200 chars for the preview;
        ``fc_count`` records how many tools the model invoked.
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
        """Emit :class:`ToolCallAttemptEvent`; ``args`` is truncated to 200 chars."""
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
        """Emit :class:`ToolCallCompleteEvent`; ``result`` is truncated to 200 chars."""
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
        """Emit :class:`ToolCallFailureEvent` for an unsuccessful tool call."""
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
        """Emit :class:`ToolPolicyDecisionEvent`."""
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
        """Emit :class:`ToolLoopWarningEvent` (unconditionally; see also ``emit_loop_warning``)."""
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
        """Emit :class:`ToolLoopBlockEvent` when the loop detector aborts a tool."""
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
        """Emit :class:`SandboxDecisionEvent` after the sandbox routes a call."""
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
        """Emit :class:`HookMutationEvent` when a hook rewrites a tool field."""
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
        """Emit :class:`ErrorEvent` for an error not tied to a tool call."""
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
        """Emit :class:`SkillUsedEvent` after a skill bundle finishes."""
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
        """Emit :class:`SkillAuthoredEvent` after a skill is created or updated."""
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
        """Emit :class:`SkillFeedbackEvent` to record a rating on a skill run."""
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
        """Flush the underlying collector under the emitter lock."""
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
        """Emit :class:`AgentSwitchEvent` when control passes between agents."""
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
        """Emit a loop warning, routing through ``hook_runner`` when present.

        If the bound ``hook_runner`` has ``on_loop_warning`` hooks they
        run instead of (not in addition to) the standard event emission.
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
