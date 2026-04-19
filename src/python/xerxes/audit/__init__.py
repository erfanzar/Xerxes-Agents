# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured audit event system for Xerxes.

This package provides a pluggable, thread-safe audit trail for every
significant decision and transition that occurs during an agent run.
It is composed of three layers:

    Events:
        Typed dataclasses (all subclasses of
        :class:`~xerxes.audit.events.AuditEvent`) that represent every
        decision point -- tool calls, policy decisions, sandbox routing,
        hook mutations, loop detection, errors, and turn boundaries.

    Collectors:
        Pluggable sinks that implement the ``emit`` / ``flush`` protocol.
        Three concrete implementations ship out-of-the-box:

        * :class:`InMemoryCollector` -- thread-safe in-memory buffer.
        * :class:`JSONLSinkCollector` -- newline-delimited JSON file sink.
        * :class:`CompositeCollector` -- fan-out to multiple child collectors.

    Emitter:
        :class:`AuditEmitter` is a high-level helper that converts
        convenient method calls into the appropriate typed events and
        forwards them to a collector.

Example:
    Basic usage with the in-memory collector::

        from xerxes.audit import AuditEmitter, InMemoryCollector

        collector = InMemoryCollector()
        emitter = AuditEmitter(collector=collector, session_id="sess-01")
        turn_id = emitter.emit_turn_start(agent_id="agent-1", prompt="Hello")
        emitter.emit_turn_end(agent_id="agent-1", turn_id=turn_id)
        assert len(collector) == 2
"""

from __future__ import annotations

from .collector import (
    AuditCollector,
    CompositeCollector,
    InMemoryCollector,
    JSONLSinkCollector,
)
from .emitter import AuditEmitter
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
from .otel_exporter import OTelCollector

__all__ = [
    "AuditCollector",
    "AuditEmitter",
    "AuditEvent",
    "CompositeCollector",
    "ErrorEvent",
    "HookMutationEvent",
    "InMemoryCollector",
    "JSONLSinkCollector",
    "OTelCollector",
    "SandboxDecisionEvent",
    "SkillAuthoredEvent",
    "SkillFeedbackEvent",
    "SkillUsedEvent",
    "ToolCallAttemptEvent",
    "ToolCallCompleteEvent",
    "ToolCallFailureEvent",
    "ToolLoopBlockEvent",
    "ToolLoopWarningEvent",
    "ToolPolicyDecisionEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
