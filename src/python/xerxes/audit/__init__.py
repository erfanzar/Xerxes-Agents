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
"""Audit event collection and emission.

Three layers compose the subsystem:

* :mod:`events` — typed dataclasses for every event the agent emits
  (turn lifecycle, tool calls, sandbox / policy decisions, skill
  events, hook mutations, agent switches, generic errors).
* :class:`AuditEmitter` — convenience facade that constructs the
  correct event subclass and forwards to a collector after injecting
  the session id.
* Collectors — :class:`InMemoryCollector`, :class:`JSONLSinkCollector`,
  :class:`CompositeCollector`, and :class:`OTelCollector` (spans /
  events via OpenTelemetry, with a noop log fallback).
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
