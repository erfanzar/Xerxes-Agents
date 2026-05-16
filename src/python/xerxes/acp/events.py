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
"""ACP wire-event shape and conversion from internal ``StreamEvent``.

Internal types live in ``xerxes.streaming.events`` and are sync
Python dataclasses. ACP clients expect a tagged-union with ``kind``
discriminator. Use ``to_acp_event(stream_event)`` to convert."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..streaming.events import (
    PermissionRequest,
    SkillSuggestion,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)


class AcpEventKind(StrEnum):
    """ACP wire event kinds Xerxes can emit."""

    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    PERMISSION_REQUEST = "permission_request"
    TURN_END = "turn_end"
    SKILL_SUGGESTION = "skill_suggestion"
    UNKNOWN = "unknown"


@dataclass
class AcpEvent:
    """Single event delivered over ACP.

    Attributes:
        kind: discriminator tagging the payload shape.
        payload: kind-specific JSON-shaped fields (text, tool info, etc.).
    """

    kind: AcpEventKind
    payload: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Render as the flat JSON-RPC dict expected on the wire."""
        return {"kind": self.kind.value, **self.payload}


def to_acp_event(event: StreamEvent) -> AcpEvent:
    """Convert an internal stream event to an ACP event.

    Unknown event types degrade to ``AcpEventKind.UNKNOWN`` with the
    raw repr so the client can at least log them."""

    if isinstance(event, TextChunk):
        return AcpEvent(AcpEventKind.TEXT_DELTA, {"text": event.text})
    if isinstance(event, ThinkingChunk):
        return AcpEvent(AcpEventKind.THINKING_DELTA, {"text": event.text})
    if isinstance(event, ToolStart):
        return AcpEvent(
            AcpEventKind.TOOL_CALL_START,
            {"name": event.name, "inputs": event.inputs, "tool_call_id": event.tool_call_id},
        )
    if isinstance(event, ToolEnd):
        return AcpEvent(
            AcpEventKind.TOOL_CALL_END,
            {
                "name": event.name,
                "result": event.result,
                "permitted": event.permitted,
                "tool_call_id": event.tool_call_id,
                "duration_ms": event.duration_ms,
            },
        )
    if isinstance(event, PermissionRequest):
        return AcpEvent(
            AcpEventKind.PERMISSION_REQUEST,
            {
                "tool_name": event.tool_name,
                "description": event.description,
                "inputs": event.inputs,
            },
        )
    if isinstance(event, TurnDone):
        return AcpEvent(
            AcpEventKind.TURN_END,
            {
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "tool_calls_count": event.tool_calls_count,
                "model": event.model,
                "cache_read_tokens": getattr(event, "cache_read_tokens", 0),
                "cache_creation_tokens": getattr(event, "cache_creation_tokens", 0),
            },
        )
    if isinstance(event, SkillSuggestion):
        return AcpEvent(
            AcpEventKind.SKILL_SUGGESTION,
            {"skill_name": event.skill_name, "description": event.description},
        )
    return AcpEvent(AcpEventKind.UNKNOWN, {"repr": repr(event)})


__all__ = ["AcpEvent", "AcpEventKind", "to_acp_event"]
