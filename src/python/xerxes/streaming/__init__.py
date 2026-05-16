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
"""Streaming subsystem: agent loop, stream events, wire-protocol events.

Re-exports the public surface used by the daemon, TUI, and bridge: the
:func:`run_agent_loop` driver, neutral message conversion helpers, the
:class:`PermissionMode` gate, internal :class:`StreamEvent` dataclasses, and
the full wire-event vocabulary.
"""

from .events import (
    AgentState,
    PermissionRequest,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from .loop import run as run_agent_loop
from .messages import (
    NeutralMessage,
    messages_to_anthropic,
    messages_to_openai,
)
from .permissions import PermissionMode, check_permission
from .wire_events import (
    ApprovalRequest,
    ApprovalResponse,
    BtwBegin,
    BtwEnd,
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    HookResolved,
    HookTriggered,
    MCPLoadingBegin,
    MCPLoadingEnd,
    Notification,
    PlanDisplay,
    QuestionItem,
    QuestionRequest,
    QuestionResponse,
    StatusUpdate,
    SteerInput,
    StepBegin,
    StepEnd,
    StepInterrupted,
    SubagentEvent,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
    WireEvent,
    WireEventType,
    event_from_dict,
)

__all__ = [
    "AgentState",
    "ApprovalRequest",
    "ApprovalResponse",
    "BtwBegin",
    "BtwEnd",
    "CompactionBegin",
    "CompactionEnd",
    "ContentPart",
    "HookResolved",
    "HookTriggered",
    "MCPLoadingBegin",
    "MCPLoadingEnd",
    "NeutralMessage",
    "Notification",
    "PermissionMode",
    "PermissionRequest",
    "PlanDisplay",
    "QuestionItem",
    "QuestionRequest",
    "QuestionResponse",
    "StatusUpdate",
    "SteerInput",
    "StepBegin",
    "StepEnd",
    "StepInterrupted",
    "StreamEvent",
    "SubagentEvent",
    "TextChunk",
    "TextPart",
    "ThinkPart",
    "ToolCall",
    "ToolCallPart",
    "ToolEnd",
    "ToolResult",
    "ToolStart",
    "TurnBegin",
    "TurnDone",
    "TurnEnd",
    "WireEvent",
    "WireEventType",
    "check_permission",
    "event_from_dict",
    "messages_to_anthropic",
    "messages_to_openai",
    "run_agent_loop",
]
