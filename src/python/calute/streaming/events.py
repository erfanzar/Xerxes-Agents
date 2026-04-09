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


"""Typed streaming events for the Calute agent loop.

Inspired by the nano-claude-code event protocol, this module defines a set of
typed dataclasses that represent discrete events during streaming agent
execution. A generator-based agent loop yields these events, allowing the
consumer (TUI, API server, etc.) to handle each event type independently.

Event types:

- :class:`TextChunk` — Incremental text from the model.
- :class:`ThinkingChunk` — Incremental thinking/reasoning text.
- :class:`ToolStart` — A tool invocation is about to begin.
- :class:`ToolEnd` — A tool invocation has completed (with result).
- :class:`PermissionRequest` — The loop is requesting user permission.
- :class:`TurnDone` — An LLM turn has completed (with token counts).

The :data:`StreamEvent` type alias is the union of all event types.

Usage::

    from calute.streaming.events import TextChunk, ToolStart, ToolEnd

    for event in agent_loop(...):
        match event:
            case TextChunk(text=t):
                print(t, end="")
            case ToolStart(name=n):
                print(f"\\n[tool] {n} ...")
            case ToolEnd(name=n, result=r):
                print(f"[tool] {n} → {r[:80]}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable session state for the streaming agent loop.

    Tracks the conversation history (in neutral message format), cumulative
    token usage, and the number of LLM turns taken.

    Attributes:
        messages: List of neutral-format messages (dicts with ``role``,
            ``content``, optionally ``tool_calls``, ``tool_call_id``, ``name``).
        total_input_tokens: Cumulative input tokens across all turns.
        total_output_tokens: Cumulative output tokens across all turns.
        turn_count: Number of LLM turns completed.
        metadata: Arbitrary key-value metadata for the session.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    thinking_content: list[str] = field(default_factory=list)
    """Accumulated thinking/reasoning text per turn (indexed by turn_count - 1)."""
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    """Full tool execution records: name, inputs, result, duration_ms, permitted."""

    @property
    def cost(self) -> float:
        """Estimated total cost in USD (requires model in metadata)."""
        from calute.llms.registry import calc_cost

        model = self.metadata.get("model", "")
        return calc_cost(model, self.total_input_tokens, self.total_output_tokens)


@dataclass
class TextChunk:
    """Incremental text output from the model.

    Attributes:
        text: The text fragment.
    """

    text: str


@dataclass
class ThinkingChunk:
    """Incremental thinking/reasoning output from the model.

    Only emitted when the model supports extended thinking (e.g. Claude
    with ``thinking`` enabled, DeepSeek-Reasoner, o1/o3).

    Attributes:
        text: The thinking text fragment.
    """

    text: str


@dataclass
class ToolStart:
    """Emitted when a tool invocation is about to begin.

    Attributes:
        name: The tool name (e.g. ``"Read"``, ``"Bash"``).
        inputs: The tool input arguments.
        tool_call_id: Provider-assigned tool call ID.
    """

    name: str
    inputs: dict[str, Any]
    tool_call_id: str = ""


@dataclass
class ToolEnd:
    """Emitted when a tool invocation has completed.

    Attributes:
        name: The tool name.
        result: The tool's output (string).
        permitted: Whether the tool was actually executed (False if denied).
        tool_call_id: Provider-assigned tool call ID.
        duration_ms: Execution duration in milliseconds (0 if not measured).
    """

    name: str
    result: str
    permitted: bool = True
    tool_call_id: str = ""
    duration_ms: float = 0.0


@dataclass
class PermissionRequest:
    """Emitted when the agent loop needs user permission to proceed.

    The consumer should set :attr:`granted` to ``True`` or ``False``
    before the loop continues.

    Attributes:
        tool_name: Name of the tool requiring permission.
        description: Human-readable description of the operation.
        inputs: The tool input arguments (for display).
        granted: Set by the consumer — ``True`` to allow, ``False`` to deny.
    """

    tool_name: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    granted: bool = False


@dataclass
class TurnDone:
    """Emitted at the end of each LLM turn.

    Attributes:
        input_tokens: Tokens consumed in this turn's input.
        output_tokens: Tokens generated in this turn's output.
        tool_calls_count: Number of tool calls made in this turn.
        model: Model used for this turn.
    """

    input_tokens: int
    output_tokens: int
    tool_calls_count: int = 0
    model: str = ""


StreamEvent = TextChunk | ThinkingChunk | ToolStart | ToolEnd | PermissionRequest | TurnDone

__all__ = [
    "AgentState",
    "PermissionRequest",
    "StreamEvent",
    "TextChunk",
    "ThinkingChunk",
    "ToolEnd",
    "ToolStart",
    "TurnDone",
]
