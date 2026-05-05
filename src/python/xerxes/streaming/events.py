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
"""Events module for Xerxes.

Exports:
    - AgentState
    - TextChunk
    - ThinkingChunk
    - ToolStart
    - ToolEnd
    - PermissionRequest
    - TurnDone
    - SkillSuggestion
    - StreamEvent"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Agent state.

    Attributes:
        messages (list[dict[str, Any]]): messages.
        total_input_tokens (int): total input tokens.
        total_output_tokens (int): total output tokens.
        turn_count (int): turn count.
        metadata (dict[str, Any]): metadata.
        thinking_content (list[str]): thinking content.
        tool_executions (list[dict[str, Any]]): tool executions."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    thinking_content: list[str] = field(default_factory=list)

    tool_executions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost(self) -> float:
        """Return Cost.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""

        from xerxes.llms.registry import calc_cost

        """Return Cost.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""
        """Return Cost.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""

        model = self.metadata.get("model", "")
        return calc_cost(model, self.total_input_tokens, self.total_output_tokens)


@dataclass
class TextChunk:
    """Text chunk.

    Attributes:
        text (str): text."""

    text: str


@dataclass
class ThinkingChunk:
    """Thinking chunk.

    Attributes:
        text (str): text."""

    text: str


@dataclass
class ToolStart:
    """Tool start.

    Attributes:
        name (str): name.
        inputs (dict[str, Any]): inputs.
        tool_call_id (str): tool call id."""

    name: str
    inputs: dict[str, Any]
    tool_call_id: str = ""


@dataclass
class ToolEnd:
    """Tool end.

    Attributes:
        name (str): name.
        result (str): result.
        permitted (bool): permitted.
        tool_call_id (str): tool call id.
        duration_ms (float): duration ms."""

    name: str
    result: str
    permitted: bool = True
    tool_call_id: str = ""
    duration_ms: float = 0.0


@dataclass
class PermissionRequest:
    """Permission request.

    Attributes:
        tool_name (str): tool name.
        description (str): description.
        inputs (dict[str, Any]): inputs.
        granted (bool): granted."""

    tool_name: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    granted: bool = False


@dataclass
class TurnDone:
    """Turn done.

    Attributes:
        input_tokens (int): input tokens.
        output_tokens (int): output tokens.
        tool_calls_count (int): tool calls count.
        model (str): model."""

    input_tokens: int
    output_tokens: int
    tool_calls_count: int = 0
    model: str = ""


@dataclass
class SkillSuggestion:
    """Skill suggestion.

    Attributes:
        skill_name (str): skill name.
        version (str): version.
        description (str): description.
        source_path (str): source path.
        tool_count (int): tool count.
        unique_tools (list[str]): unique tools."""

    skill_name: str
    version: str = "0.1.0"
    description: str = ""
    source_path: str = ""
    tool_count: int = 0
    unique_tools: list[str] = field(default_factory=list)


StreamEvent = TextChunk | ThinkingChunk | ToolStart | ToolEnd | PermissionRequest | TurnDone | SkillSuggestion

__all__ = [
    "AgentState",
    "PermissionRequest",
    "SkillSuggestion",
    "StreamEvent",
    "TextChunk",
    "ThinkingChunk",
    "ToolEnd",
    "ToolStart",
    "TurnDone",
]
