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
"""In-process stream events emitted by the agent loop.

These dataclasses are the *internal* event vocabulary: the agent loop yields
them, and the TUI / bridge translates a subset into wire events for clients.
``AgentState`` is the mutable session container that the loop appends to as
turns progress (messages, token counters, tool executions, thinking content).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable session state threaded through the agent loop.

    Holds the conversation history plus running token counters and tool
    execution audit trail. A single instance lives for the duration of a
    session and is mutated in place as each turn runs.

    Attributes:
        messages: Neutral message list (role/content dicts plus optional
            ``tool_calls``/``thinking`` fields) consumed by the LLM provider.
        total_input_tokens: Cumulative prompt tokens across all turns.
        total_output_tokens: Cumulative completion tokens across all turns.
        total_cache_read_tokens: Cumulative tokens served from Anthropic prompt
            cache (~0.1x list price).
        total_cache_creation_tokens: Cumulative tokens written to a new
            Anthropic cache entry (~1.25x list price).
        turn_count: Number of assistant turns completed (each tool round trip
            counts as one).
        metadata: Free-form session metadata; the loop sets ``model`` here so
            ``cost`` can resolve pricing.
        thinking_content: Per-turn reasoning text, parallel to assistant
            messages. Empty strings stand in for turns without thinking.
        tool_executions: Audit records for every tool invocation in this
            session (name, inputs, result, duration, permission outcome).
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    thinking_content: list[str] = field(default_factory=list)

    tool_executions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost(self) -> float:
        """Return the running session cost in USD for the active model."""
        from xerxes.llms.registry import calc_cost

        model = self.metadata.get("model", "")
        return calc_cost(model, self.total_input_tokens, self.total_output_tokens)


@dataclass
class TextChunk:
    """Streamed visible text fragment from the assistant.

    Attributes:
        text: Incremental text delta.
    """

    text: str


@dataclass
class ThinkingChunk:
    """Streamed reasoning-channel fragment (model thinking, hidden from output).

    Attributes:
        text: Incremental reasoning text.
    """

    text: str


@dataclass
class ToolStart:
    """Signals that a tool call is about to execute.

    Attributes:
        name: Tool identifier as the model emitted it.
        inputs: Parsed arguments dict that will be passed to the executor.
        tool_call_id: Provider-issued call id; ``""`` until the loop fills one in.
    """

    name: str
    inputs: dict[str, Any]
    tool_call_id: str = ""


@dataclass
class ToolEnd:
    """Signals that a tool call has finished.

    Attributes:
        name: Tool identifier.
        result: Serialized return value (or error / denial message).
        permitted: ``False`` when the user rejected the call; ``result`` then
            holds the denial reason.
        tool_call_id: Provider-issued call id correlating with the ToolStart.
        duration_ms: Wall-clock execution time; zero for denied calls.
    """

    name: str
    result: str
    permitted: bool = True
    tool_call_id: str = ""
    duration_ms: float = 0.0


@dataclass
class PermissionRequest:
    """Permission prompt yielded mid-turn for the UI to resolve.

    The loop yields this and blocks until the consumer flips ``granted`` and
    resumes iteration. The convention is in-band mutation rather than a paired
    response event.

    Attributes:
        tool_name: Tool being requested.
        description: Human-readable summary for the prompt (e.g. the shell
            command or target file path).
        inputs: Full argument dict so the UI can render a diff or preview.
        granted: Caller-mutated decision; defaults to denied.
    """

    tool_name: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    granted: bool = False


@dataclass
class TurnDone:
    """Emitted at the end of every assistant turn with usage accounting.

    Attributes:
        input_tokens: Prompt tokens for this turn only.
        output_tokens: Completion tokens for this turn only.
        tool_calls_count: Number of tool calls the assistant requested.
        model: Model id used for the turn.
        cache_read_tokens: Tokens served from a cached prefix (~0.1x cost).
        cache_creation_tokens: Tokens written to a new cache entry (~1.25x cost).
    """

    input_tokens: int
    output_tokens: int
    tool_calls_count: int = 0
    model: str = ""
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class SkillSuggestion:
    """Emitted when the authoring pipeline auto-extracts a reusable skill.

    Attributes:
        skill_name: Generated skill identifier.
        version: Semver string; pipeline initialises to ``0.1.0``.
        description: Optional human summary (currently empty by default).
        source_path: On-disk path of the authored skill bundle.
        tool_count: Number of tool events captured in the underlying candidate.
        unique_tools: De-duplicated list of tool names the skill uses.
    """

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
