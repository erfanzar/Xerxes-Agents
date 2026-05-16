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
"""Enums and event dataclasses describing agent function-execution flow.

Defines the strategy enums (:class:`FunctionCallStrategy`,
:class:`AgentSwitchTrigger`, :class:`CompactionStrategy`,
:class:`ExecutionStatus`), the in-flight call record
(:class:`RequestFunctionCall`), capability advertisement
(:class:`AgentCapability`), and the stream of events emitted by the legacy
multi-agent function-execution pipeline (chunks, detection signals,
function-execution start/complete markers, agent switches, completion, and
the reinvoke control signal).
"""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

if tp.TYPE_CHECKING:
    from google.generativeai.types.generation_types import GenerateContentResponse
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class FunctionCallStrategy(Enum):
    """Defines how the agent selects and orders tool calls.

    Attributes:
        SEQUENTIAL: Execute tools one at a time in definition order.
        PARALLEL: Issue multiple tool calls concurrently.
        CONDITIONAL: Select tools based on context and output.
        PIPELINE: Chain tool outputs as inputs to subsequent tools.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


class AgentSwitchTrigger(Enum):
    """Defines the conditions under which an agent hands off to another.

    Attributes:
        EXPLICIT: Hand off on an explicit trigger phrase or action.
        CAPABILITY_BASED: Hand off when the task matches another agent's capabilities.
        CAPABILITY_REQUIRED: Hand off when the current agent lacks required capabilities.
        LOAD_BALANCING: Distribute tasks across agents to balance workload.
        CONTEXT_BASED: Hand off based on conversation context signals.
        ERROR_RECOVERY: Switch to a recovery agent after repeated failures.
        CUSTOM: User-defined custom trigger logic.
    """

    EXPLICIT = "explicit"
    CAPABILITY_BASED = "capability_based"
    CAPABILITY_REQUIRED = "capability_required"
    LOAD_BALANCING = "load"
    CONTEXT_BASED = "context"
    ERROR_RECOVERY = "error"
    CUSTOM = "custom"


class ExecutionStatus(Enum):
    """Tracks the lifecycle state of a tool or task execution.

    Attributes:
        SUCCESS: The operation completed successfully.
        FAILURE: The operation failed (alias for FAILED).
        FAILED: The operation failed.
        PARTIAL: The operation partially succeeded or was incomplete.
        PENDING: The operation has not yet started.
        CANCELLED: The operation was cancelled before completion.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    FAILED = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    CANCELLED = "cancelled"


class CompactionStrategy(Enum):
    """Defines the strategy used to reduce conversation context size.

    Attributes:
        SUMMARIZE: Generate an LLM summary of older messages.
        SLIDING_WINDOW: Keep a fixed window of the most recent messages.
        PRIORITY_BASED: Retain high-priority messages and compress others.
        SMART: Use a model-selected strategy per situation.
        TRUNCATE: Drop the oldest messages outright.
        ADVANCED: Multi-stage prune-then-summarise pipeline.
    """

    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"
    PRIORITY_BASED = "priority_based"
    SMART = "smart"
    TRUNCATE = "truncate"
    ADVANCED = "advanced"


@dataclass
class RequestFunctionCall:
    """Encapsulates a pending or in-flight tool call request.

    Attributes:
        name: The name of the tool or function to invoke.
        arguments: Key-value arguments to pass to the tool.
        id: Unique identifier for this call.
        call_id: An optional alias for ``id`` (populated from ``id`` if unset).
        agent_id: Identifier of the agent that owns this call.
        dependencies: IDs of calls that must complete before this one executes.
        timeout: Seconds before this call is considered stalled.
        retry_count: Number of retries already attempted.
        max_retries: Maximum number of retry attempts allowed.
        status: Current execution status.
        result: The result returned by the tool, if completed.
        error: Error message if the call failed.
    """

    name: str
    arguments: dict
    id: str = field(default_factory=lambda: f"call_{hash(id(object()))}")
    call_id: str | None = None
    agent_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    timeout: float | None = None
    retry_count: int = 0
    max_retries: int = 3
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: tp.Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Set ``call_id`` to ``id`` if not already provided."""

        if self.call_id:
            self.id = self.call_id
        else:
            self.call_id = self.id


@dataclass
class AgentCapability:
    """Describes a named capability that an agent can advertise.

    Capabilities are used for introspection, agent selection, and routing decisions.

    Attributes:
        name: A human-readable identifier for the capability.
        description: A brief description of what the capability covers.
        function_names: Names of the tools or functions associated with this capability.
        context_requirements: Context keys and types needed to perform this capability.
        performance_score: A relative performance multiplier for load balancing (default 1.0).
    """

    name: str
    description: str
    function_names: list[str] = field(default_factory=list)
    context_requirements: dict[str, tp.Any] = field(default_factory=dict)
    performance_score: float = 1.0


AgentCapability.FUNCTION_CALLING = AgentCapability(
    name="function_calling",
    description="Can use tools and function calls",
)


@dataclass
class ExecutionResult:
    """The outcome of a tool or task execution.

    Attributes:
        status: The final execution status.
        result: The returned value, or None if the execution failed.
        error: Error message if the execution failed.
    """

    status: ExecutionStatus
    result: tp.Any | None = None
    error: str | None = None


@dataclass
class SwitchContext:
    """Context carried through an agent switch operation.

    Attributes:
        function_results: Results from tool calls executed before the switch.
        execution_error: Whether any tool execution errored during the switch.
        buffered_content: Accumulated text content prior to the switch.
    """

    function_results: list[ExecutionResult]
    execution_error: bool
    buffered_content: str | None = None


@dataclass
class ToolCallStreamChunk:
    """A fragment of a streaming tool call during LLM response generation.

    Attributes:
        id: Unique identifier for this tool call.
        type: Message type discriminator (default ``"function"``).
        function_name: Name of the function being called.
        arguments: Partial argument string streamed so far.
        index: Position of this tool call in the response.
        is_complete: Whether the arguments string is fully received.
    """

    id: str
    type: str = "function"
    function_name: str | None = None
    arguments: str | None = None
    index: int | None = None
    is_complete: bool = False


@dataclass
class StreamChunk:
    """A single chunk emitted during streaming LLM response generation.

    Attributes:
        type: Event type discriminator.
        chunk: The raw provider chunk object (OpenAI or Gemini).
        agent_id: Identifier of the agent producing this chunk.
        content: Text content of this chunk.
        buffered_content: Accumulated text across all chunks.
        reasoning_content: Reasoning/thinking content (if applicable).
        buffered_reasoning_content: Accumulated reasoning content.
        function_calls_detected: Whether tool calls were detected in this chunk.
        reinvoked: Whether the response was reinvoked after a tool call.
        tool_calls: Parsed tool calls from this chunk.
        streaming_tool_calls: In-progress streaming tool call fragments.
    """

    type: str = "stream_chunk"
    chunk: ChatCompletionChunk | GenerateContentResponse | None = None
    agent_id: str = ""
    content: str | None = None
    buffered_content: str | None = None
    reasoning_content: str | None = None
    buffered_reasoning_content: str | None = None
    function_calls_detected: bool | None = None
    reinvoked: bool = False
    tool_calls: list[ToolCallStreamChunk] | None = None
    streaming_tool_calls: list[ToolCallStreamChunk] | None = None

    def __post_init__(self):
        """Ensure delta content is never None to avoid downstream errors."""
        if self.chunk is not None:
            if hasattr(self.chunk, "choices"):
                for idx, chose in enumerate(self.chunk.choices):
                    if chose.delta.content is None:
                        self.chunk.choices[idx].delta.content = ""

    @property
    def gemini_content(self) -> str | None:
        """Return the text content from a Gemini streaming response.

        Returns:
            The Gemini result text, the chunk content, or None if no content is available.
        """
        if self.chunk is None:
            return None
        if hasattr(self.chunk, "_result") and self.chunk._result:
            if hasattr(self.chunk._result, "text"):
                return self.chunk._result.text
            else:
                return self.content or ""
        elif self.content:
            return self.content
        return None

    @property
    def is_thinking(self) -> bool:
        """Return whether the buffered content indicates an open think/reason block.

        Counts opening and closing tags for think/reasoning blocks in the buffered
        content and returns True if any are left open.

        Returns:
            True if an unclosed think/reasoning block is present.
        """
        if not self.buffered_content:
            return False
        opens = len(re.findall(r"<(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        closes = len(re.findall(r"</(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        return opens > closes


@dataclass
class FunctionDetection:
    """Event emitted when tool usage is detected in a response.

    Attributes:
        type: Event type discriminator (default ``"function_detection"``).
        message: Optional descriptive message accompanying the detection.
        agent_id: Identifier of the agent that detected the function call.
    """

    type: str = "function_detection"
    message: str = ""
    agent_id: str = ""


@dataclass
class FunctionCallInfo:
    """Summary of a detected tool call.

    Attributes:
        name: Name of the function that was called.
        id: Unique identifier for this call.
    """

    name: str
    id: str


@dataclass
class FunctionCallsExtracted:
    """Event emitted when one or more tool calls have been fully extracted from a response.

    Attributes:
        type: Event type discriminator (default ``"function_calls_extracted"``).
        function_calls: List of extracted tool call summaries.
        agent_id: Identifier of the agent that produced the calls.
    """

    type: str = "function_calls_extracted"
    function_calls: list[FunctionCallInfo] = field(default_factory=list)
    agent_id: str = ""


@dataclass
class FunctionExecutionStart:
    """Event emitted immediately before a tool is invoked.

    Attributes:
        type: Event type discriminator (default ``"function_execution_start"``).
        function_name: Name of the function being executed.
        function_id: Unique identifier for this call.
        progress: Human-readable progress description.
        agent_id: Identifier of the agent invoking the tool.
    """

    type: str = "function_execution_start"
    function_name: str = ""
    function_id: str = ""
    progress: str = ""
    agent_id: str = ""


@dataclass
class FunctionExecutionComplete:
    """Event emitted after a tool finishes executing.

    Attributes:
        type: Event type discriminator (default ``"function_execution_complete"``).
        function_name: Name of the function that was executed.
        function_id: Unique identifier for the call.
        status: Outcome string (e.g., ``"success"``, ``"error"``).
        result: The value returned by the tool.
        error: Error message if the execution failed.
        agent_id: Identifier of the agent that invoked the tool.
    """

    type: str = "function_execution_complete"
    function_name: str = ""
    function_id: str = ""
    status: str = ""
    result: tp.Any | None = None
    error: str | None = None
    agent_id: str = ""


@dataclass
class AgentSwitch:
    """Event emitted when an agent hands off control to another agent.

    Attributes:
        type: Event type discriminator (default ``"agent_switch"``).
        from_agent: Role or identifier of the agent giving up control.
        to_agent: Role or identifier of the agent taking over.
        reason: Human-readable reason for the switch.
    """

    type: str = "agent_switch"
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


@dataclass
class Completion:
    """Final completion event marking the end of an agent response.

    Attributes:
        type: Event type discriminator (default ``"completion"``).
        final_content: The fully assembled text response.
        reasoning_content: Accumulated reasoning/thinking content.
        function_calls_executed: Number of tool calls that were executed.
        agent_id: Identifier of the agent that produced this completion.
        execution_history: List of execution events (tool calls, switches, etc.).
    """

    type: str = "completion"
    final_content: str = ""
    reasoning_content: str = ""
    function_calls_executed: int = 0
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)


@dataclass
class ResponseResult:
    """A structured wrapper around a completed agent response.

    Attributes:
        content: The primary text content of the response.
        response: The raw provider response object.
        completion: Parsed ``Completion`` object if available.
        reasoning_content: Accumulated reasoning/thinking content.
        function_calls: List of tool calls made during this response.
        agent_id: Identifier of the agent that produced this response.
        execution_history: Full list of execution events.
        reinvoked: Whether the LLM was reinvoked after tool execution.
    """

    content: str
    response: tp.Any = None
    completion: Completion | None = None
    reasoning_content: str = ""
    function_calls: list[tp.Any] = field(default_factory=list)
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)
    reinvoked: bool = False


@dataclass
class ReinvokeSignal:
    """Internal signal requesting that the LLM be reinvoked after tool execution.

    Attributes:
        message: Optional message accompanying the signal.
        agent_id: Identifier of the agent requesting reinvocation.
        type: Event type discriminator (default ``"reinvoke_signal"``).
    """

    message: str
    agent_id: str
    type: str = "reinvoke_signal"


StreamingResponseType: tp.TypeAlias = (
    StreamChunk
    | FunctionDetection
    | FunctionCallsExtracted
    | FunctionExecutionStart
    | FunctionExecutionComplete
    | AgentSwitch
    | Completion
    | ReinvokeSignal
)
