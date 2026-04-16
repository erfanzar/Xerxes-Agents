# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Type definitions for function execution and streaming responses.

This module provides comprehensive type definitions for the function
execution system in Xerxes, including:
- Execution strategies (sequential, parallel, pipeline)
- Agent switching triggers and capabilities
- Function call request and result types
- Streaming response chunk types
- Execution status and completion types

These types form the foundation for type-safe function execution,
agent orchestration, and streaming response handling throughout
the Xerxes framework.

Example:
    >>> from xerxes_agent.types.function_execution_types import (
    ...     ExecutionStatus,
    ...     RequestFunctionCall,
    ...     StreamChunk,
    ... )
    >>> call = RequestFunctionCall(name="search", arguments={"query": "test"})
    >>> call.status = ExecutionStatus.SUCCESS
"""

import re
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

from google.generativeai.types.generation_types import GenerateContentResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class FunctionCallStrategy(Enum):
    """Enumeration of strategies for handling function calls.

    Defines how multiple function calls should be executed
    when an agent needs to invoke several functions.

    Attributes:
        SEQUENTIAL: Execute functions one after another in order.
        PARALLEL: Execute all functions concurrently.
        CONDITIONAL: Execute functions based on conditions/dependencies.
        PIPELINE: Execute functions in a pipeline where output flows to input.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


class AgentSwitchTrigger(Enum):
    """Enumeration of triggers for agent switching.

    Defines the conditions or events that can trigger
    switching from one agent to another during execution.

    Attributes:
        EXPLICIT: Manual/explicit switch requested by code or user.
        CAPABILITY_BASED: Switch based on required capabilities.
        LOAD_BALANCING: Switch to balance load across agents.
        CONTEXT_BASED: Switch based on conversation context.
        ERROR_RECOVERY: Switch triggered by error recovery mechanism.
    """

    EXPLICIT = "explicit"
    CAPABILITY_BASED = "capability"
    CAPABILITY_REQUIRED = "capability"
    LOAD_BALANCING = "load"
    CONTEXT_BASED = "context"
    ERROR_RECOVERY = "error"
    CUSTOM = "custom"


class ExecutionStatus(Enum):
    """Enumeration of function/agent execution status values.

    Represents the current state or outcome of a function
    or agent execution operation.

    Attributes:
        SUCCESS: Execution completed successfully.
        FAILURE: Execution failed with an error.
        PARTIAL: Execution partially completed (some steps succeeded).
        PENDING: Execution is pending/not yet started.
        CANCELLED: Execution was cancelled before completion.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    FAILED = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    CANCELLED = "cancelled"


class CompactionStrategy(Enum):
    """Enumeration of strategies for context compaction.

    Defines how conversation context should be reduced when
    it exceeds token limits or needs optimization.

    Attributes:
        SUMMARIZE: Compress context by generating a summary.
        SLIDING_WINDOW: Keep only recent context within a window.
        PRIORITY_BASED: Keep high-priority context, remove low-priority.
        TRUNCATE: Simply truncate context to fit limits.
    """

    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"
    PRIORITY_BASED = "priority_based"
    TRUNCATE = "truncate"


@dataclass
class RequestFunctionCall:
    """Enhanced representation of a function call request.

    Encapsulates all information needed to execute a function call,
    including execution parameters, retry configuration, and status tracking.

    Attributes:
        name: Name of the function to call.
        arguments: Dictionary of arguments to pass to the function.
        id: Unique identifier for this function call, auto-generated if not provided.
        call_id: Optional explicit call ID that, when provided, overrides ``id``.
            After initialization, ``id`` and ``call_id`` are kept in sync.
        agent_id: ID of the agent making the call, if applicable.
        dependencies: List of function call IDs this call depends on.
        timeout: Optional timeout in seconds for execution.
        retry_count: Current number of retry attempts made.
        max_retries: Maximum number of retry attempts allowed.
        status: Current execution status of the function call.
        result: Result value from successful execution.
        error: Error message if execution failed.
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
        """Synchronize the ``id`` and ``call_id`` fields after initialization.

        Ensures both identifiers are consistent: if ``call_id`` was explicitly
        provided, it takes precedence and overwrites ``id``; otherwise ``call_id``
        is set to the auto-generated ``id`` value.
        """
        if self.call_id:
            self.id = self.call_id
        else:
            self.call_id = self.id


@dataclass
class AgentCapability:
    """Definition of an agent's capability.

    Describes a specific capability that an agent possesses,
    including the functions it can execute and performance metrics.

    Attributes:
        name: Human-readable name of the capability.
        description: Detailed description of what the capability enables.
        function_names: List of function names associated with this capability.
        context_requirements: Required context variables for this capability.
        performance_score: Performance rating for this capability (0.0-1.0).
    """

    name: str
    description: str
    function_names: list[str] = field(default_factory=list)
    context_requirements: dict[str, tp.Any] = field(default_factory=dict)
    performance_score: float = 1.0


AgentCapability.FUNCTION_CALLING = AgentCapability(  # type: ignore[attr-defined]
    name="function_calling",
    description="Can use tools and function calls",
)


@dataclass
class ExecutionResult:
    """Result container for function execution.

    Holds the outcome of a function execution attempt,
    including status, result value, and any error information.

    Attributes:
        status: The execution status (success, failure, etc.).
        result: The return value from successful execution.
        error: Error message if execution failed.
    """

    status: ExecutionStatus
    result: tp.Any | None = None
    error: str | None = None


@dataclass
class SwitchContext:
    """Context information for agent switching decisions.

    Provides the context needed to make decisions about
    switching between agents during execution.

    Attributes:
        function_results: Results from recent function executions.
        execution_error: Whether an error occurred during execution.
        buffered_content: Any buffered response content.
    """

    function_results: list[ExecutionResult]
    execution_error: bool
    buffered_content: str | None = None


@dataclass
class ToolCallStreamChunk:
    """Streaming chunk for a tool/function call.

    Represents a partial chunk of a tool or function call
    received during streaming responses.

    Attributes:
        id: Unique identifier for this tool call.
        type: Type of tool call (default: "function").
        function_name: Name of the function being called.
        arguments: JSON string of arguments (may be partial during streaming).
        index: Index of this tool call in the response.
        is_complete: Whether the tool call is complete.
    """

    id: str
    type: str = "function"
    function_name: str | None = None
    arguments: str | None = None
    index: int | None = None
    is_complete: bool = False


@dataclass
class StreamChunk:
    """Streaming chunk response from an LLM provider.

    Encapsulates a streaming response chunk, supporting both
    OpenAI and Gemini response formats with tool call tracking.

    Attributes:
        type: Type identifier for this chunk (default: "stream_chunk").
        chunk: Raw chunk from OpenAI or Gemini API.
        agent_id: ID of the agent that produced this chunk.
        content: Text content extracted from the chunk.
        buffered_content: Accumulated content from all chunks so far.
        reasoning_content: Reasoning/thinking content from the current chunk.
        buffered_reasoning_content: Accumulated reasoning content from all chunks so far.
        function_calls_detected: Whether function calls were detected.
        reinvoked: Whether the agent was reinvoked after function execution.
        tool_calls: Completed tool calls extracted from the response.
        streaming_tool_calls: Tool calls still being streamed.
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
        """Normalize the chunk after initialization.

        Ensures that ``delta.content`` on each choice in an OpenAI-style chunk
        is set to an empty string rather than ``None``. This prevents downstream
        consumers from needing to handle ``None`` content during string
        concatenation of streamed responses.
        """
        if self.chunk is not None:
            if hasattr(self.chunk, "choices"):
                for idx, chose in enumerate(self.chunk.choices):
                    if chose.delta.content is None:
                        self.chunk.choices[idx].delta.content = ""

    @property
    def gemini_content(self) -> str | None:
        """Extract content from a Gemini response chunk.

        Returns:
            The text content from the Gemini response, or None if unavailable.
        """
        if hasattr(self.chunk, "_result") and self.chunk._result:
            if hasattr(self.chunk._result, "text"):
                return self.chunk._result.text
            else:
                return self.content or ""
        elif self.content:
            return self.content

    @property
    def is_thinking(self) -> bool:
        """Check if currently inside thinking/reasoning tags.

        Detects whether the buffered content is within thinking
        or reasoning XML-style tags (e.g., <think>, <reasoning>).

        Returns:
            True if inside thinking tags, False otherwise.
        """
        if not self.buffered_content:
            return False
        opens = len(re.findall(r"<(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        closes = len(re.findall(r"</(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        return opens > closes


@dataclass
class FunctionDetection:
    """Notification event for function call detection.

    Emitted when the system detects that an LLM response
    contains function/tool calls that need to be executed.

    Attributes:
        type: Event type identifier (default: "function_detection").
        message: Human-readable message about the detection.
        agent_id: ID of the agent whose response contained the calls.
    """

    type: str = "function_detection"
    message: str = ""
    agent_id: str = ""


@dataclass
class FunctionCallInfo:
    """Basic identifying information for a function call.

    Lightweight container for function call identification,
    used in event notifications and tracking.

    Attributes:
        name: Name of the function being called.
        id: Unique identifier for this specific call.
    """

    name: str
    id: str


@dataclass
class FunctionCallsExtracted:
    """Event containing extracted function call information.

    Emitted after function calls have been parsed and extracted
    from an LLM response, before execution begins.

    Attributes:
        type: Event type identifier (default: "function_calls_extracted").
        function_calls: List of extracted function call information.
        agent_id: ID of the agent that requested the function calls.
    """

    type: str = "function_calls_extracted"
    function_calls: list[FunctionCallInfo] = field(default_factory=list)
    agent_id: str = ""


@dataclass
class FunctionExecutionStart:
    """Event notification for function execution start.

    Emitted when a function begins execution, allowing
    for progress tracking and monitoring.

    Attributes:
        type: Event type identifier (default: "function_execution_start").
        function_name: Name of the function being executed.
        function_id: Unique identifier for this function call.
        progress: Progress indicator (e.g., "1/3" for first of three).
        agent_id: ID of the agent executing the function.
    """

    type: str = "function_execution_start"
    function_name: str = ""
    function_id: str = ""
    progress: str = ""
    agent_id: str = ""


@dataclass
class FunctionExecutionComplete:
    """Event notification for function execution completion.

    Emitted when a function finishes execution, containing
    the result or error information.

    Attributes:
        type: Event type identifier (default: "function_execution_complete").
        function_name: Name of the executed function.
        function_id: Unique identifier for this function call.
        status: Execution status string (e.g., "success", "error").
        result: Return value from successful execution.
        error: Error message if execution failed.
        agent_id: ID of the agent that executed the function.
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
    """Event notification for an agent switch.

    Emitted when execution switches from one agent to another,
    recording the transition details.

    Attributes:
        type: Event type identifier (default: "agent_switch").
        from_agent: ID of the agent being switched from.
        to_agent: ID of the agent being switched to.
        reason: Human-readable reason for the switch.
    """

    type: str = "agent_switch"
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


@dataclass
class Completion:
    """Final completion event for a response.

    Emitted when the entire response cycle completes,
    including all function executions and agent responses.

    Attributes:
        type: Event type identifier (default: "completion").
        final_content: The final accumulated response content.
        reasoning_content: Accumulated reasoning/thinking tokens from the model.
        function_calls_executed: Total number of function calls executed.
        agent_id: ID of the agent that produced the final response.
        execution_history: List of execution events that occurred.
    """

    type: str = "completion"
    final_content: str = ""
    reasoning_content: str = ""
    function_calls_executed: int = 0
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)


@dataclass
class ResponseResult:
    """Complete result from a non-streaming response.

    Contains all information from a completed non-streaming
    agent response, including content and execution details.

    Attributes:
        content: The text content of the response.
        response: Raw ChatCompletion response from the LLM.
        completion: Completion event with summary information.
        reasoning_content: Reasoning/thinking content produced by the model, if any.
        function_calls: List of function calls that were executed.
        agent_id: ID of the agent that produced the response.
        execution_history: List of all execution events.
        reinvoked: Whether the agent was reinvoked after function execution.
    """

    content: str
    response: ChatCompletion
    completion: Completion
    reasoning_content: str = ""
    function_calls: list[RequestFunctionCall] = field(default_factory=list)
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)
    reinvoked: bool = False


@dataclass
class ReinvokeSignal:
    """Signal that the agent is being reinvoked with function results.

    Emitted when the agent is called again after function execution,
    allowing it to process the function results and continue.

    Attributes:
        message: Informational message about the reinvocation.
        agent_id: ID of the agent being reinvoked.
        type: Event type identifier (default: "reinvoke_signal").
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
