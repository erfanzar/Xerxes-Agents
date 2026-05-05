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
"""Function execution types module for Xerxes.

Exports:
    - FunctionCallStrategy
    - AgentSwitchTrigger
    - ExecutionStatus
    - CompactionStrategy
    - RequestFunctionCall
    - AgentCapability
    - ExecutionResult
    - SwitchContext
    - ToolCallStreamChunk
    - StreamChunk
    - ... and 9 more."""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

if tp.TYPE_CHECKING:
    from google.generativeai.types.generation_types import GenerateContentResponse
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class FunctionCallStrategy(Enum):
    """Function call strategy.

    Inherits from: Enum
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


class AgentSwitchTrigger(Enum):
    """Agent switch trigger.

    Inherits from: Enum
    """

    EXPLICIT = "explicit"
    CAPABILITY_BASED = "capability_based"
    CAPABILITY_REQUIRED = "capability_required"
    LOAD_BALANCING = "load"
    CONTEXT_BASED = "context"
    ERROR_RECOVERY = "error"
    CUSTOM = "custom"


class ExecutionStatus(Enum):
    """Execution status.

    Inherits from: Enum
    """

    SUCCESS = "success"
    FAILURE = "failure"
    FAILED = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    CANCELLED = "cancelled"


class CompactionStrategy(Enum):
    """Compaction strategy.

    Inherits from: Enum
    """

    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"
    PRIORITY_BASED = "priority_based"
    SMART = "smart"
    TRUNCATE = "truncate"
    HERMES = "hermes"


@dataclass
class RequestFunctionCall:
    """Request function call.

    Attributes:
        name (str): name.
        arguments (dict): arguments.
        id (str): id.
        call_id (str | None): call id.
        agent_id (str | None): agent id.
        dependencies (list[str]): dependencies.
        timeout (float | None): timeout.
        retry_count (int): retry count.
        max_retries (int): max retries.
        status (ExecutionStatus): status.
        result (tp.Any): result.
        error (str | None): error."""

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
        """Dunder method for post init.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.call_id:
            self.id = self.call_id
        else:
            self.call_id = self.id


@dataclass
class AgentCapability:
    """Agent capability.

    Attributes:
        name (str): name.
        description (str): description.
        function_names (list[str]): function names.
        context_requirements (dict[str, tp.Any]): context requirements.
        performance_score (float): performance score."""

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
    """Execution result.

    Attributes:
        status (ExecutionStatus): status.
        result (tp.Any | None): result.
        error (str | None): error."""

    status: ExecutionStatus
    result: tp.Any | None = None
    error: str | None = None


@dataclass
class SwitchContext:
    """Switch context.

    Attributes:
        function_results (list[ExecutionResult]): function results.
        execution_error (bool): execution error.
        buffered_content (str | None): buffered content."""

    function_results: list[ExecutionResult]
    execution_error: bool
    buffered_content: str | None = None


@dataclass
class ToolCallStreamChunk:
    """Tool call stream chunk.

    Attributes:
        id (str): id.
        type (str): type.
        function_name (str | None): function name.
        arguments (str | None): arguments.
        index (int | None): index.
        is_complete (bool): is complete."""

    id: str
    type: str = "function"
    function_name: str | None = None
    arguments: str | None = None
    index: int | None = None
    is_complete: bool = False


@dataclass
class StreamChunk:
    """Stream chunk.

    Attributes:
        type (str): type.
        chunk (ChatCompletionChunk | GenerateContentResponse | None): chunk.
        agent_id (str): agent id.
        content (str | None): content.
        buffered_content (str | None): buffered content.
        reasoning_content (str | None): reasoning content.
        buffered_reasoning_content (str | None): buffered reasoning content.
        function_calls_detected (bool | None): function calls detected.
        reinvoked (bool): reinvoked.
        tool_calls (list[ToolCallStreamChunk] | None): tool calls.
        streaming_tool_calls (list[ToolCallStreamChunk] | None): streaming tool calls."""

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
        """Dunder method for post init.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        if self.chunk is not None:
            if hasattr(self.chunk, "choices"):
                for idx, chose in enumerate(self.chunk.choices):
                    if chose.delta.content is None:
                        self.chunk.choices[idx].delta.content = ""

    @property
    def gemini_content(self) -> str | None:
        """Return Gemini content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str | None: OUT: Result of the operation."""

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
        """Return Check whether thinking.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        if not self.buffered_content:
            return False
        opens = len(re.findall(r"<(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        closes = len(re.findall(r"</(think|thinking|reason|reasoning)>", self.buffered_content, re.I))
        return opens > closes


@dataclass
class FunctionDetection:
    """Function detection.

    Attributes:
        type (str): type.
        message (str): message.
        agent_id (str): agent id."""

    type: str = "function_detection"
    message: str = ""
    agent_id: str = ""


@dataclass
class FunctionCallInfo:
    """Function call info.

    Attributes:
        name (str): name.
        id (str): id."""

    name: str
    id: str


@dataclass
class FunctionCallsExtracted:
    """Function calls extracted.

    Attributes:
        type (str): type.
        function_calls (list[FunctionCallInfo]): function calls.
        agent_id (str): agent id."""

    type: str = "function_calls_extracted"
    function_calls: list[FunctionCallInfo] = field(default_factory=list)
    agent_id: str = ""


@dataclass
class FunctionExecutionStart:
    """Function execution start.

    Attributes:
        type (str): type.
        function_name (str): function name.
        function_id (str): function id.
        progress (str): progress.
        agent_id (str): agent id."""

    type: str = "function_execution_start"
    function_name: str = ""
    function_id: str = ""
    progress: str = ""
    agent_id: str = ""


@dataclass
class FunctionExecutionComplete:
    """Function execution complete.

    Attributes:
        type (str): type.
        function_name (str): function name.
        function_id (str): function id.
        status (str): status.
        result (tp.Any | None): result.
        error (str | None): error.
        agent_id (str): agent id."""

    type: str = "function_execution_complete"
    function_name: str = ""
    function_id: str = ""
    status: str = ""
    result: tp.Any | None = None
    error: str | None = None
    agent_id: str = ""


@dataclass
class AgentSwitch:
    """Agent switch.

    Attributes:
        type (str): type.
        from_agent (str): from agent.
        to_agent (str): to agent.
        reason (str): reason."""

    type: str = "agent_switch"
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


@dataclass
class Completion:
    """Completion.

    Attributes:
        type (str): type.
        final_content (str): final content.
        reasoning_content (str): reasoning content.
        function_calls_executed (int): function calls executed.
        agent_id (str): agent id.
        execution_history (list[tp.Any]): execution history."""

    type: str = "completion"
    final_content: str = ""
    reasoning_content: str = ""
    function_calls_executed: int = 0
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)


@dataclass
class ResponseResult:
    """Response result.

    Attributes:
        content (str): content.
        response (tp.Any): response.
        completion (Completion | None): completion.
        reasoning_content (str): reasoning content.
        function_calls (list[tp.Any]): function calls.
        agent_id (str): agent id.
        execution_history (list[tp.Any]): execution history.
        reinvoked (bool): reinvoked."""

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
    """Reinvoke signal.

    Attributes:
        message (str): message.
        agent_id (str): agent id.
        type (str): type."""

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
