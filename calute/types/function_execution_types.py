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

import typing as tp
from dataclasses import dataclass, field
from enum import Enum

from google.generativeai.types.generation_types import GenerateContentResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class FunctionCallStrategy(Enum):
    """Strategies for handling function calls"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


class AgentSwitchTrigger(Enum):
    """Triggers for agent switching"""

    EXPLICIT = "explicit"  # Manual switch via function
    CAPABILITY_BASED = "capability"  # Switch based on required capabilities
    LOAD_BALANCING = "load"  # Switch based on agent availability
    CONTEXT_BASED = "context"  # Switch based on conversation context
    ERROR_RECOVERY = "error"  # Switch on function execution errors


class ExecutionStatus(Enum):
    """Status of function/agent execution"""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class RequestFunctionCall:
    """Enhanced function call representation"""

    name: str
    arguments: dict
    id: str = field(default_factory=lambda: f"call_{hash(id(object()))}")
    agent_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    timeout: float | None = None
    retry_count: int = 0
    max_retries: int = 3
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: tp.Any = None
    error: str | None = None


@dataclass
class AgentCapability:
    """Defines what an agent is capable of doing"""

    name: str
    description: str
    function_names: list[str] = field(default_factory=list)
    context_requirements: dict[str, tp.Any] = field(default_factory=dict)
    performance_score: float = 1.0  # For capability-based routing


@dataclass
class ExecutionResult:
    """Result of function execution"""

    status: ExecutionStatus
    result: tp.Any | None = None
    error: str | None = None


@dataclass
class SwitchContext:
    """Context for agent switching decisions"""

    function_results: list[ExecutionResult]
    execution_error: bool
    buffered_content: str | None = None


@dataclass
class StreamChunk:
    """Represents a streaming chunk response"""

    type: str = "stream_chunk"
    chunk: ChatCompletionChunk | GenerateContentResponse | None = None
    agent_id: str = ""
    content: str | None = None
    buffered_content: str | None = None
    function_calls_detected: bool | None = None
    reinvoked: bool = False

    def __post_init__(self):
        if self.chunk is not None:
            if hasattr(self.chunk, "choices"):
                for idx, chose in enumerate(self.chunk.choices):
                    if chose.delta.content is None:
                        self.chunk.choices[idx].delta.content = ""

    @property
    def gemini_content(self):
        if hasattr(self.chunk, "_result") and self.chunk._result:
            if hasattr(self.chunk._result, "text"):
                return self.chunk._result.text
            else:
                return self.content or ""
        elif self.content:
            return self.content


@dataclass
class FunctionDetection:
    """Notification of function call detection"""

    type: str = "function_detection"
    message: str = ""
    agent_id: str = ""


@dataclass
class FunctionCallInfo:
    """Basic information about a function call"""

    name: str
    id: str


@dataclass
class FunctionCallsExtracted:
    """Information about extracted function calls"""

    type: str = "function_calls_extracted"
    function_calls: list[FunctionCallInfo] = field(default_factory=list)
    agent_id: str = ""


@dataclass
class FunctionExecutionStart:
    """Notification of function execution start"""

    type: str = "function_execution_start"
    function_name: str = ""
    function_id: str = ""
    progress: str = ""
    agent_id: str = ""


@dataclass
class FunctionExecutionComplete:
    """Result of function execution"""

    type: str = "function_execution_complete"
    function_name: str = ""
    function_id: str = ""
    status: str = ""
    result: tp.Any | None = None
    error: str | None = None
    agent_id: str = ""


@dataclass
class AgentSwitch:
    """Information about an agent switch"""

    type: str = "agent_switch"
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""


@dataclass
class Completion:
    """Final completion information"""

    type: str = "completion"
    final_content: str = ""
    function_calls_executed: int = 0
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)


@dataclass
class ResponseResult:
    """Result of a non-streaming response"""

    content: str
    response: ChatCompletion
    function_calls: list[RequestFunctionCall] = field(default_factory=list)
    agent_id: str = ""
    execution_history: list[tp.Any] = field(default_factory=list)
    reinvoked: bool = False


@dataclass
class ReinvokeSignal:
    """Signal that the agent is being reinvoked with function results"""

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
