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
from .agent_types import Agent, AgentBaseFn, AgentFunction, Response
from .converters import convert_openai_messages, convert_openai_tools
from .function_execution_types import (
    AgentCapability,
    AgentSwitch,
    AgentSwitchTrigger,
    Completion,
    ExecutionResult,
    ExecutionStatus,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionCallStrategy,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    RequestFunctionCall,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
)
from .messages import (
    AssistantMessage,
    AssistantMessageType,
    BaseContentChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageType,
    ChunkTypes,
    ImageChunk,
    ImageURL,
    ImageURLChunk,
    MessagesHistory,
    Roles,
    SystemMessage,
    SystemMessageType,
    TextChunk,
    ToolMessage,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)
from .tool_calls import Function, FunctionCall, Tool, ToolCall, ToolChoice, ToolType, ToolTypes

__all__ = (
    "Agent",
    "AgentBaseFn",
    "AgentCapability",
    "AgentFunction",
    "AgentSwitch",
    "AgentSwitchTrigger",
    "AssistantMessage",
    "AssistantMessageType",
    "BaseContentChunk",
    "BaseMessage",
    "ChatMessage",
    "ChatMessageType",
    "ChunkTypes",
    "Completion",
    "ExecutionResult",
    "ExecutionStatus",
    "Function",
    "FunctionCall",
    "FunctionCallInfo",
    "FunctionCallStrategy",
    "FunctionCallsExtracted",
    "FunctionDetection",
    "FunctionExecutionComplete",
    "FunctionExecutionStart",
    "ImageChunk",
    "ImageURL",
    "ImageURLChunk",
    "MessagesHistory",
    "RequestFunctionCall",
    "Response",
    "ResponseResult",
    "Roles",
    "StreamChunk",
    "StreamingResponseType",
    "SwitchContext",
    "SystemMessage",
    "SystemMessageType",
    "TextChunk",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolMessage",
    "ToolMessageType",
    "ToolType",
    "ToolTypes",
    "UserMessage",
    "UserMessageType",
    "convert_openai_messages",
    "convert_openai_tools",
)
