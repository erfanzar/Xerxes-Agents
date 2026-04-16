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


"""Type definitions and data structures for the Xerxes framework.

This module provides core type definitions, data classes, and enumerations used
throughout the Xerxes agent orchestration framework. It includes message types,
agent definitions, function execution types, tool call structures, and utility
converters for interoperability with external APIs.

Categories:
    Agent Types:
        - Agent: Base agent definition with capabilities and configuration
        - AgentFunction: Callable function definitions for agent tools
        - AgentCapability: Enumeration of agent capabilities

    Message Types:
        - SystemMessage, UserMessage, AssistantMessage, ToolMessage: Chat message types
        - MessagesHistory: Container for conversation history
        - Roles: Enumeration of message roles

    Function Execution:
        - FunctionCallInfo: Metadata about function calls
        - ExecutionResult: Results from function execution
        - StreamChunk: Streaming response chunks

    Tool Definitions:
        - Tool, ToolCall, ToolChoice: Tool specification types
        - Function, FunctionCall: Function specification types

    Converters:
        - convert_openai_messages: Convert OpenAI message format
        - convert_openai_tools: Convert OpenAI tool format

Example:
    >>> from xerxes_agent.types import UserMessage, AssistantMessage, Roles
    >>>
    >>> user_msg = UserMessage(content="Hello!")
    >>> print(user_msg.role)  # Roles.USER
"""

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
    ReinvokeSignal,
    RequestFunctionCall,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
    ToolCallStreamChunk,
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
    "ReinvokeSignal",
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
    "ToolCallStreamChunk",
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
