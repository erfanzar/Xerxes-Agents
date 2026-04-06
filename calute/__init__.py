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


"""Calute: An advanced AI agent orchestration framework.

This module provides a comprehensive framework for building and managing AI agents
with sophisticated capabilities including memory management, task execution, tool
integration, and multi-agent orchestration. The framework supports multiple LLM
providers and offers both synchronous and asynchronous execution modes.

Key Features:
    - Multi-agent orchestration with dynamic switching
    - Advanced memory management (short-term, long-term, contextual)
    - Tool and function integration
    - Support for multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)
    - Streaming and batch processing capabilities
    - Cortex system for complex agent workflows

Example:
    >>> from calute import Calute, OpenAILLM
    >>>
    >>> llm = OpenAILLM(api_key="your-api-key")
    >>> agent = Calute(llm=llm, name="Assistant")
    >>> response = agent.query("Hello, how can you help me?")
    >>> print(response.content)
"""

from .calute import Calute
from .core import PromptTemplate
from .core.streamer_buffer import StreamerBuffer
from .cortex import (
    ChainLink,
    ChainType,
    Cortex,
    CortexAgent,
    CortexMemory,
    CortexOutput,
    CortexTask,
    CortexTaskOutput,
    CortexTool,
    ProcessType,
)
from .executors import AgentOrchestrator
from .llms import (
    COSTS,
    PROVIDERS,
    AnthropicLLM,
    BaseLLM,
    CustomLLM,
    DeepSeekLLM,
    GeminiLLM,
    KimiLLM,
    LLMConfig,
    LMStudioLLM,
    LocalLLM,
    OllamaLLM,
    OpenAICompatLLM,
    OpenAILLM,
    ProviderConfig,
    QwenLLM,
    ZhipuLLM,
    calc_cost,
    create_llm,
    detect_provider,
    get_context_limit,
    list_all_models,
)
from .mcp import MCPClient, MCPManager, MCPResource, MCPServerConfig, MCPTool
from .memory import MemoryEntry, MemoryStore, MemoryType
from .operators.config import OperatorRuntimeConfig
from .runtime.features import AgentRuntimeOverrides, RuntimeFeaturesConfig
from .runtime.profiles import PromptProfile, PromptProfileConfig
from .streaming import (
    AgentState,
    NeutralMessage,
    PermissionMode,
    PermissionRequest,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
    check_permission,
    messages_to_anthropic,
    messages_to_openai,
    run_agent_loop,
)
from .tui import launch_tui
from .types import (
    Agent,
    AgentCapability,
    AgentFunction,
    AgentSwitch,
    AgentSwitchTrigger,
    AssistantMessage,
    AssistantMessageType,
    ChatMessageType,
    Completion,
    ExecutionResult,
    ExecutionStatus,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionCallStrategy,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    MessagesHistory,
    RequestFunctionCall,
    Roles,
    StreamChunk,
    SwitchContext,
    SystemMessage,
    SystemMessageType,
    ToolMessage,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)

__all__ = (
    "COSTS",
    "PROVIDERS",
    "Agent",
    "AgentCapability",
    "AgentFunction",
    "AgentOrchestrator",
    "AgentRuntimeOverrides",
    "AgentState",
    "AgentSwitch",
    "AgentSwitchTrigger",
    "AnthropicLLM",
    "AssistantMessage",
    "AssistantMessageType",
    "BaseLLM",
    "Calute",
    "ChainLink",
    "ChainType",
    "ChatMessageType",
    "Completion",
    "Cortex",
    "CortexAgent",
    "CortexMemory",
    "CortexOutput",
    "CortexTask",
    "CortexTaskOutput",
    "CortexTool",
    "CustomLLM",
    "DeepSeekLLM",
    "ExecutionResult",
    "ExecutionStatus",
    "FunctionCallInfo",
    "FunctionCallStrategy",
    "FunctionCallsExtracted",
    "FunctionDetection",
    "FunctionExecutionComplete",
    "FunctionExecutionStart",
    "GeminiLLM",
    "KimiLLM",
    "LLMConfig",
    "LMStudioLLM",
    "LocalLLM",
    "MCPClient",
    "MCPManager",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
    "MemoryEntry",
    "MemoryStore",
    "MemoryType",
    "MessagesHistory",
    "NeutralMessage",
    "OllamaLLM",
    "OpenAICompatLLM",
    "OpenAILLM",
    "OperatorRuntimeConfig",
    "PermissionMode",
    "PermissionRequest",
    "ProcessType",
    "PromptProfile",
    "PromptProfileConfig",
    "PromptTemplate",
    "ProviderConfig",
    "QwenLLM",
    "RequestFunctionCall",
    "Roles",
    "RuntimeFeaturesConfig",
    "StreamChunk",
    "StreamEvent",
    "StreamerBuffer",
    "SwitchContext",
    "SystemMessage",
    "SystemMessageType",
    "TextChunk",
    "ThinkingChunk",
    "ToolEnd",
    "ToolMessage",
    "ToolMessageType",
    "ToolStart",
    "TurnDone",
    "UserMessage",
    "UserMessageType",
    "ZhipuLLM",
    "calc_cost",
    "check_permission",
    "create_llm",
    "detect_provider",
    "get_context_limit",
    "launch_tui",
    "list_all_models",
    "messages_to_anthropic",
    "messages_to_openai",
    "run_agent_loop",
)

__version__ = "0.2.0"
