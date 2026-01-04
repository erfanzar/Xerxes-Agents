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


"""OpenAI API protocol definitions for Calute.

This module provides Pydantic models that mirror the OpenAI API protocol
structures. It includes:
- Request models for chat completions and text completions
- Response models for both streaming and non-streaming responses
- Message types for chat conversations
- Tool and function calling structures
- Usage and metrics tracking

These models enable Calute to provide OpenAI-compatible API endpoints
and facilitate integration with OpenAI-compatible clients and tools.

Example:
    >>> from calute.types.oai_protocols import ChatCompletionRequest, ChatMessage
    >>> request = ChatCompletionRequest(
    ...     model="gpt-4",
    ...     messages=[ChatMessage(role="user", content="Hello!")]
    ... )
"""

import time
import typing as tp
import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    """Base model for OpenAI-compatible protocol structures.

    Extends Pydantic BaseModel with support for extra fields and
    automatic field name caching for validation purposes.

    Attributes:
        field_names: Class-level cache of valid field names including aliases.
    """

    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """Validate and cache field names for the model.

        Wraps the standard validation to build a cache of valid field names
        including any aliases defined on fields.

        Args:
            data: The input data to validate.
            handler: The standard Pydantic validation handler.

        Returns:
            The validated model instance.
        """
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names
        return result


class ChatMessage(OpenAIBaseModel):
    """Represents a single message in a chat conversation.

    Attributes:
        role: Message role (system, user, assistant, function)
        content: Message content (text or structured)
        name: Optional name for the message sender
        function_call: Optional function call made by assistant
    """

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(OpenAIBaseModel):
    """Represents a change (delta) in a chat message.

    Used in streaming responses to send incremental updates.

    Attributes:
        role: Optional role if starting new message
        content: Incremental content to append
        function_call: Optional function call updates
    """

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling.

    Attributes:
        name: The function name.
        description: Description of what the function does.
        parameters: JSON Schema defining the function parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """Tool definition supporting function calling.

    Attributes:
        type: Tool type, currently only "function" is supported.
        function: The function definition for this tool.
    """

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    """Represents incremental updates to a function call during streaming.

    Attributes:
        name: Function name (sent in first chunk only).
        arguments: Incremental arguments string to append.
    """

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    """Represents incremental updates to a tool call during streaming.

    Used in streaming responses to send tool call information in chunks.

    Attributes:
        id: Tool call ID (sent in first chunk only).
        type: Tool type, always "function".
        index: Index of this tool call in the list.
        function: Incremental function call updates.
    """

    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
    """Token usage and performance metrics.

    Tracks computational resources used for a request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens generated
        total_tokens: Sum of prompt and completion tokens
        tokens_per_second: Generation speed
        processing_time: Total processing time in seconds
    """

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(OpenAIBaseModel):
    """Defines a function that can be called by the model.

    Attributes:
        name: Function name
        description: Function description for the model
        parameters: JSON Schema for function parameters (includes 'required' field inside)
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolDefinition(OpenAIBaseModel):
    """Defines a tool that can be called by the model."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(OpenAIBaseModel):
    """Represents a request to the chat completion endpoint.

    Mirrors the OpenAI ChatCompletion request structure with additional
    parameters for extended functionality.

    Attributes:
        model: Model identifier to use for completion.
        messages: List of messages in the conversation.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (0.0-2.0).
        frequency_penalty: Penalty for token frequency (0.0-2.0).
        repetition_penalty: Repetition penalty multiplier.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability threshold for sampling.
        suppress_tokens: Token IDs to suppress during generation.
        functions: Legacy function definitions (deprecated).
        function_call: Legacy function call control (deprecated).
        tools: Tool definitions for function calling.
        tool_choice: Tool selection strategy.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        stop: Stop sequences to end generation.
        logit_bias: Token logit adjustments.
        user: Unique user identifier for tracking.
        chat_template_kwargs: Additional template rendering arguments.
    """

    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
    functions: list[FunctionDefinition] | None = None
    function_call: str | dict[str, tp.Any] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, tp.Any] | None = None
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    chat_template_kwargs: dict[str, int | float | str | bool] | None = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a non-streaming chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Represents a non-streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming chat completion response chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Represents a single chunk in a streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo


class CountTokenRequest(OpenAIBaseModel):
    """Represents a request to the token counting endpoint."""

    model: str
    conversation: str | list[ChatMessage]


class CompletionRequest(OpenAIBaseModel):
    """Represents a request to the completions endpoint.

    Mirrors the OpenAI Completion request structure for text completion
    (non-chat) endpoints.

    Attributes:
        model: Model identifier to use for completion.
        prompt: Text prompt(s) to complete.
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (0.0-2.0).
        frequency_penalty: Penalty for token frequency (0.0-2.0).
        repetition_penalty: Repetition penalty multiplier.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability threshold for sampling.
        suppress_tokens: Token IDs to suppress during generation.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        stop: Stop sequences to end generation.
        logit_bias: Token logit adjustments.
        user: Unique user identifier for tracking.
    """

    model: str
    prompt: str | list[str]
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class CompletionLogprobs(OpenAIBaseModel):
    """Log probabilities for token generation."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a completion response."""

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Represents a response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming completion response chunk."""

    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Represents a streaming response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None


class FunctionCall(OpenAIBaseModel):
    """Represents a function call in the OpenAI format.

    Attributes:
        name: Name of the function to call.
        arguments: JSON-encoded string of function arguments.
    """

    name: str
    arguments: str


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible function calling.

    Attributes:
        name: The function name.
        description: Description of what the function does.
        parameters: JSON Schema defining the function parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolCall(OpenAIBaseModel):
    """Represents a tool call in responses.

    Attributes:
        id: Unique identifier for this tool call.
        type: Tool type, always "function".
        function: The function call details.
    """

    id: str
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(str, Enum):
    """Supported function call formats.

    Different models and frameworks use different formats for function calling.

    Attributes:
        OPENAI: OpenAI's standard format
        JSON_SCHEMA: Direct JSON schema format
        HERMES: Hermes model function calling format
        GORILLA: Gorilla model function calling format
        QWEN: Qwen's special token format (✿FUNCTION✿)
        NOUS: Nous XML-style format (<tool_call>)
    """

    OPENAI = "openai"
    JSON_SCHEMA = "json_schema"
    HERMES = "hermes"
    GORILLA = "gorilla"
    QWEN = "qwen"
    NOUS = "nous"


class ExtractedToolCallInformation(OpenAIBaseModel):
    """Container for extracted tool call information from model output.

    Used to parse and store tool calls extracted from various model output
    formats.

    Attributes:
        tools_called: Whether any tools were called.
        tool_calls: List of extracted tool calls.
        content: Any remaining content after tool extraction.
    """

    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None
