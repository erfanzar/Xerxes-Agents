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
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    """Base Pydantic model for all OpenAI-compatible protocol structures.

    Extends Pydantic's ``BaseModel`` with ``extra="allow"`` configuration to
    accept additional fields beyond those explicitly defined, and provides
    automatic caching of valid field names (including aliases) for efficient
    validation in downstream code.

    All OpenAI protocol models in this module inherit from this base class
    to ensure consistent behavior and forward compatibility with new API fields.

    Attributes:
        field_names: Class-variable cache of valid field names including aliases.
            Populated lazily on first model validation and reused for subsequent
            instances.
    """

    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """Validate input data and lazily cache field names for the model class.

        Wraps the standard Pydantic validation handler. On first invocation for
        each class, builds a set of all recognized field names (including any
        Pydantic field aliases) and stores it as a class variable for efficient
        lookup by validation utilities.

        Args:
            data: The raw input data (typically a dictionary) to validate.
            handler: The Pydantic validation handler to delegate to for actual
                model construction.

        Returns:
            The fully validated model instance produced by the handler.
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
    """Represents a single message in a chat conversation (OpenAI protocol).

    Used within ``ChatCompletionRequest`` to represent messages in the
    conversation history. Supports both plain text and structured multimodal
    content.

    Attributes:
        role: The role of the message sender. One of ``"system"``, ``"user"``,
            ``"assistant"``, or ``"function"``.
        content: The message content, either a plain text string or a list of
            content part mappings for multimodal messages.
        name: Optional name identifier for the message sender, used to
            distinguish between multiple participants with the same role.
        function_call: Optional dictionary containing function call information
            when the assistant requests a function invocation (legacy format).

    Example:
        >>> msg = ChatMessage(role="user", content="What is the weather?")
    """

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(OpenAIBaseModel):
    """Represents an incremental change (delta) in a chat message during streaming.

    Used within ``ChatCompletionStreamResponseChoice`` to convey partial
    updates as they are generated. The first chunk typically includes the
    ``role``, and subsequent chunks contain incremental ``content``.

    Attributes:
        role: The message role, included only in the first chunk of a new
            message (e.g., ``"assistant"``). None for subsequent chunks.
        content: Incremental text content to append to the message being
            built, or a list of structured content parts. None when only
            metadata is being sent.
        function_call: Optional dictionary with incremental function call
            updates (legacy format). Used for streaming function name and
            arguments progressively.
    """

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class Function(OpenAIBaseModel):
    """Function definition for OpenAI-compatible tool/function calling.

    Describes a callable function that the model can invoke, including its
    name, purpose, and the JSON Schema for its parameters.

    Attributes:
        name: The unique name of the function, used by the model to identify
            which function to call.
        description: Optional human-readable description of what the function
            does, helping the model decide when to call it.
        parameters: JSON Schema object describing the function's accepted
            parameters, including types, descriptions, and required fields.

    Example:
        >>> func = Function(
        ...     name="get_weather",
        ...     description="Get current weather for a city",
        ...     parameters={"type": "object", "properties": {"city": {"type": "string"}}}
        ... )
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """Tool definition wrapping a function for model-driven invocation.

    Wraps a ``Function`` definition with a type discriminator, following
    the OpenAI tools API format where ``type`` is always ``"function"``.

    Attributes:
        type: The tool type identifier, currently always ``"function"``.
        function: The ``Function`` definition describing the callable tool.
    """

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    """Represents incremental updates to a function call during streaming.

    Carries partial function call data within streaming tool call chunks.
    The ``name`` is typically sent in the first chunk only, while
    ``arguments`` are progressively appended across multiple chunks.

    Attributes:
        name: The function name, included only in the first chunk of the
            tool call. None for subsequent chunks.
        arguments: Incremental JSON string fragment of the function arguments
            to append. None when no argument data is included in this chunk.
    """

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    """Represents incremental updates to a tool call during streaming.

    Used in streaming chat completion responses to progressively build up
    tool call information across multiple chunks. The ``id`` and ``type``
    are sent in the first chunk, while ``function`` data arrives
    incrementally.

    Attributes:
        id: Unique tool call identifier, included only in the first chunk.
            None for subsequent chunks.
        type: The tool type literal, always ``"function"`` when present.
            None for subsequent chunks after the first.
        index: Zero-based index of this tool call in the response's tool
            calls list. Always present to identify which tool call is
            being updated.
        function: Incremental ``DeltaFunctionCall`` with partial function
            name and/or arguments data. None when no function data is
            included in this chunk.
    """

    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
    """Token usage statistics and performance metrics for a request.

    Tracks computational resources consumed during a chat or text
    completion request, including token counts and timing information.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the response.
            May be None for streaming responses where the count is
            not yet finalized.
        total_tokens: Total token count (sum of prompt and completion tokens).
        tokens_per_second: Token generation throughput in tokens per second.
        processing_time: Total wall-clock processing time in seconds.
    """

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(OpenAIBaseModel):
    """Defines a function that can be called by the model (legacy format).

    Used within the deprecated ``functions`` field of ``ChatCompletionRequest``.
    For new implementations, prefer ``ToolDefinition`` with the ``tools`` field.

    Attributes:
        name: The unique function name used by the model for invocation.
        description: Optional human-readable description to help the model
            understand when and how to call this function.
        parameters: JSON Schema object describing the function's parameters,
            including property types, descriptions, and ``required`` fields.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolDefinition(OpenAIBaseModel):
    """Defines a tool that can be called by the model.

    Wraps a ``FunctionDefinition`` with a type discriminator, following the
    OpenAI tools API format for use in the ``tools`` field of
    ``ChatCompletionRequest``.

    Attributes:
        type: The tool type identifier, currently always ``"function"``.
        function: The ``FunctionDefinition`` describing the callable tool.
    """

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
    """Represents a single choice within a non-streaming chat completion response.

    Each choice contains a complete message and metadata about why generation
    stopped.

    Attributes:
        index: Zero-based index of this choice in the response's choices list.
        message: The complete ``ChatMessage`` generated for this choice.
        finish_reason: The reason generation stopped. One of ``"stop"`` (natural
            end), ``"length"`` (token limit reached), ``"function_call"`` (legacy
            function invocation), ``"tool_calls"`` (tool invocation requested),
            or ``"abort"`` (generation was aborted). None if not yet finished.
    """

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Represents a complete non-streaming response from the chat completion endpoint.

    Contains one or more choices (completions), usage statistics, and metadata
    about the request. Compatible with the OpenAI chat completion response format.

    Attributes:
        id: Unique identifier for this completion, auto-generated with a
            ``"chat-"`` prefix followed by a UUID hex string.
        object: Object type identifier, always ``"chat.completion"``.
        created: Unix timestamp (seconds) when the response was created.
        model: The model identifier that generated the response.
        choices: List of ``ChatCompletionResponseChoice`` objects, one per
            requested completion (controlled by the ``n`` parameter).
        usage: Token usage statistics for the request and response.
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming chat completion response chunk.

    Contains a delta (incremental update) rather than a complete message,
    allowing clients to progressively build the response.

    Attributes:
        index: Zero-based index of this choice in the chunk's choices list.
        delta: The ``DeltaMessage`` containing incremental content or role
            information for this chunk.
        finish_reason: The reason generation stopped, or None if generation
            is still in progress. One of ``"stop"``, ``"length"``, or
            ``"function_call"``.
    """

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Represents a single chunk in a streaming response from the chat completion endpoint.

    Sent as a Server-Sent Events (SSE) data payload during streaming. Each chunk
    contains incremental updates to one or more choices being generated.

    Attributes:
        id: Unique identifier for the streaming session, auto-generated with a
            ``"chat-"`` prefix followed by a UUID hex string.
        object: Object type identifier, always ``"chat.completion.chunk"``.
        created: Unix timestamp (seconds) when the chunk was created.
        model: The model identifier generating the streamed response.
        choices: List of ``ChatCompletionStreamResponseChoice`` objects with
            incremental delta updates.
        usage: Token usage statistics. May be populated only in the final chunk.
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo


class CountTokenRequest(OpenAIBaseModel):
    """Represents a request to the token counting endpoint.

    Used to count the number of tokens in a conversation or text string
    for a specific model, without generating a completion.

    Attributes:
        model: The model identifier to use for tokenization.
        conversation: The input to tokenize, either a plain text string
            or a list of ``ChatMessage`` objects representing a conversation.
    """

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
    """Log probability information for generated tokens.

    Contains per-token log probabilities and optionally the top alternative
    tokens with their probabilities, useful for analyzing model confidence
    and debugging generation behavior.

    Attributes:
        tokens: List of generated token strings.
        token_logprobs: Log probability of each generated token.
        top_logprobs: Optional list of dictionaries mapping alternative tokens
            to their log probabilities for each position.
        text_offset: Optional list of character offsets for each token in the
            original text.
    """

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a text completion response.

    Attributes:
        text: The generated text completion string.
        index: Zero-based index of this choice in the response's choices list.
        logprobs: Optional ``CompletionLogprobs`` with per-token log probability
            information, included when requested via the ``logprobs`` parameter.
        finish_reason: The reason generation stopped. One of ``"stop"`` (hit a
            stop sequence), ``"length"`` (reached max tokens), or
            ``"function_call"``. None if not yet finished.
    """

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Represents a complete non-streaming response from the text completions endpoint.

    Attributes:
        id: Unique identifier for this completion, auto-generated with a
            ``"cmpl-"`` prefix followed by a UUID hex string.
        object: Object type identifier, always ``"text_completion"``.
        created: Unix timestamp (seconds) when the response was created.
        model: The model identifier that generated the completion.
        choices: List of ``CompletionResponseChoice`` objects, one per
            requested completion.
        usage: Token usage statistics for the request and response.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """Represents a single choice within a streaming text completion response chunk.

    Attributes:
        index: Zero-based index of this choice in the chunk's choices list.
        text: Incremental text content for this chunk.
        logprobs: Optional ``CompletionLogprobs`` with per-token log probability
            information for the tokens in this chunk.
        finish_reason: The reason generation stopped, or None if generation is
            still in progress. One of ``"stop"``, ``"length"``, or
            ``"function_call"``.
    """

    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Represents a single chunk in a streaming response from the text completions endpoint.

    Sent as a Server-Sent Events (SSE) data payload during streaming text
    completion requests.

    Attributes:
        id: Unique identifier for the streaming session, auto-generated with a
            ``"cmpl-"`` prefix followed by a UUID hex string.
        object: Object type identifier, always ``"text_completion.chunk"``.
        created: Unix timestamp (seconds) when the chunk was created.
        model: The model identifier generating the streamed completion.
        choices: List of ``CompletionStreamResponseChoice`` objects with
            incremental text content.
        usage: Optional token usage statistics, typically populated only in
            the final chunk.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None


class FunctionCall(OpenAIBaseModel):
    """Represents a function call extracted from a model response.

    Contains the function name and a JSON-encoded arguments string as
    produced by the model during function/tool calling.

    Attributes:
        name: The name of the function to call, matching a function defined
            in the request's ``tools`` or ``functions`` list.
        arguments: JSON-encoded string of the function arguments as generated
            by the model. Must be parsed with ``json.loads()`` before use.
    """

    name: str
    arguments: str


class Function(OpenAIBaseModel):
    """Function definition used within ``ToolCall`` for response-side tool calls.

    This is the response-side counterpart to the request-side ``FunctionDefinition``.
    It appears within ``ToolCall`` objects in model responses to describe the
    function being invoked.

    Attributes:
        name: The function name matching a defined tool.
        description: Optional description of the function's purpose.
        parameters: JSON Schema object for the function's parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolCall(OpenAIBaseModel):
    """Represents a tool call made by the model in a response.

    Contains the full identification and function call details for a single
    tool invocation requested by the model.

    Attributes:
        id: Unique identifier for this tool call, used to correlate with
            the corresponding tool result message.
        type: The tool type, always ``"function"``.
        function: The ``FunctionCall`` containing the function name and
            JSON-encoded arguments string.
    """

    id: str
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(StrEnum):
    """Supported function call formats.

    Different models and frameworks use different formats for function calling.

    Attributes:
        OPENAI: OpenAI's standard format
        JSON_SCHEMA: Direct JSON schema format
        HERMES: Hermes model function calling format
        GORILLA: Gorilla model function calling format
        QWEN: Qwen's special token format (✿FUNCTION✿)
        NOUS: Nous XML-style wrapper format
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
