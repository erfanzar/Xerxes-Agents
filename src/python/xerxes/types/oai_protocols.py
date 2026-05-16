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
"""Pydantic models for the OpenAI-compatible HTTP API surface.

These models back the OpenAI-compatible endpoints (chat completions,
completions, token counting) the bridge and gateway expose. They accept
unknown fields via ``extra="allow"`` so forward-compatible clients can pass
additional parameters without failing validation.
"""

import time
import typing as tp
import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    """Pydantic base for OpenAI-compatible protocol objects.

    Permits unknown fields and lazily caches the set of declared field
    names (plus aliases) on first validation so request handlers can do quick
    membership checks without re-introspecting the model.

    Attributes:
        field_names: Cached set of declared field names plus aliases. Populated
            on first call to the wrapping model validator.
    """

    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """Validate and populate :attr:`field_names` from ``model_fields`` once.

        Returns the handler-validated result unchanged; the side effect is the
        first-call population of the class-level ``field_names`` cache.
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
    """One message in an OpenAI chat-completions ``messages`` array.

    Attributes:
        role: Role discriminator (``system`` / ``user`` / ``assistant`` /
            ``tool`` / ``function``).
        content: Plain text or a list of structured content parts.
        name: Optional speaker name (used by function-style messages).
        function_call: Optional legacy ``function_call`` payload.
    """

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(OpenAIBaseModel):
    """Streamed partial message delta in a chat-completion chunk.

    Attributes:
        role: Role for the first chunk; ``None`` thereafter.
        content: Incremental text fragment or structured parts.
        function_call: Streamed legacy ``function_call`` payload.
    """

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class Function(OpenAIBaseModel):
    """OpenAI legacy ``function`` definition.

    Attributes:
        name: Function identifier.
        description: Human-readable summary.
        parameters: JSON-schema dict describing the arguments.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """OpenAI ``tool`` definition wrapping a :class:`Function`.

    Attributes:
        type: Tool kind discriminator; always ``"function"`` today.
        function: The wrapped function definition.
    """

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    """Streamed delta for a function-call payload.

    Attributes:
        name: Function name fragment (typically only in the first chunk).
        arguments: Incremental JSON-argument fragment.
    """

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    """Streamed delta for one entry of the ``tool_calls`` list.

    Attributes:
        id: Tool-call id (first chunk only).
        type: Tool kind; always ``"function"`` today.
        index: Position in the ``tool_calls`` array; used to merge deltas.
        function: Function-call delta payload.
    """

    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
    """Token usage and timing report.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens emitted in the completion.
        total_tokens: Sum of prompt and completion tokens.
        tokens_per_second: Observed throughput.
        processing_time: Total wall-clock processing time in seconds.
    """

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(OpenAIBaseModel):
    """Function definition used by the modern ``tools`` field.

    Attributes:
        name: Function identifier.
        description: Human-readable summary.
        parameters: JSON-schema dict describing the arguments.
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolDefinition(OpenAIBaseModel):
    """Modern OpenAI tool definition wrapping a :class:`FunctionDefinition`.

    Attributes:
        type: Tool kind discriminator; always ``"function"`` today.
        function: The wrapped function definition.
    """

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(OpenAIBaseModel):
    """Request body for ``POST /v1/chat/completions``.

    Attributes:
        model: Target model identifier.
        messages: Conversation history.
        max_tokens: Maximum tokens to generate.
        presence_penalty: OpenAI presence penalty.
        frequency_penalty: OpenAI frequency penalty.
        repetition_penalty: Extra repetition penalty (vLLM extension).
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.
        top_k: Top-k sampling cutoff.
        min_p: Minimum probability cutoff.
        suppress_tokens: Token ids that the sampler must never emit.
        functions: Legacy function definitions.
        function_call: Legacy function-call selector.
        tools: Modern tool definitions.
        tool_choice: Modern tool-choice selector.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        stop: Stop sequence(s).
        logit_bias: Per-token logit bias map.
        user: End-user identifier for abuse tracking.
        chat_template_kwargs: Extra kwargs forwarded to the chat template.
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
    """One element of the ``choices`` array on a non-streaming chat response.

    Attributes:
        index: Choice index within the response.
        message: Assembled assistant message.
        finish_reason: Why generation stopped.
    """

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Non-streaming chat-completions response body.

    Attributes:
        id: Response identifier (auto-generated).
        object: API object discriminator (always ``"chat.completion"``).
        created: Unix timestamp at creation.
        model: Model that produced the response.
        choices: Generated choices.
        usage: Token usage report.
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """One element of the ``choices`` array on a streaming chat chunk.

    Attributes:
        index: Choice index within the response.
        delta: Streamed message delta.
        finish_reason: Final finish reason on the terminating chunk only.
    """

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """One SSE chunk in a streaming chat-completions response.

    Attributes:
        id: Response identifier (auto-generated, stable across chunks of a
            single response when the server reuses it).
        object: API object discriminator (always ``"chat.completion.chunk"``).
        created: Unix timestamp at chunk creation.
        model: Model that produced the response.
        choices: Streamed choice deltas.
        usage: Token usage (typically only on the final chunk).
    """

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo


class CountTokenRequest(OpenAIBaseModel):
    """Request body for the token-counting endpoint.

    Attributes:
        model: Model whose tokenizer should be used.
        conversation: Plain prompt string or full chat history.
    """

    model: str
    conversation: str | list[ChatMessage]


class CompletionRequest(OpenAIBaseModel):
    """Request body for ``POST /v1/completions`` (legacy text completion).

    Attributes:
        model: Target model identifier.
        prompt: Prompt text or list of prompts for batch generation.
        max_tokens: Maximum tokens to generate.
        presence_penalty: OpenAI presence penalty.
        frequency_penalty: OpenAI frequency penalty.
        repetition_penalty: Extra repetition penalty.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.
        top_k: Top-k sampling cutoff.
        min_p: Minimum probability cutoff.
        suppress_tokens: Token ids that must never be emitted.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        stop: Stop sequence(s).
        logit_bias: Per-token logit bias map.
        user: End-user identifier.
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
    """Per-token logprob report for a completion choice.

    Attributes:
        tokens: Decoded token strings.
        token_logprobs: Logprob of each emitted token.
        top_logprobs: Top-k alternatives per position.
        text_offset: Character offset of each token within the response text.
    """

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """One element of the ``choices`` array on a non-streaming completion.

    Attributes:
        text: Generated text.
        index: Choice index.
        logprobs: Optional logprob report.
        finish_reason: Why generation stopped.
    """

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Non-streaming completion response body.

    Attributes:
        id: Response identifier (auto-generated).
        object: API object discriminator (always ``"text_completion"``).
        created: Unix timestamp at creation.
        model: Model that produced the response.
        choices: Generated choices.
        usage: Token usage report.
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """One element of the ``choices`` array on a streaming completion chunk.

    Attributes:
        index: Choice index.
        text: Streamed text fragment.
        logprobs: Optional logprob report.
        finish_reason: Final finish reason on the terminating chunk only.
    """

    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """One SSE chunk in a streaming completion response.

    Attributes:
        id: Response identifier.
        object: API object discriminator (always ``"text_completion.chunk"``).
        created: Unix timestamp at chunk creation.
        model: Model that produced the response.
        choices: Streamed choice deltas.
        usage: Token usage (typically only on the final chunk).
    """

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None


class FunctionCall(OpenAIBaseModel):
    """Legacy assembled function-call payload.

    Attributes:
        name: Function identifier.
        arguments: JSON-encoded argument string.
    """

    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    """Assembled modern tool-call payload.

    Attributes:
        id: Tool-call identifier.
        type: Tool kind; always ``"function"`` today.
        function: Function-call payload.
    """

    id: str
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(StrEnum):
    """Wire format families used by self-hosted models for tool calls.

    Attributes:
        OPENAI: Structured OpenAI ``tool_calls`` payload.
        JSON_SCHEMA: Plain JSON object validating against a tool's schema.
        XML_TAG: ``<tool_call>...</tool_call>`` XML-tag form.
        GORILLA: Gorilla function-call format.
        QWEN: Qwen-family ``<tool_call>`` format.
    """

    OPENAI = "openai"
    JSON_SCHEMA = "json_schema"
    XML_TAG = "xml_tag"
    GORILLA = "gorilla"
    QWEN = "qwen"


class ExtractedToolCallInformation(OpenAIBaseModel):
    """Result returned by a server-side tool-call extractor.

    Attributes:
        tools_called: Whether any tool calls were detected.
        tool_calls: Parsed tool calls (may be empty).
        content: Non-tool text portion of the response, if any.
    """

    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None
