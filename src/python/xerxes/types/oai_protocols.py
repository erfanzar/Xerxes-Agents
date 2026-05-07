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
"""Oai protocols module for Xerxes.

Exports:
    - OpenAIBaseModel
    - ChatMessage
    - DeltaMessage
    - Function
    - Tool
    - DeltaFunctionCall
    - DeltaToolCall
    - UsageInfo
    - FunctionDefinition
    - ToolDefinition
    - ... and 16 more."""

import time
import typing as tp
import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenAIBaseModel(BaseModel):
    """Extended BaseModel for OpenAI-compatible protocol objects.

    Allows extra fields via ``model_config`` and tracks declared field names
    as a class variable for validation.

    Attributes:
        field_names: Cached set of declared field names (including aliases) for
            this class. Lazily populated on first validation.

    Inherits from:
        BaseModel: Pydantic base for serialization and validation.
    """

    model_config = ConfigDict(extra="allow")
    field_names: tp.ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """Validate and normalize input data, building the field-names cache if needed."""
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
    """Chat message.

    Inherits from: OpenAIBaseModel

    Attributes:
        role (str): role.
        content (str | list[tp.Mapping[str, str]]): content.
        name (str | None): name.
        function_call (dict[str, tp.Any] | None): function call."""

    role: str
    content: str | list[tp.Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(OpenAIBaseModel):
    """Delta message.

    Inherits from: OpenAIBaseModel

    Attributes:
        role (str | None): role.
        content (str | list[tp.Mapping[str, str]] | None): content.
        function_call (dict[str, tp.Any] | None): function call."""

    role: str | None = None
    content: str | list[tp.Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class Function(OpenAIBaseModel):
    """Function.

    Inherits from: OpenAIBaseModel

    Attributes:
        name (str): name.
        description (str | None): description.
        parameters (dict[str, tp.Any]): parameters."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(OpenAIBaseModel):
    """Tool.

    Inherits from: OpenAIBaseModel

    Attributes:
        type (str): type.
        function (Function): function."""

    type: str = "function"
    function: Function


class DeltaFunctionCall(OpenAIBaseModel):
    """Delta function call.

    Inherits from: OpenAIBaseModel

    Attributes:
        name (str | None): name.
        arguments (str | None): arguments."""

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    """Delta tool call.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str | None): id.
        type (tp.Literal['function'] | None): type.
        index (int): index.
        function (DeltaFunctionCall | None): function."""

    id: str | None = None
    type: tp.Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class UsageInfo(OpenAIBaseModel):
    """Usage info.

    Inherits from: OpenAIBaseModel

    Attributes:
        prompt_tokens (int): prompt tokens.
        completion_tokens (int | None): completion tokens.
        total_tokens (int): total tokens.
        tokens_per_second (float): tokens per second.
        processing_time (float): processing time."""

    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(OpenAIBaseModel):
    """Function definition.

    Inherits from: OpenAIBaseModel

    Attributes:
        name (str): name.
        description (str | None): description.
        parameters (dict[str, tp.Any]): parameters."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class ToolDefinition(OpenAIBaseModel):
    """Tool definition.

    Inherits from: OpenAIBaseModel

    Attributes:
        type (str): type.
        function (FunctionDefinition): function."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(OpenAIBaseModel):
    """Chat completion request.

    Inherits from: OpenAIBaseModel

    Attributes:
        model (str): model.
        messages (list[ChatMessage]): messages.
        max_tokens (int): max tokens.
        presence_penalty (float): presence penalty.
        frequency_penalty (float): frequency penalty.
        repetition_penalty (float): repetition penalty.
        temperature (float): temperature.
        top_p (float): top p.
        top_k (int): top k.
        min_p (float): min p.
        suppress_tokens (list[int]): suppress tokens.
        functions (list[FunctionDefinition] | None): functions.
        function_call (str | dict[str, tp.Any] | None): function call.
        tools (list[ToolDefinition] | None): tools.
        tool_choice (str | dict[str, tp.Any] | None): tool choice.
        n (int | None): n.
        stream (bool | None): stream.
        stop (str | list[str] | None): stop.
        logit_bias (dict[str, float] | None): logit bias.
        user (str | None): user.
        chat_template_kwargs (dict[str, int | float | str | bool] | None): chat template kwargs."""

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
    """Chat completion response choice.

    Inherits from: OpenAIBaseModel

    Attributes:
        index (int): index.
        message (ChatMessage): message.
        finish_reason (tp.Literal['stop', 'length', 'function_call', 'tool_calls', 'abort'] | None): finish reason."""

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "tool_calls", "abort"] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Chat completion response.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str): id.
        object (str): object.
        created (int): created.
        model (str): model.
        choices (list[ChatCompletionResponseChoice]): choices.
        usage (UsageInfo): usage."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(OpenAIBaseModel):
    """Chat completion stream response choice.

    Inherits from: OpenAIBaseModel

    Attributes:
        index (int): index.
        delta (DeltaMessage): delta.
        finish_reason (tp.Literal['stop', 'length', 'function_call'] | None): finish reason."""

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Chat completion stream response.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str): id.
        object (str): object.
        created (int): created.
        model (str): model.
        choices (list[ChatCompletionStreamResponseChoice]): choices.
        usage (UsageInfo): usage."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo


class CountTokenRequest(OpenAIBaseModel):
    """Count token request.

    Inherits from: OpenAIBaseModel

    Attributes:
        model (str): model.
        conversation (str | list[ChatMessage]): conversation."""

    model: str
    conversation: str | list[ChatMessage]


class CompletionRequest(OpenAIBaseModel):
    """Completion request.

    Inherits from: OpenAIBaseModel

    Attributes:
        model (str): model.
        prompt (str | list[str]): prompt.
        max_tokens (int): max tokens.
        presence_penalty (float): presence penalty.
        frequency_penalty (float): frequency penalty.
        repetition_penalty (float): repetition penalty.
        temperature (float): temperature.
        top_p (float): top p.
        top_k (int): top k.
        min_p (float): min p.
        suppress_tokens (list[int]): suppress tokens.
        n (int | None): n.
        stream (bool | None): stream.
        stop (str | list[str] | None): stop.
        logit_bias (dict[str, float] | None): logit bias.
        user (str | None): user."""

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
    """Completion logprobs.

    Inherits from: OpenAIBaseModel

    Attributes:
        tokens (list[str]): tokens.
        token_logprobs (list[float]): token logprobs.
        top_logprobs (list[dict[str, float]] | None): top logprobs.
        text_offset (list[int] | None): text offset."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(OpenAIBaseModel):
    """Completion response choice.

    Inherits from: OpenAIBaseModel

    Attributes:
        text (str): text.
        index (int): index.
        logprobs (CompletionLogprobs | None): logprobs.
        finish_reason (tp.Literal['stop', 'length', 'function_call'] | None): finish reason."""

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(OpenAIBaseModel):
    """Completion response.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str): id.
        object (str): object.
        created (int): created.
        model (str): model.
        choices (list[CompletionResponseChoice]): choices.
        usage (UsageInfo): usage."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(OpenAIBaseModel):
    """Completion stream response choice.

    Inherits from: OpenAIBaseModel

    Attributes:
        index (int): index.
        text (str): text.
        logprobs (CompletionLogprobs | None): logprobs.
        finish_reason (tp.Literal['stop', 'length', 'function_call'] | None): finish reason."""

    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Completion stream response.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str): id.
        object (str): object.
        created (int): created.
        model (str): model.
        choices (list[CompletionStreamResponseChoice]): choices.
        usage (UsageInfo | None): usage."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None


class FunctionCall(OpenAIBaseModel):
    """Function call.

    Inherits from: OpenAIBaseModel

    Attributes:
        name (str): name.
        arguments (str): arguments."""

    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    """Tool call.

    Inherits from: OpenAIBaseModel

    Attributes:
        id (str): id.
        type (str): type.
        function (FunctionCall): function."""

    id: str
    type: str = "function"
    function: FunctionCall


class FunctionCallFormat(StrEnum):
    """Function call format.

    Inherits from: StrEnum
    """

    OPENAI = "openai"
    JSON_SCHEMA = "json_schema"
    HERMES = "hermes"
    GORILLA = "gorilla"
    QWEN = "qwen"
    NOUS = "nous"


class ExtractedToolCallInformation(OpenAIBaseModel):
    """Extracted tool call information.

    Inherits from: OpenAIBaseModel

    Attributes:
        tools_called (bool): tools called.
        tool_calls (list[ToolCall]): tool calls.
        content (str | None): content."""

    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None
