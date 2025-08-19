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
"""Defines Pydantic models for the vSurge API, mimicking OpenAI's structure."""

import time
import typing as tp
import uuid
from collections.abc import Mapping

from pydantic import BaseModel, Field


class RequestChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""

    role: str
    content: str | list[Mapping[str, str]]
    name: str | None = None
    function_call: dict[str, tp.Any] | None = None


class DeltaMessage(BaseModel):
    """Represents a change (delta) in a chat message, used in streaming responses."""

    role: str | None = None
    content: str | list[Mapping[str, str]] | None = None
    function_call: dict[str, tp.Any] | None = None


class UsageInfo(BaseModel):
    """Provides information about token usage and processing time for a request."""

    completion_tokens: int | None = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: dict | None = 0
    prompt_tokens_details: dict | None = 0
    tokens_per_second: float = 0
    processing_time: float = 0.0


class FunctionDefinition(BaseModel):
    """Defines a function that can be called by the model."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)
    required: list[str] | None = None


class ToolDefinition(BaseModel):
    """Defines a tool that can be called by the model."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionResponseChoice(BaseModel):
    """Represents a single choice within a non-streaming chat completion response."""

    index: int
    message: RequestChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionResponse(BaseModel):
    """Represents a non-streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponseChoice(BaseModel):
    """Represents a single choice within a streaming chat completion response chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class ChatCompletionStreamResponse(BaseModel):
    """Represents a single chunk in a streaming response from the chat completion endpoint."""

    id: str = Field(default_factory=lambda: f"chat-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]
    usage: UsageInfo


class CountTokenRequest(BaseModel):
    """Represents a request to the token counting endpoint."""

    model: str
    conversation: str | list[RequestChatMessage]


class CompletionLogprobs(BaseModel):
    """Log probabilities for token generation."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] | None = None


class CompletionResponseChoice(BaseModel):
    """Represents a single choice within a completion response."""

    text: str
    index: int
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    """Represents a response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponseChoice(BaseModel):
    """Represents a single choice within a streaming completion response chunk."""

    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length"] | None = None


class CompletionStreamResponse(BaseModel):
    """Represents a streaming response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]
    usage: UsageInfo | None = None
