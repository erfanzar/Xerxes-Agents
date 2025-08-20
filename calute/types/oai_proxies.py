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

import json
import re
import time
import typing as tp
import uuid
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
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


class DeltaMessage(BaseModel):
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


class UsageInfo(BaseModel):
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


class FunctionDefinition(BaseModel):
    """Defines a function that can be called by the model.

    Attributes:
        name: Function name
        description: Function description for the model
        parameters: JSON Schema for function parameters
        required: List of required parameter names
    """

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)
    required: list[str] | None = None


class ToolDefinition(BaseModel):
    """Defines a tool that can be called by the model."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """
    Represents a request to the chat completion endpoint.
    Mirrors the OpenAI ChatCompletion request structure.
    """

    # Core parameters
    model: str
    messages: list[ChatMessage]

    # Sampling parameters (mirroring OpenAI)
    max_tokens: int = 128
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    suppress_tokens: list[int] = Field(default_factory=list)
    # Added for potential EasyDeL support

    # OpenAI native parameters (some may be ignored by vInference)
    functions: list[FunctionDefinition] | None = None
    function_call: str | dict[str, tp.Any] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, tp.Any] | None = None
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] | None = None  # Ignored by EasyDeL
    user: str | None = None  # Ignored by EasyDeL
    chat_template_kwargs: dict[str, int | float | str | bool] | None = None


class ChatCompletionResponseChoice(BaseModel):
    """Represents a single choice within a non-streaming chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: tp.Literal["stop", "length", "function_call", "abort"] | None = None


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
    usage: UsageInfo  # Usage info might be included in chunks, often zero until the end


class CountTokenRequest(BaseModel):
    """Represents a request to the token counting endpoint."""

    model: str
    conversation: str | list[ChatMessage]  # Can count tokens for a string or a list of messages


class CompletionRequest(BaseModel):
    """
    Represents a request to the completions endpoint.
    Mirrors the OpenAI Completion request structure.
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
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionResponse(BaseModel):
    """Represents a response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


# New model for streaming completion choices (OAI compatible)
class CompletionStreamResponseChoice(BaseModel):
    """Represents a single choice within a streaming completion response chunk."""

    index: int
    text: str  # The delta text content
    logprobs: CompletionLogprobs | None = None
    finish_reason: tp.Literal["stop", "length", "function_call"] | None = None


class CompletionStreamResponse(BaseModel):
    """Represents a streaming response from the completions endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion.chunk"  # Correct object type for streaming
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamResponseChoice]  # Use the new streaming choice model
    usage: UsageInfo | None = None
    # Usage is often None until the final chunk in OAI


class FunctionCall(BaseModel):
    """Represents a function call in the OpenAI format."""

    name: str
    arguments: str  # JSON string of arguments


class Function(BaseModel):
    """Function definition for OpenAI-compatible function calling."""

    name: str
    description: str | None = None
    parameters: dict[str, tp.Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """Tool definition supporting function calling."""

    type: str = "function"
    function: Function


class ToolCall(BaseModel):
    """Represents a tool call in responses."""

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

    OPENAI = "openai"  # OpenAI's format
    JSON_SCHEMA = "json_schema"  # Direct JSON schema
    HERMES = "hermes"  # Hermes function calling format
    GORILLA = "gorilla"  # Gorilla function calling format
    QWEN = "qwen"  # Qwen's special token format
    NOUS = "nous"  # Nous XML-style format


@dataclass
class FunctionCallParser:
    """Parser for extracting function calls from generated text.

    Supports multiple function calling formats and can extract
    structured function calls from model outputs.

    Attributes:
        format: Function call format to parse
        strict: If True, require exact format matching

    Methods:
        parse: Extract function calls from text
    """

    format: FunctionCallFormat = FunctionCallFormat.OPENAI
    strict: bool = False  # If True, require exact format matching

    def parse(self, text: str) -> list[FunctionCall] | None:
        """Parse function calls from generated text.

        Args:
            text: Generated text containing function calls

        Returns:
            List of parsed FunctionCall objects, or None if no calls found

        Raises:
            ValueError: If format is unsupported
            json.JSONDecodeError: If strict mode and JSON is invalid
        """
        if self.format == FunctionCallFormat.OPENAI:
            return self._parse_openai_format(text)
        elif self.format == FunctionCallFormat.JSON_SCHEMA:
            return self._parse_json_schema_format(text)
        elif self.format == FunctionCallFormat.HERMES:
            return self._parse_hermes_format(text)
        elif self.format == FunctionCallFormat.GORILLA:
            return self._parse_gorilla_format(text)
        elif self.format == FunctionCallFormat.QWEN:
            return self._parse_qwen_format(text)
        elif self.format == FunctionCallFormat.NOUS:
            return self._parse_nous_format(text)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _parse_openai_format(self, text: str) -> list[FunctionCall] | None:
        """Parse OpenAI-style function calls."""
        function_calls = []

        # Look for function call patterns
        # Pattern 1: Direct JSON after specific markers
        patterns = [
            r"<function_call>\s*({.*?})\s*</function_call>",
            r"Function call:\s*({.*?})",
            r"```json\s*({.*?})\s*```",
            r'({.*?"name".*?"arguments".*?})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "name" in data and ("arguments" in data or "parameters" in data):
                        args = data.get("arguments", data.get("parameters", {}))
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        function_calls.append(FunctionCall(name=data["name"], arguments=args))
                except json.JSONDecodeError:
                    if self.strict:
                        raise
                    continue

        # Pattern 2: Natural language function calls
        if not function_calls and not self.strict:
            # Look for patterns like "call function_name with ..."
            nl_pattern = r"call\s+(\w+)\s+with\s+({.*?}|\(.*?\))"
            nl_matches = re.findall(nl_pattern, text, re.IGNORECASE | re.DOTALL)
            for name, args in nl_matches:
                try:
                    # Try to parse arguments
                    args = args.strip("()")
                    if args.startswith("{"):
                        args_dict = json.loads(args)
                    else:
                        # Simple key=value parsing
                        args_dict = {}
                        for pair in args.split(","):
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                args_dict[k.strip()] = v.strip().strip("\"'")

                    function_calls.append(FunctionCall(name=name, arguments=json.dumps(args_dict)))
                except Exception:
                    if self.strict:
                        raise
                    continue

        return function_calls if function_calls else None

    def _parse_json_schema_format(self, text: str) -> list[FunctionCall] | None:
        """Parse direct JSON schema format."""
        try:
            # Extract JSON from text
            json_match = re.search(r"{.*}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "function" in data:
                    func_data = data["function"]
                    return [
                        FunctionCall(
                            name=func_data["name"],
                            arguments=json.dumps(func_data.get("arguments", {})),
                        )
                    ]
                elif "name" in data:
                    return [
                        FunctionCall(
                            name=data["name"],
                            arguments=json.dumps(data.get("arguments", data.get("parameters", {}))),
                        )
                    ]
        except json.JSONDecodeError:
            if self.strict:
                raise
        return None

    def _parse_hermes_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Hermes-style function calls."""
        function_calls = []

        # Hermes format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                function_calls.append(
                    FunctionCall(
                        name=data["name"],
                        arguments=json.dumps(data.get("arguments", {})),
                    )
                )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_gorilla_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Gorilla-style function calls."""
        function_calls = []

        pattern = r"<<<(\w+)\((.*?)\)>>>"
        matches = re.findall(pattern, text)

        for name, args_str in matches:
            try:
                # Parse arguments
                args_dict = {}
                if args_str:
                    for arg in args_str.split(","):
                        if "=" in arg:
                            k, v = arg.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("\"'")
                            try:
                                v = json.loads(v)
                            except Exception:
                                pass
                            args_dict[k] = v

                function_calls.append(FunctionCall(name=name, arguments=json.dumps(args_dict)))
            except Exception:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_qwen_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Qwen-style function calls with special tokens."""
        function_calls = []

        # Look for Qwen special token patterns
        # Pattern: ✿FUNCTION✿: function_name\n✿ARGS✿: {...}
        pattern = r"✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*({.*?})"
        matches = re.findall(pattern, text, re.DOTALL)

        for name, args_str in matches:
            try:
                args_dict = json.loads(args_str)
                function_calls.append(
                    FunctionCall(
                        name=name,
                        arguments=json.dumps(args_dict),
                    )
                )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None

    def _parse_nous_format(self, text: str) -> list[FunctionCall] | None:
        """Parse Nous-style function calls with XML tags."""
        function_calls = []

        # Pattern: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    args = data.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    function_calls.append(
                        FunctionCall(
                            name=data["name"],
                            arguments=args,
                        )
                    )
            except json.JSONDecodeError:
                if self.strict:
                    raise
                continue

        return function_calls if function_calls else None
