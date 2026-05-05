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
"""Converters module for Xerxes.

Exports:
    - convert_openai_messages
    - convert_openai_tools
    - check_openai_fields_names
    - is_openai_field_name"""

from typing import Any

from .messages import AssistantMessage, ChatMessage, SystemMessage, ToolMessage, UserMessage
from .tool_calls import Tool


def convert_openai_messages(messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]) -> list[ChatMessage]:
    """Convert openai messages.

    Args:
        messages (list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[ChatMessage]: OUT: Result of the operation."""

    converted_messages: list[ChatMessage] = []
    """Convert openai messages.

    Args:
        messages (list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[ChatMessage]: OUT: Result of the operation."""
    """Convert openai messages.

    Args:
        messages (list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[ChatMessage]: OUT: Result of the operation."""
    for openai_message in messages:
        message_role = openai_message.get("role")
        message: ChatMessage
        if message_role == "user":
            message = UserMessage.from_openai(openai_message)
        elif message_role == "assistant":
            message = AssistantMessage.from_openai(openai_message)
        elif message_role == "tool":
            message = ToolMessage.from_openai(openai_message)
        elif message_role == "system":
            message = SystemMessage.from_openai(openai_message)
        else:
            raise ValueError(f"Unknown message role: {message_role}")
        converted_messages.append(message)
    return converted_messages


def convert_openai_tools(tools: list[dict[str, Any]]) -> list[Tool]:
    """Convert openai tools.

    Args:
        tools (list[dict[str, Any]]): IN: tools. OUT: Consumed during execution.
    Returns:
        list[Tool]: OUT: Result of the operation."""

    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    """Convert openai tools.

    Args:
        tools (list[dict[str, Any]]): IN: tools. OUT: Consumed during execution.
    Returns:
        list[Tool]: OUT: Result of the operation."""
    """Convert openai tools.

    Args:
        tools (list[dict[str, Any]]): IN: tools. OUT: Consumed during execution.
    Returns:
        list[Tool]: OUT: Result of the operation."""
    return converted_tools


def check_openai_fields_names(valid_fields_names: set[str], names: set[str]) -> None:
    """Check openai fields names.

    Args:
        valid_fields_names (set[str]): IN: valid fields names. OUT: Consumed during execution.
        names (set[str]): IN: names. OUT: Consumed during execution."""

    openai_valid_params = set()
    """Check openai fields names.

    Args:
        valid_fields_names (set[str]): IN: valid fields names. OUT: Consumed during execution.
        names (set[str]): IN: names. OUT: Consumed during execution."""
    """Check openai fields names.

    Args:
        valid_fields_names (set[str]): IN: valid fields names. OUT: Consumed during execution.
        names (set[str]): IN: names. OUT: Consumed during execution."""
    non_valid_params = set()

    for name in names:
        if name in valid_fields_names:
            continue
        elif name in _OPENAI_COMPLETION_FIELDS:
            openai_valid_params.add(name)
        else:
            non_valid_params.add(name)

    if openai_valid_params or non_valid_params:
        raise ValueError(
            "Invalid parameters passed to `ChatCompletionRequest.from_openai`:\n"
            f"OpenAI valid parameters but not in `ChatCompletionRequest`: {openai_valid_params}\n"
            f"Non valid parameters: {non_valid_params}"
        )


def is_openai_field_name(name: str) -> bool:
    """Check whether openai field name.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""

    return name in _OPENAI_COMPLETION_FIELDS
    """Check whether openai field name.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    """Check whether openai field name.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""


_OPENAI_COMPLETION_FIELDS: set[str] = {
    "messages",
    "model",
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
}
