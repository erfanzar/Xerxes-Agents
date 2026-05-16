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
"""Conversions between OpenAI wire dicts and typed Xerxes message/tool objects.

Includes :func:`convert_openai_messages` (dispatches by role to the right
``ChatMessage`` subclass), :func:`convert_openai_tools`, and helpers that
validate that an incoming kwargs dict only contains recognised OpenAI
chat-completion parameter names.
"""

from typing import Any

from .messages import AssistantMessage, ChatMessage, SystemMessage, ToolMessage, UserMessage
from .tool_calls import Tool


def convert_openai_messages(messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]) -> list[ChatMessage]:
    """Convert OpenAI-style message dictionaries into Xerxes ``ChatMessage`` objects.

    Each message is classified by its ``role`` field and dispatched to the
    appropriate message constructor (``UserMessage``, ``AssistantMessage``, etc.).

    Args:
        messages: A list of message dictionaries conforming to the OpenAI
            ``messages`` API format.

    Returns:
        A list of Xerxes ``ChatMessage`` subclasses corresponding to each input
        message.

    Raises:
        ValueError: If a message has an unrecognized ``role`` value.
    """

    converted_messages: list[ChatMessage] = []
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
    """Convert OpenAI-style tool definitions into Xerxes ``Tool`` objects.

    Args:
        tools: A list of tool dictionaries in OpenAI's ``tools`` format.

    Returns:
        A list of Xerxes ``Tool`` instances.
    """

    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    return converted_tools


def check_openai_fields_names(valid_fields_names: set[str], names: set[str]) -> None:
    """Validate that field names are compatible with OpenAI completion APIs.

    Checks each name in *names* against the set of OpenAI completion parameters
    and the provided valid field names. Raises ``ValueError`` if any names are
    not recognized.

    Args:
        valid_fields_names: The set of field names that are valid for this context.
        names: The field names to validate.

    Raises:
        ValueError: If any field names are not in ``valid_fields_names`` or
            ``_OPENAI_COMPLETION_FIELDS``.
    """

    openai_valid_params = set()
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
    """Return whether *name* is a valid OpenAI completion parameter.

    Args:
        name: The field name to check.

    Returns:
        True if *name* is in ``_OPENAI_COMPLETION_FIELDS``, False otherwise.
    """

    return name in _OPENAI_COMPLETION_FIELDS


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
