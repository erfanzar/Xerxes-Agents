# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""OpenAI format conversion utilities for Xerxes.

This module provides utilities for converting between OpenAI API formats
and Xerxes's internal message and tool representations. It includes:
- Message conversion from OpenAI format to Xerxes ChatMessage types
- Tool conversion from OpenAI format to Xerxes Tool types
- Field name validation for OpenAI compatibility checks

The converters ensure seamless interoperability with OpenAI-compatible
APIs and models while maintaining Xerxes's type safety.

Example:
    >>> from xerxes.types.converters import convert_openai_messages, convert_openai_tools
    >>> openai_messages = [{"role": "user", "content": "Hello"}]
    >>> xerxes_messages = convert_openai_messages(openai_messages)
    >>> openai_tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    >>> xerxes_tools = convert_openai_tools(openai_tools)
"""

from typing import Any

from .messages import AssistantMessage, ChatMessage, SystemMessage, ToolMessage, UserMessage
from .tool_calls import Tool


def convert_openai_messages(messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]]) -> list[ChatMessage]:
    """Convert a list of OpenAI-format message dictionaries to Xerxes ChatMessage objects.

    Iterates through each message dictionary and dispatches to the appropriate
    Xerxes message class (UserMessage, AssistantMessage, ToolMessage, or
    SystemMessage) based on the 'role' field.

    Args:
        messages: A list of message dictionaries in OpenAI chat completion format.
            Each dictionary must contain a 'role' key with one of the values:
            'user', 'assistant', 'tool', or 'system'. Additional keys depend
            on the role (e.g., 'content', 'tool_calls', 'tool_call_id').

    Returns:
        A list of typed Xerxes ChatMessage objects corresponding to the input
        messages, preserving order.

    Raises:
        ValueError: If a message has an unrecognized role value.

    Example:
        >>> openai_msgs = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> xerxes_msgs = convert_openai_messages(openai_msgs)
        >>> len(xerxes_msgs)
        3
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
    """Convert a list of OpenAI-format tool dictionaries to Xerxes Tool objects.

    Each tool dictionary is expected to follow the OpenAI tool definition format
    with a 'type' field (typically "function") and a 'function' field containing
    the function name, description, and JSON Schema parameters.

    Args:
        tools: A list of tool definition dictionaries in OpenAI format. Each
            dictionary should contain 'type' and 'function' keys matching
            the OpenAI tool specification.

    Returns:
        A list of Xerxes Tool instances, one per input dictionary.

    Example:
        >>> openai_tools = [{
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "Get weather info",
        ...         "parameters": {"type": "object", "properties": {}}
        ...     }
        ... }]
        >>> xerxes_tools = convert_openai_tools(openai_tools)
        >>> xerxes_tools[0].function.name
        'get_weather'
    """
    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    return converted_tools


def check_openai_fields_names(valid_fields_names: set[str], names: set[str]) -> None:
    """Validate that parameter names are recognized field names.

    Checks each name against two sets: the caller-provided ``valid_fields_names``
    and the internal ``_OPENAI_COMPLETION_FIELDS`` set of standard OpenAI chat
    completion parameters. Names that appear in neither set are flagged as invalid.

    The error message distinguishes between names that are valid OpenAI parameters
    (but not in the caller's valid set) and names that are entirely unrecognized,
    making it easier to diagnose configuration issues.

    Args:
        valid_fields_names: A set of field names that the caller considers valid
            for its specific context (e.g., fields on a custom request model).
        names: The set of field names to validate against both the caller's
            valid set and the OpenAI completion fields.

    Raises:
        ValueError: If any name is not found in ``valid_fields_names``. The error
            message separates OpenAI-valid-but-unsupported parameters from
            completely unrecognized parameters.

    Example:
        >>> valid = {"model", "messages", "temperature"}
        >>> check_openai_fields_names(valid, {"model", "temperature"})
        >>> check_openai_fields_names(valid, {"invalid_param"})
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
    """Check whether a name is a recognized OpenAI chat completion field.

    Looks up the given name in the internal ``_OPENAI_COMPLETION_FIELDS`` set,
    which contains all standard parameter names accepted by the OpenAI chat
    completion API (e.g., 'model', 'messages', 'temperature', 'tools').

    Args:
        name: The field name string to check against the known OpenAI
            completion parameter names.

    Returns:
        True if the name is a recognized OpenAI completion field name,
        False otherwise.

    Example:
        >>> is_openai_field_name("temperature")
        True
        >>> is_openai_field_name("not_a_real_field")
        False
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
