# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Message conversion utilities for the API server.

This module provides utilities for converting between OpenAI-format
messages and Xerxes's internal message format. It includes:
- Bidirectional message format conversion
- Support for system, user, and assistant message roles
- Proper handling of message content extraction

The conversion utilities enable seamless interoperability between
OpenAI-compatible API requests and Xerxes's agent processing pipeline.

Example:
    >>> from xerxes.api_server.converters import MessageConverter
    >>> from xerxes.types.oai_protocols import ChatMessage
    >>> messages = [ChatMessage(role="user", content="Hello")]
    >>> history = MessageConverter.convert_openai_to_xerxes(messages)
"""

from xerxes.types import MessagesHistory
from xerxes.types.messages import AssistantMessage, SystemMessage, UserMessage
from xerxes.types.oai_protocols import ChatMessage


class MessageConverter:
    """Converts between OpenAI and Xerxes message formats.

    Utility class providing static methods for bidirectional conversion
    between OpenAI-format ``ChatMessage`` objects and Xerxes's internal
    ``MessagesHistory`` format. Handles role mapping and content extraction
    for all supported message types (system, user, assistant).

    This converter is used internally by the API server routers to
    translate incoming OpenAI-compatible requests into the format
    expected by Xerxes's agent execution pipeline.

    Example:
        >>> from xerxes.api_server.converters import MessageConverter
        >>> from xerxes.types.oai_protocols import ChatMessage
        >>> msgs = [
        ...     ChatMessage(role="system", content="You are helpful."),
        ...     ChatMessage(role="user", content="Hello!"),
        ... ]
        >>> history = MessageConverter.convert_openai_to_xerxes(msgs)
        >>> len(history.messages)
        2
    """

    @staticmethod
    def convert_openai_to_xerxes(messages: list[ChatMessage]) -> MessagesHistory:
        """Convert a list of OpenAI-format messages to Xerxes's internal format.

        Iterates through each ``ChatMessage`` and maps it to the corresponding
        Xerxes message type based on the role field:

        - ``"system"`` -> ``SystemMessage``
        - ``"user"`` -> ``UserMessage``
        - ``"assistant"`` -> ``AssistantMessage``

        Any ``None`` content is converted to an empty string.

        Args:
            messages: List of OpenAI ``ChatMessage`` objects to convert. Each
                message must have a ``role`` field set to one of ``"system"``,
                ``"user"``, or ``"assistant"``.

        Returns:
            A ``MessagesHistory`` instance containing the converted messages
            in the same order as the input list.

        Raises:
            ValueError: If a message has an unrecognized role (i.e., not
                ``"system"``, ``"user"``, or ``"assistant"``).

        Example:
            >>> from xerxes.api_server.converters import MessageConverter
            >>> from xerxes.types.oai_protocols import ChatMessage
            >>> messages = [ChatMessage(role="user", content="Hi")]
            >>> history = MessageConverter.convert_openai_to_xerxes(messages)
            >>> history.messages[0].content
            'Hi'
        """
        xerxes_messages = []

        for msg in messages:
            content = str(msg.content) if msg.content else ""

            if msg.role == "system":
                xerxes_messages.append(SystemMessage(content=content))
            elif msg.role == "user":
                xerxes_messages.append(UserMessage(content=content))
            elif msg.role == "assistant":
                xerxes_messages.append(AssistantMessage(content=content))
            else:
                raise ValueError(f"Unknown message role: {msg.role}")

        return MessagesHistory(messages=xerxes_messages)
