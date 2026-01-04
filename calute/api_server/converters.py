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


"""Message conversion utilities for the API server.

This module provides utilities for converting between OpenAI-format
messages and Calute's internal message format. It includes:
- Bidirectional message format conversion
- Support for system, user, and assistant message roles
- Proper handling of message content extraction

The conversion utilities enable seamless interoperability between
OpenAI-compatible API requests and Calute's agent processing pipeline.

Example:
    >>> from calute.api_server.converters import MessageConverter
    >>> from calute.types.oai_protocols import ChatMessage
    >>> messages = [ChatMessage(role="user", content="Hello")]
    >>> history = MessageConverter.convert_openai_to_calute(messages)
"""

from calute.types import MessagesHistory
from calute.types.messages import AssistantMessage, SystemMessage, UserMessage
from calute.types.oai_protocols import ChatMessage


class MessageConverter:
    """Converts between OpenAI and Calute message formats.

    Utility class providing static methods for bidirectional
    conversion between OpenAI-format ChatMessage objects and
    Calute's internal MessagesHistory format. Handles role
    mapping and content extraction for all supported message types.
    """

    @staticmethod
    def convert_openai_to_calute(messages: list[ChatMessage]) -> MessagesHistory:
        """Convert OpenAI messages to Calute format.

        Args:
            messages: List of OpenAI ChatMessage objects

        Returns:
            MessagesHistory with converted messages

        Raises:
            ValueError: If an unknown message role is encountered
        """
        calute_messages = []

        for msg in messages:
            content = str(msg.content) if msg.content else ""

            if msg.role == "system":
                calute_messages.append(SystemMessage(content=content))
            elif msg.role == "user":
                calute_messages.append(UserMessage(content=content))
            elif msg.role == "assistant":
                calute_messages.append(AssistantMessage(content=content))
            else:
                raise ValueError(f"Unknown message role: {msg.role}")

        return MessagesHistory(messages=calute_messages)
