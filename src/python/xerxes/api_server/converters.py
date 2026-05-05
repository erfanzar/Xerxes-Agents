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
"""Message format converters for the API server.

This module converts between OpenAI-style chat messages and Xerxes internal
message types.
"""

from xerxes.types import MessagesHistory
from xerxes.types.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from xerxes.types.oai_protocols import ChatMessage


class MessageConverter:
    """Converts OpenAI protocol messages to Xerxes internal message types."""

    @staticmethod
    def convert_openai_to_xerxes(messages: list[ChatMessage]) -> MessagesHistory:
        """Convert a list of OpenAI ChatMessages to a Xerxes MessagesHistory.

        Args:
            messages (list[ChatMessage]): IN: OpenAI-format messages with roles
                ``"system"``, ``"user"``, or ``"assistant"``. OUT: Mapped to Xerxes
                message types.

        Returns:
            MessagesHistory: OUT: The converted message history.

        Raises:
            ValueError: If an unsupported message role is encountered.
        """
        xerxes_messages: list[SystemMessage | UserMessage | AssistantMessage | ToolMessage] = []

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
