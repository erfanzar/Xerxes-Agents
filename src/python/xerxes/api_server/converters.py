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
"""Convert OpenAI-style chat messages into Xerxes internal types.

Used by both the standard and Cortex completion services to translate
incoming requests before handing them to the agent runtime.
"""

from xerxes.types import MessagesHistory
from xerxes.types.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from xerxes.types.oai_protocols import ChatMessage


class MessageConverter:
    """Namespace for OpenAI <-> Xerxes message translation helpers."""

    @staticmethod
    def convert_openai_to_xerxes(messages: list[ChatMessage]) -> MessagesHistory:
        """Map OpenAI chat messages to a Xerxes :class:`MessagesHistory`.

        Only the ``system``, ``user``, and ``assistant`` roles are
        recognised; anything else raises ``ValueError``. Tool/function
        messages are handled by the completion services, not here.
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
