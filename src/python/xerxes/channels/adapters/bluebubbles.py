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
"""BlueBubbles channel adapter.

Connects to a self-hosted BlueBubbles server for iMessage bridging.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class BlueBubblesChannel(WebhookChannel):
    """Channel implementation for BlueBubbles iMessage bridge."""

    name = "bluebubbles"

    def __init__(
        self,
        server_url: str,
        password: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the BlueBubbles channel.

        Args:
            server_url (str): IN: base URL of the BlueBubbles server.
                OUT: stored with trailing slash removed.
            password (str): IN: server API password.
                OUT: stored for authenticating outbound requests.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.password = password
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a BlueBubbles webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg_data = data.get("data", data)
        text = msg_data.get("text") or msg_data.get("body") or ""
        chat = msg_data.get("chats", [{}])[0] if isinstance(msg_data.get("chats"), list) else msg_data.get("chat", {})
        handle = msg_data.get("handle") or {}
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=handle.get("address", ""),
                room_id=chat.get("guid", ""),
                platform_message_id=msg_data.get("guid", ""),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """Send a text message via the BlueBubbles API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is used
                as the target chat GUID and ``text`` as the message body.
        """
        url = f"{self.server_url}/api/v1/message/text?password={self.password}"
        body = {"chatGuid": message.room_id, "message": message.text, "method": "private-api"}
        http_post(url, json_body=body, http_client=self._http)
