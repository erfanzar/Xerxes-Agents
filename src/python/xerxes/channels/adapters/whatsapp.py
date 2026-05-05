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
"""WhatsApp channel adapter.

Connects to the WhatsApp Business API (via Meta Graph API) for messaging.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WhatsAppChannel(WebhookChannel):
    """Channel implementation for WhatsApp Business API."""

    name = "whatsapp"

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        http_client: tp.Any = None,
        api_version: str = "v18.0",
    ) -> None:
        """Initialize the WhatsApp channel.

        Args:
            access_token (str): IN: Meta Graph API access token.
                OUT: stored for API authorization.
            phone_number_id (str): IN: WhatsApp Business phone number ID.
                OUT: used in the API request path.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
            api_version (str): IN: Meta Graph API version. Defaults to "v18.0".
                OUT: used in the API URL.
        """
        super().__init__()
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a WhatsApp webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        out = []
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value") or {}
                for msg in value.get("messages", []):
                    text = (msg.get("text") or {}).get("body", "") or msg.get("button", {}).get("text", "")
                    out.append(
                        ChannelMessage(
                            text=text,
                            channel=self.name,
                            channel_user_id=msg.get("from", ""),
                            room_id=msg.get("from", ""),
                            platform_message_id=msg.get("id", ""),
                            direction=MessageDirection.INBOUND,
                        )
                    )
        return out

    async def _send_outbound(self, message):
        """Send a text message via the WhatsApp Business API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` or
                ``channel_user_id`` is the recipient, and ``text`` the body.
        """
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "to": message.room_id or message.channel_user_id,
            "type": "text",
            "text": {"body": message.text},
        }
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
