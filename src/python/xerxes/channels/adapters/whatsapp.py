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

Targets the WhatsApp Business Cloud API on Meta's Graph endpoint. Parses
inbound webhook payloads (entry → changes → value → messages) and sends
``text`` messages through ``POST /<phone_number_id>/messages``. Meta's
``hub.verify_token`` handshake is the operator's responsibility — the
adapter only handles the JSON event format.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WhatsAppChannel(WebhookChannel):
    """WhatsApp Business Cloud API (Meta Graph) adapter."""

    name = "whatsapp"

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        http_client: tp.Any = None,
        api_version: str = "v18.0",
    ) -> None:
        """Build the channel.

        Args:
            access_token: Meta Graph API access token.
            phone_number_id: WhatsApp Business phone-number id; the leading
                path segment of every outbound API call.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
            api_version: Meta Graph API version. Defaults to ``"v18.0"``.
        """
        super().__init__()
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Walk the WhatsApp webhook envelope and emit one message per entry.

        The Cloud API can batch many messages into one webhook: each
        ``entry`` may have multiple ``changes``, and each change's ``value``
        may have multiple ``messages``. We unpack the full tree.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            Every parsed user message in delivery order; empty if the
            envelope contained none.
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
        """Send a ``text`` message via ``POST /<phone_number_id>/messages``.

        Args:
            message: Outbound message. ``room_id`` or ``channel_user_id``
                becomes the recipient phone number (E.164 without ``+``);
                ``text`` carries the body.
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
