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
"""DingTalk (钉钉) channel adapter.

Supports inbound and outbound messaging via DingTalk webhooks.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DingTalkChannel(WebhookChannel):
    """Channel implementation for DingTalk."""

    name = "dingtalk"

    def __init__(
        self,
        webhook_url: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the DingTalk channel.

        Args:
            webhook_url (str): IN: DingTalk incoming webhook URL.
                OUT: stored for outbound message delivery.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.webhook_url = webhook_url
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a DingTalk webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        if not data:
            return []
        text = (data.get("text") or {}).get("content", "") or data.get("content", "")
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=data.get("senderId", "") or data.get("senderStaffId", ""),
                room_id=data.get("conversationId", ""),
                platform_message_id=str(data.get("msgId", "")),
                direction=MessageDirection.INBOUND,
                metadata={"sender_nick": data.get("senderNick", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send a text message via the DingTalk webhook.

        Args:
            message (ChannelMessage): IN: message to send. ``text`` is used
                as the message content.
        """
        body = {"msgtype": "text", "text": {"content": message.text}}
        http_post(self.webhook_url, json_body=body, http_client=self._http)
