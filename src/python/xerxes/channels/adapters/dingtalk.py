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

Inbound: parses DingTalk's outgoing-webhook JSON. Outbound: posts plain
``text`` messages back to the configured incoming-webhook URL. Webhook
sign / keyword authentication is the operator's responsibility — the
adapter does not enforce it.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DingTalkChannel(WebhookChannel):
    """DingTalk webhook adapter."""

    name = "dingtalk"

    def __init__(
        self,
        webhook_url: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            webhook_url: DingTalk incoming-webhook URL (includes the
                ``access_token`` query parameter).
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.webhook_url = webhook_url
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a DingTalk outgoing-webhook payload into ``ChannelMessage``.

        Accepts both the official ``{"text": {"content": ...}}`` shape and
        the simplified top-level ``content`` shape used by some bridges.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or empty when the payload is empty.
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
        """Post ``message.text`` to the configured DingTalk webhook.

        DingTalk incoming webhooks are conversation-scoped, so ``room_id``
        is ignored — the URL itself determines the destination.
        """
        body = {"msgtype": "text", "text": {"content": message.text}}
        http_post(self.webhook_url, json_body=body, http_client=self._http)
