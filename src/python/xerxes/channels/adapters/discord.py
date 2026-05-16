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
"""Discord channel adapter.

Authenticates with a bot token, parses inbound webhook payloads, and
sends replies via Discord's REST v10 ``messages`` endpoint. Webhook
authenticity (Ed25519 signature) is out of scope here — front the
webhook URL with a verifier if you need defence against forged inbound
traffic.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DiscordChannel(WebhookChannel):
    """Discord bot-token adapter using REST v10."""

    name = "discord"

    def __init__(self, bot_token: str, *, http_client: tp.Any = None) -> None:
        """Build the channel.

        Args:
            bot_token: Discord bot authentication token.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.bot_token = bot_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a Discord webhook payload into ``ChannelMessage`` instances.

        Tolerates two shapes — payloads with a nested ``message`` object
        and payloads where the message fields sit at the top level.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or an empty list if the payload
            decoded to an empty dict.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg = data.get("message") or data
        author = msg.get("author") or {}
        return [
            ChannelMessage(
                text=msg.get("content", ""),
                channel=self.name,
                channel_user_id=str(author.get("id", "")),
                room_id=str(msg.get("channel_id", data.get("channel_id", ""))),
                platform_message_id=str(msg.get("id", "")),
                direction=MessageDirection.INBOUND,
                metadata={"guild_id": msg.get("guild_id", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send one message via ``POST /channels/{channel_id}/messages``.

        Args:
            message: Outbound message. ``room_id`` is the channel id,
                ``text`` the content, and ``reply_to`` (when set) becomes a
                ``message_reference`` so the reply quotes the original.
        """
        url = f"https://discord.com/api/v10/channels/{message.room_id}/messages"
        body = {"content": message.text}
        if message.reply_to:
            body["message_reference"] = {"message_id": message.reply_to}
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bot {self.bot_token}"},
            http_client=self._http,
        )
