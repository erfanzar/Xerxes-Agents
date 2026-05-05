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

Connects to Discord via bot token for sending and receiving messages.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DiscordChannel(WebhookChannel):
    """Channel implementation for Discord."""

    name = "discord"

    def __init__(self, bot_token: str, *, http_client: tp.Any = None) -> None:
        """Initialize the Discord channel.

        Args:
            bot_token (str): IN: Discord bot authentication token.
                OUT: stored for authorizing API requests.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.bot_token = bot_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Discord webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
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
        """Send a message to a Discord channel.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is the
                target channel ID, ``text`` is the content, and ``reply_to``
                (if set) becomes a message reference.
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
